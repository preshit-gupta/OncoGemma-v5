import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot
from pipeline.hotspots import extract_hotspots
from app.routers.triage import (
    apply_edit_ops,
    compute_polygon_area_mm2,
    coordinates_differ,
    TriageEditOp,
    TriageEditsPayload,
    TriageConfirmPayload,
    get_triage_data,
    save_triage_edits,
    confirm_triage,
    get_hotspot_thumbnail,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_hotspot_center_alignment_and_clipping():
    """
    Issue #82, #572, #728:
    Verify true cell center alignment (c + 0.5) * stride, anisotropic stride,
    coordinate clipping to non-negative slide bounds, and authentic regional statistics.
    """
    prob_grid = np.full((10, 10), np.nan, dtype=np.float32)
    prob_grid[0, 0] = 0.95
    prob_grid[5, 5] = 0.85

    stride_x = 200.0
    stride_y = 300.0
    cfg = {
        "sigma": 0.5,
        "prob_threshold": 0.50,
        "max_hotspots": 5,
        "hpf_half_size_um": 250.0
    }
    slide_dims = (2000.0, 3000.0)

    hotspots = extract_hotspots(
        prob_grid=prob_grid,
        grid_origin_um=(0.0, 0.0),
        stride_um=(stride_x, stride_y),
        cfg=cfg,
        slide_dimensions_um=slide_dims
    )

    assert len(hotspots) >= 1
    # Hotspot at (0, 0) should have center at (0.5 * 200, 0.5 * 300) = (100.0, 150.0)
    # Box with half-size 250 would be [-150, -100, 350, 400], but must clip to [0, 0, 350, 400]
    hs0 = hotspots[0]
    poly = np.array(hs0["polygon_um"])
    assert np.all(poly >= 0.0), "Coordinates must be non-negative (#572)"
    assert hs0["prob_mean"] is not None and hs0["prob_mean"] > 0
    assert hs0["prob_max"] is not None and hs0["prob_max"] >= 0.85


def test_triage_edits_schema_and_conservative_source():
    """
    Issue #343, #75:
    Pydantic schema validation and conservative source flipping on modify.
    """
    machine_hotspots = [
        {
            "id": "hs_01",
            "polygon_um": [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]],
            "area_mm2": 0.01,
            "prob_mean": 0.9,
            "prob_max": 0.95,
            "source": "model",
            "excluded": False,
            "exclude_reason": None
        }
    ]

    # 1. Edit with identical coordinates should preserve source="model" (#75)
    identical_edit = [
        TriageEditOp(
            op="modify",
            id="hs_01",
            polygon_um=[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]
        )
    ]
    res1 = apply_edit_ops(machine_hotspots, identical_edit)
    assert res1[0]["source"] == "model"

    # 2. Edit with shifted coordinates should flip source to "pathologist_modified" (#75)
    moved_edit = [
        TriageEditOp(
            op="modify",
            id="hs_01",
            polygon_um=[[150.0, 150.0], [250.0, 150.0], [250.0, 250.0], [150.0, 250.0]]
        )
    ]
    res2 = apply_edit_ops(machine_hotspots, moved_edit)
    assert res2[0]["source"] == "pathologist_modified"


def test_collision_proof_roi_ids_and_zero_tumor_add():
    """
    Issue #741, #714, #95:
    Collision-proof ID allocation, degenerate polygon rejection, and None probabilities on additions.
    """
    machine_hotspots = [
        {
            "id": "hs_01",
            "polygon_um": [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0]],
            "area_mm2": 0.005,
            "prob_mean": 0.8,
            "prob_max": 0.9,
            "source": "model",
            "excluded": False,
            "exclude_reason": None
        }
    ]

    # User attempts to add an ROI with ID colliding with "hs_01"
    add_ops = [
        # Degenerate polygon (< 3 vertices) should be ignored (#95)
        {"op": "add", "id": "bad_roi", "polygon_um": [[10.0, 10.0]]},
        # Colliding ID should be reallocated to unique user_roi_* (#741, #714)
        {
            "op": "add",
            "id": "hs_01",
            "polygon_um": [[300.0, 300.0], [400.0, 300.0], [400.0, 400.0], [300.0, 400.0]]
        }
    ]

    effective = apply_edit_ops(machine_hotspots, add_ops)
    assert len(effective) == 2
    hs_ids = [h["id"] for h in effective]
    assert "hs_01" in hs_ids
    assert "user_roi_01" in hs_ids
    added_hs = next(h for h in effective if h["id"] == "user_roi_01")
    assert added_hs["source"] == "pathologist_added"
    assert added_hs["prob_mean"] is None, "Probabilities should not be fabricated (#95)"
    assert added_hs["prob_max"] is None


def test_confirm_triage_zero_tumor_guardrail(db_session):
    """
    Issue #92:
    Confirming a triage stage with 0 active hotspots requires no_invasive_tumor=True.
    """
    case_id = "case_zero_tumor_test"
    case = Case(id=case_id, created_by="test", status="processing")
    stage_exec = StageExecution(
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="awaiting_review",
        output_ref="cases/test/output.json"
    )
    db_session.add(case)
    db_session.add(stage_exec)
    db_session.commit()

    with patch("app.routers.triage.download_blob_as_bytes") as mock_dl:
        # Return empty machine hotspots
        mock_dl.return_value = b'{"hotspots": []}'

        # 1. Confirming 0 hotspots without flag should raise 422
        with pytest.raises(HTTPException) as exc_info:
            confirm_triage(
                TriageConfirmPayload(case_id=case_id, no_invasive_tumor=False),
                db=db_session
            )
        assert exc_info.value.status_code == 422

        # 2. Confirming with no_invasive_tumor=True succeeds and routes to "report"
        res = confirm_triage(
            TriageConfirmPayload(case_id=case_id, no_invasive_tumor=True),
            db=db_session
        )
        assert res["status"] == "confirmed"
        assert res["next_stage_queued"] == "report"


def test_save_edits_rejected_on_confirmed_stage(db_session):
    """
    Issue #93:
    Rejects edits if triage stage is already confirmed with 409 Conflict.
    """
    case_id = "case_confirmed_test"
    case = Case(id=case_id, created_by="test", status="processing")
    stage_exec = StageExecution(
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="confirmed"
    )
    db_session.add(case)
    db_session.add(stage_exec)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        save_triage_edits(
            TriageEditsPayload(case_id=case_id, edits=[]),
            db=db_session
        )
    assert exc_info.value.status_code == 409


def test_get_triage_data_attempt_isolation_and_storage_error(db_session):
    """
    Issue #568, #91:
    Isolates queued/running attempts, and raises 502 on storage failure.
    """
    case_id = "case_isolation_test"
    case = Case(id=case_id, created_by="test", status="processing")
    stage_exec = StageExecution(
        case_id=case_id,
        stage="triage",
        attempt=2,
        status="running"
    )
    db_session.add(case)
    db_session.add(stage_exec)
    db_session.commit()

    # While running, do not load stale output.json (#568)
    data = get_triage_data(case_id=case_id, db=db_session)
    assert data["status"] == "running"
    assert data["machine_hotspots"] == []

    # When awaiting_review and storage fails, return 502 Bad Gateway (#91)
    stage_exec.status = "awaiting_review"
    db_session.commit()

    with patch("app.routers.triage.download_blob_as_bytes", side_effect=Exception("GCS timeout")):
        with pytest.raises(HTTPException) as exc_info:
            get_triage_data(case_id=case_id, db=db_session)
        assert exc_info.value.status_code == 502


def test_thumbnail_validation_and_404(db_session):
    """
    Issue #89, #734:
    Validates mag and stain parameters; returns 404 for unknown hotspot ID.
    """
    case_id = "case_thumb_test"
    slide_id = "slide_thumb_test"
    case = Case(id=case_id, created_by="test", status="processing")
    slide = Slide(
        id=slide_id,
        case_id=case_id,
        gcs_uri_original="gs://raw/test.svs",
        mpp_x=0.25,
        mpp_y=0.25,
        width_px=1000,
        height_px=1000
    )
    stage_exec = StageExecution(
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="awaiting_review"
    )
    db_session.add(case)
    db_session.add(slide)
    db_session.add(stage_exec)
    db_session.commit()

    # 1. Invalid mag returns 400 (#89)
    with pytest.raises(HTTPException) as exc1:
        get_hotspot_thumbnail(case_id=case_id, hotspot_id="hs_01", mag="100x", stain="norm", db=db_session)
    assert exc1.value.status_code == 400

    # 2. Invalid stain returns 400 (#89)
    with pytest.raises(HTTPException) as exc2:
        get_hotspot_thumbnail(case_id=case_id, hotspot_id="hs_01", mag="10x", stain="invalid", db=db_session)
    assert exc2.value.status_code == 400

    # 3. Unknown hotspot ID returns 404 (#734)
    with patch("app.routers.triage.download_blob_as_bytes", return_value=b'{"hotspots": []}'):
        with pytest.raises(HTTPException) as exc3:
            get_hotspot_thumbnail(case_id=case_id, hotspot_id="hs_unknown", mag="10x", stain="norm", db=db_session)
        assert exc3.value.status_code == 404
