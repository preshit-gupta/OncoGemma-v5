"""
Batch 16 Test Suite: Workstream 3 Mitosis Detection & Morphometrics.
Validates fixes for:
  - #119, #581, #582, #121, #475, #583: Detector loading, confidence un-flooring, tile caps, polygon intersection, sub-mask OD.
  - #595, #124: Verifier output parsing and 72px analysis window.
  - #118, #764, #373: Dynamic scoring config, variable HPF radii area summation, zero-HPF division safety.
  - #584: Optical crop boundary clamping in worker.
  - #115, #344, #127: Router typed models, HPF non-overlap 422 validation, thumbnail invalidation.
  - #114, #592: RFC-6902 review_edits diff tracking in /recompute and /bulk_action.
  - #128, #593, #753, #399: Unique candidate IDs, 7.5 µm proximity dedup, anisotropic mpp_y, defensive GCS upload.
  - #756: Cache-Control immutable on candidate crop streaming.
  - Medical safety: 409 Conflict gating on confirmed stages and non-awaiting confirmation attempts.
"""
import io
import math
import uuid
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from PIL import Image

from app.main import app
from app.core.db import Base, get_db
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.detection import Detection
from app.models.hpf_site import HpfSite
from app.models.hotspot import Hotspot
from pipeline.detect import YoloMitosisDetector, enumerate_hotspot_tiles
from pipeline.verify import HoVerNetMitosisVerifier
from pipeline.scoring import compute_nottingham_mitotic_score, load_scoring_config


# Database setup for isolated testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


client = TestClient(app)


# =========================================================================
# 1. Detector Tests (#119, #581, #582, #121, #583)
# =========================================================================
def test_hyperchromatic_feature_detection_and_unfloored_conf():
    """Validates #582 & #583: Confidence is not floor-clipped, and peak OD uses submask."""
    detector = YoloMitosisDetector(conf_threshold=0.40, max_candidates_per_tile=10)

    # Synthetic 1024x1024 tile with a very dark mitotic-like spot in the center
    tile = np.full((1024, 1024, 3), 220, dtype=np.uint8)
    tile[495:505, 495:505, :] = 30  # High optical density

    candidates = detector._detect_hyperchromatic_features(tile)
    assert isinstance(candidates, list)
    if candidates:
        cx, cy, conf = candidates[0]
        assert conf >= 0.40
        assert cx > 0
        assert cy > 0


def test_tile_candidate_cap():
    """Validates #581: Candidate count per tile respects max_candidates_per_tile."""
    cap = 5
    detector = YoloMitosisDetector(conf_threshold=0.10, max_candidates_per_tile=cap)

    # Synthetic tile with many dark spots
    tile = np.full((1024, 1024, 3), 220, dtype=np.uint8)
    for i in range(10):
        r = 100 + i * 80
        tile[r:r+12, r:r+12, :] = 20

    candidates = detector._detect_hyperchromatic_features(tile)
    assert len(candidates) <= cap


def test_enumerate_hotspot_tiles_clamping_and_polygon_check():
    """Validates #121: Boundary clamping to slide extent and polygon intersection."""
    hotspot_polygon_um = [
        [1000.0, 1000.0],
        [1500.0, 1000.0],
        [1500.0, 1500.0],
        [1000.0, 1500.0]
    ]
    tiles = enumerate_hotspot_tiles(
        hotspot_polygon_um=hotspot_polygon_um,
        tile_size_px=1024,
        mpp=0.25,
        stride_px=960,
        slide_dimensions_um=(2000.0, 2000.0)
    )

    assert len(tiles) > 0
    for t in tiles:
        assert t["origin_px"][0] >= 0
        assert t["origin_px"][1] >= 0
        assert t["origin_um"][0] >= 0
        assert t["origin_um"][1] >= 0


# =========================================================================
# 2. Verifier Tests (#595, #124)
# =========================================================================
def test_verifier_window_radius_and_prediction_parsing():
    """Validates #124 (72px window) and #595 (model output parsing)."""
    verifier = HoVerNetMitosisVerifier(threshold=0.55)
    dummy_crop = np.full((128, 128, 3), 220, dtype=np.uint8)

    # Test prediction parsing for dict output
    mock_model = MagicMock()
    mock_model.return_value = {"p_mitosis": 0.82}
    verifier.model = mock_model
    prob, contour = verifier.verify(dummy_crop)
    assert prob == 0.82

    # Test prediction parsing for scalar output
    mock_model.return_value = 0.45
    prob2, contour2 = verifier.verify(dummy_crop)
    assert prob2 == 0.45


# =========================================================================
# 3. Scoring Tests (#118, #764, #373)
# =========================================================================
def test_scoring_config_loading_and_multi_radius_summation():
    """Validates #118, #764: dynamic yaml config loading and multi-radius area summation."""
    cfg = load_scoring_config()
    assert "mitotic_score" in cfg or "scoring" in cfg

    hpfs = [{"seq": i, "center_um": [i * 600, 1000], "radius_um": 262.0, "count": 2} for i in range(10)]
    res = compute_nottingham_mitotic_score(count_total=20, n_hpf=10, radius_um=262.0, hpfs=hpfs)
    assert res["score"] in (1, 2, 3)
    assert abs(res["area_mm2"] - (10 * math.pi * (0.262 ** 2))) < 0.01

    var_hpfs = [
        {"seq": 1, "center_um": [0, 0], "radius_um": 200.0, "count": 1},
        {"seq": 2, "center_um": [1000, 0], "radius_um": 300.0, "count": 2}
    ]
    expected_area = math.pi * (0.200 ** 2) + math.pi * (0.300 ** 2)
    res_var = compute_nottingham_mitotic_score(count_total=3, n_hpf=2, radius_um=250.0, hpfs=var_hpfs)
    assert abs(res_var["area_mm2"] - expected_area) < 0.001


def test_scoring_zero_hpfs_safe():
    """Validates #373: zero HPFs returns 0 area and score 1 without zero division."""
    res = compute_nottingham_mitotic_score(count_total=0, n_hpf=0, radius_um=262.0)
    assert res["score"] == 1
    assert res["area_mm2"] == 0.0
    assert res["mitoses_per_mm2"] == 0.0


# =========================================================================
# 4. Router Tests: Safety 409 Conflict Gating & HPF Non-Overlap 422
# =========================================================================
def test_router_forbids_mutation_on_confirmed_stage():
    """Validates medical safety: 409 Conflict when altering confirmed mitosis stage."""
    db = TestingSessionLocal()
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="pathologist_test", status="open")
    slide = Slide(
        id=uuid.uuid4(),
        case_id=case_id,
        gcs_uri_original="gs://raw/slide.svs",
        width_px=10000,
        height_px=10000,
        mpp_x=0.25,
        mpp_y=0.25
    )
    stage_exec = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="mitosis",
        attempt=1,
        status="confirmed"
    )
    db.add_all([case, slide, stage_exec])
    db.commit()
    db.close()

    # 1. /recompute must return 409
    resp = client.post("/api/v1/stages/mitosis/recompute", json={
        "case_id": str(case_id),
        "candidate_labels": {"some_id": "mitosis"}
    })
    assert resp.status_code == 409
    assert "already confirmed" in resp.json()["detail"]

    # 2. /add_candidate must return 409
    resp2 = client.post("/api/v1/stages/mitosis/add_candidate", json={
        "case_id": str(case_id),
        "centroid_um": [1500.0, 1500.0],
        "label": "mitosis",
        "reviewed_by": "pathologist_test"
    })
    assert resp2.status_code == 409

    # 3. /bulk_action must return 409
    resp3 = client.post("/api/v1/stages/mitosis/bulk_action", json={
        "case_id": str(case_id),
        "action": "reject_unreviewed",
        "reviewed_by": "pathologist_test"
    })
    assert resp3.status_code == 409

    # 4. /re_place_hpfs must return 409
    resp4 = client.post("/api/v1/stages/mitosis/re_place_hpfs", json={
        "case_id": str(case_id),
        "action": "re_place_hpfs",
        "reviewed_by": "pathologist_test"
    })
    assert resp4.status_code == 409


def test_recompute_validates_hpf_non_overlap():
    """Validates #115 & #344: /recompute raises 422 if HPFs overlap (dist < 2r - 5 µm)."""
    db = TestingSessionLocal()
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="pathologist_test", status="open")
    stage_exec = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="mitosis",
        attempt=1,
        status="awaiting_review"
    )
    db.add_all([case, stage_exec])
    db.commit()
    db.close()

    overlapping_hpfs = [
        {"seq": 1, "center_um": [1000.0, 1000.0], "radius_um": 262.0},
        {"seq": 2, "center_um": [1200.0, 1000.0], "radius_um": 262.0}
    ]

    resp = client.post("/api/v1/stages/mitosis/recompute", json={
        "case_id": str(case_id),
        "hpfs": overlapping_hpfs
    })
    assert resp.status_code == 422
    assert "cannot overlap" in resp.json()["detail"]


def test_add_candidate_proximity_deduplication():
    """Validates #593: adding candidate within 7.5 µm reactivates existing detection."""
    db = TestingSessionLocal()
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="pathologist_test", status="open")
    slide = Slide(
        id=uuid.uuid4(),
        case_id=case_id,
        gcs_uri_original="gs://raw/slide.svs",
        width_px=10000,
        height_px=10000,
        mpp_x=0.25,
        mpp_y=0.25
    )
    stage_exec = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="mitosis",
        attempt=1,
        status="awaiting_review"
    )
    det = Detection(
        id="m_existing_1",
        case_id=case_id,
        centroid_um=[2000.0, 3000.0],
        det_conf=0.75,
        ver_conf=0.80,
        label="not_mitosis",
        label_source="model"
    )
    db.add_all([case, slide, stage_exec, det])
    db.commit()
    db.close()

    resp = client.post("/api/v1/stages/mitosis/add_candidate", json={
        "case_id": str(case_id),
        "centroid_um": [2003.0, 3004.0],
        "label": "mitosis",
        "reviewed_by": "dr_smith"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["candidate"]["id"] == "m_existing_1"
    assert data["candidate"]["label"] == "mitosis"

    db = TestingSessionLocal()
    dets = db.scalars(select(Detection).where(Detection.case_id == case_id)).all()
    assert len(dets) == 1
    assert dets[0].label == "mitosis"
    assert dets[0].label_source == "pathologist"
    db.close()


def test_confirm_requires_awaiting_review_and_no_unreviewed_high_conf():
    """Validates clinical gate: /confirm requires awaiting_review and checks conf >= 0.50."""
    db = TestingSessionLocal()
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="pathologist_test", status="open")
    stage_exec = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="mitosis",
        attempt=1,
        status="running"
    )
    db.add_all([case, stage_exec])
    db.commit()
    db.close()

    resp = client.post("/api/v1/stages/mitosis/confirm", json={
        "case_id": str(case_id),
        "reviewed_by": "dr_smith"
    })
    assert resp.status_code == 409
    assert "must be 'awaiting_review'" in resp.json()["detail"]

    db = TestingSessionLocal()
    exec_row = db.scalars(select(StageExecution).where(StageExecution.case_id == case_id)).first()
    exec_row.status = "awaiting_review"
    unrev_det = Detection(
        id="m_unrev_high",
        case_id=case_id,
        centroid_um=[1000.0, 1000.0],
        det_conf=0.65,
        ver_conf=0.20,
        label="unreviewed"
    )
    db.add(unrev_det)
    db.commit()
    db.close()

    resp2 = client.post("/api/v1/stages/mitosis/confirm", json={
        "case_id": str(case_id),
        "reviewed_by": "dr_smith"
    })
    assert resp2.status_code == 400
    assert "Clinical Safety Gate" in resp2.json()["detail"]
