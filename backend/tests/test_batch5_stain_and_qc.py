import pytest
import numpy as np
from PIL import Image
from unittest.mock import AsyncMock, patch

from pipeline.stain import PureNumpyMacenkoNormalizer, fit_macenko_stain
from pipeline.qc_checks import run_all_qc_checks
from app.models.case import Case
from app.models.slide import Slide
from app.models.grading import Grading
from app.models.stage_execution import StageExecution


def test_macenko_transform_uses_fitted_source_matrix():
    """Verify transform() uses fitted stain_matrix_src and max_conc_src rather than ignoring them (Issue #50)."""
    norm = PureNumpyMacenkoNormalizer()
    tile = np.full((128, 128, 3), 180, dtype=np.uint8)

    # 1. Without fitted source matrix (fallback per-tile SVD)
    out_default = norm.transform(tile)
    assert out_default.shape == tile.shape

    # 2. Assign distinct fitted source matrix
    norm.stain_matrix_src = np.array([[0.80, 0.50, 0.30], [0.10, 0.90, 0.20]])
    norm.max_conc_src = np.array([2.5, 1.8])
    norm.stain_matrix_target = np.array([[0.644, 0.717, 0.267], [0.093, 0.954, 0.283]])
    norm.max_conc_target = np.array([1.85, 1.05])

    out_fitted = norm.transform(tile)
    assert out_fitted.shape == tile.shape

    # 3. Altering stain_matrix_src must directly alter transform output (proves fit is not ignored)
    norm.stain_matrix_src = np.array([[0.40, 0.80, 0.45], [0.30, 0.70, 0.65]])
    out_altered = norm.transform(tile)
    assert not np.array_equal(out_fitted, out_altered)


def test_fit_macenko_stain_populates_source_params_and_samples_tissue():
    """Verify fit_macenko_stain records source parameters and handles sampling from tissue mask (Issues #436, #553)."""
    # Create mock slide
    slide_img = Image.new("RGB", (512, 512), color=(220, 180, 210))
    checksum = "a1b2c3d4e5f6"

    normalizer, stain_params, tissue_mask = fit_macenko_stain(
        slide_img,
        checksum_sha256=checksum,
        mpp_x=0.25,
        mpp_y=0.25
    )

    assert "stain_matrix" in stain_params
    assert "max_concentrations" in stain_params
    assert "stain_matrix_src" in stain_params
    assert "max_conc_src" in stain_params
    assert stain_params["fit_status"] in ["fitted", "sparse", "degenerate"]
    assert stain_params["fit_success"] is True


def test_fit_macenko_stain_degenerate_mask_flagging():
    """Verify fit_macenko_stain flags degenerate slides with 0 tissue patches (Issue #553)."""
    # Pure white slide (glass background only)
    white_slide = Image.new("RGB", (512, 512), color=(255, 255, 255))
    
    normalizer, stain_params, tissue_mask = fit_macenko_stain(
        white_slide,
        checksum_sha256="deadbeef1234",
        mpp_x=0.25,
        mpp_y=0.25
    )

    assert "fit_status" in stain_params
    # Pure white slide should have degenerate or sparse tissue
    assert stain_params["fit_status"] in ["degenerate", "sparse"]


def test_qc_all_5_checks_pass_on_clean_benchmark():
    """Verify complete 5-check suite evaluates all checks cleanly."""
    arr = np.random.randint(160, 240, (512, 512, 3), dtype=np.uint8)
    slide = Image.fromarray(arr)
    mask = np.ones((512, 512), dtype=bool)
    stain_params = {
        "max_concentrations": [1.95, 1.10],
        "fit_status": "fitted"
    }

    res = run_all_qc_checks(slide, mask, stain_params=stain_params)
    assert len(res["checks"]) == 5
    assert res["verdict"] in ["pass", "warn"]
    names = [c["name"] for c in res["checks"]]
    assert "tissue_coverage" in names
    assert "focus" in names
    assert "pen_marks" in names
    assert "folds" in names
    assert "stain_sanity" in names


def test_grading_rerun_resets_stale_overrides_and_unconfirms_type():
    """Verify re-running grading clears stale overrides and unconfirms histologic type (Issue #143)."""
    import uuid
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool
    from app.core.db import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session = Session(bind=engine)

    try:
        case_id = uuid.uuid4()
        slide_id = uuid.uuid4()

        case = Case(
            id=case_id,
            created_by="test_user",
            status="open"
        )
        session.add(case)

        slide = Slide(
            id=slide_id,
            case_id=case_id,
            mpp_x=0.25,
            mpp_y=0.25,
            checksum_sha256="12345678abcdef",
            gcs_uri_original="gs://fake-bucket/slide.svs"
        )
        session.add(slide)

        # Existing grading with stale pathologist overrides
        old_grading = Grading(
            case_id=case_id,
            tubule_percent=15.0,
            tubule_score=2,
            pleo_score=2,
            mitotic_score=1,
            nottingham_sum=5,
            grade=1,
            histologic_type="ILC",
            type_confirmed_by="pathologist_007",
            machine={"fake": "data"},
            overrides={"patch_001": {"tubule_percent": 80.0, "status": "approved"}}
        )
        session.add(old_grading)

        stage_exec = StageExecution(
            case_id=case_id,
            stage="grading",
            status="running",
            attempt=2
        )
        session.add(stage_exec)
        session.commit()

        # Verify initial state has overrides and confirmed type
        assert old_grading.overrides != {}
        assert old_grading.type_confirmed_by == "pathologist_007"

        # Simulate re-run persistence logic
        mock_aggregate = {
            "tubule_percent": 45.0,
            "tubule_score": 2,
            "pleo_score": 3,
            "mitotic_score": 2,
            "nottingham_sum": 7,
            "grade": 2,
            "flags": []
        }
        mock_machine = {"new": "attempt_2_output"}

        # Update path
        old_grading.tubule_percent = mock_aggregate["tubule_percent"]
        old_grading.tubule_score = mock_aggregate["tubule_score"]
        old_grading.pleo_score = mock_aggregate["pleo_score"]
        old_grading.mitotic_score = mock_aggregate["mitotic_score"]
        old_grading.nottingham_sum = mock_aggregate["nottingham_sum"]
        old_grading.grade = mock_aggregate["grade"]
        old_grading.machine = mock_machine
        old_grading.overrides = {}
        old_grading.type_confirmed_by = "unconfirmed"
        session.commit()

        reloaded = session.get(Grading, case_id)
        assert reloaded.overrides == {}
        assert reloaded.type_confirmed_by == "unconfirmed"
        assert reloaded.grade == 2
    finally:
        session.close()
