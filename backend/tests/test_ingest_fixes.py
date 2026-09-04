import os
import uuid
import json
import struct
import shutil
import tempfile
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.main import app
from app.core.db import Base, get_db
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from worker.ingest import (
    strip_label_and_macro_images,
    extract_openslide_metadata,
    generate_dzi_pyramid,
    run_ingest
)
from worker.preprocess import run_preprocess
from worker.qc import run_qc
from worker.triage import run_triage
from worker.mitosis import run_mitosis
from worker.grading import run_grading

# Isolated in-memory DB for tests
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
def setup_db_override():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)

client = TestClient(app)

@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()


def test_issue_37_strip_label_and_macro_phi(tmp_path):
    """
    Issue #37: Verify label and macro IFDs are stripped and unlinked from TIFF/SVS files,
    scrubbing PHI bytes from storage. Verify clean slides return False (no stripping).
    """
    import tifffile

    slide_path = str(tmp_path / "test_phi_slide.tif")
    data_main = np.full((100, 100, 3), 128, dtype=np.uint8)
    data_label = np.full((30, 30, 3), 200, dtype=np.uint8) # PHI label
    data_macro = np.full((40, 40, 3), 250, dtype=np.uint8) # Macro image

    with tifffile.TiffWriter(slide_path) as tif:
        tif.write(data_main, description="Baseline WSI level 0")
        tif.write(data_label, description="label - patient MRN 12345")
        tif.write(data_macro, description="macro camera view")

    # Pre-strip check: 3 pages
    with tifffile.TiffFile(slide_path) as tif:
        assert len(tif.pages) == 3

    stripped = strip_label_and_macro_images(slide_path)
    assert stripped is True

    # Post-strip check: only 1 page remains, label and macro are gone
    with tifffile.TiffFile(slide_path) as tif:
        assert len(tif.pages) == 1
        assert "Baseline" in tif.pages[0].description
        assert "label" not in (tif.pages[0].description or "").lower()

    # Stripping an already-clean slide returns False
    stripped_again = strip_label_and_macro_images(slide_path)
    assert stripped_again is False


def test_issue_39_unreadable_file_fails_fast(tmp_path):
    """
    Issue #39: When OpenSlide and pyvips fail to open the file, fail fast and raise RuntimeError.
    Do NOT fabricate a flat pink 2048x2048 pyramid or default dimensions.
    """
    corrupt_file = str(tmp_path / "corrupt_slide.svs")
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_WSI_HEADER_CORRUPT_BYTES")

    # extract_openslide_metadata must raise RuntimeError
    with pytest.raises(RuntimeError) as exc_meta:
        extract_openslide_metadata(corrupt_file)
    assert "Failed to open slide" in str(exc_meta.value)

    # generate_dzi_pyramid must raise RuntimeError (no pink fallback!)
    out_dir = str(tmp_path / "dzi_out")
    with pytest.raises(RuntimeError) as exc_pyr:
        generate_dzi_pyramid(corrupt_file, out_dir)
    assert "Failed to generate pyramid" in str(exc_pyr.value)


def test_issue_38_missing_mpp_sets_needs_mpp_and_halts_pipeline(db_session, tmp_path):
    """
    Issue #38: Never guess 0.25 µm/px.
    If MPP is missing, set slide and case status to 'needs_mpp' and halt downstream stage chaining.
    All downstream workers must reject execution without valid MPP.
    """
    # Create a TIFF without OpenSlide MPP properties
    import tifffile
    slide_path = str(tmp_path / "slide_no_mpp.tif")
    img_data = np.full((128, 128, 3), 150, dtype=np.uint8)
    with tifffile.TiffWriter(slide_path) as tif:
        tif.write(img_data, description="Generic TIFF without MPP metadata")

    case_id = uuid.uuid4()
    slide_id = uuid.uuid4()

    case_obj = Case(id=case_id, created_by="pathologist_test", status="open")
    slide_obj = Slide(
        id=slide_id,
        case_id=case_id,
        gcs_uri_original=f"gs://test-bucket/{slide_id}.tif",
        status="ready"
    )
    stage_exec = StageExecution(
        case_id=case_id,
        stage="ingest",
        attempt=1,
        status="running",
        input_ref={"slide_id": str(slide_id), "gcs_uri_original": f"gs://test-bucket/{slide_id}.tif"}
    )
    db_session.add(case_obj)
    db_session.add(slide_obj)
    db_session.add(stage_exec)
    db_session.commit()

    # Mock OpenSlide opening a slide that lacks MPP properties
    mock_os = MagicMock()
    mock_os.dimensions = (1024, 1024)
    mock_os.properties = {"openslide.vendor": "test_scanner"} # No openslide.mpp-x or mpp-y

    with patch("worker.ingest.download_blob_to_filename", side_effect=lambda b, k, dest: shutil.copyfile(slide_path, dest)), \
         patch("openslide.OpenSlide", return_value=mock_os), \
         patch("worker.ingest.generate_dzi_pyramid", return_value="pyramid.dzi"), \
         patch("worker.ingest.upload_dzi_tree_to_gcs"), \
         patch("worker.ingest.upload_blob_from_bytes"), \
         patch("worker.ingest.upload_blob_from_file"):

        out_ref, model_versions = run_ingest(stage_exec, db_session)

    # 1. Slide status must be 'needs_mpp', MPP must NOT be defaulted to 0.25
    db_session.refresh(slide_obj)
    db_session.refresh(case_obj)
    assert slide_obj.status == "needs_mpp"
    assert case_obj.status == "needs_mpp"
    assert slide_obj.mpp_x is None
    assert slide_obj.mpp_y is None

    # 2. Downstream stage 'preprocess' must NOT be queued (halted!)
    prep_stage = db_session.scalars(
        select(StageExecution).where(StageExecution.case_id == case_id, StageExecution.stage == "preprocess")
    ).first()
    assert prep_stage is None, "Downstream stage 'preprocess' must not be created when MPP is missing"

    # 3. Downstream workers must fail fast if called while MPP is missing
    prep_exec = StageExecution(case_id=case_id, stage="preprocess", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError) as exc_prep:
        run_preprocess(prep_exec, db_session)
    assert "missing valid MPP" in str(exc_prep.value)

    qc_exec = StageExecution(case_id=case_id, stage="qc", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError) as exc_qc:
        run_qc(qc_exec, db_session)
    assert "missing valid MPP" in str(exc_qc.value)

    triage_exec = StageExecution(case_id=case_id, stage="triage", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError) as exc_tri:
        run_triage(triage_exec, db_session)
    assert "missing valid MPP" in str(exc_tri.value)

    mitosis_exec = StageExecution(case_id=case_id, stage="mitosis", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError) as exc_mit:
        run_mitosis(mitosis_exec, db_session)
    assert "missing valid MPP" in str(exc_mit.value)

    grading_exec = StageExecution(case_id=case_id, stage="grading", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError) as exc_grad:
        run_grading(grading_exec, db_session)
    assert "missing valid MPP" in str(exc_grad.value)


def test_issue_38_manual_mpp_update_endpoint(db_session):
    """
    Verify pathologist can update missing MPP via API endpoint, unblocking case status and queuing preprocess.
    """
    case_id = uuid.uuid4()
    slide_id = uuid.uuid4()

    case_obj = Case(id=case_id, created_by="path_001", status="needs_mpp")
    slide_obj = Slide(
        id=slide_id,
        case_id=case_id,
        gcs_uri_original=f"gs://test-bucket/{slide_id}.svs",
        status="needs_mpp",
        mpp_x=None,
        mpp_y=None
    )
    db_session.add(case_obj)
    db_session.add(slide_obj)
    db_session.commit()

    # Attempt with viewer role -> 403 Forbidden
    res_bad = client.patch(
        f"/api/v1/cases/{case_id}/slides/{slide_id}/mpp",
        json={"mpp_x": 0.50},
        headers={"X-User-Role": "viewer"}
    )
    assert res_bad.status_code == 403

    # Invalid MPP <= 0 -> 400
    res_neg = client.patch(
        f"/api/v1/cases/{case_id}/slides/{slide_id}/mpp",
        json={"mpp_x": -0.25},
        headers={"X-User-Role": "pathologist"}
    )
    assert res_neg.status_code == 400

    # Valid MPP update -> 200, slide status='ready', case status='open'
    with patch("app.core.cloud_tasks.dispatch_stage_task"):
        res = client.patch(
            f"/api/v1/cases/{case_id}/slides/{slide_id}/mpp",
            json={"mpp_x": 0.50, "mpp_y": 0.50},
            headers={"X-User-Role": "pathologist"}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["mpp_x"] == 0.50
    assert data["mpp_y"] == 0.50

    db_session.refresh(case_obj)
    db_session.refresh(slide_obj)
    assert slide_obj.status == "ready"
    assert case_obj.status == "open"
    assert slide_obj.mpp_x == 0.50

    # Verify preprocess stage was auto-queued
    prep = db_session.scalars(
        select(StageExecution).where(StageExecution.case_id == case_id, StageExecution.stage == "preprocess")
    ).first()
    assert prep is not None
    assert prep.status == "queued"


def test_issue_635_full_depth_pyramid_pregeneration(tmp_path):
    """
    Issue #635: Verify pyramid generation is not hard-capped at level 11 (min(12, dz.level_count)).
    Allows pregenerating all levels up to dz.level_count.
    """
    out_dir = str(tmp_path / "dzi_full_depth")
    files_dir = os.path.join(out_dir, "pyramid_files")

    # Mock DeepZoomGenerator to report 15 levels (levels 0..14)
    mock_dz = MagicMock()
    mock_dz.level_count = 15
    mock_dz.level_tiles = {lvl: (1, 1) for lvl in range(15)}
    mock_dz.get_tile.return_value = Image.new("RGB", (256, 256), color=(200, 100, 100))

    mock_slide = MagicMock()

    with patch("openslide.OpenSlide", return_value=mock_slide), \
         patch("openslide.deepzoom.DeepZoomGenerator", return_value=mock_dz):
        
        dzi_res = generate_dzi_pyramid("mock_slide.svs", out_dir)

    assert dzi_res.endswith("pyramid.dzi")
    assert os.path.exists(dzi_res), "DZI XML file must exist on disk"
    # Verify levels beyond 11 (e.g. 12, 13, 14) were generated
    for lvl in range(15):
        lvl_dir = os.path.join(files_dir, str(lvl))
        assert os.path.exists(lvl_dir), f"Level {lvl} directory should exist"
        assert os.path.exists(os.path.join(lvl_dir, "0_0.jpg"))
        assert os.path.exists(os.path.join(lvl_dir, "0_0.png"))


def test_base_mag_calculation_accuracy():
    """
    Verify base magnification calculation from MPP:
    0.25 um/px -> 40x objective
    0.50 um/px -> 20x objective
    1.00 um/px -> 10x objective
    """
    mock_slide = MagicMock()
    mock_slide.dimensions = (20000, 20000)
    mock_slide.properties = {
        "openslide.mpp-x": "0.25",
        "openslide.mpp-y": "0.25"
        # No openslide.objective-power
    }

    with patch("openslide.OpenSlide", return_value=mock_slide):
        meta_40x = extract_openslide_metadata("dummy.svs")
    assert meta_40x["base_mag"] == 40.0, f"Expected 40.0x for 0.25 um/px, got {meta_40x['base_mag']}"

    mock_slide.properties["openslide.mpp-x"] = "0.50"
    mock_slide.properties["openslide.mpp-y"] = "0.50"
    with patch("openslide.OpenSlide", return_value=mock_slide):
        meta_20x = extract_openslide_metadata("dummy.svs")
    assert meta_20x["base_mag"] == 20.0, f"Expected 20.0x for 0.50 um/px, got {meta_20x['base_mag']}"


def test_negative_and_zero_mpp_rejected_by_all_workers(db_session):
    """
    Ensure all downstream workers strictly reject non-positive MPP (<= 0).
    """
    case_id = uuid.uuid4()
    slide_id = uuid.uuid4()
    case_obj = Case(id=case_id, created_by="test", status="needs_mpp")
    slide_obj = Slide(
        id=slide_id,
        case_id=case_id,
        gcs_uri_original=f"gs://test/{slide_id}.svs",
        status="needs_mpp",
        mpp_x=-0.25,
        mpp_y=-0.25
    )
    db_session.add(case_obj)
    db_session.add(slide_obj)
    db_session.commit()

    exec_prep = StageExecution(case_id=case_id, stage="preprocess", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError, match="missing valid MPP"):
        run_preprocess(exec_prep, db_session)

    exec_qc = StageExecution(case_id=case_id, stage="qc", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError, match="missing valid MPP"):
        run_qc(exec_qc, db_session)

    exec_triage = StageExecution(case_id=case_id, stage="triage", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError, match="missing valid MPP"):
        run_triage(exec_triage, db_session)

    exec_mitosis = StageExecution(case_id=case_id, stage="mitosis", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError, match="missing valid MPP"):
        run_mitosis(exec_mitosis, db_session)

    exec_grading = StageExecution(case_id=case_id, stage="grading", attempt=1, status="running", input_ref={"slide_id": str(slide_id)})
    with pytest.raises(ValueError, match="missing valid MPP"):
        run_grading(exec_grading, db_session)


def test_mitosis_router_endpoints_reject_missing_mpp(db_session):
    """
    Verify candidate crop and HPF thumbnail endpoints reject with HTTP 400 when MPP is missing.
    """
    case_id = uuid.uuid4()
    slide_id = uuid.uuid4()
    case_obj = Case(id=case_id, created_by="path_001", status="needs_mpp")
    slide_obj = Slide(
        id=slide_id,
        case_id=case_id,
        gcs_uri_original=f"gs://test/{slide_id}.svs",
        status="needs_mpp",
        mpp_x=None,
        mpp_y=None
    )
    db_session.add(case_obj)
    db_session.add(slide_obj)
    db_session.commit()

    cand_id = str(uuid.uuid4())
    res_crop = client.get(
        f"/api/v1/stages/mitosis/{case_id}/candidates/{cand_id}/crop",
        headers={"X-User-Role": "pathologist"}
    )
    assert res_crop.status_code == 400
    assert "missing valid MPP" in res_crop.json()["detail"]

    res_hpf = client.get(
        f"/api/v1/stages/mitosis/{case_id}/hpfs/1/thumbnail",
        headers={"X-User-Role": "pathologist"}
    )
    assert res_hpf.status_code == 400
    assert "missing valid MPP" in res_hpf.json()["detail"]


def test_strip_label_and_macro_does_not_expand_truncated_file(tmp_path):
    """
    Verify strip_label_and_macro_images handles truncated files safely without writing beyond EOF.
    """
    import tifffile
    slide_path = str(tmp_path / "truncated_phi.tif")
    data_main = np.full((64, 64, 3), 120, dtype=np.uint8)
    data_label = np.full((32, 32, 3), 220, dtype=np.uint8)

    with tifffile.TiffWriter(slide_path) as tif:
        tif.write(data_main, description="Baseline Level 0")
        tif.write(data_label, description="label barcode 98765")

    orig_size = os.path.getsize(slide_path)
    res = strip_label_and_macro_images(slide_path)
    assert res is True
    post_size = os.path.getsize(slide_path)
    # File size must not have grown
    assert post_size <= orig_size

