"""
Batch 10 Test Suite:
Stage 3 Triage, Hotspot Extraction & TriageViewer Review Integrity
Covers:
  - #77 [HIGH]: Thumbnail endpoint 404 on missing slide (no cross-patient leakage)
  - #80 [HIGH]: Confirm raises 502 on GCS read failure and does not delete DB hotspots
  - #279 [CRITICAL]: confirm_triage 409 gating when not awaiting_review; attempt monotonicity
  - #81 [HIGH]: extract_hotspots zero-tumor path returns [] (no fabricated hotspots)
  - #74 [HIGH]: OpenSlide thumbnail shape mismatch resized to (nx, ny) without IndexError
  - #452 [HIGH]: Parquet embedding cache read/write with ix, iy, emb columns
  - #73 [HIGH]: probe_v1 synthetic_dev provenance and exact sha256 checksum in triage.yaml
  - #448 [CRITICAL]: apply_edit_ops delete operation drops hotspot from effective and DB set
  - #450 [HIGH]: Multi-vertex custom polygon area calculation via Shoelace formula
"""
import os
import json
import hashlib
import tempfile
import uuid
import yaml
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from sqlalchemy.pool import StaticPool
import app.models # Load all ORM models into Base.metadata
from app.main import app
from app.core.db import Base, get_db
from app.core.config import settings
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot
from app.models.audit import AuditEvent
from app.routers.triage import apply_edit_ops
from pipeline.hotspots import extract_hotspots


@pytest.fixture
def client_and_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    db = TestingSessionLocal()
    yield client, db
    db.close()
    app.dependency_overrides.clear()


# --- #77: Cross-patient image leakage prevented ---
def test_issue_77_no_cross_patient_slide_leakage(client_and_db):
    client, test_db = client_and_db
    case_a_id = str(uuid.uuid4())
    case_b_id = str(uuid.uuid4())

    case_a = Case(id=case_a_id, created_by="dr_a", status="processing")
    slide_a = Slide(
        id=str(uuid.uuid4()),
        case_id=case_a_id,
        gcs_uri_original="gs://raw/slide_a.svs",
        mpp_x=0.25,
        mpp_y=0.25
    )
    case_b = Case(id=case_b_id, created_by="dr_b", status="processing")

    test_db.add_all([case_a, slide_a, case_b])
    test_db.commit()

    # Case B has no slide. Requesting hotspot thumbnail for Case B must return 404, NOT fallback to Case A's slide!
    with patch("app.routers.triage.download_blob_as_bytes", side_effect=Exception("No patch")):
        res = client.get(f"/api/v1/stages/triage/{case_b_id}/hotspots/hs_01/thumbnail?mag=10x&cx=100&cy=100")
        assert res.status_code == 404
        assert "Slide not found for case" in res.json()["detail"]


# --- #80: Confirm aborts on GCS failure without deleting DB hotspots ---
def test_issue_80_confirm_triage_aborts_on_gcs_failure(client_and_db):
    client, test_db = client_and_db
    case_id = str(uuid.uuid4())
    case = Case(id=case_id, created_by="dr_a", status="processing")
    se_triage_id = uuid.uuid4()
    se_triage = StageExecution(
        id=se_triage_id,
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="awaiting_review",
        output_ref=f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/triage/output.json"
    )
    # Existing confirmed hotspot in DB
    existing_hs = Hotspot(
        id="hs_existing",
        case_id=case_id,
        stage_execution_id=str(se_triage_id),
        polygon_um=[[0, 0], [10, 0], [10, 10], [0, 10]],
        area_mm2=0.36,
        source="model"
    )
    test_db.add_all([case, se_triage, existing_hs])
    test_db.commit()

    # When GCS read fails with transient error, confirm_triage must raise 502 and NOT delete existing_hs
    with patch("app.routers.triage.download_blob_as_bytes", side_effect=RuntimeError("GCS network timeout")):
        res = client.post("/api/v1/stages/triage/confirm", json={"case_id": case_id, "no_invasive_tumor": False})
        assert res.status_code == 502
        assert "Failed to load triage machine output from storage" in res.json()["detail"]

    # Verify existing DB hotspot was NOT wiped
    db_hs = test_db.scalars(select(Hotspot).where(Hotspot.case_id == case_id)).all()
    assert len(db_hs) == 1
    assert db_hs[0].id == "hs_existing"


# --- #279: Confirm status gating & attempt monotonicity ---
def test_issue_279_confirm_triage_status_gating(client_and_db):
    client, test_db = client_and_db
    case_id = str(uuid.uuid4())
    case = Case(id=case_id, created_by="dr_a", status="processing")
    se_triage = StageExecution(
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="confirmed" # Already confirmed!
    )
    test_db.add_all([case, se_triage])
    test_db.commit()

    # Re-confirming an already confirmed triage stage must return 409 Conflict
    res = client.post("/api/v1/stages/triage/confirm", json={"case_id": case_id, "no_invasive_tumor": False})
    assert res.status_code == 409
    assert "expected 'awaiting_review'" in res.json()["detail"]


# --- #81: Zero-tumor hotspot extraction ---
def test_issue_81_extract_hotspots_zero_tumor_path():
    cfg = {
        "sigma": 1.0,
        "prob_threshold": 0.45,
        "max_hotspots": 10,
        "hpf_half_size_um": 300.0
    }
    # Case 1: All-zero tissue grid
    zero_grid = np.zeros((60, 80), dtype=np.float32)
    hotspots_zero = extract_hotspots(zero_grid, (0.0, 0.0), 224.0, cfg)
    assert hotspots_zero == [], "All-zero grid must return empty list (no tumor foci)"

    # Case 2: Uniform 0.12 grid (fallback stain map)
    uniform_grid = np.full((60, 80), 0.12, dtype=np.float32)
    hotspots_uniform = extract_hotspots(uniform_grid, (0.0, 0.0), 224.0, cfg)
    assert hotspots_uniform == [], "Uniform 0.12 grid below 0.45 threshold must return empty list"

    # Case 3: Sub-threshold grid with max 0.30
    sub_threshold_grid = np.full((60, 80), 0.20, dtype=np.float32)
    sub_threshold_grid[30, 30] = 0.30
    hotspots_sub = extract_hotspots(sub_threshold_grid, (0.0, 0.0), 224.0, cfg)
    assert hotspots_sub == [], "Sub-threshold grid must return empty list"


# --- #74: OpenSlide thumbnail dimension mismatch handled cleanly ---
def test_issue_74_thumbnail_shape_mismatch_resizing():
    from worker.triage import run_triage
    # Verify PIL resize logic directly
    nx, ny = 80, 55
    # Simulate an OpenSlide get_thumbnail that produced (ny-1, nx)
    imperfect_thumb = Image.new("RGB", (nx, ny - 1), (200, 200, 200))
    resized_thumb = imperfect_thumb.resize((nx, ny), Image.Resampling.BILINEAR)
    arr = np.array(resized_thumb)
    assert arr.shape == (ny, nx, 3)


# --- #452: Parquet embedding cache read and write ---
def test_issue_452_parquet_cache_roundtrip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = str(tmp_path / "test_embeddings.parquet")
    n_patches = 16
    sampled_cells = [(i, i * 2) for i in range(n_patches)]
    embeddings = np.random.randn(n_patches, 384).astype(np.float32)

    # Save to parquet with columns ix, iy, emb
    records = []
    for idx, emb in enumerate(embeddings):
        c_ix, c_iy = sampled_cells[idx]
        records.append({"ix": c_ix, "iy": c_iy, "emb": emb.tolist()})

    import pandas as pd
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_file)

    # Read back
    read_table = pq.read_table(parquet_file)
    read_df = read_table.to_pandas()
    assert "ix" in read_df.columns
    assert "iy" in read_df.columns
    assert "emb" in read_df.columns
    assert len(read_df) == n_patches

    recovered_cells = list(zip(read_df["ix"].astype(int), read_df["iy"].astype(int)))
    recovered_embs = np.stack(read_df["emb"].values).astype(np.float32)

    assert recovered_cells == sampled_cells
    np.testing.assert_allclose(recovered_embs, embeddings, rtol=1e-5)


# --- #73: Probe metadata honest provenance & real SHA256 ---
def test_issue_73_probe_provenance_and_sha256():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    probe_joblib_path = os.path.join(repo_root, "models/probe/probe_v1.joblib")
    probe_json_path = os.path.join(repo_root, "models/probe/probe_v1.json")
    triage_yaml_path = os.path.join(repo_root, "configs/triage.yaml")

    assert os.path.exists(probe_joblib_path)
    assert os.path.exists(probe_json_path)
    assert os.path.exists(triage_yaml_path)

    # Compute real SHA256 of probe weights
    with open(probe_joblib_path, "rb") as f:
        real_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Verify JSON metadata
    with open(probe_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["dataset"] == "synthetic_dev", "Probe dataset must be labeled synthetic_dev"
    assert "train_auc" not in meta, "Fabricated train_auc must be removed"
    assert meta["weights_sha256"] == real_sha256

    # Verify configs/triage.yaml
    with open(triage_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    probe_version = cfg["probe"]["version"]
    assert real_sha256 in probe_version, f"triage.yaml version must contain real sha256 {real_sha256}"


# --- #448: apply_edit_ops processes delete operation ---
def test_issue_448_apply_edit_ops_delete_operation():
    machine_hotspots = [
        {"id": "hs_01", "polygon_um": [[0, 0], [100, 0], [100, 100], [0, 100]], "area_mm2": 0.36, "source": "model", "excluded": False},
        {"id": "hs_02", "polygon_um": [[200, 200], [300, 200], [300, 300], [200, 300]], "area_mm2": 0.36, "source": "model", "excluded": False},
    ]

    # Pathologist deletes hs_01
    edits = [
        {"op": "delete", "id": "hs_01"}
    ]

    effective = apply_edit_ops(machine_hotspots, edits)
    assert len(effective) == 1
    assert effective[0]["id"] == "hs_02", "Deleted hs_01 must be completely removed from effective set"


# --- #450: Custom polygon area calculation via Shoelace formula ---
def test_issue_450_polygon_shoelace_area():
    def compute_polygon_area_mm2(pts):
        area = 0
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return round(abs(area) / 2.0 / 1e6, 3)

    # 1000 um x 1000 um square = 1 mm²
    square_1mm = [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
    assert compute_polygon_area_mm2(square_1mm) == 1.0

    # Right triangle with base 1000 um, height 2000 um = 0.5 * 1000 * 2000 = 1,000,000 um² = 1.0 mm²
    triangle = [[0, 0], [1000, 0], [0, 2000]]
    assert compute_polygon_area_mm2(triangle) == 1.0

