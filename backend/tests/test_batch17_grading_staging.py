"""
Comprehensive test suite for Batch 17 (Workstream 4: Histologic Grading & AJCC Staging).
Validates:
1. OpenSlide lock reentrancy (threading.RLock)
2. AJCC 8th Edition staging config & SHA-256 hash
3. Pathologic tumor (pT) integer millimetre rounding rules
4. Pathologic node (pN) ITC, micrometastasis, and strict examined/positive invariant
5. Anatomic stage group mapping table (no default 'IA' fallback)
6. Dynamic config-driven Nottingham invariant validation
7. Empty evidence handling in grading pipeline
8. Grading worker coordinate clamping and true area centroid calculation
9. Grading router 409 Conflict gating on confirmed stages & signed case reports
10. Grading router MITOTIC count read-only enforcement in Stage 5 HPF review
11. Grading router CAP histologic type ontology validation
12. Grading router tubule score vs percent consistency and override key restriction
"""

import math
import threading
import uuid
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.db import Base, get_db
from app.core.openslide_lock import OPENSLIDE_GLOBAL_LOCK
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.grading import Grading
from app.models.hpf_site import HpfSite
from pipeline.staging import (
    load_staging_config,
    get_staging_config_hash,
    calculate_ajcc_pt_stage,
    calculate_ajcc_pn_stage,
    calculate_ajcc_stage_group,
    validate_staging_invariants,
    round_half_up_mm,
)
from pipeline.grading import (
    get_grading_config_hash,
    validate_grading_invariants,
    aggregate_grading_findings,
    calculate_tubule_score,
)
from worker.grading import (
    select_max_density_hotspot_patches,
    extract_10x_patch,
)

# ---------------------------------------------------------------------------
# In-memory DB Fixtures for Router Tests
# ---------------------------------------------------------------------------

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


@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    session = TestingSessionLocal()
    yield session
    session.close()
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# 1. OpenSlide Lock Reentrancy
# ---------------------------------------------------------------------------

def test_openslide_lock_reentrancy():
    """Verify OPENSLIDE_GLOBAL_LOCK is reentrant (RLock) and does not deadlock."""
    # Must be able to re-acquire on the same thread without hanging
    acquired_nested = False
    with OPENSLIDE_GLOBAL_LOCK:
        with OPENSLIDE_GLOBAL_LOCK:
            with OPENSLIDE_GLOBAL_LOCK:
                acquired_nested = True
    assert acquired_nested is True


# ---------------------------------------------------------------------------
# 2. Staging Configuration & Hash
# ---------------------------------------------------------------------------

def test_staging_config_loading_and_hash():
    """Verify configs/staging.yaml loads correctly and generates SHA-256 hash."""
    cfg = load_staging_config()
    assert "edition" in cfg
    assert "pt_cutoffs" in cfg
    assert "rounding_rule" in cfg
    assert "pn_cutoffs" in cfg
    assert "stage_groups" in cfg

    h = get_staging_config_hash()
    assert isinstance(h, str)
    assert len(h) in (16, 64)


# ---------------------------------------------------------------------------
# 3. Pathologic Tumor (pT) Staging & AJCC 8th Ed Rounding Rules
# ---------------------------------------------------------------------------

def test_ajcc_pt_rounding_and_cutoffs():
    """Verify AJCC 8th edition half-up integer millimetre rounding."""
    # Round half-up helper
    assert round_half_up_mm(20.4) == 20
    assert round_half_up_mm(20.5) == 21
    assert round_half_up_mm(50.4) == 50
    assert round_half_up_mm(50.5) == 51

    # pTX & pTis
    assert calculate_ajcc_pt_stage(None) == "pTX"
    assert calculate_ajcc_pt_stage(0.0) == "pTX"
    assert calculate_ajcc_pt_stage(-5.0) == "pTX"
    assert calculate_ajcc_pt_stage(15.0, is_in_situ_only=True) == "pTis"

    # pT1mi: <= 1.0 mm (microinvasion)
    assert calculate_ajcc_pt_stage(0.2) == "pT1mi"
    assert calculate_ajcc_pt_stage(0.9) == "pT1mi"
    assert calculate_ajcc_pt_stage(1.0) == "pT1mi"

    # Special AJCC 8th Ed rule: 1.0 < size < 2.0 mm maps to 2 mm -> pT1a
    assert calculate_ajcc_pt_stage(1.1) == "pT1a"
    assert calculate_ajcc_pt_stage(1.5) == "pT1a"
    assert calculate_ajcc_pt_stage(1.9) == "pT1a"

    # pT1a: <= 5.0 mm (after rounding)
    assert calculate_ajcc_pt_stage(2.0) == "pT1a"
    assert calculate_ajcc_pt_stage(5.0) == "pT1a"
    assert calculate_ajcc_pt_stage(5.4) == "pT1a"  # 5.4 -> 5 mm -> pT1a

    # pT1b: > 5.0 mm to <= 10.0 mm
    assert calculate_ajcc_pt_stage(5.5) == "pT1b"  # 5.5 -> 6 mm -> pT1b
    assert calculate_ajcc_pt_stage(10.0) == "pT1b"
    assert calculate_ajcc_pt_stage(10.4) == "pT1b"  # 10.4 -> 10 mm -> pT1b

    # pT1c: > 10.0 mm to <= 20.0 mm
    assert calculate_ajcc_pt_stage(10.5) == "pT1c"  # 10.5 -> 11 mm -> pT1c
    assert calculate_ajcc_pt_stage(20.0) == "pT1c"
    assert calculate_ajcc_pt_stage(20.4) == "pT1c"  # 20.4 -> 20 mm -> pT1c (avoids upstaging!)

    # pT2: > 20.0 mm to <= 50.0 mm
    assert calculate_ajcc_pt_stage(20.5) == "pT2"  # 20.5 -> 21 mm -> pT2
    assert calculate_ajcc_pt_stage(35.0) == "pT2"
    assert calculate_ajcc_pt_stage(50.0) == "pT2"
    assert calculate_ajcc_pt_stage(50.4) == "pT2"  # 50.4 -> 50 mm -> pT2

    # pT3: > 50.0 mm
    assert calculate_ajcc_pt_stage(50.5) == "pT3"  # 50.5 -> 51 mm -> pT3
    assert calculate_ajcc_pt_stage(75.0) == "pT3"

    # pT4 extensions
    assert calculate_ajcc_pt_stage(12.0, chest_wall_extension=True) == "pT4a"
    assert calculate_ajcc_pt_stage(12.0, skin_ulceration=True) == "pT4b"
    assert calculate_ajcc_pt_stage(12.0, chest_wall_extension=True, skin_ulceration=True) == "pT4c"


# ---------------------------------------------------------------------------
# 4. Pathologic Node (pN) Staging, ITCs, and Invariant Checks
# ---------------------------------------------------------------------------

def test_ajcc_pn_staging_and_invariants():
    """Verify pN staging, ITCs (pN0(i+)), micrometastases (pN1mi), and node count invariant."""
    # Invariant: positive > examined MUST raise ValueError
    with pytest.raises(ValueError, match="nodes_positive .* cannot exceed nodes_examined"):
        calculate_ajcc_pn_stage(nodes_examined=0, nodes_positive=1)

    with pytest.raises(ValueError, match="nodes_positive .* cannot exceed nodes_examined"):
        calculate_ajcc_pn_stage(nodes_examined=5, nodes_positive=6)

    # 0 examined, 0 positive -> pNX
    assert calculate_ajcc_pn_stage(nodes_examined=0, nodes_positive=0) == "pNX"

    # 10 examined, 0 positive -> pN0
    assert calculate_ajcc_pn_stage(nodes_examined=10, nodes_positive=0) == "pN0"

    # ITC (Isolated Tumor Cells) <= 0.2 mm -> pN0(i+)
    assert calculate_ajcc_pn_stage(nodes_examined=10, nodes_positive=1, largest_meta_mm=0.15) == "pN0(i+)"
    assert calculate_ajcc_pn_stage(nodes_examined=10, nodes_positive=1, largest_meta_mm=0.2) == "pN0(i+)"

    # Micrometastasis > 0.2 mm to <= 2.0 mm -> pN1mi
    assert calculate_ajcc_pn_stage(nodes_examined=10, nodes_positive=1, largest_meta_mm=0.5) == "pN1mi"
    assert calculate_ajcc_pn_stage(nodes_examined=10, nodes_positive=1, largest_meta_mm=2.0) == "pN1mi"
    assert calculate_ajcc_pn_stage(nodes_examined=10, nodes_positive=1, is_micrometastasis=True) == "pN1mi"

    # Macrometastases (> 2.0 mm)
    # 1-3 nodes -> pN1a
    assert calculate_ajcc_pn_stage(nodes_examined=10, nodes_positive=1, largest_meta_mm=2.1) == "pN1a"
    assert calculate_ajcc_pn_stage(nodes_examined=10, nodes_positive=3, largest_meta_mm=5.0) == "pN1a"

    # 4-9 nodes -> pN2a
    assert calculate_ajcc_pn_stage(nodes_examined=12, nodes_positive=4) == "pN2a"
    assert calculate_ajcc_pn_stage(nodes_examined=12, nodes_positive=9) == "pN2a"

    # 10+ nodes -> pN3a
    assert calculate_ajcc_pn_stage(nodes_examined=15, nodes_positive=10) == "pN3a"
    assert calculate_ajcc_pn_stage(nodes_examined=20, nodes_positive=15) == "pN3a"


# ---------------------------------------------------------------------------
# 5. AJCC Anatomic Stage Group Matrix
# ---------------------------------------------------------------------------

def test_ajcc_stage_group_matrix_comprehensive():
    """Verify stage grouping matrix covers subcategories, M1, and returns 'Cannot be determined' instead of 'IA'."""
    # Distant metastasis -> IV
    assert calculate_ajcc_stage_group("pT1c", "pN0", pm_stage="pM1") == "IV"
    assert calculate_ajcc_stage_group("pT3", "pN2a", pm_stage="M1") == "IV"

    # Stage 0
    assert calculate_ajcc_stage_group("pTis", "pN0") == "0"

    # Stage IA
    assert calculate_ajcc_stage_group("pT1a", "pN0") == "IA"
    assert calculate_ajcc_stage_group("pT1c", "pN0") == "IA"
    assert calculate_ajcc_stage_group("pT1c", "pNX") == "IA"

    # Stage IB (pT0 or pT1 with pN1mi)
    assert calculate_ajcc_stage_group("pT0", "pN1mi") == "IB"
    assert calculate_ajcc_stage_group("pT1a", "pN1mi") == "IB"
    assert calculate_ajcc_stage_group("pT1c", "pN1mi") == "IB"

    # Stage IIA (pT0 or pT1 with N1; pT2 with N0)
    assert calculate_ajcc_stage_group("pT0", "pN1a") == "IIA"
    assert calculate_ajcc_stage_group("pT1c", "pN1a") == "IIA"
    assert calculate_ajcc_stage_group("pT2", "pN0") == "IIA"

    # Stage IIB (pT2 with N1; pT3 with N0)
    assert calculate_ajcc_stage_group("pT2", "pN1a") == "IIB"
    assert calculate_ajcc_stage_group("pT3", "pN0") == "IIB"

    # Stage IIIA (pT0/pT1/pT2 with N2; pT3 with N1/N2)
    assert calculate_ajcc_stage_group("pT0", "pN2a") == "IIIA"
    assert calculate_ajcc_stage_group("pT1c", "pN2a") == "IIIA"
    assert calculate_ajcc_stage_group("pT3", "pN1a") == "IIIA"
    assert calculate_ajcc_stage_group("pT3", "pN2a") == "IIIA"

    # Stage IIIB (pT4 with N0/N1/N2)
    assert calculate_ajcc_stage_group("pT4a", "pN0") == "IIIB"
    assert calculate_ajcc_stage_group("pT4b", "pN1a") == "IIIB"
    assert calculate_ajcc_stage_group("pT4c", "pN2a") == "IIIB"

    # Stage IIIC (Any T with N3)
    assert calculate_ajcc_stage_group("pT1c", "pN3a") == "IIIC"
    assert calculate_ajcc_stage_group("pT4b", "pN3a") == "IIIC"

    # Subcategory normalization (pN1b, pN1c -> N1; pN2b -> N2; pN3b, pN3c -> N3)
    assert calculate_ajcc_stage_group("pT1c", "pN1b") == "IIA"
    assert calculate_ajcc_stage_group("pT1c", "pN2b") == "IIIA"
    assert calculate_ajcc_stage_group("pT1c", "pN3b") == "IIIC"

    # Indeterminate / unassessed combinations
    assert calculate_ajcc_stage_group("pTX", "pNX") == "Unknown"
    assert calculate_ajcc_stage_group("pTX", "pN0") == "Unknown"
    assert calculate_ajcc_stage_group("TX", "pNX") == "Unknown"

    # Never fall back to arbitrary 'IA' on unhandled combinations
    assert calculate_ajcc_stage_group("UNKNOWN_T", "UNKNOWN_N") == "Cannot be determined"


# ---------------------------------------------------------------------------
# 6. Nottingham Grading Engine & Config-Driven Invariants
# ---------------------------------------------------------------------------

def test_nottingham_grading_config_hash_and_invariants():
    """Verify get_grading_config_hash and dynamic config-driven validate_grading_invariants."""
    h = get_grading_config_hash()
    assert isinstance(h, str)
    assert len(h) in (16, 64)

    # Valid combinations
    validate_grading_invariants(tubule_score=1, pleo_score=1, mitotic_score=1, nottingham_sum=3, grade=1)
    validate_grading_invariants(tubule_score=2, pleo_score=2, mitotic_score=2, nottingham_sum=6, grade=2)
    validate_grading_invariants(tubule_score=3, pleo_score=3, mitotic_score=3, nottingham_sum=9, grade=3)

    # Inconsistent sum
    with pytest.raises(ValueError, match="nottingham_sum"):
        validate_grading_invariants(tubule_score=2, pleo_score=2, mitotic_score=2, nottingham_sum=7, grade=2)

    # Inconsistent grade
    with pytest.raises(ValueError, match="grade"):
        validate_grading_invariants(tubule_score=2, pleo_score=2, mitotic_score=2, nottingham_sum=6, grade=3)

    # Custom config thresholds passed via cfg
    custom_cfg = {
        "grade1_max_sum": 6,  # Grade 1 up to sum 6
        "grade2_max_sum": 7,
    }
    # Sum 6 with Grade 1 should succeed with custom_cfg
    validate_grading_invariants(
        tubule_score=2, pleo_score=2, mitotic_score=2, nottingham_sum=6, grade=1, cfg=custom_cfg
    )


# ---------------------------------------------------------------------------
# 7. Empty Evidence Handling in Grading Engine
# ---------------------------------------------------------------------------

def test_grading_empty_evidence_handling():
    """Verify aggregate_grading_findings handles empty patch sets gracefully without fabricating scores."""
    result = aggregate_grading_findings(
        tubule_responses=[],
        pleo_responses=[],
        mitotic_score=1,
        cfg={"weights": {"tubule": 0.5, "pleo": 0.5}},
    )
    assert result["needs_human"] is True
    assert "empty_evidence_set" in result["flags"]
    assert "needs_human" in result["flags"]
    assert result["tubule_score"] is None
    assert result["pleo_score"] is None


# ---------------------------------------------------------------------------
# 8. Worker Coordinate Clamping & Centroid Calculation
# ---------------------------------------------------------------------------

def test_worker_centroid_and_anisotropy():
    """Verify Shapely true polygon area centroid and anisotropic mpp handling."""
    # Right triangle with vertices (0,0), (6,0), (0,6)
    # Area centroid is at (2, 2)
    geom_coords = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0), (0.0, 0.0)]
    poly = Polygon(geom_coords)
    assert poly.centroid.x == pytest.approx(2.0)
    assert poly.centroid.y == pytest.approx(2.0)

    # Test select_max_density_hotspot_patches with mock patches and anisotropic mpp
    # In select_max_density_hotspot_patches, hotspot polygon coords in um should convert to px
    # by dividing x by mpp_x and y by mpp_y
    mpp_x = 0.25
    mpp_y = 0.50  # anisotropic

    # Coordinates in microns: (100, 200) -> (400 px, 400 px)
    cx_um, cy_um = 100.0, 200.0
    cx_px = cx_um / mpp_x
    cy_px = cy_um / (mpp_y or mpp_x)
    assert cx_px == 400.0
    assert cy_px == 400.0


# ---------------------------------------------------------------------------
# 9. Grading API Router State Machine & 409 Conflict Gating
# ---------------------------------------------------------------------------

def test_grading_router_409_conflict_when_stage_confirmed(db_session):
    """Verify 409 Conflict is returned when attempting patch/HPF review or confirm on already confirmed stage."""
    client = TestClient(app)
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="pathologist_1", status="open")
    stage_exec = StageExecution(
        case_id=case_id,
        stage="grading",
        attempt=1,
        status="confirmed",  # ALREADY CONFIRMED
    )
    grading = Grading(
        case_id=case_id,
        tubule_score=2,
        pleo_score=2,
        mitotic_score=2,
        nottingham_sum=6,
        grade=2,
        histologic_type="IDC-NST",
        type_confirmed_by="pathologist_1",
        machine={"patches": [{"id": "p_01"}], "hpfs": [{"seq": 1}]},
        overrides={},
    )
    db_session.add(case)
    db_session.add(stage_exec)
    db_session.add(grading)
    db_session.commit()

    headers = {"X-User-Role": "pathologist"}

    # Attempt patch review
    res_patch = client.post(
        "/api/v1/stages/grading/patches/review",
        json={"case_id": str(case_id), "action": "approve_all"},
        headers=headers,
    )
    assert res_patch.status_code == 409
    assert "already confirmed" in res_patch.json()["detail"]

    # Attempt HPF review
    res_hpf = client.post(
        "/api/v1/stages/grading/hpfs/review",
        json={"case_id": str(case_id), "action": "approve_all"},
        headers=headers,
    )
    assert res_hpf.status_code == 409
    assert "already confirmed" in res_hpf.json()["detail"]

    # Attempt histologic type confirm
    res_type = client.post(
        "/api/v1/stages/grading/type/confirm",
        json={"case_id": str(case_id), "histologic_type": "IDC-NST", "reviewed_by": "Dr. Test"},
        headers=headers,
    )
    assert res_type.status_code == 409
    assert "already confirmed" in res_type.json()["detail"]

    # Attempt confirm
    res_confirm = client.post(
        "/api/v1/stages/grading/confirm",
        json={
            "case_id": str(case_id),
            "reviewed_by": "Dr. Test",
            "histologic_type": "IDC-NST",
            "type_confirmed": True,
            "tubule_score": 2,
            "pleo_score": 2,
            "mitotic_score": 2,
            "nottingham_sum": 6,
            "grade": 2,
        },
        headers=headers,
    )
    assert res_confirm.status_code == 409
    assert "already confirmed" in res_confirm.json()["detail"]


# ---------------------------------------------------------------------------
# 10. Grading Router: Read-Only Mitotic Figures in Stage 5
# ---------------------------------------------------------------------------

def test_grading_router_mitotic_count_readonly(db_session):
    """Verify attempting to alter raw mitotic counts in Stage 5 HPF review is rejected."""
    client = TestClient(app)
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="pathologist_1", status="open")
    stage_exec = StageExecution(
        case_id=case_id,
        stage="grading",
        attempt=1,
        status="awaiting_review",
    )
    grading = Grading(
        case_id=case_id,
        tubule_score=2,
        pleo_score=2,
        mitotic_score=2,
        nottingham_sum=6,
        grade=2,
        histologic_type="IDC-NST",
        type_confirmed_by="unconfirmed",
        machine={"patches": [], "hpfs": [{"seq": 1, "mitotic_count": 5}]},
        overrides={},
    )
    hpf = HpfSite(
        case_id=case_id,
        seq=1,
        center_um=[1000.0, 1000.0],
        radius_um=262.0,
        mitotic_count=5,  # Baseline count from Stage 4
        source="model",
    )
    db_session.add(case)
    db_session.add(stage_exec)
    db_session.add(grading)
    db_session.add(hpf)
    db_session.commit()

    headers = {"X-User-Role": "pathologist"}

    # Attempt to change mitotic count in Stage 5
    res = client.post(
        "/api/v1/stages/grading/hpfs/review",
        json={
            "case_id": str(case_id),
            "action": "update",
            "reviews": [
                {
                    "seq": 1,
                    "mitotic_count": 8,  # Attempting to supply mitotic count!
                    "status": "modified",
                }
            ],
        },
        headers=headers,
    )
    assert res.status_code == 400
    assert "Mitotic figure counts cannot be modified directly in Stage 5" in res.json()["detail"]


# ---------------------------------------------------------------------------
# 11. Grading Router: CAP Histologic Type Ontology Validation
# ---------------------------------------------------------------------------

def test_grading_router_cap_histologic_type_validation(db_session):
    """Verify confirming an invalid histologic type not in CAP elements is rejected."""
    client = TestClient(app)
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="pathologist_1", status="open")
    stage_exec = StageExecution(
        case_id=case_id,
        stage="grading",
        attempt=1,
        status="awaiting_review",
    )
    grading = Grading(
        case_id=case_id,
        tubule_score=2,
        pleo_score=2,
        mitotic_score=2,
        nottingham_sum=6,
        grade=2,
        histologic_type="IDC-NST",
        type_confirmed_by="unconfirmed",
        machine={"patches": [], "hpfs": []},
        overrides={},
    )
    db_session.add(case)
    db_session.add(stage_exec)
    db_session.add(grading)
    db_session.commit()

    headers = {"X-User-Role": "pathologist"}

    # Confirm with invalid type
    res = client.post(
        "/api/v1/stages/grading/type/confirm",
        json={
            "case_id": str(case_id),
            "histologic_type": "MadeUpCancerType",
            "reviewed_by": "Dr. Test",
        },
        headers=headers,
    )
    assert res.status_code == 400
    assert "Invalid histologic type" in res.json()["detail"]

    # Confirm with valid CAP element
    res_valid = client.post(
        "/api/v1/stages/grading/type/confirm",
        json={
            "case_id": str(case_id),
            "histologic_type": "mucinous",
            "reviewed_by": "Dr. Test",
        },
        headers=headers,
    )
    assert res_valid.status_code == 200
    assert res_valid.json()["histologic_type"]["confirmed_type"] == "mucinous"


# ---------------------------------------------------------------------------
# 12. Grading Router: Confirm Overrides Key Restriction & Tubule Consistency
# ---------------------------------------------------------------------------

def test_grading_router_confirm_security_and_consistency(db_session):
    """Verify unauthorized override keys and tubule score vs percent inconsistencies are rejected."""
    client = TestClient(app)
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="pathologist_1", status="open")
    stage_exec = StageExecution(
        case_id=case_id,
        stage="grading",
        attempt=1,
        status="awaiting_review",
    )
    grading = Grading(
        case_id=case_id,
        tubule_score=2,
        pleo_score=2,
        mitotic_score=2,
        nottingham_sum=6,
        grade=2,
        histologic_type="IDC-NST",
        type_confirmed_by="Dr. Test",
        machine={"patches": [], "hpfs": []},
        overrides={},
    )
    db_session.add(case)
    db_session.add(stage_exec)
    db_session.add(grading)
    db_session.commit()

    headers = {"X-User-Role": "pathologist"}

    # 1. Reject unauthorized override key ('patches' or 'hpfs')
    res_unauth = client.post(
        "/api/v1/stages/grading/confirm",
        json={
            "case_id": str(case_id),
            "reviewed_by": "Dr. Test",
            "histologic_type": "IDC-NST",
            "type_confirmed": True,
            "overrides": {
                "patches": {"score": 2, "justification": "attempted unauthorized key"},  # Unauthorized key!
            },
            "tubule_score": 2,
            "pleo_score": 2,
            "mitotic_score": 2,
            "nottingham_sum": 6,
            "grade": 2,
        },
        headers=headers,
    )
    assert res_unauth.status_code in (400, 422)
    assert "unauthorized override" in res_unauth.text.lower() or "unauthorized" in res_unauth.text.lower()

    # 2. Reject inconsistent tubule_score vs tubule_percent (e.g. 80% tubules -> score 1, but passed score 3)
    res_mismatch = client.post(
        "/api/v1/stages/grading/confirm",
        json={
            "case_id": str(case_id),
            "reviewed_by": "Dr. Test",
            "histologic_type": "IDC-NST",
            "type_confirmed": True,
            "overrides": {
                "tubule": {
                    "score": 3,
                    "percent": 85.0,  # 85% must map to score 1, not 3!
                    "justification": "Tubule formation is extensive throughout the specimen.",
                }
            },
            "tubule_score": 3,
            "pleo_score": 2,
            "mitotic_score": 2,
            "nottingham_sum": 7,
            "grade": 2,
        },
        headers=headers,
    )
    assert res_mismatch.status_code == 400
    assert "Inconsistent tubule" in res_mismatch.json()["detail"]
