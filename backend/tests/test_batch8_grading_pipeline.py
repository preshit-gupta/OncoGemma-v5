"""
Batch 8 Test Suite: Stage 5 Nottingham Grading, Histologic Type Confirmation & MedGemma Fallback Integrity.

Validates:
1. Finding #722: HistologicTypeResponse requires type and confidence; empty/malformed inputs fail schema validation.
2. Finding #597: MedGemma fallback and client use explicit task parameter to prevent keyword hijacking.
3. Finding #598: Findings narrative fallback extracts Nottingham grade and sum from aggregate nesting.
4. Finding #599: CAP synoptic fallback grounds pleomorphism in p_score and LVI in case lvi_status.
5. Finding #136: Grading worker raises ValueError on empty hotspots; zero fabricated 24 diagonal coordinates.
6. Finding #144: API eliminates 10 fake HPF synthesis on empty cases; review returns 400; can_confirm is False.
7. Finding #600: ScoreOverrideItem validates scores 1..3 and min 10-char justification; prevents GET 500 crashes.
8. Finding #138: Confirm endpoint enforces pathologist role RBAC, stage status gating, and report signed immutability (409).
9. Finding #142: Server-side histologic type confirmation endpoint stamps type_confirmed_by and emits audit event.
10. Finding #139: Sub-score divergence without justification rejected with 400; authoritatively recomputes grade.
11. Finding #140: Missing patch image returns 404, never a synthetic drawn canvas image.
"""

import uuid
import asyncio
import numpy as np
import unittest.mock as mock
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.db import Base, get_db
from app.models.case import Case
from app.models.slide import Slide
from app.models.grading import Grading
from app.models.hotspot import Hotspot
from app.models.detection import Detection
from app.models.hpf_site import HpfSite
from app.models.stage_execution import StageExecution
from app.models.report import Report
from app.models.audit import AuditEvent
from pipeline.medgemma import HistologicTypeResponse, MedGemmaClient
from worker.grading import select_max_density_hotspot_patches, run_grading

# Shared in-memory test database
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


def _create_approved_grading_data():
    """Helper creating 24 approved patches and 10 approved HPFs for stage confirmation."""
    patches = [
        {
            "id": f"p{i}",
            "index": i,
            "hotspot_id": "h1",
            "tubule": {"tubule_percent": 25, "tumor_present": True, "confidence": "high"},
            "pleo": {"pleomorphism_score": 2, "confidence": "high", "rationale": "mild to moderate atypia"},
            "review_status": "approved"
        }
        for i in range(1, 25)
    ]
    # 0 mitoses across 10 HPFs -> mitotic score = 1
    hpfs = [
        {
            "seq": i,
            "mitotic_count": 0,
            "review_status": "approved"
        }
        for i in range(1, 11)
    ]
    return {
        "patches": patches,
        "hpfs": hpfs,
        "tubule_score": 2,
        "pleo_score": 2,
        "mitotic_score": 1,
        "tubule_percent": 25.0,
        "nottingham_sum": 5,
        "grade": 1
    }


# ============================================================================
# 1. MedGemma Schema Validation Tests (#722)
# ============================================================================

def test_histologic_type_response_validation_failure_on_empty():
    """Empty dictionary must not default to IDC-NST / medium; it must fail validation."""
    with pytest.raises(ValidationError):
        HistologicTypeResponse.model_validate({})


def test_histologic_type_response_validation_missing_confidence():
    """Missing confidence field must raise ValidationError."""
    with pytest.raises(ValidationError):
        HistologicTypeResponse.model_validate({"type": "IDC-NST"})


def test_histologic_type_response_valid_schema_error():
    """Explicit unassessed_schema_error confidence value is valid."""
    resp = HistologicTypeResponse.model_validate({
        "type": "other",
        "confidence": "unassessed_schema_error",
        "rationale": "Model timeout degraded"
    })
    assert resp.type == "other"
    assert resp.confidence == "unassessed_schema_error"


# ============================================================================
# 2. MedGemma Task Routing & Mock Dispatch Tests (#597)
# ============================================================================

def test_medgemma_task_routing_priority():
    """Prompt containing 'tubule' must not be hijacked if task is histologic_type or cap_report."""
    client = MedGemmaClient()
    prompt_with_tubule = "Review tubule formation patterns and classify the primary histologic subtype."
    
    resp_type = client._mock_fallback_response(prompt_with_tubule, task="histologic_type")
    assert "IDC-NST" in resp_type or "histologic" in resp_type.lower()
    assert "tubule_percent" not in resp_type

    resp_narrative = client._mock_fallback_response(prompt_with_tubule, task="findings_narrative")
    assert "invasive breast carcinoma" in resp_narrative.lower()
    assert "tubule_percent" not in resp_narrative

    resp_cap = client._mock_fallback_response(prompt_with_tubule, task="cap_report")
    assert "diagnosis_line" in resp_cap
    assert "tubule_percent" not in resp_cap


# ============================================================================
# 3. Findings Narrative & CAP Grounding Tests (#598, #599)
# ============================================================================

def test_generate_findings_narrative_grounding_in_aggregate():
    """Findings narrative must extract Nottingham grade and sum from aggregate data dict (#598)."""
    client = MedGemmaClient()
    with mock.patch.object(client, "_call_vertex_endpoint", side_effect=RuntimeError("Endpoint unavailable")):
        aggregated_data = {
            "aggregate": {
                "grade": 3,
                "nottingham_sum": 8,
                "tubule_score": 3,
                "pleo_score": 3,
                "mitotic_score": 2,
                "tubule_percent": 12.5,
                "mitoses_per_mm2": 15.0
            }
        }
        narrative = asyncio.run(client.generate_findings_narrative(aggregated_data, "Prompt {input_json}"))
        assert "Grade 3" in narrative
        assert "8/9" in narrative
        assert "Grade 2" not in narrative


def test_generate_cap_report_grounding_pleomorphism_and_lvi():
    """CAP synoptic narrative must ground pleomorphism in p_score and LVI in case lvi_status (#599)."""
    client = MedGemmaClient()
    with mock.patch.object(client, "_call_vertex_endpoint", side_effect=RuntimeError("Endpoint unavailable")):
        case_marked_present = {
            "histologic_type": "IDC-NST",
            "nottingham_grade": {
                "grade": 3,
                "nottingham_sum": 8,
                "tubule_score": 3,
                "pleo_score": 3,
                "mitotic_score": 2,
                "tubule_percent": 10.0
            },
            "lvi_status": "present",
            "procedure": "Excision",
            "laterality": "left"
        }
        cap_1 = asyncio.run(client.generate_cap_report_narrative(case_marked_present))
        assert "marked nuclear pleomorphism" in cap_1["microscopic_findings"].lower()
        assert "lymphovascular invasion is identified" in cap_1["microscopic_findings"].lower()

        case_mild_absent = {
            "histologic_type": "Tubular Carcinoma",
            "nottingham_grade": {
                "grade": 1,
                "nottingham_sum": 3,
                "tubule_score": 1,
                "pleo_score": 1,
                "mitotic_score": 1,
                "tubule_percent": 85.0
            },
            "lvi_status": "absent"
        }
        cap_2 = asyncio.run(client.generate_cap_report_narrative(case_mild_absent))
        assert "mild nuclear pleomorphism" in cap_2["microscopic_findings"].lower()
        assert "lymphovascular invasion is not identified" in cap_2["microscopic_findings"].lower()


# ============================================================================
# 4. Zero Hotspot Fabrication Tests (#136)
# ============================================================================

def test_select_max_density_hotspot_patches_empty_hotspots_raises():
    """select_max_density_hotspot_patches must raise ValueError when hotspots list is empty."""
    mask = np.ones((100, 100), dtype=bool)
    with pytest.raises(ValueError, match="No confirmed tumor hotspots provided"):
        select_max_density_hotspot_patches([], mask, (10000.0, 10000.0), 0.25, "case_test")


def test_select_max_density_hotspot_patches_invalid_polygons_raises():
    """Polygons with fewer than 3 vertices must be rejected, not replaced with fake 5000,5000 coordinates."""
    mask = np.ones((100, 100), dtype=bool)
    invalid_hotspots = [
        {"id": "h1", "polygon": [[100, 100]], "density": 0.9},
        {"id": "h2", "polygon": [[100, 100], [200, 200]], "density": 0.8}
    ]
    with pytest.raises(ValueError, match="No valid tumor tissue patches could be sampled"):
        select_max_density_hotspot_patches(invalid_hotspots, mask, (10000.0, 10000.0), 0.25, "case_test")


def test_run_grading_worker_fails_fast_on_zero_hotspots():
    """run_grading must fail with ValueError when a case has zero confirmed hotspots in DB."""
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    slide_uid = uuid.uuid4()

    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    slide = Slide(id=slide_uid, case_id=case_uid, gcs_uri_original="gs://test/test.svs", mpp_x=0.25, mpp_y=0.25)
    stage_exec = StageExecution(case_id=case_uid, stage="grading", attempt=1, status="in_progress")
    db.add_all([case, slide, stage_exec])
    db.commit()

    with pytest.raises(ValueError, match="No confirmed tumor hotspots"):
        run_grading(stage_exec, db)
    db.close()


def test_run_grading_worker_queries_detections_without_name_error():
    """run_grading must query Detection and HpfSite without raising NameError."""
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    slide_uid = uuid.uuid4()

    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    slide = Slide(id=slide_uid, case_id=case_uid, gcs_uri_original="gs://test/test.svs", mpp_x=0.25, mpp_y=0.25)
    stage_exec_uid = uuid.uuid4()
    stage_exec = StageExecution(id=stage_exec_uid, case_id=case_uid, stage="grading", attempt=1, status="in_progress")
    hotspot = Hotspot(
        id="hs_01",
        case_id=case_uid,
        stage_execution_id=stage_exec_uid,
        polygon_um=[[1000.0, 1000.0], [2000.0, 1000.0], [2000.0, 2000.0], [1000.0, 2000.0]],
        prob_mean=0.9
    )
    det = Detection(
        id="m_001",
        case_id=case_uid,
        hotspot_id="hs_01",
        centroid_um=[1500.0, 1500.0],
        label="mitosis",
        label_source="model"
    )
    hpf = HpfSite(
        seq=1,
        case_id=case_uid,
        center_um=[1500.0, 1500.0],
        radius_um=262.0,
        mitotic_count=1
    )
    db.add_all([case, slide, stage_exec, hotspot, det, hpf])
    db.commit()

    with mock.patch("worker.grading.download_blob_to_filename"), \
         mock.patch("os.path.exists", return_value=True), \
         mock.patch("openslide.OpenSlide") as mock_os:
        # Halt execution right after the Stage 4 mitotic score retrieval step
        mock_os.side_effect = RuntimeError("OpenSlide stopped after detection query")
        with pytest.raises(RuntimeError, match="OpenSlide stopped after detection query"):
            run_grading(stage_exec, db)
    db.close()


# ============================================================================
# 5. Zero HPF Fabrication Tests (#144)
# ============================================================================

def test_grading_data_endpoint_no_fake_hpfs_on_empty_case():
    """When no HpfSite rows exist, GET /grading must return hpfs: [] and can_confirm: False."""
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    stage_grading = StageExecution(case_id=case_uid, stage="grading", attempt=1, status="in_progress")
    db.add_all([case, stage_grading])
    db.commit()
    db.close()

    res = client.get(f"/api/v1/stages/grading/{case_uid}")
    assert res.status_code == 200
    data = res.json()
    assert data["hpfs"] == []
    assert data["review_summary"]["total_hpfs"] == 0
    assert data["can_confirm"] is False


def test_hpf_review_endpoint_rejects_empty_hpfs():
    """review_grading_hpfs must reject with 400 when no HPF sites exist."""
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    grading_rec = Grading(case_id=case_uid, machine={})
    db.add_all([case, grading_rec])
    db.commit()
    db.close()

    res = client.post(
        "/api/v1/stages/grading/hpfs/review",
        json={
            "case_id": str(case_uid),
            "reviews": [{"seq": 1, "mitotic_count": 0, "status": "approved"}]
        }
    )
    assert res.status_code == 400
    assert "No HPF sites found" in res.json()["detail"]


# ============================================================================
# 6. Override Score Validation & Crash Prevention Tests (#600)
# ============================================================================

def test_confirm_grading_payload_validation_rejects_invalid_score():
    """Overrides with out-of-range score or malformed type must fail validation with 422."""
    client = TestClient(app)
    case_uid = str(uuid.uuid4())

    res_bad_score = client.post(
        "/api/v1/stages/grading/confirm",
        json={
            "case_id": case_uid,
            "reviewed_by": "Dr. Test",
            "histologic_type": "IDC-NST",
            "type_confirmed": True,
            "tubule_score": 2,
            "pleo_score": 2,
            "mitotic_score": 1,
            "nottingham_sum": 5,
            "grade": 1,
            "overrides": {
                "tubule": {
                    "score": 5,
                    "justification": "Invalid high score justification test"
                }
            }
        },
        headers={"X-User-Role": "pathologist"}
    )
    assert res_bad_score.status_code == 422

    res_bad_type = client.post(
        "/api/v1/stages/grading/confirm",
        json={
            "case_id": case_uid,
            "reviewed_by": "Dr. Test",
            "histologic_type": "IDC-NST",
            "type_confirmed": True,
            "tubule_score": 2,
            "pleo_score": 2,
            "mitotic_score": 1,
            "nottingham_sum": 5,
            "grade": 1,
            "overrides": {
                "tubule": {
                    "score": "three",
                    "justification": "Valid length justification string"
                }
            }
        },
        headers={"X-User-Role": "pathologist"}
    )
    assert res_bad_type.status_code == 422


# ============================================================================
# 7. Authentication, Role RBAC, State Gating & Confirmation (#138, #139, #142)
# ============================================================================

def test_confirm_grading_stage_rejects_non_pathologist_role():
    """confirm_grading_stage must reject non-pathologist/admin users (e.g. technician) with 403."""
    client = TestClient(app)
    case_uid = str(uuid.uuid4())

    res = client.post(
        "/api/v1/stages/grading/confirm",
        json={
            "case_id": case_uid,
            "reviewed_by": "Technician User",
            "histologic_type": "IDC-NST",
            "type_confirmed": True,
            "tubule_score": 2,
            "pleo_score": 2,
            "mitotic_score": 1,
            "nottingham_sum": 5,
            "grade": 1
        },
        headers={"X-User-Role": "technician"}
    )
    assert res.status_code == 403
    assert "Only pathologists or administrators" in res.json()["detail"]


def test_histologic_type_dedicated_confirm_endpoint():
    """POST /stages/grading/type/confirm must stamp type_confirmed_by and emit audit event (#142)."""
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()

    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    grading_rec = Grading(
        case_id=case_uid,
        histologic_type="IDC-NST",
        type_confirmed_by="unconfirmed",
        tubule_score=2,
        pleo_score=2,
        mitotic_score=1,
        nottingham_sum=5,
        grade=1,
        machine=_create_approved_grading_data()
    )
    db.add_all([case, grading_rec])
    db.commit()
    db.close()

    res = client.post(
        "/api/v1/stages/grading/type/confirm",
        json={
            "case_id": str(case_uid),
            "histologic_type": "ILC",
            "reviewed_by": "Dr. Attending Pathologist"
        },
        headers={"X-User-Role": "pathologist", "X-User-Id": "Dr. Attending Pathologist"}
    )
    assert res.status_code == 200

    db = TestingSessionLocal()
    updated_rec = db.query(Grading).filter(Grading.case_id == case_uid).first()
    assert updated_rec.histologic_type == "ILC"
    assert updated_rec.type_confirmed_by == "Dr. Attending Pathologist"

    audit_evt = db.query(AuditEvent).filter(
        AuditEvent.case_id == str(case_uid),
        AuditEvent.event_type == "histologic_type_confirmed"
    ).first()
    assert audit_evt is not None
    assert audit_evt.payload["histologic_type"] == "ILC"
    db.close()


def test_confirm_grading_divergence_without_justification_rejected():
    """Divergent score between payload and server effective data without justification returns 400 (#139)."""
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()

    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    stage_exec = StageExecution(case_id=case_uid, stage="grading", attempt=1, status="in_progress")
    grading_rec = Grading(
        case_id=case_uid,
        histologic_type="IDC-NST",
        type_confirmed_by="Dr. Pathologist",
        tubule_score=2,
        pleo_score=2,
        mitotic_score=1,
        nottingham_sum=5,
        grade=1,
        machine=_create_approved_grading_data()
    )
    db.add_all([case, stage_exec, grading_rec])
    db.commit()
    db.close()

    # Client claims tubule_score = 3, but server has 2, without overrides justification
    res = client.post(
        "/api/v1/stages/grading/confirm",
        json={
            "case_id": str(case_uid),
            "reviewed_by": "Dr. Pathologist",
            "histologic_type": "IDC-NST",
            "type_confirmed": True,
            "tubule_score": 3,
            "pleo_score": 2,
            "mitotic_score": 1,
            "nottingham_sum": 6,
            "grade": 2,
            "overrides": {}
        },
        headers={"X-User-Role": "pathologist"}
    )
    assert res.status_code == 400
    assert "justification" in res.json()["detail"].lower()


def test_confirm_grading_stage_successful_and_locked_against_reconfirmation():
    """Successful confirmation locks Stage 5; repeat confirmation returns 409 Conflict (#138, #674)."""
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()

    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    stage_exec = StageExecution(case_id=case_uid, stage="grading", attempt=1, status="in_progress")
    grading_rec = Grading(
        case_id=case_uid,
        histologic_type="IDC-NST",
        type_confirmed_by="Dr. Pathologist",
        tubule_score=2,
        pleo_score=2,
        mitotic_score=1,
        nottingham_sum=5,
        grade=1,
        machine=_create_approved_grading_data()
    )
    db.add_all([case, stage_exec, grading_rec])
    db.commit()
    db.close()

    payload = {
        "case_id": str(case_uid),
        "reviewed_by": "Dr. Pathologist",
        "histologic_type": "IDC-NST",
        "type_confirmed": True,
        "tubule_score": 2,
        "pleo_score": 2,
        "mitotic_score": 1,
        "nottingham_sum": 5,
        "grade": 1
    }

    # 1. First confirmation succeeds
    res_first = client.post(
        "/api/v1/stages/grading/confirm",
        json=payload,
        headers={"X-User-Role": "pathologist"}
    )
    assert res_first.status_code == 200
    assert res_first.json()["status"] == "success"

    # 2. Second confirmation is rejected with 409 Conflict
    res_second = client.post(
        "/api/v1/stages/grading/confirm",
        json=payload,
        headers={"X-User-Role": "pathologist"}
    )
    assert res_second.status_code == 409
    assert "already confirmed" in res_second.json()["detail"].lower()


def test_confirm_grading_stage_rejects_mutation_on_signed_report():
    """If case report is signed, confirm_grading_stage must return 409 Conflict (#138)."""
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()

    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    stage_exec = StageExecution(case_id=case_uid, stage="grading", attempt=1, status="in_progress")
    grading_rec = Grading(
        case_id=case_uid,
        histologic_type="IDC-NST",
        type_confirmed_by="Dr. Pathologist",
        tubule_score=2,
        pleo_score=2,
        mitotic_score=1,
        nottingham_sum=5,
        grade=1,
        machine=_create_approved_grading_data()
    )
    signed_report = Report(
        case_id=case_uid,
        version=1,
        status="signed",
        signed_by="Dr. Attending"
    )
    db.add_all([case, stage_exec, grading_rec, signed_report])
    db.commit()
    db.close()

    res = client.post(
        "/api/v1/stages/grading/confirm",
        json={
            "case_id": str(case_uid),
            "reviewed_by": "Dr. Pathologist",
            "histologic_type": "IDC-NST",
            "type_confirmed": True,
            "tubule_score": 2,
            "pleo_score": 2,
            "mitotic_score": 1,
            "nottingham_sum": 5,
            "grade": 1
        },
        headers={"X-User-Role": "pathologist"}
    )
    assert res.status_code == 409
    assert "signed or amended case report" in res.json()["detail"].lower()


# ============================================================================
# 8. Patch Image Endpoint 404 (#140)
# ============================================================================

def test_patch_image_returns_404_on_missing_file():
    """Non-existent patch image must return 404, never a synthetic drawn histology PNG."""
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)
    db.commit()
    db.close()

    res = client.get(f"/api/v1/stages/grading/{case_uid}/patches/{uuid.uuid4()}/image")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()
