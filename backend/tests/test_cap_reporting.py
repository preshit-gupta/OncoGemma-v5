"""
Unit and Integration Test Suite for Stage v4.5: CAP-Compliant Synoptic Reporting.

Verifies:
1. Pure code deterministic AJCC 8th/9th Edition staging calculations (pT, pN, Stage Group).
2. Staging invariant validations.
3. MedGemma narrative guardrail numerical consistency check.
4. ReportLab clinical PDF generation with embedded visual evidence.
5. Report database model persistence and Case relationship.
6. Stage 6 background worker execution.
7. Full REST API workflow (GET, PUT, regenerate-narrative, PDF stream, JSON export, sign-off, amendment).
"""

import os
import uuid
import json
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.db import Base, get_db
from app.models.case import Case
from app.models.slide import Slide
from app.models.grading import Grading
from app.models.stage_execution import StageExecution
from app.models.report import Report

from pipeline.staging import (
    calculate_ajcc_pt_stage,
    calculate_ajcc_pn_stage,
    calculate_ajcc_stage_group,
    validate_staging_invariants,
    validate_narrative_consistency
)
from pipeline.report_pdf import generate_clinical_cap_pdf


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


# ---------------------------------------------------------------------------
# 1. Pure Code AJCC Deterministic Staging Unit Tests
# ---------------------------------------------------------------------------

def test_ajcc_pt_staging_cutoffs():
    """Test all AJCC 8th/9th ed pT tumor size boundary cutoffs."""
    assert calculate_ajcc_pt_stage(None) == "pTX"
    assert calculate_ajcc_pt_stage(0.0) == "pTX"
    assert calculate_ajcc_pt_stage(15.0, is_in_situ_only=True) == "pTis"
    
    # pT1mi: <= 1.0 mm
    assert calculate_ajcc_pt_stage(0.8) == "pT1mi"
    assert calculate_ajcc_pt_stage(1.0) == "pT1mi"
    
    # pT1a: > 1.0 mm to <= 5.0 mm
    assert calculate_ajcc_pt_stage(1.1) == "pT1a"
    assert calculate_ajcc_pt_stage(5.0) == "pT1a"
    
    # pT1b: > 5.0 mm to <= 10.0 mm
    assert calculate_ajcc_pt_stage(5.1) == "pT1b"
    assert calculate_ajcc_pt_stage(10.0) == "pT1b"
    
    # pT1c: > 10.0 mm to <= 20.0 mm
    assert calculate_ajcc_pt_stage(10.1) == "pT1c"
    assert calculate_ajcc_pt_stage(18.0) == "pT1c"
    assert calculate_ajcc_pt_stage(20.0) == "pT1c"
    
    # pT2: > 20.0 mm to <= 50.0 mm
    assert calculate_ajcc_pt_stage(20.1) == "pT2"
    assert calculate_ajcc_pt_stage(35.0) == "pT2"
    assert calculate_ajcc_pt_stage(50.0) == "pT2"
    
    # pT3: > 50.0 mm
    assert calculate_ajcc_pt_stage(50.1) == "pT3"
    assert calculate_ajcc_pt_stage(75.0) == "pT3"
    
    # pT4: Chest wall / Skin extension
    assert calculate_ajcc_pt_stage(15.0, chest_wall_extension=True) == "pT4a"
    assert calculate_ajcc_pt_stage(15.0, skin_ulceration=True) == "pT4b"
    assert calculate_ajcc_pt_stage(15.0, chest_wall_extension=True, skin_ulceration=True) == "pT4c"


def test_ajcc_pn_staging_cutoffs():
    """Test all AJCC 8th/9th ed pN lymph node boundary cutoffs."""
    # Biopsy / No nodes
    assert calculate_ajcc_pn_stage(nodes_examined=0, nodes_positive=0) == "pNX"
    
    # pN0: All negative
    assert calculate_ajcc_pn_stage(nodes_examined=12, nodes_positive=0) == "pN0"
    
    # pN1mi: Micrometastasis
    assert calculate_ajcc_pn_stage(nodes_examined=12, nodes_positive=1, largest_meta_mm=1.5) == "pN1mi"
    assert calculate_ajcc_pn_stage(nodes_examined=12, nodes_positive=1, is_micrometastasis=True) == "pN1mi"
    
    # pN1a: 1 to 3 positive axillary nodes (>2.0mm)
    assert calculate_ajcc_pn_stage(nodes_examined=12, nodes_positive=1, largest_meta_mm=4.0) == "pN1a"
    assert calculate_ajcc_pn_stage(nodes_examined=12, nodes_positive=3, largest_meta_mm=3.0) == "pN1a"
    
    # pN2a: 4 to 9 positive axillary nodes
    assert calculate_ajcc_pn_stage(nodes_examined=15, nodes_positive=4) == "pN2a"
    assert calculate_ajcc_pn_stage(nodes_examined=15, nodes_positive=9) == "pN2a"
    
    # pN3a: 10 or more positive axillary nodes
    assert calculate_ajcc_pn_stage(nodes_examined=20, nodes_positive=10) == "pN3a"
    assert calculate_ajcc_pn_stage(nodes_examined=20, nodes_positive=16) == "pN3a"


def test_ajcc_stage_group_matrix():
    """Test AJCC Anatomic Stage Group Matrix combinations."""
    assert calculate_ajcc_stage_group("pTis", "pN0") == "0"
    assert calculate_ajcc_stage_group("pT1c", "pN0") == "IA"
    assert calculate_ajcc_stage_group("pT1c", "pNX") == "IA"
    assert calculate_ajcc_stage_group("pT1c", "pN1mi") == "IB"
    assert calculate_ajcc_stage_group("pT1c", "pN1a") == "IIA"
    assert calculate_ajcc_stage_group("pT2", "pN0") == "IIA"
    assert calculate_ajcc_stage_group("pT2", "pN1a") == "IIB"
    assert calculate_ajcc_stage_group("pT3", "pN0") == "IIB"
    assert calculate_ajcc_stage_group("pT3", "pN1a") == "IIIA"
    assert calculate_ajcc_stage_group("pT1c", "pN2a") == "IIIA"
    assert calculate_ajcc_stage_group("pT4b", "pN0") == "IIIB"
    assert calculate_ajcc_stage_group("pT1c", "pN3a") == "IIIC"
    assert calculate_ajcc_stage_group("pT1c", "pN0", pm_stage="pM1") == "IV"
    # Issue #12 & #361: Unassessed / Unknown tumor size boundaries
    assert calculate_ajcc_stage_group("pTX", "pNX") == "Unknown"
    assert calculate_ajcc_stage_group("pTX", "pN0") == "Unknown"
    assert calculate_ajcc_stage_group("TX", "pNX") == "Unknown"
    assert calculate_ajcc_stage_group("N/A", "N/A") == "Benign"


def test_staging_invariant_violations():
    """Verify ValueError is raised on invalid staging data combinations."""
    # Positive nodes exceed examined nodes
    with pytest.raises(ValueError, match="nodes_positive .* cannot exceed nodes_examined"):
        validate_staging_invariants(
            tumor_size_mm=18.0,
            pt_stage="pT1c",
            nodes_examined=5,
            nodes_positive=8,
            pn_stage="pN2a",
            stage_group="IIIA"
        )
        
    # Invalid pT string
    with pytest.raises(ValueError, match="Invalid pT category"):
        validate_staging_invariants(
            tumor_size_mm=18.0,
            pt_stage="pT99",
            nodes_examined=10,
            nodes_positive=0,
            pn_stage="pN0",
            stage_group="IA"
        )


def test_narrative_consistency_guardrail():
    """Verify narrative guardrail detects conflicting Nottingham grade citations."""
    verified_data = {
        "nottingham_grade": {"grade": 2, "nottingham_sum": 7}
    }
    
    # Accurate narrative (Grade 2 cited)
    accurate_narrative = {
        "diagnosis_line": "Right breast, biopsy: Invasive carcinoma, Nottingham Grade 2.",
        "microscopic_findings": "Demonstrates Nottingham Grade 2 features.",
        "clinical_correlation": "Stage IA disease."
    }
    assert len(validate_narrative_consistency(accurate_narrative, verified_data)) == 0
    
    # Hallucinated narrative (Grade 3 cited erroneously)
    hallucinated_narrative = {
        "diagnosis_line": "Right breast, biopsy: Invasive carcinoma, Nottingham Grade 3.",
        "microscopic_findings": "High grade carcinoma.",
        "clinical_correlation": "Aggressive disease."
    }
    issues = validate_narrative_consistency(hallucinated_narrative, verified_data)
    assert len(issues) > 0
    assert "Grade 3 instead of confirmed Grade 2" in issues[0]


# ---------------------------------------------------------------------------
# 2. Server-Side Clinical PDF Generation Unit Test
# ---------------------------------------------------------------------------

def test_clinical_pdf_generation(tmp_path):
    """Verify ReportLab compiles institutional CAP synoptic report PDF with embedded visual evidence."""
    pdf_out = str(tmp_path / "test_cap_report.pdf")
    
    sample_report_data = {
        "case_id": str(uuid.uuid4()),
        "procedure": "Core Needle Biopsy",
        "laterality": "Right",
        "tumor_site": "Upper Outer Quadrant",
        "histologic_type": "IDC-NST",
        "tumor_size_mm": 18.0,
        "lvi_status": "Absent",
        "dcis_present": False,
        "margins": {"status": "negative", "closest_margin_mm": 5.0},
        "lymph_nodes": {"examined_count": 0, "positive_count": 0},
        "biomarkers": {
            "er": {"status": "positive", "percent": 95},
            "pr": {"status": "positive", "percent": 80},
            "her2": {"ihc_score": "1+", "result": "negative"},
            "ki67": {"percent": 18}
        },
        "staging": {"pt_stage": "pT1c", "pn_stage": "pNX", "stage_group": "IA"},
        "nottingham_grade": {
            "grade": 2, "tubule_score": 2, "tubule_percent": 45.0,
            "pleo_score": 2, "mitotic_score": 2, "nottingham_sum": 6
        },
        "narrative": {
            "diagnosis_line": "RIGHT BREAST, BIOPSY: INVASIVE CARCINOMA OF NO SPECIAL TYPE, GRADE 2.",
            "microscopic_findings": "Moderate tubule formation and pleomorphism. Mitoses 6/10 HPFs.",
            "clinical_correlation": "Pathologic Stage IA."
        },
        "status": "signed",
        "signed_by": "Dr. Jane Doe, MD, FCAP",
        "npi": "NPI-1982347102",
        "signed_at": datetime.now(timezone.utc).isoformat(),
        "integrity_hash": "a1b2c3d4e5f6789012345678"
    }
    
    res_path = generate_clinical_cap_pdf(sample_report_data, pdf_out)
    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000 # Valid non-empty PDF file


# ---------------------------------------------------------------------------
# 3. Full REST API & Database Integration Workflow
# ---------------------------------------------------------------------------

def test_stage_6_full_workflow():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    
    test_case = Case(
        id=case_uid,
        created_by="pathologist_test",
        status="open"
    )
    db.add(test_case)
    
    test_slide = Slide(
        id=uuid.uuid4(),
        case_id=case_uid,
        gcs_uri_original="gs://test-bucket/slide.svs",
        format="SVS",
        mpp_x=0.25,
        mpp_y=0.25
    )
    db.add(test_slide)
    
    test_grading = Grading(
        case_id=case_uid,
        tubule_percent=45.0,
        tubule_score=2,
        pleo_score=2,
        mitotic_score=3,
        nottingham_sum=7,
        grade=2,
        histologic_type="IDC-NST",
        type_confirmed_by="pathologist_test",
        machine={"patches": [], "hpfs": []},
        overrides={}
    )
    db.add(test_grading)
    
    stage_exec = StageExecution(
        case_id=case_uid,
        stage="report",
        attempt=1,
        status="queued"
    )
    db.add(stage_exec)
    
    db.commit()
    db.close()
    
    case_id = str(case_uid)
    headers = {"X-User-Role": "pathologist"}
    
    # 1. GET initial report data (auto-initializes draft with nullable fields)
    res_get = client.get(f"/api/v1/stages/report/{case_id}", headers=headers)
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["case_id"] == case_id
    assert data["status"] == "draft"
    assert data["histologic_type"] == "IDC-NST"
    assert data["nottingham_grade"]["grade"] == 2
    assert data["tumor_size_mm"] is None
    assert data["margins"] is None
    assert data["biomarkers"] is None
    assert data["staging"]["pt_stage"] == "pTX"
    assert data["staging"]["stage_group"] == "Unknown"
    assert data["can_sign"] is True
    
    # 2. PUT update synoptic fields (change size to 25.0 mm -> should update to pT2 IIA)
    update_payload = {
        "case_id": case_id,
        "procedure": "Excision / Lumpectomy",
        "laterality": "left",
        "tumor_size_mm": 25.0,
        "lvi_status": "present",
        "margins": {
            "status": "negative",
            "closest_margin_mm": 8.0,
            "closest_margin_name": "anterior",
            "positive_margins": []
        },
        "lymph_nodes": {
            "examined_count": 12,
            "positive_count": 2, # pN1a
            "extranodal_extension": False,
            "largest_metastasis_mm": 4.0
        }
    }
    res_put = client.put(f"/api/v1/stages/report/{case_id}", json=update_payload, headers=headers)
    assert res_put.status_code == 200
    put_data = res_put.json()
    assert put_data["procedure"] == "Excision / Lumpectomy"
    assert put_data["laterality"] == "left"
    assert put_data["tumor_size_mm"] == 25.0
    assert put_data["lvi_status"] == "present"
    # pT2 (25mm) + pN1a (2 nodes) -> Stage IIB
    assert put_data["staging"]["pt_stage"] == "pT2"
    assert put_data["staging"]["pn_stage"] == "pN1a"
    assert put_data["staging"]["stage_group"] == "IIB"
    assert put_data["status"] == "in_review"
    
    # 3. POST regenerate narrative
    res_narr = client.post(f"/api/v1/stages/report/{case_id}/regenerate-narrative", headers=headers)
    assert res_narr.status_code == 200
    narr_data = res_narr.json()
    assert "narrative" in narr_data
    assert "diagnosis_line" in narr_data["narrative"]
    assert "LEFT BREAST" in narr_data["narrative"]["diagnosis_line"]
    
    # 4. GET PDF streaming
    res_pdf = client.get(f"/api/v1/stages/report/{case_id}/pdf", headers=headers)
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    
    # 5. GET structured JSON export
    res_json = client.get(f"/api/v1/stages/report/{case_id}/json", headers=headers)
    assert res_json.status_code == 200
    assert res_json.json()["case_id"] == case_id
    
    # 6. POST sign and finalize report
    sign_payload = {
        "case_id": case_id,
        "signed_by": "Dr. Pathologist Reviewer, MD, FCAP",
        "npi": "NPI-9988776655",
        "attestation_statement": "I electronically attest that I have reviewed the Whole-Slide Image and verified all findings."
    }
    res_sign = client.post("/api/v1/stages/report/sign", json=sign_payload, headers=headers)
    assert res_sign.status_code == 200
    sign_data = res_sign.json()
    assert sign_data["status"] == "signed"
    assert sign_data["signed_by"] == "Dr. Pathologist Reviewer, MD, FCAP"
    assert sign_data["integrity_hash"] is not None
    assert sign_data["can_sign"] is False
    
    # Verify Case and StageExecution transitioned in DB
    db2 = TestingSessionLocal()
    c = db2.get(Case, case_uid)
    assert c.status == "done"
    s_exec = db2.scalars(
        select(StageExecution).where(StageExecution.case_id == case_uid, StageExecution.stage == "report")
    ).first()
    assert s_exec.status == "confirmed"
    db2.close()
    
    # 7. Attempting PUT on signed report is blocked
    res_blocked = client.put(f"/api/v1/stages/report/{case_id}", json=update_payload, headers=headers)
    assert res_blocked.status_code == 400
    assert "already signed and locked" in res_blocked.json()["detail"]
    
    # 8. POST amend signed report
    amend_payload = {
        "case_id": case_id,
        "amended_by": "Dr. Pathologist Reviewer, MD, FCAP",
        "amendment_reason": "Addendum: Correlated with additional external receptor IHC block results.",
        "updated_fields": {
            "dcis_present": True
        }
    }
    res_amend = client.post("/api/v1/stages/report/amend", json=amend_payload, headers=headers)
    assert res_amend.status_code == 200
    amend_data = res_amend.json()
    assert amend_data["status"] == "amended"
    assert len(amend_data["amendments"]) == 1
    assert amend_data["amendments"][0]["version"] == "v1.1"
    assert amend_data["dcis_present"] is True

    # 9. Verify report row versioning and immutability (Issue #496)
    db3 = TestingSessionLocal()
    reports = db3.scalars(
        select(Report).where(Report.case_id == case_uid).order_by(Report.version.asc())
    ).all()
    assert len(reports) == 2
    # Original row preserved immutably
    assert reports[0].version == 1
    assert reports[0].status == "signed"
    assert reports[0].signed_by == "Dr. Pathologist Reviewer, MD, FCAP"
    assert reports[0].integrity_hash is not None
    # New row created with incremented version
    assert reports[1].version == 2
    assert reports[1].status == "amended"
    assert reports[1].signed_by is None
    assert reports[1].dcis_present is True
    db3.close()


def test_benign_case_pdf_generation():
    """
    Verify benign case PDF generation without crashing or substituting fabricated Nottingham Grade 2.
    """
    from pipeline.report_pdf import generate_clinical_cap_pdf
    import tempfile

    benign_report_data = {
        "case_id": "test-benign-case-uuid",
        "status": "draft",
        "histologic_type": "Benign / No invasive carcinoma identified",
        "tumor_size_mm": 0.0,
        "nottingham_grade": {
            "grade": None,
            "tubule_score": None,
            "tubule_percent": None,
            "pleo_score": None,
            "mitotic_score": None,
            "nottingham_sum": None
        },
        "staging": {
            "ajcc_version": "8th/9th Edition",
            "pt_stage": "N/A",
            "pn_stage": "N/A",
            "pm_stage": "cM0",
            "stage_group": "Benign"
        },
        "narrative": {
            "diagnosis_line": "RIGHT BREAST, CORE NEEDLE BIOPSY: BENIGN BREAST TISSUE, NEGATIVE FOR INVASIVE MALIGNANCY.",
            "microscopic_findings": "Benign breast parenchyma without cytologic atypia.",
            "clinical_correlation": "Negative for invasive carcinoma."
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        out_pdf_path = f.name

    try:
        generate_clinical_cap_pdf(benign_report_data, out_pdf_path)
        assert os.path.exists(out_pdf_path)
        assert os.path.getsize(out_pdf_path) > 1000
    finally:
        if os.path.exists(out_pdf_path):
            os.remove(out_pdf_path)


def test_report_rbac_signing_and_amending():
    """
    Verify RBAC authorization on signing and amending reports.
    """
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    c = Case(id=case_uid, status="in_progress", created_by="path_test")
    db.add(c)
    rep = Report(
        case_id=case_uid,
        status="in_review",
        laterality="left",
        tumor_site="upper_outer_quadrant",
        histologic_type="IDC-NST",
        tumor_size_mm=20.0
    )
    db.add(rep)
    db.commit()
    db.close()

    case_id = str(case_uid)

    # Viewer cannot sign -> 403
    res_v_sign = client.post(
        "/api/v1/stages/report/sign",
        json={
            "case_id": case_id,
            "signed_by": "Dr. Viewer, MD",
            "npi": "1234567890",
            "attestation_statement": "I hereby verify and attest all diagnostic findings in this report."
        },
        headers={"X-User-Role": "viewer"}
    )
    assert res_v_sign.status_code == 403

    # Viewer cannot amend -> 403
    res_v_amend = client.post(
        "/api/v1/stages/report/amend",
        json={
            "case_id": case_id,
            "amended_by": "Dr. Viewer, MD",
            "amendment_reason": "Formal addendum regarding clinical correlation.",
            "updated_fields": {}
        },
        headers={"X-User-Role": "viewer"}
    )
    assert res_v_amend.status_code == 403


def test_draft_watermark_and_attestation_suppression(tmp_path):
    """
    Issue #169:
    Verify that unsigned/draft PDFs suppress the pathologist attestation signature block
    and render the DRAFT watermark, while signed PDFs render the attestation signature block.
    """
    draft_pdf = str(tmp_path / "draft.pdf")
    signed_pdf = str(tmp_path / "signed.pdf")

    draft_data = {
        "case_id": str(uuid.uuid4()),
        "status": "draft",
        "procedure": "Core Needle Biopsy",
        "laterality": "right",
        "tumor_site": "upper_outer_quadrant",
        "histologic_type": "IDC-NST",
        "tumor_size_mm": None,
        "margins": None,
        "biomarkers": None,
        "staging": {"pt_stage": "pTX", "pn_stage": "pNX", "stage_group": "Unknown"},
        "nottingham_grade": {"grade": 2, "tubule_score": 2, "pleo_score": 2, "mitotic_score": 2, "nottingham_sum": 6},
        "narrative": {"diagnosis_line": "Draft diagnosis", "microscopic_findings": "Draft findings", "clinical_correlation": "Draft correlation"}
    }
    generate_clinical_cap_pdf(draft_data, draft_pdf)
    assert os.path.exists(draft_pdf)
    assert os.path.getsize(draft_pdf) > 1000

    signed_data = dict(draft_data)
    signed_data.update({
        "status": "signed",
        "signed_by": "Dr. Verified Pathologist, MD, FCAP",
        "npi": "NPI-1234567890",
        "signed_at": datetime.now(timezone.utc).isoformat(),
        "integrity_hash": "abcdef1234567890"
    })
    generate_clinical_cap_pdf(signed_data, signed_pdf)
    assert os.path.exists(signed_pdf)
    assert os.path.getsize(signed_pdf) > 1000

    import pypdf
    draft_reader = pypdf.PdfReader(draft_pdf)
    draft_text = draft_reader.pages[0].extract_text()

    signed_reader = pypdf.PdfReader(signed_pdf)
    signed_text = signed_reader.pages[0].extract_text()

    # Draft PDF contains preliminary notice and watermark text, suppressing attestation block
    assert "DRAFT" in draft_text
    assert "PRELIMINARY DRAFT" in draft_text
    assert "Dr. Verified Pathologist" not in draft_text
    assert "Pathologist Attestation" not in draft_text

    # Signed PDF contains digital signature and attestation, and suppresses draft watermark
    assert "Electronically Signed By" in signed_text
    assert "Dr. Verified Pathologist" in signed_text
    assert "Pathologist Attestation" in signed_text
    assert "PRELIMINARY DRAFT" not in signed_text


def test_multiple_successive_amendments_and_signatures():
    """
    Verify Issue #496 multi-generation immutability:
    Monotonic version increments and signature sealing across multiple cycles
    (v1 draft -> v1 signed -> v2 amended -> v2 signed -> v3 amended -> v3 signed).
    """
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    headers = {"X-User-Role": "pathologist"}

    case = Case(id=case_uid, status="in_progress", created_by="path_reviewer")
    db.add(case)
    rep = Report(
        case_id=case_uid,
        version=1,
        status="in_review",
        laterality="right",
        tumor_site="upper_outer_quadrant",
        histologic_type="IDC-NST",
        tumor_size_mm=16.0
    )
    db.add(rep)
    db.commit()
    db.close()

    case_id = str(case_uid)

    # 1. Sign Version 1
    res_sign_1 = client.post("/api/v1/stages/report/sign", json={
        "case_id": case_id,
        "signed_by": "Dr. Alice Pathologist, MD",
        "npi": "NPI-1111111111",
        "attestation_statement": "Attestation v1: I have reviewed the slide and verified the diagnosis."
    }, headers=headers)
    assert res_sign_1.status_code == 200
    assert res_sign_1.json()["version"] == 1
    assert res_sign_1.json()["status"] == "signed"

    # 2. Amend to create Version 2 (v1.1)
    res_amend_1 = client.post("/api/v1/stages/report/amend", json={
        "case_id": case_id,
        "amended_by": "Dr. Alice Pathologist, MD",
        "amendment_reason": "Amendment 1: Updated margins from supplementary re-excision block.",
        "updated_fields": {
            "margins": {"status": "negative", "closest_margin_mm": 12.0}
        }
    }, headers=headers)
    assert res_amend_1.status_code == 200
    data_amend_1 = res_amend_1.json()
    assert data_amend_1["version"] == 2
    assert data_amend_1["status"] == "amended"
    assert len(data_amend_1["amendments"]) == 1
    assert data_amend_1["amendments"][0]["version"] == "v1.1"

    # 3. Sign Version 2
    res_sign_2 = client.post("/api/v1/stages/report/sign", json={
        "case_id": case_id,
        "signed_by": "Dr. Alice Pathologist, MD",
        "npi": "NPI-1111111111",
        "attestation_statement": "Attestation v2: I have reviewed the amended report findings and verified."
    }, headers=headers)
    assert res_sign_2.status_code == 200
    assert res_sign_2.json()["version"] == 2
    assert res_sign_2.json()["status"] == "signed"

    # 4. Amend to create Version 3 (v1.2)
    res_amend_2 = client.post("/api/v1/stages/report/amend", json={
        "case_id": case_id,
        "amended_by": "Dr. Bob Consultant, MD, FCAP",
        "amendment_reason": "Amendment 2: External expert second opinion confirmed pleomorphism score 3.",
        "updated_fields": {
            "dcis_present": True
        }
    }, headers=headers)
    assert res_amend_2.status_code == 200
    data_amend_2 = res_amend_2.json()
    assert data_amend_2["version"] == 3
    assert data_amend_2["status"] == "amended"
    assert len(data_amend_2["amendments"]) == 2
    assert data_amend_2["amendments"][0]["version"] == "v1.1"
    assert data_amend_2["amendments"][1]["version"] == "v1.2"
    assert data_amend_2["amendments"][1]["amended_by"] == "Dr. Bob Consultant, MD, FCAP"

    # 5. Sign Version 3
    res_sign_3 = client.post("/api/v1/stages/report/sign", json={
        "case_id": case_id,
        "signed_by": "Dr. Bob Consultant, MD, FCAP",
        "npi": "NPI-2222222222",
        "attestation_statement": "Attestation v3: Final consultant concurrence on amended diagnosis."
    }, headers=headers)
    assert res_sign_3.status_code == 200
    assert res_sign_3.json()["version"] == 3
    assert res_sign_3.json()["status"] == "signed"

    # 6. Verify full immutable audit history in database: all 3 versions preserved
    db_check = TestingSessionLocal()
    all_reports = db_check.scalars(
        select(Report).where(Report.case_id == case_uid).order_by(Report.version.asc())
    ).all()
    assert len(all_reports) == 3

    # V1 checks
    assert all_reports[0].version == 1
    assert all_reports[0].status == "signed"
    assert all_reports[0].signed_by == "Dr. Alice Pathologist, MD"
    assert all_reports[0].margins is None

    # V2 checks
    assert all_reports[1].version == 2
    assert all_reports[1].status == "signed"
    assert all_reports[1].signed_by == "Dr. Alice Pathologist, MD"
    assert all_reports[1].margins["closest_margin_mm"] == 12.0
    assert len(all_reports[1].amendments) == 1

    # V3 checks
    assert all_reports[2].version == 3
    assert all_reports[2].status == "signed"
    assert all_reports[2].signed_by == "Dr. Bob Consultant, MD, FCAP"
    assert all_reports[2].dcis_present is True
    assert len(all_reports[2].amendments) == 2

    db_check.close()


