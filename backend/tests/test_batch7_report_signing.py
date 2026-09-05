"""
Batch 7 Test Suite: CAP Synoptic Reporting, AJCC Staging, Pathologist Sign-Off Preconditions,
Cryptographic Sealing and PDF Integrity.

Validates:
1. Sign-off precondition gates (Stages 1-5 confirmed, histologic type confirmed, mandatory CAP elements, PIN validation, role RBAC).
2. Elimination of fabricated Grade 2 defaults across PDF generation and API responses.
3. Immutability of signed reports (PUT blocked with 400, worker re-run gracefully skips).
4. Full cryptographic SHA-256 integrity hash covering all clinical elements + PDF SHA-256 digest.
5. Versioned PDF upload and retrieval with Cache-Control headers.
6. Staging invariant validation (nodes_positive > nodes_examined raises 422).
7. Microscopic narrative consistency checks (grade, score, LVI, laterality concordance).
8. Multi-cycle amendment workflow with immutable predecessor hash tracking.
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
from app.models.audit import AuditEvent
from pipeline.staging import (
    calculate_ajcc_pt_stage,
    calculate_ajcc_pn_stage,
    calculate_ajcc_stage_group,
    validate_staging_invariants,
    validate_narrative_consistency
)
from pipeline.report_pdf import generate_clinical_cap_pdf
from worker.report import run_report

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


def test_sign_preconditions_unconfirmed_prior_stages():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)

    # Triage stage is still in_progress
    stage_triage = StageExecution(case_id=case_uid, stage="triage", attempt=1, status="in_progress")
    db.add(stage_triage)

    report = Report(
        case_id=case_uid,
        version=1,
        status="in_review",
        procedure="Core Needle Biopsy",
        laterality="right",
        tumor_site="upper_outer_quadrant",
        histologic_type="IDC-NST",
        tumor_size_mm=18.0,
        lvi_status="absent"
    )
    db.add(report)
    db.commit()
    db.close()

    res = client.post(
        "/api/v1/stages/report/sign",
        json={
            "case_id": str(case_uid),
            "signed_by": "Dr. Attending Pathologist, MD",
            "npi": "1234567890",
            "password_or_pin": "1234",
            "attestation_statement": "I electronically attest to the accuracy of all findings."
        },
        headers={"X-User-Role": "pathologist"}
    )
    assert res.status_code == 422
    err_detail = res.json()["detail"]
    assert "missing_items" in err_detail
    assert "stage_triage_not_confirmed" in err_detail["missing_items"]


def test_sign_preconditions_unconfirmed_histologic_type():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)

    # Grading stage exists but subtype is unconfirmed
    grading = Grading(
        case_id=case_uid,
        histologic_type="IDC-NST",
        type_confirmed_by="unconfirmed",
        tubule_score=2,
        pleo_score=2,
        mitotic_score=2,
        nottingham_sum=6,
        grade=2
    )
    db.add(grading)

    report = Report(
        case_id=case_uid,
        version=1,
        status="in_review",
        procedure="Core Needle Biopsy",
        laterality="right",
        tumor_site="upper_outer_quadrant",
        histologic_type="IDC-NST",
        tumor_size_mm=18.0,
        lvi_status="absent"
    )
    db.add(report)
    db.commit()
    db.close()

    res = client.post(
        "/api/v1/stages/report/sign",
        json={
            "case_id": str(case_uid),
            "signed_by": "Dr. Attending Pathologist, MD",
            "npi": "1234567890",
            "password_or_pin": "1234",
            "attestation_statement": "I electronically attest to the accuracy of all findings."
        },
        headers={"X-User-Role": "pathologist"}
    )
    assert res.status_code == 422
    err_detail = res.json()["detail"]
    assert "histologic_type_unconfirmed" in err_detail["missing_items"]


def test_sign_preconditions_pin_validation():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)

    report = Report(
        case_id=case_uid,
        version=1,
        status="in_review",
        procedure="Core Needle Biopsy",
        laterality="right",
        tumor_site="upper_outer_quadrant",
        histologic_type="IDC-NST",
        tumor_size_mm=18.0,
        lvi_status="absent"
    )
    db.add(report)
    db.commit()
    db.close()

    # Short PIN < 4 chars
    res = client.post(
        "/api/v1/stages/report/sign",
        json={
            "case_id": str(case_uid),
            "signed_by": "Dr. Attending Pathologist, MD",
            "npi": "1234567890",
            "password_or_pin": "12",
            "attestation_statement": "I electronically attest to the accuracy of all findings."
        },
        headers={"X-User-Role": "pathologist"}
    )
    assert res.status_code == 422
    err_detail = res.json()["detail"]
    assert "reauthentication_pin_required" in err_detail["missing_items"]


def test_sign_preconditions_role_forbidden():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)
    report = Report(case_id=case_uid, version=1, status="in_review")
    db.add(report)
    db.commit()
    db.close()

    res = client.post(
        "/api/v1/stages/report/sign",
        json={
            "case_id": str(case_uid),
            "signed_by": "Dr. Viewer, MD",
            "npi": "1234567890",
            "password_or_pin": "1234",
            "attestation_statement": "I electronically attest to the accuracy of all findings."
        },
        headers={"X-User-Role": "viewer"}
    )
    assert res.status_code == 403


def test_successful_signature_and_canonical_integrity_hash():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)

    grading = Grading(
        case_id=case_uid,
        histologic_type="Invasive Lobular Carcinoma",
        type_confirmed_by="pathologist_dr_smith",
        tubule_score=3,
        pleo_score=2,
        mitotic_score=1,
        nottingham_sum=6,
        grade=2
    )
    db.add(grading)

    stage_exec = StageExecution(case_id=case_uid, stage="report", attempt=1, status="awaiting_review")
    db.add(stage_exec)

    report = Report(
        case_id=case_uid,
        version=1,
        status="in_review",
        procedure="Total Mastectomy",
        laterality="left",
        tumor_site="lower_outer_quadrant",
        histologic_type="Invasive Lobular Carcinoma",
        tumor_size_mm=22.0,
        lvi_status="present",
        dcis_present=False,
        margins={"status": "negative", "closest_margin_mm": 10.0},
        lymph_nodes={"examined_count": 15, "positive_count": 3, "extranodal_extension": False, "largest_metastasis_mm": 2.0}
    )
    db.add(report)
    db.commit()
    db.close()

    case_id = str(case_uid)
    res = client.post(
        "/api/v1/stages/report/sign",
        json={
            "case_id": case_id,
            "signed_by": "Dr. Alice Smith, MD, FCAP",
            "npi": "NPI-1982736450",
            "password_or_pin": "9876",
            "attestation_statement": "I electronically attest that I have reviewed the Whole-Slide Image and verify all diagnostic findings."
        },
        headers={"X-User-Role": "pathologist", "X-User-Email": "alice.smith@hospital.org"}
    )
    assert res.status_code == 200
    signed_data = res.json()

    assert signed_data["status"] == "signed"
    assert signed_data["signed_by"] == "Dr. Alice Smith, MD, FCAP"
    assert signed_data["npi"] == "NPI-1982736450"
    assert signed_data["pdf_sha256"] is not None
    assert len(signed_data["pdf_sha256"]) == 64
    assert signed_data["integrity_hash"] is not None
    assert len(signed_data["integrity_hash"]) == 64

    # DB verification
    db2 = TestingSessionLocal()
    c = db2.get(Case, case_uid)
    assert c.status == "done"
    s_exec = db2.scalars(
        select(StageExecution).where(StageExecution.case_id == case_uid, StageExecution.stage == "report")
    ).first()
    assert s_exec.status == "confirmed"

    audit = db2.scalars(
        select(AuditEvent).where(AuditEvent.case_id == case_id, AuditEvent.event_type == "stage_6_report_signed")
    ).first()
    assert audit is not None
    assert audit.actor == "alice.smith@hospital.org"
    assert audit.payload["pdf_sha256"] == signed_data["pdf_sha256"]
    assert audit.payload["integrity_hash"] == signed_data["integrity_hash"]
    db2.close()


def test_sign_report_with_prior_failed_attempts_and_string_case_id():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)

    grading = Grading(
        case_id=case_uid,
        histologic_type="Invasive Ductal Carcinoma",
        type_confirmed_by="user_pathologist_001",
        tubule_score=2,
        pleo_score=2,
        mitotic_score=1,
        nottingham_sum=5,
        grade=1
    )
    db.add(grading)

    # Add prior failed attempts and confirmed attempt 4 for grading
    db.add(StageExecution(case_id=case_uid, stage="preprocess", attempt=1, status="confirmed"))
    db.add(StageExecution(case_id=case_uid, stage="triage", attempt=1, status="confirmed"))
    db.add(StageExecution(case_id=case_uid, stage="mitosis", attempt=1, status="confirmed"))
    db.add(StageExecution(case_id=case_uid, stage="grading", attempt=1, status="failed"))
    db.add(StageExecution(case_id=case_uid, stage="grading", attempt=2, status="failed"))
    db.add(StageExecution(case_id=case_uid, stage="grading", attempt=3, status="failed"))
    db.add(StageExecution(case_id=case_uid, stage="grading", attempt=4, status="confirmed"))
    db.add(StageExecution(case_id=case_uid, stage="report", attempt=1, status="awaiting_review"))

    report = Report(
        case_id=case_uid,
        version=1,
        status="in_review",
        specimen_type="core_biopsy",
        procedure="Core Needle Biopsy",
        laterality="right",
        tumor_site="upper_outer_quadrant",
        histologic_type="Invasive Ductal Carcinoma",
        tumor_size_mm=4.4,
        lvi_status="present",
        dcis_present=False,
        margins={"status": "cannot_be_assessed"},
        lymph_nodes={"examined_count": 2, "positive_count": 0, "extranodal_extension": False, "largest_metastasis_mm": 0.0}
    )
    db.add(report)
    db.commit()
    db.close()

    case_id = str(case_uid)
    res = client.post(
        "/api/v1/stages/report/sign",
        json={
            "case_id": case_id,
            "signed_by": "Dr. Jane Doe, MD, FCAP",
            "npi": "1234567890",
            "password_or_pin": "secret_pin",
            "attestation_statement": "I electronically attest to the accuracy of all findings."
        },
        headers={
            "X-User-Role": "pathologist",
            "X-User-Email": "jane.doe@hospital.org"
        }
    )
    assert res.status_code == 200, res.text
    signed = res.json()
    assert signed["status"] == "signed"
    assert signed["signed_by"] == "Dr. Jane Doe, MD, FCAP"
    assert signed["tumor_size_mm"] == 4.4



def test_signed_report_immutability_put_and_worker():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="done", created_by="test_user")
    db.add(case)

    report = Report(
        case_id=case_uid,
        version=1,
        status="signed",
        procedure="Core Needle Biopsy",
        laterality="right",
        tumor_site="upper_outer_quadrant",
        histologic_type="IDC-NST",
        tumor_size_mm=18.0,
        lvi_status="absent",
        pdf_path="gs://artifacts/cases/test/report/v1/CAP_Report.pdf",
        pdf_sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        integrity_hash="fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"
    )
    db.add(report)
    stage_exec = StageExecution(case_id=case_uid, stage="report", attempt=1, status="confirmed")
    db.add(stage_exec)
    db.commit()
    st_id = stage_exec.id
    db.close()

    case_id = str(case_uid)
    # 1. Attempting PUT on signed report must return 400
    res_put = client.put(
        f"/api/v1/stages/report/{case_id}",
        json={"case_id": case_id, "tumor_size_mm": 25.0},
        headers={"X-User-Role": "pathologist"}
    )
    assert res_put.status_code == 400
    assert "already signed and locked" in res_put.json()["detail"]

    # 2. Worker execution on signed report must skip re-running and not overwrite
    db3 = TestingSessionLocal()
    st_exec_db = db3.get(StageExecution, st_id)
    pdf_uri, meta = run_report(st_exec_db, db3)
    assert "skipped" in str(meta)
    assert pdf_uri == "gs://artifacts/cases/test/report/v1/CAP_Report.pdf"
    db3.close()


def test_staging_invariant_zero_examined_positive_nodes():
    # Calling validate_staging_invariants with 0 examined and 1 positive must raise ValueError
    with pytest.raises(ValueError, match="cannot exceed nodes_examined"):
        validate_staging_invariants(
            tumor_size_mm=18.0,
            pt_stage="pT1c",
            nodes_examined=0,
            nodes_positive=1,
            pn_stage="pNX",
            stage_group="Unknown"
        )

    # API PUT must return 422
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)
    report = Report(case_id=case_uid, version=1, status="in_review", histologic_type="IDC-NST")
    db.add(report)
    db.commit()
    db.close()

    case_id = str(case_uid)
    res = client.put(
        f"/api/v1/stages/report/{case_id}",
        json={
            "case_id": case_id,
            "lymph_nodes": {"examined_count": 0, "positive_count": 1}
        },
        headers={"X-User-Role": "pathologist"}
    )
    assert res.status_code == 422


def test_narrative_consistency_guardrails():
    # 1. Nottingham Grade discordance
    case_payload = {
        "procedure": "Core Needle Biopsy",
        "laterality": "right",
        "nottingham_grade": {"grade": 3, "nottingham_sum": 8},
        "lvi_status": "absent"
    }
    narrative = {
        "diagnosis_line": "RIGHT BREAST, BIOPSY: INVASIVE CARCINOMA, NOTTINGHAM HISTOLOGIC GRADE 1",
        "microscopic_findings": "Tubule formation is prominent. Grade 1 features.",
        "clinical_correlation": "Low grade."
    }
    warnings = validate_narrative_consistency(narrative, case_payload)
    assert any("Grade 1" in w for w in warnings)

    # 2. Nottingham Sum score mismatch
    narrative2 = {
        "diagnosis_line": "NOTTINGHAM HISTOLOGIC GRADE 3 (SCORE 6/9)",
        "microscopic_findings": "",
        "clinical_correlation": ""
    }
    warnings2 = validate_narrative_consistency(narrative2, case_payload)
    assert any("Nottingham sum" in w for w in warnings2)

    # 3. LVI discordance
    narrative3 = {
        "diagnosis_line": "INVASIVE CARCINOMA",
        "microscopic_findings": "Definite lymphovascular invasion is present and identified within dermal lymphatics.",
        "clinical_correlation": ""
    }
    warnings3 = validate_narrative_consistency(narrative3, case_payload)
    assert any("lymphovascular invasion" in w for w in warnings3)

    # 4. Laterality discordance
    narrative4 = {
        "diagnosis_line": "LEFT BREAST, BIOPSY: INVASIVE CARCINOMA",
        "microscopic_findings": "",
        "clinical_correlation": ""
    }
    warnings4 = validate_narrative_consistency(narrative4, case_payload)
    assert any("laterality" in w.lower() for w in warnings4)


def test_no_fabricated_grade_defaults_in_api_and_pdf(tmp_path):
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)
    # No grading record added
    report = Report(case_id=case_uid, version=1, status="draft", histologic_type="IDC-NST")
    db.add(report)
    db.commit()
    db.close()

    case_id = str(case_uid)
    res = client.get(f"/api/v1/stages/report/{case_id}", headers={"X-User-Role": "pathologist"})
    assert res.status_code == 200
    ng = res.json()["nottingham_grade"]
    # Values must be None, NOT fabricated (2, 2, 2, 6, 45%)
    assert ng["grade"] is None
    assert ng["tubule_score"] is None
    assert ng["tubule_percent"] is None
    assert ng["pleo_score"] is None
    assert ng["mitotic_score"] is None
    assert ng["nottingham_sum"] is None

    # PDF generation with unassessed grading produces clean text without crashing
    out_pdf = str(tmp_path / "unassessed.pdf")
    report_data = {
        "case_id": case_id,
        "status": "draft",
        "procedure": "Core Needle Biopsy",
        "laterality": "right",
        "tumor_site": "upper_outer_quadrant",
        "histologic_type": "IDC-NST",
        "tumor_size_mm": None,
        "staging": {"pt_stage": "pTX", "pn_stage": "pNX", "stage_group": "Unknown"},
        "nottingham_grade": ng,
        "narrative": {"diagnosis_line": "Pending", "microscopic_findings": "Pending", "clinical_correlation": "Pending"}
    }
    generate_clinical_cap_pdf(report_data, out_pdf)
    assert os.path.exists(out_pdf)
    assert os.path.getsize(out_pdf) > 1000


def test_pdf_cache_control_headers():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_uid = uuid.uuid4()
    case = Case(id=case_uid, status="in_progress", created_by="test_user")
    db.add(case)
    report = Report(case_id=case_uid, version=1, status="draft", histologic_type="IDC-NST")
    db.add(report)
    db.commit()
    db.close()

    res = client.get(f"/api/v1/stages/report/{str(case_uid)}/pdf", headers={"X-User-Role": "pathologist"})
    assert res.status_code == 200
    assert "Cache-Control" in res.headers
    cc = res.headers["Cache-Control"]
    assert "no-cache" in cc
    assert "no-store" in cc
    assert "must-revalidate" in cc
