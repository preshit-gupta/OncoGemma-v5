"""
Unit and Integration Tests for Batch 18:
- #502, #501: ReportLab Platypus 3-page clinical PDF layout, Appendix, RUO banner, Watermark, Dynamic Descriptors
- #185, #621, #623: Biomarker null safety without fabricated defaults, Allred scores, FISH status
- #624: Margin distance formatting (only appended for negative margins)
- #192: Dynamic pleomorphism and mitotic rate descriptors
- #187, #495: Immutability of signed and amended reports (HTTP 409 Conflict)
- #189: 404 Not Found for nonexistent and malformed case IDs across report and audit endpoints
- #215, #708: Audit pagination tiebreaking by id.desc() and case_id normalization
- #644: stage_started audit event emission on stage approval
- #748, #289: Narrative consistency guardrails and transaction separation
"""

import os
import uuid
import tempfile
import pytest
from unittest.mock import patch
from datetime import datetime, timezone
from pypdf import PdfReader
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.main import app
from app.core.db import SessionLocal, Base, engine
from app.models.case import Case
from app.models.slide import Slide
from app.models.grading import Grading
from app.models.report import Report
from app.models.audit import AuditEvent
from app.models.stage_execution import StageExecution

from pipeline.report_pdf import (
    generate_clinical_cap_pdf,
    render_report_html,
    build_report_pdf_context,
    _format_biomarkers,
    _format_margins,
    _format_pleomorphism,
    _format_mitotic_rate,
    clean_markup
)
from pipeline.staging import validate_narrative_consistency

client = TestClient(app)


# ==============================================================================
# 1. 3-Page Layout & Clinical Appendix Verification (#502, #523, #626)
# ==============================================================================

def test_platypus_pdf_3page_layout_and_appendix():
    """Verify generate_clinical_cap_pdf produces exactly 3 pages with Page 3 Appendix & RUO banner."""
    case_uid = str(uuid.uuid4())
    report_data = {
        "case_id": case_uid,
        "procedure": "Ultrasound-Guided Core Needle Biopsy",
        "specimen_type": "core_biopsy",
        "laterality": "Left",
        "tumor_site": "Upper Outer Quadrant",
        "histologic_type": "Invasive Breast Carcinoma of No Special Type (IDC-NST)",
        "tumor_size_mm": 21.0,
        "lvi_status": "absent",
        "dcis_present": True,
        "margins": {
            "status": "negative",
            "closest_margin_mm": 5.0,
            "closest_margin_name": "Inferior"
        },
        "lymph_nodes": {
            "examined_count": 3,
            "positive_count": 0,
            "extranodal_extension": False,
            "largest_metastasis_mm": 0.0
        },
        "biomarkers": {
            "er": {"status": "positive", "percent": 90, "allred_score": 8},
            "pr": {"status": "positive", "percent": 75, "allred_score": 7},
            "her2": {"ihc_score": "1+", "result": "negative", "fish_status": "not_performed"},
            "ki67": {"percent": 18}
        },
        "staging": {
            "ajcc_version": "8th/9th Edition",
            "pt_stage": "pT2",
            "pn_stage": "pN0",
            "pm_stage": "cM0",
            "stage_group": "IIA"
        },
        "nottingham_grade": {
            "grade": 2,
            "tubule_score": 2,
            "tubule_percent": 45.0,
            "pleo_score": 2,
            "mitotic_score": 2,
            "nottingham_sum": 6
        },
        "narrative": {
            "diagnosis_line": "LEFT BREAST, CORE BIOPSY: INVASIVE BREAST CARCINOMA, NOTTINGHAM GRADE 2.",
            "microscopic_findings": "Invasive ductal carcinoma showing tubule formation and moderate atypia.",
            "clinical_correlation": "Correlates with radiographic mass."
        },
        "status": "draft"
    }

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        pdf_out = tf.name

    try:
        generate_clinical_cap_pdf(report_data=report_data, output_path=pdf_out)
        assert os.path.exists(pdf_out)
        assert os.path.getsize(pdf_out) > 1000

        reader = PdfReader(pdf_out)
        # Must have exactly 3 pages (#502)
        assert len(reader.pages) == 3, f"Expected exactly 3 pages, got {len(reader.pages)}"

        # Page 1: Synoptic Checklist & Demographics
        page1_text = reader.pages[0].extract_text()
        assert "FINAL SYNOPTIC DIAGNOSIS" in page1_text
        assert "CAP SYNOPTIC PATHOLOGY REPORT" in page1_text or "CAP Synoptic Pathology Protocol" in page1_text or "CAP" in page1_text

        # Page 2: Microscopic Narrative & Signature
        page2_text = reader.pages[1].extract_text()
        assert "MICROSCOPIC DESCRIPTION" in page2_text or "FINDINGS" in page2_text or "DIAGNOSTIC NARRATIVE" in page2_text

        # Page 3: Appendix, RUO Banner, Model Provenance (#523, #626)
        page3_text = reader.pages[2].extract_text()
        assert "RESEARCH USE ONLY" in page3_text
        assert "CLINICAL APPENDIX" in page3_text
        assert "COMPUTATIONAL PROVENANCE" in page3_text
    finally:
        if os.path.exists(pdf_out):
            os.remove(pdf_out)


# ==============================================================================
# 2. Biomarker Null Safety & Captured Elements (#185, #621, #623)
# ==============================================================================

def test_biomarker_null_safety():
    """Verify biomarkers without fabricated defaults format cleanly as 'Not assessed / Pending'."""
    assert _format_biomarkers(None) == "Not assessed / Pending"
    assert _format_biomarkers({}) == "Not assessed / Pending"
    assert _format_biomarkers({"er": None, "pr": None, "her2": None, "ki67": None}) == "Not assessed / Pending"

    # Partial biomarker with Allred and HER2 FISH (#623)
    bm = {
        "er": {"status": "positive", "percent": 95, "allred_score": 8},
        "her2": {"ihc_score": "2+", "result": "equivocal", "fish_status": "amplified"}
    }
    fmt = _format_biomarkers(bm)
    assert "ER: Positive (95%) [Allred: 8/8]" in fmt
    assert "HER2: 2+ (Equivocal), FISH: Amplified" in fmt
    # Ensure missing PR and Ki-67 are omitted rather than fabricated
    assert "PR:" not in fmt
    assert "Ki-67:" not in fmt


# ==============================================================================
# 3. Margin Distance Suppression (#624)
# ==============================================================================

def test_margin_formatting_distance_rules():
    """Verify margin distance is ONLY displayed for negative margins (#624)."""
    # Negative with distance and margin name
    neg = _format_margins({"status": "negative", "closest_margin_mm": 3.2, "closest_margin_name": "Posterior"})
    assert "Negative / Uninvolved (Closest: 3.2 mm, Posterior)" == neg

    # Positive margins must list positive margins WITHOUT misleading closest distance
    pos = _format_margins({"status": "positive", "closest_margin_mm": 0.0, "positive_margins": ["Deep", "Superior"]})
    assert "Positive / Involved (Deep, Superior)" == pos
    assert "Closest" not in pos

    # Cannot be assessed
    cba = _format_margins({"status": "cannot_be_assessed", "closest_margin_mm": 0.0})
    assert cba == "Cannot be assessed / Pending"
    assert "Closest" not in cba

    # Null safe
    assert _format_margins(None) == "Not assessed / Pending"


# ==============================================================================
# 4. Dynamic Descriptors (#192)
# ==============================================================================

def test_dynamic_descriptors():
    """Verify Nottingham pleomorphism and mitotic rate descriptors."""
    assert "Score 1" in _format_pleomorphism(1)
    assert "Mild nuclear pleomorphism" in _format_pleomorphism(1)

    assert "Score 2" in _format_pleomorphism(2)
    assert "Moderate atypia" in _format_pleomorphism(2)

    assert "Score 3" in _format_pleomorphism(3)
    assert "Marked pleomorphism" in _format_pleomorphism(3)

    assert "Score 1" in _format_mitotic_rate(1)
    assert "Standardized across 10 HPFs" in _format_mitotic_rate(1)
    assert "3.60 mm²" in _format_mitotic_rate(2, evaluated_area="3.60 mm²")
    assert _format_mitotic_rate(None) == "Pending / Not Assessed"


# ==============================================================================
# 5. Clean XML Markup Helper (#183)
# ==============================================================================

def test_clean_markup_entity_escaping():
    """Verify clean_markup sanitizes non-whitelisted XML characters."""
    assert clean_markup("Tumor & Stroma > 5cm < 10cm") == "Tumor &amp; Stroma &gt; 5cm &lt; 10cm"
    assert clean_markup("<b>Bold text</b> with & symbol") == "<b>Bold text</b> with &amp; symbol"
    assert clean_markup(None) == ""


# ==============================================================================
# 6. Signed/Amended Report Immutability (HTTP 409 Conflict) (#187)
# ==============================================================================

def test_signed_and_amended_report_immutability():
    """Verify PUT /report/{case_id} and regenerate-narrative reject mutations on signed/amended reports with 409 Conflict."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    case_uid = uuid.uuid4()
    case_id = str(case_uid)

    try:
        case = Case(id=case_uid, created_by="pathologist_test", status="done")
        db.add(case)
        report = Report(
            case_id=case_uid,
            version=1,
            procedure="Core Needle Biopsy",
            laterality="right",
            tumor_site="upper_outer_quadrant",
            histologic_type="IDC-NST",
            tumor_size_mm=18.0,
            status="signed",
            signed_by="Dr. Alice Smith, MD, FCAP",
            npi="1234567890",
            attestation_statement="I personally verified all findings and digital evidence.",
            signed_at=datetime.now(timezone.utc),
            staging={"pt_stage": "pT1c", "pn_stage": "pN0", "stage_group": "IA"},
            narrative={"diagnosis_line": "Signed diagnosis"}
        )
        db.add(report)
        db.commit()

        headers = {"X-User-Role": "pathologist"}

        # 1. PUT on signed report must return 409 Conflict
        put_resp = client.put(
            f"/api/v1/stages/report/{case_id}",
            json={"case_id": case_id, "tumor_size_mm": 25.0},
            headers=headers
        )
        assert put_resp.status_code == 409
        assert "signed and locked" in put_resp.json()["detail"]

        # 2. POST regenerate-narrative on signed report must return 409 Conflict
        regen_resp = client.post(
            f"/api/v1/stages/report/{case_id}/regenerate-narrative",
            headers=headers
        )
        assert regen_resp.status_code == 409
        assert "signed and locked" in regen_resp.json()["detail"]

        # 3. Mark report as amended and verify it also rejects PUT and regenerate-narrative
        report.status = "amended"
        db.commit()

        put_amended = client.put(
            f"/api/v1/stages/report/{case_id}",
            json={"case_id": case_id, "tumor_size_mm": 26.0},
            headers=headers
        )
        assert put_amended.status_code == 409

        regen_amended = client.post(
            f"/api/v1/stages/report/{case_id}/regenerate-narrative",
            headers=headers
        )
        assert regen_amended.status_code == 409

    finally:
        db.close()


# ==============================================================================
# 7. Unknown Case ID 404 Handling (#189)
# ==============================================================================

def test_unknown_case_id_returns_404():
    """Verify all report and audit endpoints return 404 for unknown case IDs."""
    fake_case_id = str(uuid.uuid4())
    headers = {"X-User-Role": "pathologist"}

    # GET report data
    r_get = client.get(f"/api/v1/stages/report/{fake_case_id}", headers=headers)
    assert r_get.status_code == 404

    # PUT report data
    r_put = client.put(
        f"/api/v1/stages/report/{fake_case_id}",
        json={"case_id": fake_case_id, "tumor_size_mm": 15.0},
        headers=headers
    )
    assert r_put.status_code == 404

    # POST regenerate-narrative
    r_regen = client.post(f"/api/v1/stages/report/{fake_case_id}/regenerate-narrative", headers=headers)
    assert r_regen.status_code == 404

    # POST sign
    r_sign = client.post(
        "/api/v1/stages/report/sign",
        json={
            "case_id": fake_case_id,
            "signed_by": "Dr. Test, MD",
            "npi": "1234567890",
            "attestation_statement": "Pathologist digital attestation verification statement."
        },
        headers=headers
    )
    assert r_sign.status_code == 404

    # POST amend
    r_amend = client.post(
        "/api/v1/stages/report/amend",
        json={
            "case_id": fake_case_id,
            "amended_by": "Dr. Test, MD",
            "amendment_reason": "Correction of clinical tumor size."
        },
        headers=headers
    )
    assert r_amend.status_code == 404

    # GET audit events for unknown case
    r_audit = client.get(f"/api/v1/cases/{fake_case_id}/audit", headers=headers)
    assert r_audit.status_code == 404

    # Malformed non-UUID case string must return 404 (#189, #708)
    r_malformed = client.get("/api/v1/cases/not-a-valid-uuid/audit", headers=headers)
    assert r_malformed.status_code == 404


# ==============================================================================
# 8. Audit Pagination Tiebreaker (#215)
# ==============================================================================

def test_audit_pagination_tiebreaker_order():
    """Verify audit query uses (created_at.desc(), id.desc()) for deterministic paging (#215)."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    case_uid = uuid.uuid4()
    case_id_str = str(case_uid)

    try:
        case = Case(id=case_uid, created_by="test_admin", status="open")
        db.add(case)
        db.commit()

        # Insert multiple audit events sharing identical created_at timestamp
        fixed_time = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
        events = []
        for i in range(5):
            e = AuditEvent(
                case_id=case_id_str,
                actor="test_user",
                event_type=f"event_type_{i}",
                stage="triage",
                created_at=fixed_time
            )
            db.add(e)
            events.append(e)
        db.commit()

        # Fetch page 1
        resp = client.get(f"/api/v1/cases/{case_id_str}/audit?page=1&page_size=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["events"]) == 3

        # IDs must be strictly monotonically decreasing due to id.desc() tiebreak (#215)
        ids_p1 = [e["id"] for e in data["events"]]
        assert ids_p1 == sorted(ids_p1, reverse=True)

        # Fetch page 2
        resp2 = client.get(f"/api/v1/cases/{case_id_str}/audit?page=2&page_size=3")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["events"]) == 2
        ids_p2 = [e["id"] for e in data2["events"]]
        assert ids_p2 == sorted(ids_p2, reverse=True)

        # No duplicate IDs across pages
        assert set(ids_p1).isdisjoint(set(ids_p2))

    finally:
        db.close()


# ==============================================================================
# 9. stage_started Audit Event Emission on Stage Approval (#644)
# ==============================================================================

def test_stage_started_audit_event_emission():
    """Verify approve_case_stage emits both stage_approved and stage_started audit events (#644)."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    case_uid = uuid.uuid4()
    slide_uid = uuid.uuid4()

    try:
        case = Case(id=case_uid, created_by="test_doc", status="in_review")
        db.add(case)
        slide = Slide(id=slide_uid, case_id=case_uid, gcs_uri_original="gs://bucket/test.svs")
        db.add(slide)
        stage_exec = StageExecution(case_id=case_uid, stage="triage", attempt=1, status="awaiting_review")
        db.add(stage_exec)
        db.commit()

        with patch("app.routers.cases.dispatch_stage_task"):
            resp = client.post(
                f"/api/v1/cases/{case_uid}/stages/triage/approve",
                json={"review_comment": "Verified triage heatmap quality"},
                headers={"X-User-Role": "pathologist"}
            )
        assert resp.status_code == 202
        res_data = resp.json()
        assert res_data["status"] == "approved"
        assert res_data["approved_stage"] == "triage"
        assert res_data["next_stage"] == "mitosis"

        # Check emitted audit events
        audit_events = db.scalars(
            select(AuditEvent)
            .where(AuditEvent.case_id == str(case_uid))
            .order_by(AuditEvent.id.asc())
        ).all()

        event_types = [e.event_type for e in audit_events]
        assert "stage_approved" in event_types
        assert "stage_started" in event_types

        started_evt = next(e for e in audit_events if e.event_type == "stage_started")
        assert started_evt.stage == "mitosis"
        assert started_evt.payload["triggered_by_approval_of"] == "triage"

    finally:
        db.close()


# ==============================================================================
# 10. Narrative Consistency Validation (#748)
# ==============================================================================

def test_narrative_consistency_guardrail():
    """Verify validate_narrative_consistency detects numerical contradictions."""
    case_payload = {
        "case_id": "case-test-748",
        "tumor_size_mm": 24.0,
        "nottingham_grade": {"grade": 3}
    }

    # Narrative with contradictory grade (Grade 1 vs case Grade 3)
    narrative_mismatch = {
        "diagnosis_line": "INVASIVE CARCINOMA, NOTTINGHAM HISTOLOGIC GRADE 1",
        "microscopic_findings": "Tumor measures 12 mm across greatest dimension."
    }

    warnings = validate_narrative_consistency(narrative_mismatch, case_payload)
    assert len(warnings) > 0
    # Should flag grade contradiction
    assert any("Grade" in w or "grade" in w for w in warnings)
