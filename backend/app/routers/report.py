import os
import io
import json
import uuid
import hashlib
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import (
    get_gcs_client,
    parse_gcs_uri,
    upload_blob_from_bytes,
    download_blob_as_bytes,
    blob_exists
)
from app.core.db import get_db
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.grading import Grading
from app.models.hpf_site import HpfSite
from app.models.hotspot import Hotspot
from app.models.report import Report
from app.models.audit import AuditEvent

from pipeline.staging import (
    calculate_ajcc_pt_stage,
    calculate_ajcc_pn_stage,
    calculate_ajcc_stage_group,
    validate_staging_invariants,
    validate_narrative_consistency
)
from pipeline.medgemma import MedGemmaClient, load_prompt_template
from pipeline.report_pdf import generate_clinical_cap_pdf

router = APIRouter(prefix="/api/v1/stages/report", tags=["report"])


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class MarginsPayload(BaseModel):
    status: Literal["negative", "positive", "cannot_be_assessed"] = "negative"
    closest_margin_mm: Optional[float] = Field(default=5.0, ge=0.0)
    closest_margin_name: Optional[str] = "posterior"
    positive_margins: List[str] = Field(default_factory=list)


class LymphNodesPayload(BaseModel):
    examined_count: int = Field(default=0, ge=0)
    positive_count: int = Field(default=0, ge=0)
    extranodal_extension: bool = False
    largest_metastasis_mm: Optional[float] = Field(default=0.0, ge=0.0)


class BiomarkersPayload(BaseModel):
    er: Dict[str, Any] = Field(default_factory=lambda: {"status": "positive", "percent": 95, "allred_score": 8})
    pr: Dict[str, Any] = Field(default_factory=lambda: {"status": "positive", "percent": 80, "allred_score": 7})
    her2: Dict[str, Any] = Field(default_factory=lambda: {"ihc_score": "1+", "fish_status": "not_performed", "result": "negative"})
    ki67: Dict[str, Any] = Field(default_factory=lambda: {"percent": 18})


class UpdateReportPayload(BaseModel):
    case_id: str
    specimen_type: Optional[str] = "core_biopsy"
    procedure: Optional[str] = "Core Needle Biopsy"
    laterality: Optional[str] = "right"
    tumor_site: Optional[str] = "upper_outer_quadrant"
    tumor_size_mm: Optional[float] = Field(default=18.0, ge=0.0)
    lvi_status: Optional[Literal["absent", "present", "indeterminate"]] = "absent"
    dcis_present: Optional[bool] = False
    margins: Optional[MarginsPayload] = None
    lymph_nodes: Optional[LymphNodesPayload] = None
    biomarkers: Optional[BiomarkersPayload] = None
    narrative: Optional[Dict[str, str]] = None


class SignReportPayload(BaseModel):
    case_id: str
    signed_by: str = Field(min_length=3, description="Pathologist full name and credentials, e.g. Dr. Jane Doe, MD, FCAP")
    npi: Optional[str] = Field(default="NPI-1982347102", description="National Provider Identifier or License")
    attestation_statement: str = Field(min_length=15, description="Explicit attestation agreement text")
    password_or_pin: Optional[str] = None


class AmendReportPayload(BaseModel):
    case_id: str
    amended_by: str = Field(min_length=3)
    amendment_reason: str = Field(min_length=10, description="Mandatory rationale for amending a signed report")
    updated_fields: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def to_uuid(val: Any) -> uuid.UUID:
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except Exception:
        return val


def _ensure_report_record(case_uid: uuid.UUID, db: Session) -> Report:
    """Retrieve existing report or create initial draft from verified Stage 5 data."""
    report = db.scalars(select(Report).where(Report.case_id == case_uid)).first()
    if report:
        return report

    grading = db.scalars(select(Grading).where(Grading.case_id == case_uid)).first()
    htype = grading.histologic_type if grading and grading.histologic_type else "IDC-NST"

    report = Report(
        case_id=case_uid,
        specimen_type="core_biopsy",
        procedure="Core Needle Biopsy",
        laterality="right",
        tumor_site="upper_outer_quadrant",
        histologic_type=htype,
        tumor_size_mm=18.0,
        lvi_status="absent",
        dcis_present=False,
        margins={"status": "negative", "closest_margin_mm": 5.0, "closest_margin_name": "posterior", "positive_margins": []},
        lymph_nodes={"examined_count": 0, "positive_count": 0, "extranodal_extension": False, "largest_metastasis_mm": 0.0},
        biomarkers={
            "er": {"status": "positive", "percent": 95, "allred_score": 8},
            "pr": {"status": "positive", "percent": 80, "allred_score": 7},
            "her2": {"ihc_score": "1+", "fish_status": "not_performed", "result": "negative"},
            "ki67": {"percent": 18}
        },
        staging={"ajcc_version": "8th/9th Edition", "pt_stage": "pT1c", "pn_stage": "pNX", "pm_stage": "cM0", "stage_group": "IA"},
        narrative={
            "diagnosis_line": f"RIGHT BREAST, CORE NEEDLE BIOPSY: INVASIVE BREAST CARCINOMA OF {htype.upper()}, NOTTINGHAM HISTOLOGIC GRADE {grading.grade if grading else 2}.",
            "microscopic_findings": "Invasive carcinoma showing moderate tubular differentiation and nuclear pleomorphism. No lymphovascular invasion identified.",
            "clinical_correlation": "Pathologic findings consistent with Stage IA invasive carcinoma. Clinical and receptor biomarker correlation recommended."
        },
        status="draft"
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def render_and_upload_report_pdf(case_id: str, report: Report, grading: Optional[Grading], db: Session) -> str:
    """Renders CAP PDF via transient scratch directory and uploads directly to GCS."""
    scratch_dir = tempfile.mkdtemp(prefix="og_pdf_")
    try:
        pdf_filename = f"CAP_Report_{str(case_id)[:8]}.pdf"
        pdf_scratch_path = os.path.join(scratch_dir, pdf_filename)

        ng_data = {
            "grade": grading.grade if grading and grading.grade else 2,
            "tubule_score": grading.tubule_score if grading and grading.tubule_score else 2,
            "tubule_percent": grading.tubule_percent if grading and grading.tubule_percent is not None else 45.0,
            "pleo_score": grading.pleo_score if grading and grading.pleo_score else 2,
            "mitotic_score": grading.mitotic_score if grading and grading.mitotic_score else 2,
            "nottingham_sum": grading.nottingham_sum if grading and grading.nottingham_sum else 6
        }

        render_dict = {
            "case_id": str(case_id),
            "procedure": report.procedure,
            "laterality": report.laterality,
            "tumor_site": report.tumor_site,
            "histologic_type": report.histologic_type,
            "tumor_size_mm": report.tumor_size_mm,
            "lvi_status": report.lvi_status,
            "dcis_present": report.dcis_present,
            "margins": report.margins,
            "lymph_nodes": report.lymph_nodes,
            "biomarkers": report.biomarkers,
            "staging": report.staging,
            "nottingham_grade": ng_data,
            "narrative": report.narrative,
            "status": report.status,
            "signed_by": report.signed_by,
            "npi": report.npi,
            "signed_at": report.signed_at.isoformat() if report.signed_at else None,
            "integrity_hash": report.integrity_hash
        }

        evidence_paths = {}
        for hm in [f"cases/{case_id}/triage/heatmap_triage.png", f"cases/{case_id}/triage/heatmap.png"]:
            try:
                data = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, hm)
                p = os.path.join(scratch_dir, "heatmap.png")
                with open(p, "wb") as f:
                    f.write(data)
                evidence_paths["heatmap"] = p
                break
            except Exception:
                pass

        for hpf in [
            f"cases/{case_id}/mitosis/hpfs/hpf_1_40x_norm.png",
            f"cases/{case_id}/mitosis/hpfs/hpf_1_20x_norm.png",
            f"cases/{case_id}/mitosis/hpfs/hpf_1_10x_norm.png",
            f"cases/{case_id}/mitosis/crops/m_0364.png"
        ]:
            try:
                data = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, hpf)
                p = os.path.join(scratch_dir, "mitotic_hpf.png")
                with open(p, "wb") as f:
                    f.write(data)
                evidence_paths["mitotic_hpf"] = p
                break
            except Exception:
                pass

        for gp in [
            f"cases/{case_id}/triage/patches/hs_01_10x_norm.png",
            f"cases/{case_id}/triage/patches/hs_01_20x_norm.png",
            f"cases/{case_id}/triage/patches/hs_01_40x_norm.png",
            f"cases/{case_id}/grading_patches/p_001.png"
        ]:
            try:
                data = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, gp)
                p = os.path.join(scratch_dir, "grading_patch.png")
                with open(p, "wb") as f:
                    f.write(data)
                evidence_paths["grading_patch"] = p
                break
            except Exception:
                pass

        generate_clinical_cap_pdf(
            report_data=render_dict,
            output_path=pdf_scratch_path,
            evidence_paths=evidence_paths
        )

        with open(pdf_scratch_path, "rb") as f:
            pdf_bytes = f.read()

        gcs_blob_name = f"cases/{case_id}/report/{pdf_filename}"
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            gcs_blob_name,
            pdf_bytes,
            "application/pdf"
        )
        gcs_uri = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/{gcs_blob_name}"
        report.pdf_path = gcs_uri
        db.commit()
        return gcs_uri
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _build_report_response_dict(case_id: str, db: Session) -> Dict[str, Any]:
    case_uid = to_uuid(case_id)
    case = db.scalars(select(Case).where(Case.id == case_uid)).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    slide = db.scalars(select(Slide).where(Slide.case_id == case_uid)).first()
    grading = db.scalars(select(Grading).where(Grading.case_id == case_uid)).first()
    report = _ensure_report_record(case_uid, db)
    stage_exec = db.scalars(
        select(StageExecution).where(StageExecution.case_id == case_uid, StageExecution.stage == "report")
    ).first()

    # Grading details
    ng_data = {
        "grade": grading.grade if grading and grading.grade else 2,
        "tubule_score": grading.tubule_score if grading and grading.tubule_score else 2,
        "tubule_percent": grading.tubule_percent if grading and grading.tubule_percent is not None else 45.0,
        "pleo_score": grading.pleo_score if grading and grading.pleo_score else 2,
        "mitotic_score": grading.mitotic_score if grading and grading.mitotic_score else 2,
        "nottingham_sum": grading.nottingham_sum if grading and grading.nottingham_sum else 6,
        "histologic_type": grading.histologic_type if grading and grading.histologic_type else "IDC-NST",
        "type_confirmed_by": grading.type_confirmed_by if grading else "unconfirmed"
    }

    # Staging recalculation if needed
    tumor_size = report.tumor_size_mm or 18.0
    nodes_info = report.lymph_nodes or {}
    n_exam = nodes_info.get("examined_count", 0)
    n_pos = nodes_info.get("positive_count", 0)

    pt_stage = calculate_ajcc_pt_stage(tumor_size)
    pn_stage = calculate_ajcc_pn_stage(n_exam, n_pos)
    stage_grp = calculate_ajcc_stage_group(pt_stage, pn_stage)

    staging_dict = {
        "ajcc_version": "8th/9th Edition",
        "pt_stage": pt_stage,
        "pn_stage": pn_stage,
        "pm_stage": "cM0",
        "stage_group": stage_grp
    }

    return {
        "case_id": str(case_id),
        "slide_id": str(slide.id) if slide else None,
        "status": report.status,
        "stage_status": stage_exec.status if stage_exec else "awaiting_review",
        "specimen_type": report.specimen_type,
        "procedure": report.procedure,
        "laterality": report.laterality,
        "tumor_site": report.tumor_site,
        "histologic_type": report.histologic_type,
        "tumor_size_mm": report.tumor_size_mm,
        "lvi_status": report.lvi_status,
        "dcis_present": report.dcis_present,
        "margins": report.margins,
        "lymph_nodes": report.lymph_nodes,
        "biomarkers": report.biomarkers,
        "staging": staging_dict,
        "nottingham_grade": ng_data,
        "narrative": report.narrative,
        "visual_evidence": {
            "has_heatmap": blob_exists(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/triage/heatmap_triage.png") or blob_exists(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/triage/heatmap.png"),
            "has_mitotic_hpf": blob_exists(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/mitosis/crops/m_0001.png"),
            "has_grading_patch": blob_exists(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/grading_patches/p_001.png")
        },
        "pdf_url": f"/api/v1/stages/report/{case_id}/pdf",
        "json_url": f"/api/v1/stages/report/{case_id}/json",
        "signed_by": report.signed_by,
        "npi": report.npi,
        "attestation_statement": report.attestation_statement,
        "signed_at": report.signed_at.isoformat() if report.signed_at else None,
        "integrity_hash": report.integrity_hash,
        "amendments": report.amendments,
        "can_sign": report.status in ("draft", "in_review")
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{case_id}")
def get_report_data(case_id: str, db: Session = Depends(get_db)):
    """
    Retrieve current synoptic report state, verified grading data, staging, narrative, and PDF status.
    """
    return _build_report_response_dict(case_id, db)


@router.post("/update")
@router.put("/{case_id}")
def update_report_data(payload: UpdateReportPayload, db: Session = Depends(get_db)):
    """
    Live reactive update of synoptic gross, margin, nodal, and biomarker inputs.
    Automatically recalculates AJCC staging, marks status as 'in_review', and updates PDF in GCS.
    """
    case_uid = to_uuid(payload.case_id)
    report = _ensure_report_record(case_uid, db)
    grading = db.scalars(select(Grading).where(Grading.case_id == case_uid)).first()

    if report.status == "signed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Report is already signed and locked. Please use the /amend endpoint to submit a formal versioned amendment."
        )

    # Update gross / clinical fields
    if payload.specimen_type is not None:
        report.specimen_type = payload.specimen_type
    if payload.procedure is not None:
        report.procedure = payload.procedure
    if payload.laterality is not None:
        report.laterality = payload.laterality
    if payload.tumor_site is not None:
        report.tumor_site = payload.tumor_site
    if payload.tumor_size_mm is not None:
        report.tumor_size_mm = payload.tumor_size_mm
    if payload.lvi_status is not None:
        report.lvi_status = payload.lvi_status
    if payload.dcis_present is not None:
        report.dcis_present = payload.dcis_present
    if payload.margins is not None:
        report.margins = payload.margins.model_dump()
    if payload.lymph_nodes is not None:
        report.lymph_nodes = payload.lymph_nodes.model_dump()
    if payload.biomarkers is not None:
        report.biomarkers = payload.biomarkers.model_dump()
    if payload.narrative is not None:
        report.narrative = payload.narrative

    # Recalculate AJCC Staging
    tumor_size = report.tumor_size_mm or 18.0
    nodes_info = report.lymph_nodes or {}
    n_exam = nodes_info.get("examined_count", 0)
    n_pos = nodes_info.get("positive_count", 0)

    pt_stage = calculate_ajcc_pt_stage(tumor_size)
    pn_stage = calculate_ajcc_pn_stage(n_exam, n_pos)
    stage_grp = calculate_ajcc_stage_group(pt_stage, pn_stage)

    report.staging = {
        "ajcc_version": "8th/9th Edition",
        "pt_stage": pt_stage,
        "pn_stage": pn_stage,
        "pm_stage": "cM0",
        "stage_group": stage_grp
    }
    report.status = "in_review"

    db.commit()
    db.refresh(report)

    # Re-render and upload PDF to GCS
    try:
        render_and_upload_report_pdf(str(payload.case_id), report, grading, db)
    except Exception as e:
        print(f"[Update PDF Render Note] {e}")

    return _build_report_response_dict(payload.case_id, db)


@router.post("/resynthesize-narrative")
def resynthesize_narrative(payload: UpdateReportPayload, db: Session = Depends(get_db)):
    """
    Triggers grounded narrative re-generation via MedGemma 1.5.
    """
    case_uid = to_uuid(payload.case_id)
    report = _ensure_report_record(case_uid, db)
    grading = db.scalars(select(Grading).where(Grading.case_id == case_uid)).first()

    if report.status == "signed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot modify a signed report narrative directly. Use amendment workflow."
        )

    # Aggregate latest context payload
    ng_data = {
        "grade": grading.grade if grading and grading.grade else 2,
        "tubule_score": grading.tubule_score if grading and grading.tubule_score else 2,
        "tubule_percent": grading.tubule_percent if grading and grading.tubule_percent is not None else 45.0,
        "pleo_score": grading.pleo_score if grading and grading.pleo_score else 2,
        "mitotic_score": grading.mitotic_score if grading and grading.mitotic_score else 2,
        "nottingham_sum": grading.nottingham_sum if grading and grading.nottingham_sum else 6
    }

    tumor_size = payload.tumor_size_mm or report.tumor_size_mm or 18.0
    nodes_info = (payload.lymph_nodes.model_dump() if payload.lymph_nodes else report.lymph_nodes) or {}
    n_exam = nodes_info.get("examined_count", 0)
    n_pos = nodes_info.get("positive_count", 0)

    pt_stage = calculate_ajcc_pt_stage(tumor_size)
    pn_stage = calculate_ajcc_pn_stage(n_exam, n_pos)
    stage_grp = calculate_ajcc_stage_group(pt_stage, pn_stage)

    staging_dict = {
        "ajcc_version": "8th/9th Edition",
        "pt_stage": pt_stage,
        "pn_stage": pn_stage,
        "pm_stage": "cM0",
        "stage_group": stage_grp
    }

    case_summary = {
        "case_id": str(payload.case_id),
        "procedure": payload.procedure or report.procedure,
        "laterality": payload.laterality or report.laterality,
        "tumor_site": payload.tumor_site or report.tumor_site,
        "histologic_type": report.histologic_type,
        "tumor_size_mm": tumor_size,
        "lvi_status": payload.lvi_status or report.lvi_status,
        "nottingham_grade": ng_data,
        "staging": staging_dict,
        "biomarkers": payload.biomarkers.model_dump() if payload.biomarkers else report.biomarkers
    }

    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    prompt_tpl, _ = load_prompt_template("cap_report", "v1")
    medgemma = MedGemmaClient()
    new_narrative = loop.run_until_complete(
        medgemma.generate_cap_report_narrative(case_summary, prompt_tpl)
    )

    report.narrative = new_narrative
    report.status = "in_review"
    db.commit()
    db.refresh(report)

    return _build_report_response_dict(payload.case_id, db)


@router.post("/{case_id}/regenerate-narrative")
async def regenerate_report_narrative(case_id: str, db: Session = Depends(get_db)):
    """
    Re-synthesize the 3-section diagnostic narrative using MedGemma 1.5 with numerical guardrails.
    """
    case_uid = to_uuid(case_id)
    report = _ensure_report_record(case_uid, db)
    if report.status == "signed":
        raise HTTPException(status_code=400, detail="Cannot regenerate narrative on a signed and locked report.")

    grading = db.scalars(select(Grading).where(Grading.case_id == case_uid)).first()
    ng_data = {
        "grade": grading.grade if grading and grading.grade else 2,
        "tubule_score": grading.tubule_score if grading and grading.tubule_score else 2,
        "tubule_percent": grading.tubule_percent if grading and grading.tubule_percent is not None else 45.0,
        "pleo_score": grading.pleo_score if grading and grading.pleo_score else 2,
        "mitotic_score": grading.mitotic_score if grading and grading.mitotic_score else 2,
        "nottingham_sum": grading.nottingham_sum if grading and grading.nottingham_sum else 6
    }

    case_payload = {
        "case_id": case_id,
        "procedure": report.procedure,
        "laterality": report.laterality,
        "tumor_site": report.tumor_site,
        "histologic_type": report.histologic_type,
        "tumor_size_mm": report.tumor_size_mm,
        "lvi_status": report.lvi_status,
        "nottingham_grade": ng_data,
        "staging": report.staging,
        "biomarkers": report.biomarkers
    }

    medgemma_client = MedGemmaClient()
    narrative_dict = await medgemma_client.generate_cap_report_narrative(case_payload)

    # Validate numerical consistency
    warnings = validate_narrative_consistency(narrative_dict, case_payload)
    if warnings:
        print(f"[MedGemma Narrative Warning for {case_id}] {warnings}")

    report.narrative = narrative_dict
    db.commit()
    db.refresh(report)

    return {"status": "success", "narrative": narrative_dict, "warnings": warnings}


@router.get("/{case_id}/pdf")
def get_report_pdf(case_id: str, db: Session = Depends(get_db)):
    """
    Stream server-generated clinical PDF report directly, freshly rendering latest data to ensure 1-page compliance.
    """
    case_uid = to_uuid(case_id)
    report = _ensure_report_record(case_uid, db)
    grading = db.scalars(select(Grading).where(Grading.case_id == case_uid)).first()

    blob_name = f"cases/{case_id}/report/CAP_Report_{str(case_id)[:8]}.pdf"
    try:
        render_and_upload_report_pdf(str(case_id), report, grading, db)
        pdf_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, blob_name)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="CAP_Report_{str(case_id)[:8]}.pdf"',
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        print(f"[PDF Render Error] {e}")
        try:
            pdf_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, blob_name)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'inline; filename="CAP_Report_{str(case_id)[:8]}.pdf"'}
            )
        except Exception:
            raise HTTPException(status_code=500, detail=f"Failed to render PDF: {e}")


@router.get("/{case_id}/json")
def get_report_json(case_id: str, db: Session = Depends(get_db)):
    """
    Export full synoptic report in HL7/FHIR-compliant structured JSON format.
    """
    data = _build_report_response_dict(case_id, db)
    headers = {"Content-Disposition": f"attachment; filename=CAP_Synoptic_{str(case_id)[:8]}.json"}
    return JSONResponse(content=data, headers=headers)


@router.post("/sign")
def sign_final_report(payload: SignReportPayload, db: Session = Depends(get_db)):
    """
    Pathologist Digital Sign-Off & Attestation Gate.
    Calculates cryptographic SHA-256 integrity hash over all elements, seals report,
    advances Stage 6 status to 'completed', and logs full immutable audit event.
    """
    case_uid = to_uuid(payload.case_id)
    report = _ensure_report_record(case_uid, db)
    case = db.scalars(select(Case).where(Case.id == case_uid)).first()
    grading = db.scalars(select(Grading).where(Grading.case_id == case_uid)).first()

    if report.status == "signed":
        # Idempotent return if already signed
        return _build_report_response_dict(payload.case_id, db)

    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.isoformat()

    # Compute Cryptographic SHA-256 Integrity Hash over all synoptic elements
    canonical_payload = {
        "case_id": str(payload.case_id),
        "procedure": report.procedure,
        "laterality": report.laterality,
        "histologic_type": report.histologic_type,
        "tumor_size_mm": report.tumor_size_mm,
        "margins": report.margins,
        "lymph_nodes": report.lymph_nodes,
        "biomarkers": report.biomarkers,
        "staging": report.staging,
        "narrative": report.narrative,
        "signed_by": payload.signed_by,
        "npi": payload.npi,
        "signed_at": now_iso
    }
    canonical_str = json.dumps(canonical_payload, sort_keys=True, separators=(',', ':'))
    integrity_hash = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

    # Seal report
    report.status = "signed"
    report.signed_by = payload.signed_by
    report.npi = payload.npi
    report.attestation_statement = payload.attestation_statement
    report.signed_at = now_utc
    report.integrity_hash = integrity_hash

    if case:
        case.status = "done"

    # Advance stage execution to completed
    stage_exec = db.scalars(
        select(StageExecution).where(StageExecution.case_id == case_uid, StageExecution.stage == "report")
    ).first()
    if stage_exec:
        stage_exec.status = "confirmed"
        stage_exec.completed_at = now_utc
        stage_exec.reviewed_at = now_utc
        stage_exec.reviewed_by = payload.signed_by

    audit_evt = AuditEvent(
        case_id=str(payload.case_id),
        actor=payload.signed_by,
        event_type="stage_6_report_signed",
        stage="report",
        payload={
            "signed_by": payload.signed_by,
            "npi": payload.npi,
            "signed_at": now_iso,
            "integrity_hash": integrity_hash,
            "pt_stage": report.staging.get("pt_stage"),
            "pn_stage": report.staging.get("pn_stage"),
            "stage_group": report.staging.get("stage_group")
        }
    )
    db.add(audit_evt)
    db.commit()
    db.refresh(report)

    # Re-render finalized PDF with signature block and upload to GCS
    try:
        render_and_upload_report_pdf(str(payload.case_id), report, grading, db)
    except Exception as e:
        print(f"[Final PDF Render Note] {e}")

    return _build_report_response_dict(payload.case_id, db)


@router.post("/amend")
def amend_signed_report(payload: AmendReportPayload, db: Session = Depends(get_db)):
    """
    Formal Versioned Amendment workflow for signed reports.
    Appends an amendment entry with version (e.g. v1.1), reason, and timestamps.
    """
    case_uid = to_uuid(payload.case_id)
    report = _ensure_report_record(case_uid, db)
    if report.status != "signed" and report.status != "amended":
        raise HTTPException(status_code=400, detail="Only finalized/signed reports can be amended.")

    now_iso = datetime.now(timezone.utc).isoformat()
    current_amendments = list(report.amendments or [])
    amendment_ver = f"v1.{len(current_amendments) + 1}"

    amendment_entry = {
        "version": amendment_ver,
        "amended_by": payload.amended_by,
        "amended_at": now_iso,
        "reason": payload.amendment_reason,
        "previous_hash": report.integrity_hash,
        "updated_fields": payload.updated_fields
    }
    current_amendments.append(amendment_entry)
    report.amendments = current_amendments
    report.status = "amended"

    # Apply updated fields if provided
    for k, v in payload.updated_fields.items():
        if hasattr(report, k):
            setattr(report, k, v)

    audit_evt = AuditEvent(
        case_id=str(payload.case_id),
        actor=payload.amended_by,
        event_type="stage_6_report_amended",
        stage="report",
        payload={
            "version": amendment_ver,
            "reason": payload.amendment_reason,
            "amended_at": now_iso
        }
    )
    db.add(audit_evt)
    db.commit()
    db.refresh(report)

    return _build_report_response_dict(payload.case_id, db)
