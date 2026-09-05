import os
import io
import json
import asyncio
import tempfile
import shutil
from typing import Tuple, Dict, Any
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import (
    get_gcs_client,
    upload_blob_from_bytes,
    download_blob_as_bytes,
    download_blob_to_filename
)
from app.models.stage_execution import StageExecution
from app.models.case import Case
from app.models.slide import Slide
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
import hashlib


def run_report(stage_exec: StageExecution, db: Session) -> Tuple[str, Dict[str, str]]:
    """
    Executes Stage 6 (Report Generation) pipeline:
    1. Gathers clinical, grading, and staging parameters.
    2. Runs MedGemma CAP-compliant synoptic report generation.
    3. Renders high-fidelity clinical PDF using GCS-backed evidence assets.
    4. Uploads final PDF directly to GCS artifacts bucket.
    5. Cleans up temporary scratch directory.
    """
    case_id = str(stage_exec.case_id)
    case_uid = stage_exec.case_id
    
    print(f"[Stage 6 Worker] Generating CAP-compliant report for Case {case_id}...")

    # Guard: Do not overwrite or recalculate signed or amended reports (#170)
    existing_report = db.scalars(
        select(Report).where(Report.case_id == case_uid).order_by(Report.version.desc())
    ).first()
    if existing_report and existing_report.status in ("signed", "amended"):
        print(f"[Stage 6 Worker] Case {case_id} report is already {existing_report.status}. Skipping regeneration.")
        stage_exec.status = "awaiting_review"
        stage_exec.output_ref = existing_report.pdf_path
        db.commit()
        return existing_report.pdf_path or "", {"status": f"skipped_already_{existing_report.status}"}

    case = db.scalars(select(Case).where(Case.id == case_uid)).first()
    slide = db.scalars(select(Slide).where(Slide.case_id == case_uid)).first()
    grading = db.scalars(select(Grading).where(Grading.case_id == case_uid)).first()
    hpfs = list(db.scalars(select(HpfSite).where(HpfSite.case_id == case_uid)).all())
    hotspots = list(db.scalars(select(Hotspot).where(Hotspot.case_id == case_uid)).all())

    # 1. Check if triage or case indicated benign (no invasive tumor)
    input_ref = stage_exec.input_ref or {}
    is_benign = bool(input_ref.get("benign_flag", False))

    # Extract verified Stage 4 & 5 values (no fabricated defaults, #532)
    if is_benign:
        histologic_type = "Benign / No invasive carcinoma identified"
        grade_val = None
        tubule_score = None
        tubule_pct = None
        pleo_score = None
        mitotic_score = None
        nottingham_sum = None
        tumor_size = 0.0
        pt_stage = "N/A"
        pn_stage = "N/A"
        stage_grp = "Benign"
    else:
        grade_val = grading.grade if grading and grading.grade else None
        tubule_score = grading.tubule_score if grading and grading.tubule_score is not None else None
        tubule_pct = grading.tubule_percent if grading and grading.tubule_percent is not None else None
        pleo_score = grading.pleo_score if grading and grading.pleo_score is not None else None
        mitotic_score = grading.mitotic_score if grading and grading.mitotic_score is not None else None
        nottingham_sum = grading.nottingham_sum if grading and grading.nottingham_sum is not None else (
            (tubule_score + pleo_score + mitotic_score)
            if (tubule_score is not None and pleo_score is not None and mitotic_score is not None)
            else None
        )
        histologic_type = grading.histologic_type if grading and grading.histologic_type else "Invasive Carcinoma of No Special Type (NST)"

    # 2. Check existing report record or initialize default
    report_record = existing_report
    if not report_record:
        report_record = Report(
            case_id=case_uid,
            version=1,
            specimen_type="core_biopsy",
            procedure="Core Needle Biopsy",
            laterality="right",
            tumor_site="upper_outer_quadrant",
            histologic_type=histologic_type,
            tumor_size_mm=0.0 if is_benign else None,
            lvi_status="absent",
            dcis_present=False,
            margins=None,
            lymph_nodes={"examined_count": 0, "positive_count": 0, "extranodal_extension": False, "largest_metastasis_mm": 0.0},
            biomarkers=None,
            status="draft"
        )
        db.add(report_record)
        db.flush()

    # 3. Deterministic AJCC Staging Calculation
    if is_benign:
        report_record.histologic_type = histologic_type
        report_record.tumor_size_mm = 0.0
        report_record.staging = {
            "ajcc_version": "8th/9th Edition",
            "pt_stage": "N/A",
            "pn_stage": "N/A",
            "pm_stage": "cM0",
            "stage_group": "Benign"
        }
    else:
        tumor_size = report_record.tumor_size_mm
        nodes_info = report_record.lymph_nodes or {}
        n_exam = nodes_info.get("examined_count", 0)
        n_pos = nodes_info.get("positive_count", 0)

        pt_stage = calculate_ajcc_pt_stage(tumor_size)
        pn_stage = calculate_ajcc_pn_stage(n_exam, n_pos)
        stage_grp = calculate_ajcc_stage_group(pt_stage, pn_stage)

        report_record.staging = {
            "ajcc_version": "8th/9th Edition",
            "pt_stage": pt_stage,
            "pn_stage": pn_stage,
            "pm_stage": "cM0",
            "stage_group": stage_grp
        }


    # 4. Synthesize Grounded Narrative via MedGemma 1.5
    prompt_tpl, prompt_hash = load_prompt_template("cap_report", "v1")
    medgemma_client = MedGemmaClient()

    case_summary_payload = {
        "case_id": case_id,
        "procedure": report_record.procedure,
        "laterality": report_record.laterality,
        "tumor_site": report_record.tumor_site,
        "histologic_type": report_record.histologic_type,
        "tumor_size_mm": tumor_size,
        "lvi_status": report_record.lvi_status,
        "nottingham_grade": {
            "grade": grade_val,
            "tubule_score": tubule_score,
            "tubule_percent": tubule_pct,
            "pleo_score": pleo_score,
            "mitotic_score": mitotic_score,
            "nottingham_sum": nottingham_sum
        },
        "staging": report_record.staging,
        "biomarkers": report_record.biomarkers
    }

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if is_benign:
        narrative_dict = {
            "diagnosis_line": f"{report_record.laterality.upper()} BREAST, BIOPSY: BENIGN BREAST TISSUE, NEGATIVE FOR INVASIVE CARCINOMA.",
            "microscopic_findings": "Sections show benign breast parenchyma without evidence of cytologic atypia, architectural disruption, or invasive carcinoma. No mitotic figures suspicious for malignancy identified.",
            "clinical_correlation": "Negative for invasive or in-situ carcinoma. Follow-up as clinically indicated."
        }
    else:
        narrative_dict = loop.run_until_complete(
            medgemma_client.generate_cap_report_narrative(case_summary_payload, prompt_tpl)
        )
        # Check narrative consistency against structured parameters (#176)
        consistency_warnings = validate_narrative_consistency(narrative_dict, case_summary_payload)
        if consistency_warnings:
            print(f"[Stage 6 Worker] Narrative consistency warnings for Case {case_id}: {consistency_warnings}")
    report_record.narrative = narrative_dict

    # 5. Generate Clinical PDF via temporary scratch directory
    scratch_dir = tempfile.mkdtemp(prefix="og_report_")

    try:
        version_num = report_record.version or 1
        pdf_filename = f"CAP_Report_{case_id[:8]}_v{version_num}.pdf"
        pdf_out_path = os.path.join(scratch_dir, pdf_filename)
        
        # Download evidence files if available in GCS
        evidence_paths = {}
        try:
            hm_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/triage/heatmap_triage.png")
            hm_path = os.path.join(scratch_dir, "heatmap.png")
            with open(hm_path, "wb") as f:
                f.write(hm_bytes)
            evidence_paths["heatmap"] = hm_path
        except Exception:
            pass

        try:
            hpf_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/mitosis/crops/m_0001.png")
            hpf_path = os.path.join(scratch_dir, "mitotic_hpf.png")
            with open(hpf_path, "wb") as f:
                f.write(hpf_bytes)
            evidence_paths["mitotic_hpf"] = hpf_path
        except Exception:
            pass

        try:
            patch_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/grading_patches/p_001.png")
            patch_path = os.path.join(scratch_dir, "grading_patch.png")
            with open(patch_path, "wb") as f:
                f.write(patch_bytes)
            evidence_paths["grading_patch"] = patch_path
        except Exception:
            pass

        report_render_dict = {
            "case_id": case_id,
            "procedure": report_record.procedure,
            "laterality": report_record.laterality,
            "tumor_site": report_record.tumor_site,
            "histologic_type": report_record.histologic_type,
            "tumor_size_mm": tumor_size,
            "lvi_status": report_record.lvi_status,
            "dcis_present": report_record.dcis_present,
            "margins": report_record.margins,
            "lymph_nodes": report_record.lymph_nodes,
            "biomarkers": report_record.biomarkers,
            "staging": report_record.staging,
            "nottingham_grade": case_summary_payload["nottingham_grade"],
            "narrative": narrative_dict,
            "status": report_record.status,
            "signed_by": report_record.signed_by,
            "npi": report_record.npi,
            "signed_at": report_record.signed_at.isoformat() if report_record.signed_at else None,
            "integrity_hash": report_record.integrity_hash
        }

        generate_clinical_cap_pdf(
            report_data=report_render_dict,
            output_path=pdf_out_path,
            evidence_paths=evidence_paths
        )

        with open(pdf_out_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()

        # Upload versioned PDF
        versioned_path = f"cases/{case_id}/report/v{version_num}/{pdf_filename}"
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            versioned_path,
            pdf_content,
            "application/pdf"
        )

        # Upload unversioned latest pointer
        latest_path = f"cases/{case_id}/report/CAP_Report_{case_id[:8]}.pdf"
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            latest_path,
            pdf_content,
            "application/pdf"
        )

        report_record.pdf_sha256 = hashlib.sha256(pdf_content).hexdigest()
        gcs_pdf_uri = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/{versioned_path}"
        report_record.pdf_path = gcs_pdf_uri
        stage_exec.status = "awaiting_review"
        stage_exec.output_ref = gcs_pdf_uri

        # Audit event
        audit_evt = AuditEvent(
            case_id=case_id,
            actor="system_worker",
            event_type="stage_6_report_drafted",
            stage="report",
            payload={
                "status": "awaiting_review",
                "pt_stage": pt_stage,
                "pn_stage": pn_stage,
                "stage_group": stage_grp,
                "pdf_path": gcs_pdf_uri
            }
        )
        db.add(audit_evt)
        db.commit()

        model_versions = {
            "medgemma": "1.5",
            "prompt_cap_report": prompt_hash[:12],
            "cap_checklist": "2026.06",
            "ajcc": "8th/9th"
        }

        print(f"[Stage 6 Worker] Report generation completed for Case {case_id}. Uploaded to GCS {gcs_pdf_uri}. Ready for Pathologist Review.")
        return gcs_pdf_uri, model_versions

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
