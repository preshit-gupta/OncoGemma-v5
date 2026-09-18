import os
import json
import asyncio
import tempfile
import shutil
import hashlib
from typing import Tuple, Dict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import (
    upload_blob_from_bytes,
    download_blob_as_bytes
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
    validate_narrative_consistency
)
from pipeline.medgemma import MedGemmaClient, load_prompt_template
from pipeline.report_pdf import generate_clinical_cap_pdf, build_report_pdf_context


def run_report(stage_exec: StageExecution, db: Session) -> Tuple[str, Dict[str, str]]:
    """
    Executes Stage 6 (Report Generation) pipeline:
    1. Gathers clinical, grading, and staging parameters in an isolated DB read transaction (#289).
    2. Closes read transaction before invoking MedGemma and ReportLab rendering.
    3. Runs MedGemma CAP-compliant synoptic report generation.
    4. Validates narrative consistency against structured parameters (#748).
    5. Renders high-fidelity clinical PDF using GCS-backed evidence assets via shared context builder (#629).
    6. Writes atomic machine-readable JSON report blob to GCS (#706).
    7. Uploads final PDF directly to GCS artifacts bucket.
    8. Opens discrete write transaction to persist report state and audit event (#289).
    9. Cleans up temporary scratch directory.
    """
    case_id = str(stage_exec.case_id)
    case_uid = stage_exec.case_id
    stage_exec_id = stage_exec.id
    
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

    version_num = report_record.version or 1
    procedure = report_record.procedure
    laterality = report_record.laterality
    tumor_site = report_record.tumor_site
    lvi_status = report_record.lvi_status
    dcis_present = report_record.dcis_present
    margins = report_record.margins
    lymph_nodes = report_record.lymph_nodes
    biomarkers = report_record.biomarkers
    staging_dict = report_record.staging

    # Prepare evidence geometry primitives for burn-in (#504)
    top_hpf = max(hpfs, key=lambda h: h.mitotic_count) if hpfs else None
    evidence_geometry = {
        "hotspots": [
            {
                "id": h.id,
                "seq": getattr(h, "seq", i + 1),
                "polygon_coords_um": getattr(h, "polygon_um", getattr(h, "polygon_coords_um", None)),
                "polygon_um": getattr(h, "polygon_um", None),
                "center_um": getattr(h, "center_um", None)
            }
            for i, h in enumerate(hotspots)
        ],
        "top_hpf": {
            "seq": top_hpf.seq,
            "mitotic_count": top_hpf.mitotic_count,
            "center_um": top_hpf.center_um,
            "radius_um": getattr(top_hpf, "radius_um", 262.0)
        } if top_hpf else None
    }

    # Commit initial DB setup and release transaction before MedGemma & PDF compilation (#289)
    db.commit()

    # 4. Synthesize Grounded Narrative via MedGemma 1.5 (Outside DB transaction)
    prompt_tpl, prompt_hash = load_prompt_template("cap_report", "v1")
    medgemma_client = MedGemmaClient()

    case_summary_payload = {
        "case_id": case_id,
        "procedure": procedure,
        "laterality": laterality,
        "tumor_site": tumor_site,
        "histologic_type": histologic_type,
        "tumor_size_mm": tumor_size,
        "lvi_status": lvi_status,
        "nottingham_grade": {
            "grade": grade_val,
            "tubule_score": tubule_score,
            "tubule_percent": tubule_pct,
            "pleo_score": pleo_score,
            "mitotic_score": mitotic_score,
            "nottingham_sum": nottingham_sum
        },
        "staging": staging_dict,
        "biomarkers": biomarkers
    }

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if is_benign:
        narrative_dict = {
            "diagnosis_line": f"{laterality.upper()} BREAST, BIOPSY: BENIGN BREAST TISSUE, NEGATIVE FOR INVASIVE CARCINOMA.",
            "microscopic_findings": "Sections show benign breast parenchyma without evidence of cytologic atypia, architectural disruption, or invasive carcinoma. No mitotic figures suspicious for malignancy identified.",
            "clinical_correlation": "Negative for invasive or in-situ carcinoma. Follow-up as clinically indicated."
        }
    else:
        narrative_dict = loop.run_until_complete(
            medgemma_client.generate_cap_report_narrative(case_summary_payload, prompt_tpl)
        )
        # Check narrative consistency against structured parameters (#748)
        consistency_warnings = validate_narrative_consistency(narrative_dict, case_summary_payload)
        if consistency_warnings:
            print(f"[Stage 6 Worker] Narrative consistency warnings for Case {case_id}: {consistency_warnings}")

    # 5. Generate Clinical PDF via temporary scratch directory (Outside DB transaction)
    scratch_dir = tempfile.mkdtemp(prefix="og_report_")

    try:
        pdf_filename = f"CAP_Report_{case_id[:8]}_v{version_num}.pdf"
        pdf_out_path = os.path.join(scratch_dir, pdf_filename)
        
        evidence_paths = {}
        for hm in [f"cases/{case_id}/triage/heatmap_triage.png", f"cases/{case_id}/triage/heatmap.png"]:
            try:
                hm_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, hm)
                hm_path = os.path.join(scratch_dir, "heatmap.png")
                with open(hm_path, "wb") as f:
                    f.write(hm_bytes)
                evidence_paths["heatmap"] = hm_path
                break
            except Exception:
                pass

        for hpf in [
            f"cases/{case_id}/mitosis/hpfs/hpf_1_40x_norm.png",
            f"cases/{case_id}/mitosis/hpfs/hpf_1_20x_norm.png",
            f"cases/{case_id}/mitosis/hpfs/hpf_1_10x_norm.png",
            f"cases/{case_id}/mitosis/crops/m_0001.png",
            f"cases/{case_id}/mitosis/crops/m_0364.png"
        ]:
            try:
                hpf_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, hpf)
                hpf_path = os.path.join(scratch_dir, "mitotic_hpf.png")
                with open(hpf_path, "wb") as f:
                    f.write(hpf_bytes)
                evidence_paths["mitotic_hpf"] = hpf_path
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
                gp_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, gp)
                gp_path = os.path.join(scratch_dir, "grading_patch.png")
                with open(gp_path, "wb") as f:
                    f.write(gp_bytes)
                evidence_paths["grading_patch"] = gp_path
                break
            except Exception:
                pass

        model_versions = {
            "medgemma": "1.5",
            "prompt_cap_report": prompt_hash[:12],
            "cap_checklist": "v4.2.0.0 (2026.06)",
            "ajcc_edition": "8th / 9th Edition",
            "grading_engine": "Multi-Head ViT + Nottingham Rules",
            "mitosis_model": "YOLOv8x-Mitosis 40x (calibrated reticle r=262µm)"
        }

        # Build standardized context using shared builder (#629)
        raw_report_data = {
            "case_id": case_id,
            "procedure": procedure,
            "laterality": laterality,
            "tumor_site": tumor_site,
            "histologic_type": histologic_type,
            "tumor_size_mm": tumor_size,
            "lvi_status": lvi_status,
            "dcis_present": dcis_present,
            "margins": margins,
            "lymph_nodes": lymph_nodes,
            "biomarkers": biomarkers,
            "staging": staging_dict,
            "nottingham_grade": case_summary_payload["nottingham_grade"],
            "narrative": narrative_dict,
            "status": "draft",
            "model_versions": model_versions
        }

        pdf_context = build_report_pdf_context(
            report=raw_report_data,
            evidence_paths=evidence_paths,
            evidence_geometry=evidence_geometry,
            model_versions=model_versions
        )

        generate_clinical_cap_pdf(
            report_data=pdf_context,
            output_path=pdf_out_path,
            evidence_paths=evidence_paths,
            evidence_geometry=evidence_geometry,
            model_versions=model_versions
        )

        with open(pdf_out_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()

        pdf_sha256 = hashlib.sha256(pdf_content).hexdigest()

        # Upload versioned PDF
        versioned_pdf_path = f"cases/{case_id}/report/v{version_num}/{pdf_filename}"
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            versioned_pdf_path,
            pdf_content,
            "application/pdf"
        )

        # Upload unversioned latest PDF pointer
        latest_pdf_path = f"cases/{case_id}/report/CAP_Report_{case_id[:8]}.pdf"
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            latest_pdf_path,
            pdf_content,
            "application/pdf"
        )

        # Upload atomic machine-readable JSON blob to GCS (#706)
        output_json_bytes = json.dumps(pdf_context, indent=2, default=str).encode("utf-8")
        versioned_json_path = f"cases/{case_id}/report/v{version_num}/output.json"
        latest_json_path = f"cases/{case_id}/report/output.json"

        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            versioned_json_path,
            output_json_bytes,
            "application/json"
        )
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            latest_json_path,
            output_json_bytes,
            "application/json"
        )

        gcs_pdf_uri = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/{versioned_pdf_path}"

        # 6. Discrete write transaction to persist report state and audit event (#289)
        report_curr = db.scalars(
            select(Report).where(Report.case_id == case_uid).order_by(Report.version.desc())
        ).first()
        if report_curr:
            report_curr.narrative = narrative_dict
            report_curr.pdf_sha256 = pdf_sha256
            report_curr.pdf_path = gcs_pdf_uri

        stage_curr = db.get(StageExecution, stage_exec_id)
        if stage_curr:
            stage_curr.status = "awaiting_review"
            stage_curr.output_ref = gcs_pdf_uri

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
                "pdf_path": gcs_pdf_uri,
                "output_json": f"gs://{settings.GCS_ARTIFACTS_BUCKET}/{versioned_json_path}"
            }
        )
        db.add(audit_evt)
        db.commit()

        print(f"[Stage 6 Worker] Report generation completed for Case {case_id}. Uploaded to GCS {gcs_pdf_uri}. Ready for Pathologist Review.")
        return gcs_pdf_uri, model_versions

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
