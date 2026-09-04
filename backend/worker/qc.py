import os
import json
import shutil
import tempfile
import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import (
    get_gcs_client,
    parse_gcs_uri,
    upload_blob_from_bytes,
    download_blob_to_filename,
    resolve_slide_raw_uri
)
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.audit import AuditEvent
from pipeline.stain import fit_macenko_stain
from pipeline.qc_checks import run_all_qc_checks

def run_qc(stage_execution: StageExecution, session: Session) -> tuple[str, dict]:
    """
    QC worker handler:
    1. Downloads slide directly from GCS.
    2. Runs simplified QC checks suite (coverage & focus sharpness).
    3. Evaluates overall verdict ('pass', 'warn', 'fail').
    4. Uploads qc/output.json directly to GCS.
    5. Updates stage status.
    """
    input_ref = stage_execution.input_ref or {}
    slide_id = input_ref.get("slide_id")
    case_id = stage_execution.case_id

    if not slide_id:
        slide_obj = session.scalars(select(Slide).where(Slide.case_id == case_id)).first()
        if slide_obj:
            slide_id = str(slide_obj.id)

    if not slide_id:
        raise ValueError(f"Slide not found for case {case_id}")

    slide_obj = session.get(Slide, str(slide_id))
    if not slide_obj:
        raise ValueError(f"Slide object {slide_id} not found in database")

    # Halt QC stage if MPP is missing per PRD 01-stage-v4.0 §2.3 step 4
    if not getattr(slide_obj, "mpp_x", None) or slide_obj.mpp_x <= 0 or not getattr(slide_obj, "mpp_y", None) or slide_obj.mpp_y <= 0:
        raise ValueError(f"Slide {slide_obj.id} is missing valid MPP (status='needs_mpp'). Cannot execute QC stage.")

    mpp_x = float(slide_obj.mpp_x)
    mpp_y = float(slide_obj.mpp_y)
    checksum = getattr(slide_obj, "checksum_sha256", "default_checksum") or "default_checksum"

    scratch_dir = tempfile.mkdtemp(prefix="og_qc_")

    try:
        gcs_uri_original = resolve_slide_raw_uri(case_id, slide_obj) or slide_obj.gcs_uri_original or f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_id}/{slide_id}.svs"
        raw_bucket_name, blob_name = parse_gcs_uri(gcs_uri_original)
        
        ext = os.path.splitext(blob_name)[1] or ".svs"
        local_slide_path = os.path.join(scratch_dir, f"slide{ext}")

        # Download directly from GCS raw bucket to transient scratch file
        download_blob_to_filename(raw_bucket_name, blob_name, local_slide_path)

        if not os.path.exists(local_slide_path):
            raise FileNotFoundError(f"Raw slide file not found in GCS for QC stage in case {case_id}")

        try:
            import openslide
            slide = openslide.OpenSlide(local_slide_path)
        except Exception:
            slide = Image.open(local_slide_path)

        # Obtain stain matrix & tissue mask
        normalizer, stain_params, tissue_mask_1bit = fit_macenko_stain(
            slide,
            checksum_sha256=checksum,
            ref_image_path="configs/stain_reference.png",
            mpp_x=mpp_x,
            mpp_y=mpp_y
        )

        # Execute simplified QC check suite
        qc_result = run_all_qc_checks(
            slide,
            tissue_mask_1bit=tissue_mask_1bit,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            config_path="configs/qc.yaml"
        )

        if hasattr(slide, "close"):
            slide.close()

        verdict = qc_result["verdict"]

        # Persist qc/output.json directly to GCS artifacts bucket
        output_ref = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/qc/output.json"
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            f"cases/{case_id}/qc/output.json",
            json.dumps(qc_result, indent=2).encode("utf-8"),
            "application/json"
        )

        # Update stage status & case status based on QC verdict
        case_obj = session.get(Case, case_id)

        if verdict == "pass":
            stage_execution.status = "awaiting_review"
        elif verdict == "warn":
            stage_execution.status = "awaiting_review"
        elif verdict == "fail":
            stage_execution.status = "failed"
            stage_execution.error = f"QC Hard Failure: {[c['message'] for c in qc_result['checks'] if c['status'] == 'fail']}"
            if case_obj:
                case_obj.status = "needs_rescan"

        # Emit audit event
        audit = AuditEvent(
            case_id=str(case_id),
            actor="worker_qc",
            event_type="stage_output",
            stage="qc",
            payload={
                "verdict": verdict,
                "config_hash": qc_result["config_hash"],
                "failed_checks": [c["name"] for c in qc_result["checks"] if c["status"] == "fail"]
            }
        )
        session.add(audit)
        session.commit()

        return output_ref, {"opencv": "4.13.0"}

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

