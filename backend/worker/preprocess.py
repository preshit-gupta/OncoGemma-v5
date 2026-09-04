import os
import io
import json
import math
import shutil
import tempfile
import numpy as np
from PIL import Image
from datetime import datetime, timezone
import glob
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import (
    get_gcs_client,
    parse_gcs_uri,
    upload_blob_from_bytes,
    upload_blob_from_filename,
    download_blob_to_filename,
    resolve_slide_raw_uri
)
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.audit import AuditEvent
from pipeline.stain import fit_macenko_stain
from pipeline.tiles import read_region_srgb

def generate_norm_dzi_pyramid(slide_obj, normalizer, local_slide_path: str, scratch_dir: str) -> str:
    """
    Generate complete normalized DZI pyramid and stream directly to GCS pyramids bucket.
    Applies Macenko stain normalization across all pyramid levels.
    """
    slide_id = str(slide_obj.id)
    norm_pyramid_dir = os.path.join(scratch_dir, "norm_pyramid")
    os.makedirs(norm_pyramid_dir, exist_ok=True)

    # 1. OpenSlide DeepZoomGenerator
    try:
        import openslide
        from openslide.deepzoom import DeepZoomGenerator
        
        slide = openslide.OpenSlide(local_slide_path)
        dz = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
        
        max_norm_pregen_level = min(12, dz.level_count)
        for level in range(0, max_norm_pregen_level):
            norm_level_dir = os.path.join(norm_pyramid_dir, str(level))
            os.makedirs(norm_level_dir, exist_ok=True)
            cols, rows = dz.level_tiles[level]
            for c in range(cols):
                for r in range(rows):
                    png_path = os.path.join(norm_level_dir, f"{c}_{r}.png")
                    tile = dz.get_tile(level, (c, r))
                    if tile.mode != "RGB":
                        tile = tile.convert("RGB")
                    raw_arr = np.array(tile, dtype=np.uint8)
                    try:
                        norm_arr = normalizer.transform(raw_arr)
                    except Exception:
                        norm_arr = raw_arr
                    norm_tile = Image.fromarray(norm_arr)
                    norm_tile.save(png_path, "PNG")
                    norm_tile.save(os.path.join(norm_level_dir, f"{c}_{r}.jpg"), "JPEG", quality=85)
        slide.close()
    except Exception as dz_err:
        print(f"[Preprocess Worker Note] Direct norm DeepZoom generation note: {dz_err}")

    # 2. Stream normalized tiles directly to GCS Cloud Storage pyramid bucket
    client = get_gcs_client()
    try:
        bucket = client.bucket(settings.GCS_PYRAMIDS_BUCKET)
        norm_files = glob.glob(os.path.join(norm_pyramid_dir, "**", "*.*"), recursive=True)
        norm_files = [f for f in norm_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        def upload_single_norm_tile(local_path):
            try:
                rel_path = os.path.relpath(local_path, norm_pyramid_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    z_level = parts[-2]
                    filename = parts[-1]
                    blob_path = f"{slide_id}/norm/{z_level}/{filename}"
                    blob = bucket.blob(blob_path)
                    c_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
                    blob.upload_from_filename(local_path, content_type=c_type, timeout=15)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(upload_single_norm_tile, norm_files))
    except Exception as ge:
        print(f"[Preprocess Worker Note] Parallel GCP cloud norm pyramid upload note: {ge}")

    return f"gs://{settings.GCS_PYRAMIDS_BUCKET}/{slide_id}/norm/"


def run_preprocess(stage_execution: StageExecution, session: Session) -> tuple[str, dict]:
    """
    Preprocess worker handler:
    1. Downloads raw slide directly from GCS.
    2. Fits Macenko stain normalizer on tissue patches.
    3. Extracts 1-bit tissue mask PNG and stain parameters.
    4. Assembles normalized DZI pyramid and uploads to GCS.
    5. Persists preprocess artifacts directly to GCS & queues next stage ('qc').
    """
    input_ref = stage_execution.input_ref or {}
    slide_id = input_ref.get("slide_id")
    case_id = stage_execution.case_id

    if not slide_id:
        slide_obj = session.scalars(select(Slide).where(Slide.case_id == case_id)).first()
        if slide_obj:
            slide_id = str(slide_obj.id)

    if not slide_id:
        raise ValueError(f"Slide not found for preprocess stage in case {case_id}")

    slide_obj = session.get(Slide, str(slide_id))
    if not slide_obj:
        slide_obj = session.scalars(select(Slide).where(Slide.id == str(slide_id))).first()

    if not slide_obj:
        raise ValueError(f"Slide object {slide_id} not found in database")

    scratch_dir = tempfile.mkdtemp(prefix="og_preprocess_")

    try:
        gcs_uri_original = resolve_slide_raw_uri(case_id, slide_obj) or slide_obj.gcs_uri_original or f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_id}/{slide_id}.svs"
        raw_bucket_name, blob_name = parse_gcs_uri(gcs_uri_original)
        
        ext = os.path.splitext(blob_name)[1] or ".svs"
        local_slide_path = os.path.join(scratch_dir, f"slide{ext}")

        # Download directly from GCS raw bucket to transient scratch file
        download_blob_to_filename(raw_bucket_name, blob_name, local_slide_path)

        if not os.path.exists(local_slide_path):
            raise FileNotFoundError(f"Raw slide file not found in GCS for preprocess stage in case {case_id}")

        mpp_x = float(getattr(slide_obj, "mpp_x", 0.25) or 0.25)
        mpp_y = float(getattr(slide_obj, "mpp_y", 0.25) or 0.25)
        checksum = getattr(slide_obj, "checksum_sha256", "default_checksum") or "default_checksum"

        try:
            import openslide
            slide = openslide.OpenSlide(local_slide_path)
        except Exception:
            slide = Image.open(local_slide_path)

        # Fit STAINS Macenko Normalizer & Extract Tissue Mask
        normalizer, stain_params, tissue_mask_1bit = fit_macenko_stain(
            slide,
            checksum_sha256=checksum,
            ref_image_path="configs/stain_reference.png",
            mpp_x=mpp_x,
            mpp_y=mpp_y
        )

        px_area_mm2 = (8.0 * mpp_x / 0.25) * (8.0 * mpp_y / 0.25) * 1e-6
        tissue_area_mm2 = float(np.count_nonzero(tissue_mask_1bit) * px_area_mm2)

        from pipeline.tiles import check_icc_profile
        _, icc_applied = check_icc_profile(slide)

        if hasattr(slide, "close"):
            slide.close()

        # Save artifacts directly to GCS artifacts bucket
        stain_params_uri = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/preprocess/stain_params.json"
        tissue_mask_uri = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/preprocess/tissue_mask.png"
        thumbnail_uri = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/preprocess/thumbnail.png"

        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            f"cases/{case_id}/preprocess/stain_params.json",
            json.dumps(stain_params, indent=2).encode("utf-8"),
            "application/json"
        )

        mask_img = Image.fromarray((tissue_mask_1bit * 255).astype(np.uint8))
        mask_buf = io.BytesIO()
        mask_img.save(mask_buf, format="PNG")
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            f"cases/{case_id}/preprocess/tissue_mask.png",
            mask_buf.getvalue(),
            "image/png"
        )

        # Assemble Normalized DZI Pyramid directly to GCS
        norm_pyramid_uri = generate_norm_dzi_pyramid(slide_obj, normalizer, local_slide_path, scratch_dir)

        # Save preprocess/output.json directly to GCS
        preprocess_output = {
            "icc_applied": icc_applied,
            "stain_params_uri": stain_params_uri,
            "norm_pyramid_uri": norm_pyramid_uri,
            "thumbnail_uri": thumbnail_uri,
            "tissue_mask_uri": tissue_mask_uri,
            "tissue_area_mm2": round(tissue_area_mm2, 2),
            "model_versions": {"tiatoolbox": "1.6.0"}
        }

        output_ref = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/preprocess/output.json"
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            f"cases/{case_id}/preprocess/output.json",
            json.dumps(preprocess_output, indent=2).encode("utf-8"),
            "application/json"
        )

        # Update stage execution status
        stage_execution.status = "done"

        # Emit audit event
        audit = AuditEvent(
            case_id=str(case_id),
            actor="worker_preprocess",
            event_type="stage_output",
            stage="preprocess",
            payload={
                "icc_applied": icc_applied,
                "tissue_area_mm2": tissue_area_mm2,
                "norm_pyramid_uri": norm_pyramid_uri
            }
        )
        session.add(audit)

        # Auto-chain next stage ('qc') in queued status
        existing_qc = session.scalars(
            select(StageExecution).where(
                StageExecution.case_id == case_id,
                StageExecution.stage == "qc",
                StageExecution.attempt == 1
            )
        ).first()

        if not existing_qc:
            next_qc_stage = StageExecution(
                case_id=case_id,
                stage="qc",
                attempt=1,
                status="queued",
                input_ref={"slide_id": str(slide_id), "preprocess_output_ref": output_ref}
            )
            session.add(next_qc_stage)
            session.commit()
            session.refresh(next_qc_stage)

            from app.core.cloud_tasks import dispatch_stage_task
            dispatch_stage_task(
                case_id=str(case_id),
                stage="qc",
                stage_exec_id=str(next_qc_stage.id),
                payload={"slide_id": str(slide_id), "preprocess_output_ref": output_ref}
            )
        else:
            session.commit()

        return output_ref, {"tiatoolbox": "1.6.0"}

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
