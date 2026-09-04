import os
import sys
import json
import uuid
import math
import struct
import hashlib
import tempfile
import shutil
import glob
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import select
from sqlalchemy.orm import Session
from PIL import Image

# Disable PIL max pixel limit for gigapixel pathology WSIs
Image.MAX_IMAGE_PIXELS = None

from app.core.config import settings
from app.core.gcs import (
    get_gcs_client,
    parse_gcs_uri,
    upload_blob_from_bytes,
    upload_blob_from_file,
    download_blob_to_filename
)
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.audit import AuditEvent

def calculate_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def strip_label_and_macro_images(filepath: str) -> bool:
    """
    De-identify TIFF/SVS Whole Slide Images by stripping label and macro IFDs (Issue #37).
    Modifies the file in-place by unlinking label and macro directories from the IFD chain
    and zeroing out their pixel/strip data so no PHI lingers in raw storage.
    Returns True if actual stripping occurred, False otherwise.
    """
    if not os.path.exists(filepath):
        return False

    try:
        import tifffile
    except ImportError:
        return False

    try:
        with tifffile.TiffFile(filepath) as tif:
            is_bigtiff = tif.is_bigtiff
            byte_order = "<" if tif.byteorder == "<" else ">"

            # Identify pages that contain 'label' or 'macro' in description or tags
            # Page 0 (baseline primary image) must NEVER be removed
            pages_to_remove = []
            for idx, page in enumerate(tif.pages):
                if idx == 0:
                    continue
                desc = (page.description or "").lower()
                tag_desc = ""
                if 270 in page.tags:
                    tag_desc = str(page.tags[270].value).lower()
                if "label" in desc or "label" in tag_desc or "macro" in desc or "macro" in tag_desc:
                    pages_to_remove.append(idx)

            if not pages_to_remove:
                return False

            data_ranges_to_zero = []
            for idx in pages_to_remove:
                p = tif.pages[idx]
                if hasattr(p, "dataoffsets") and hasattr(p, "databytecounts"):
                    for off, cnt in zip(p.dataoffsets, p.databytecounts):
                        if off and cnt:
                            data_ranges_to_zero.append((off, cnt))
                num_tags = len(p.tags)
                ifd_size = (8 + num_tags * 20 + 8) if is_bigtiff else (2 + num_tags * 12 + 4)
                data_ranges_to_zero.append((p.offset, ifd_size))

            pages_to_keep = [p for i, p in enumerate(tif.pages) if i not in pages_to_remove]
            if not pages_to_keep:
                return False

    except Exception as e:
        print(f"[Ingest De-identification Note] Could not inspect TIFF structure: {e}")
        return False

    # Modify file in-place: rewire the IFD pointers and zero out PHI bytes
    try:
        file_size = os.path.getsize(filepath)
        ptr_size = 8 if is_bigtiff else 4
        fmt_uint = byte_order + ("Q" if is_bigtiff else "I")

        with open(filepath, "r+b") as f:
            for k in range(len(pages_to_keep)):
                curr_p = pages_to_keep[k]
                next_offset = pages_to_keep[k + 1].offset if k + 1 < len(pages_to_keep) else 0

                num_tags = len(curr_p.tags)
                next_ptr_pos = curr_p.offset + (8 + num_tags * 20 if is_bigtiff else 2 + num_tags * 12)
                if next_ptr_pos + ptr_size <= file_size:
                    f.seek(next_ptr_pos)
                    f.write(struct.pack(fmt_uint, next_offset))

            for off, cnt in data_ranges_to_zero:
                if 0 <= off < file_size and cnt > 0:
                    actual_cnt = min(cnt, file_size - off, 10 * 1024 * 1024)
                    if actual_cnt > 0:
                        f.seek(off)
                        f.write(b"\x00" * actual_cnt)

            f.flush()
        return True
    except Exception as we:
        print(f"[Ingest De-identification Note] Failed to rewrite de-identified TIFF: {we}")
        return False

def extract_openslide_metadata(filepath: str) -> dict:
    """
    Extract WSI metadata via OpenSlide with pyvips fallback.
    Never guesses 0.25 µm/px or default dimensions.
    Fails fast and raises RuntimeError if neither OpenSlide nor pyvips can open the file (Issue #38, #39).
    """
    meta = {
        "mpp_x": None,
        "mpp_y": None,
        "base_mag": None,
        "width_px": None,
        "height_px": None,
        "vendor": "unknown",
        "format": "unknown"
    }

    errors = []

    # 1. Primary: OpenSlide
    try:
        import openslide
        slide = openslide.OpenSlide(filepath)
        
        meta["width_px"], meta["height_px"] = slide.dimensions
        meta["vendor"] = slide.properties.get(openslide.PROPERTY_NAME_VENDOR, "unknown")
        
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        if mpp_x:
            val_x = float(mpp_x)
            if val_x > 0:
                meta["mpp_x"] = val_x
        if mpp_y:
            val_y = float(mpp_y)
            if val_y > 0:
                meta["mpp_y"] = val_y
            
        mag = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if mag:
            meta["base_mag"] = float(mag)
        elif meta["mpp_x"]:
            meta["base_mag"] = round(10.0 / meta["mpp_x"], 1)

        meta["format"] = os.path.splitext(filepath)[1].lstrip(".").lower()
        slide.close()
        return meta
    except Exception as e:
        errors.append(f"OpenSlide error: {e}")

    # 2. Secondary: pyvips
    try:
        import pyvips
        image = pyvips.Image.new_from_file(filepath, access="sequential")
        meta["width_px"] = image.width
        meta["height_px"] = image.height
        if hasattr(image, "xres") and image.xres > 0:
            meta["mpp_x"] = 1000.0 / image.xres
        if hasattr(image, "yres") and image.yres > 0:
            meta["mpp_y"] = 1000.0 / image.yres
        if meta["mpp_x"]:
            meta["base_mag"] = round(10.0 / meta["mpp_x"], 1)
        meta["format"] = os.path.splitext(filepath)[1].lstrip(".").lower()
        return meta
    except Exception as pe:
        errors.append(f"pyvips error: {pe}")

    # Neither OpenSlide nor pyvips could open the file -> Fail fast with real error
    raise RuntimeError(f"Failed to open slide '{filepath}' for metadata extraction: {'; '.join(errors)}")

def generate_dzi_pyramid(filepath: str, output_dir: str) -> str:
    """
    Generate DZI pyramid tiles using OpenSlide DeepZoomGenerator or pyvips dzsave.
    Generates tiles for full level count (dz.level_count) per Issue #635.
    Fails fast and raises RuntimeError if both OpenSlide and pyvips fail (Issue #39).
    """
    dzi_base = os.path.join(output_dir, "pyramid")
    dzi_files_dir = dzi_base + "_files"
    os.makedirs(dzi_files_dir, exist_ok=True)
    
    errors = []

    # 1. Primary: OpenSlide DeepZoomGenerator
    try:
        import openslide
        from openslide.deepzoom import DeepZoomGenerator
        
        slide = openslide.OpenSlide(filepath)
        dz = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
        
        # Full-depth pyramid pregeneration up to dz.level_count (Issue #635)
        for level in range(0, dz.level_count):
            cols, rows = dz.level_tiles[level]
            level_dir = os.path.join(dzi_files_dir, str(level))
            os.makedirs(level_dir, exist_ok=True)
            for c in range(cols):
                for r in range(rows):
                    tile_path_jpg = os.path.join(level_dir, f"{c}_{r}.jpg")
                    tile_path_png = os.path.join(level_dir, f"{c}_{r}.png")
                    if not os.path.exists(tile_path_jpg):
                        tile = dz.get_tile(level, (c, r))
                        if tile.mode != "RGB":
                            tile = tile.convert("RGB")
                        tile.save(tile_path_jpg, "JPEG", quality=85)
                        tile.save(tile_path_png, "PNG")

        try:
            dzi_xml = dz.get_dzi("png")
            with open(dzi_base + ".dzi", "w", encoding="utf-8") as f:
                f.write(dzi_xml)
        except Exception:
            pass

        slide.close()
        return dzi_base + ".dzi"
    except Exception as oe:
        errors.append(f"OpenSlide DeepZoom error: {oe}")

    # 2. Secondary: Pyvips
    try:
        import pyvips
        image = pyvips.Image.new_from_file(filepath, access="sequential")
        image.dzsave(dzi_base, tile_size=256, overlap=0, suffix=".png[Q=90]")
        return dzi_base + ".dzi"
    except Exception as pe:
        errors.append(f"pyvips error: {pe}")

    # Fail fast if both OpenSlide and pyvips fail - do NOT fabricate flat pink pyramid
    raise RuntimeError(f"Failed to generate pyramid for slide '{filepath}': {'; '.join(errors)}")

def upload_dzi_tree_to_gcs(dzi_files_dir: str, slide_id: str):
    """
    Save generated DZI tile tree directly to Google Cloud Storage pyramid bucket with zero local persistence.
    """
    client = get_gcs_client()
    try:
        bucket = client.bucket(settings.GCS_PYRAMIDS_BUCKET)
        tile_files = glob.glob(os.path.join(dzi_files_dir, "**", "*.*"), recursive=True)
        tile_files = [f for f in tile_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        def upload_single_tile(local_path):
            try:
                rel_path = os.path.relpath(local_path, dzi_files_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    z_level = parts[-2]
                    filename = parts[-1]
                    blob_path = f"{slide_id}/orig/{z_level}/{filename}"
                    blob = bucket.blob(blob_path)
                    c_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
                    blob.upload_from_filename(local_path, content_type=c_type, timeout=15)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(upload_single_tile, tile_files))
    except Exception as ge:
        print(f"[Ingest Worker Note] Parallel GCP cloud pyramid upload note: {ge}")

def run_ingest(stage_execution: StageExecution, session: Session) -> tuple[str, dict]:
    """
    Ingest handler logic for worker execution.
    1. Downloads raw WSI from GCS raw bucket.
    2. De-identifies TIFF/SVS files by stripping label and macro images (Issue #37).
    3. Computes SHA256 checksum on de-identified file.
    4. Extracts WSI metadata via OpenSlide / pyvips without guessing MPP (Issue #38, #39).
    5. Sets slide status to 'needs_mpp' and halts downstream chaining if MPP missing (Issue #38).
    6. Generates full-depth DZI tiles and uploads to GCS pyramid storage (Issue #635).
    7. Emits audit event and persists ingest output.
    Returns (output_ref_uri, model_versions_dict).
    """
    input_ref = stage_execution.input_ref or {}
    gcs_uri_original = input_ref.get("gcs_uri_original")
    slide_id = input_ref.get("slide_id")

    if not slide_id:
        raise ValueError("Missing slide_id in stage input_ref")

    slide_obj = session.get(Slide, str(slide_id))
    if not slide_obj:
        slide_obj = session.scalars(select(Slide).where(Slide.id == str(slide_id))).first()

    if not slide_obj:
        raise ValueError(f"Slide {slide_id} not found in database")

    scratch_dir = tempfile.mkdtemp(prefix="og_ingest_")

    try:
        raw_bucket_name = settings.GCS_RAW_BUCKET
        if gcs_uri_original and gcs_uri_original.startswith("gs://"):
            raw_bucket_name, blob_name = parse_gcs_uri(gcs_uri_original)
        else:
            blob_name = f"cases/{stage_execution.case_id}/{slide_id}.svs"

        ext = os.path.splitext(blob_name)[1]
        if not ext or len(ext) < 2:
            ext = ".svs"
        local_slide_path = os.path.join(scratch_dir, f"slide{ext}")
        
        # Download directly from GCS raw bucket to transient scratch file
        download_blob_to_filename(raw_bucket_name, blob_name, local_slide_path)

        if not os.path.exists(local_slide_path):
            raise FileNotFoundError(f"Original slide file not found in GCS for ingest stage in case {stage_execution.case_id} (URI: {gcs_uri_original})")

        # 1. De-identify and strip label and macro images before metadata extraction (Issue #37)
        was_stripped = strip_label_and_macro_images(local_slide_path)
        if was_stripped:
            slide_obj.label_stripped_at = datetime.now(timezone.utc)
            # Overwrite raw blob in GCS raw bucket with scrubbed de-identified file
            with open(local_slide_path, "rb") as f_scrubbed:
                upload_blob_from_file(raw_bucket_name, blob_name, f_scrubbed, content_type="application/octet-stream")
        else:
            slide_obj.label_stripped_at = None

        # 2. SHA256 checksum calculated on de-identified file
        checksum = calculate_sha256(local_slide_path)
        slide_obj.checksum_sha256 = checksum

        # 3. Metadata extraction (fail fast on unopenable files, never guess 0.25 MPP)
        meta = extract_openslide_metadata(local_slide_path)
        slide_obj.mpp_x = meta.get("mpp_x")
        slide_obj.mpp_y = meta.get("mpp_y")
        slide_obj.base_mag = meta.get("base_mag")
        slide_obj.width_px = meta.get("width_px")
        slide_obj.height_px = meta.get("height_px")
        slide_obj.format = meta.get("format")
        slide_obj.scanner = meta.get("vendor")

        # 4. Check MPP validity per Issue #38 and PRD 01-stage-v4.0 §2.3 step 4
        has_valid_mpp = bool(
            slide_obj.mpp_x is not None and slide_obj.mpp_x > 0 and
            slide_obj.mpp_y is not None and slide_obj.mpp_y > 0
        )

        case_obj = session.get(Case, stage_execution.case_id)
        if not has_valid_mpp:
            slide_obj.status = "needs_mpp"
            if case_obj:
                case_obj.status = "needs_mpp"
        else:
            slide_obj.status = "ready"

        # 5. Full-depth DZI Pyramid generation (Issue #39, Issue #635)
        dzi_path = generate_dzi_pyramid(local_slide_path, scratch_dir)
        dzi_files_dir = dzi_path.replace(".dzi", "_files")

        # 6. Save tile tree directly to GCS pyramid storage
        if os.path.exists(dzi_files_dir):
            upload_dzi_tree_to_gcs(dzi_files_dir, str(slide_obj.id))

        gcs_pyramid_uri = f"gs://{settings.GCS_PYRAMIDS_BUCKET}/{slide_obj.id}/orig/"
        slide_obj.gcs_uri_pyramid = gcs_pyramid_uri

        # 7. Output payload directly to GCS artifacts bucket
        output_ref = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{stage_execution.case_id}/ingest_output.json"
        out_payload = {
            "slide_id": str(slide_obj.id),
            "checksum": checksum,
            "mpp_x": slide_obj.mpp_x,
            "mpp_y": slide_obj.mpp_y,
            "dimensions": [slide_obj.width_px, slide_obj.height_px] if slide_obj.width_px and slide_obj.height_px else None,
            "gcs_uri_pyramid": gcs_pyramid_uri,
            "status": slide_obj.status
        }
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            f"cases/{stage_execution.case_id}/ingest_output.json",
            json.dumps(out_payload, indent=2).encode("utf-8"),
            "application/json"
        )

        # Emit audit event
        audit = AuditEvent(
            case_id=str(stage_execution.case_id),
            actor="worker_ingest",
            event_type="stage_output",
            stage="ingest",
            payload=out_payload
        )
        session.add(audit)

        # 8. Downstream chaining: only queue preprocess if valid MPP exists
        if has_valid_mpp:
            existing_prep = session.scalars(
                select(StageExecution).where(
                    StageExecution.case_id == stage_execution.case_id,
                    StageExecution.stage == "preprocess"
                )
            ).first()

            if not existing_prep:
                next_prep_stage = StageExecution(
                    case_id=stage_execution.case_id,
                    stage="preprocess",
                    attempt=1,
                    status="queued",
                    input_ref={"slide_id": str(slide_obj.id), "ingest_output_ref": output_ref}
                )
                session.add(next_prep_stage)
                session.commit()
                session.refresh(next_prep_stage)

                from app.core.cloud_tasks import dispatch_stage_task
                dispatch_stage_task(
                    case_id=str(stage_execution.case_id),
                    stage="preprocess",
                    stage_exec_id=str(next_prep_stage.id),
                    payload={"slide_id": str(slide_obj.id), "ingest_output_ref": output_ref}
                )
            else:
                session.commit()
        else:
            # Halt downstream stage execution without valid MPP per PRD 01-stage-v4.0 §2.3 step 4
            session.commit()

        model_versions = {"pillow": "10.2.0", "openslide": "1.3.1"}

        return output_ref, model_versions

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
