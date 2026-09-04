import uuid
import os
import io
import json
import math
import tempfile
import shutil
import threading
from io import BytesIO
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.db import get_db
from app.core.auth import get_current_user, CurrentUser
from app.core.config import settings
from app.core.gcs import (
    get_gcs_client,
    parse_gcs_uri,
    download_blob_as_bytes,
    download_blob_as_text,
    download_blob_to_filename,
    resolve_slide_raw_uri
)
from app.core.openslide_lock import OPENSLIDE_GLOBAL_LOCK
from app.models.case import Case
from app.models.slide import Slide
from pipeline.tiles import read_region_srgb

router = APIRouter(prefix="/api/v1/cases", tags=["tiles"])

def generate_tile_on_the_fly(
    slide_file_path: str,
    slide_obj: Slide,
    z: int,
    c: int,
    r: int,
    layer: str
) -> bytes | None:
    """
    On-the-fly tile rendering fallback using OpenSlide / Pillow.
    Computes exact tile bounding box at DeepZoom level z and returns PNG/JPEG bytes.
    Thread-safe to prevent concurrent OpenSlide C-library access violations.
    """
    try:
        with OPENSLIDE_GLOBAL_LOCK:
            try:
                import openslide
                slide = openslide.OpenSlide(slide_file_path)
            except Exception:
                slide = Image.open(slide_file_path)

            slide_w = float(getattr(slide_obj, "width_px", 2048) or 2048)
            slide_h = float(getattr(slide_obj, "height_px", 2048) or 2048)
            if not getattr(slide_obj, "mpp_x", None) or not getattr(slide_obj, "mpp_y", None):
                raise HTTPException(status_code=400, detail="Slide is missing valid MPP (status='needs_mpp'). Cannot render tile.")
            mpp_x = float(slide_obj.mpp_x)
            mpp_y = float(slide_obj.mpp_y)

            max_dim = max(slide_w, slide_h)
            max_level = int(math.ceil(math.log2(max_dim))) if max_dim > 0 else 11

            tile_size = 256
            level_scale = 2 ** (z - max_level)

            # Region bounding box at level 0 in pixels
            w_px_0 = tile_size / level_scale
            h_px_0 = tile_size / level_scale
            x_px_0 = c * w_px_0
            y_px_0 = r * h_px_0

            # Convert to micrometers
            x_um = x_px_0 * mpp_x
            y_um = y_px_0 * mpp_y
            w_um = w_px_0 * mpp_x
            h_um = h_px_0 * mpp_y

            tile_arr, _ = read_region_srgb(
                slide,
                x_um=x_um,
                y_um=y_um,
                w_um=w_um,
                h_um=h_um,
                out_px=(tile_size, tile_size),
                mpp_x=mpp_x,
                mpp_y=mpp_y
            )

            if hasattr(slide, "close"):
                slide.close()

        if layer == "norm":
            try:
                stain_text = download_blob_as_text(settings.GCS_ARTIFACTS_BUCKET, f"cases/{slide_obj.case_id}/preprocess/stain_params.json")
                stain_params = json.loads(stain_text)
                from pipeline.stain import PureNumpyMacenkoNormalizer
                norm_obj = PureNumpyMacenkoNormalizer()
                norm_obj.stain_matrix_target = np.array(stain_params["stain_matrix"])
                norm_obj.max_conc_target = np.array(stain_params["max_concentrations"])
                tile_arr = norm_obj.transform(tile_arr)
            except Exception as norm_err:
                print(f"[Tile Router Warning] On-the-fly norm transform note: {norm_err}")

        img = Image.fromarray(tile_arr)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"[Tile Router Warning] Dynamic tile extraction error for z={z}, c={c}, r={r}: {e}")
        return None


def stream_slide_tile(slide: Slide, layer: str, z: int, filename: str, case_id: uuid.UUID | None = None) -> Response:
    stem = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1] or ".png"

    slide_w = float(getattr(slide, "width_px", 2048) or 2048)
    slide_h = float(getattr(slide, "height_px", 2048) or 2048)
    max_dim = max(slide_w, slide_h)
    slide_max_level = int(math.ceil(math.log2(max_dim))) if max_dim > 0 else 11
    cap_10x_level = max(0, slide_max_level - 2)

    target_layer = layer
    if layer == "norm" and z > cap_10x_level:
        target_layer = "orig"

    target_z = z

    no_cache_headers = {
        "Cache-Control": "public, max-age=86400, immutable",
        "X-Tile-Layer": target_layer,
        "X-Tile-Zoom": str(target_z)
    }

    # 1. Stream directly from Real GCP Cloud Storage Bucket (oncogemma-dev-pyramids)
    try:
        client = get_gcs_client()
        bucket = client.bucket(settings.GCS_PYRAMIDS_BUCKET)
        for check_ext in [ext, ".png", ".jpg"]:
            blob_name = f"{slide.id}/{target_layer}/{target_z}/{stem}{check_ext}"
            blob = bucket.blob(blob_name)
            if hasattr(blob, "download_as_bytes"):
                try:
                    tile_bytes = blob.download_as_bytes()
                    m_type = "image/png" if check_ext.lower() == ".png" else "image/jpeg"
                    return Response(content=tile_bytes, media_type=m_type, headers=no_cache_headers)
                except Exception:
                    pass
    except Exception as gcs_err:
        print(f"[Tile Router Warning] Real GCS fetch note: {gcs_err}")

    # 2. Dynamic On-The-Fly Tile Generation Fallback from GCS raw bucket
    scratch_dir = tempfile.mkdtemp(prefix="og_tile_dyn_")
    try:
        parts = stem.split("_")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            c, r = int(parts[0]), int(parts[1])
            cid = str(case_id or slide.case_id)
            gcs_uri_original = resolve_slide_raw_uri(cid, slide) or slide.gcs_uri_original or f"gs://{settings.GCS_RAW_BUCKET}/cases/{cid}/{slide.id}.svs"
            raw_bucket_name, blob_name = parse_gcs_uri(gcs_uri_original)
            slide_ext = os.path.splitext(blob_name)[1] or ".svs"
            local_slide_path = os.path.join(scratch_dir, f"slide{slide_ext}")

            download_blob_to_filename(raw_bucket_name, blob_name, local_slide_path)
            if os.path.exists(local_slide_path):
                tile_bytes = generate_tile_on_the_fly(
                    slide_file_path=local_slide_path,
                    slide_obj=slide,
                    z=target_z,
                    c=c,
                    r=r,
                    layer=target_layer
                )
                if tile_bytes:
                    return Response(content=tile_bytes, media_type="image/png", headers=no_cache_headers)
    except Exception as dynamic_err:
        print(f"[Tile Router Warning] Dynamic tile extraction fallback error: {dynamic_err}")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    # 3. Fallback for 'norm' to 'orig' in GCS
    if target_layer == "norm":
        try:
            client = get_gcs_client()
            bucket = client.bucket(settings.GCS_PYRAMIDS_BUCKET)
            for check_ext in [ext, ".jpg", ".png"]:
                blob_name = f"{slide.id}/orig/{target_z}/{stem}{check_ext}"
                blob = bucket.blob(blob_name)
                if hasattr(blob, "download_as_bytes"):
                    try:
                        tile_bytes = blob.download_as_bytes()
                        m_type = "image/png" if check_ext.lower() == ".png" else "image/jpeg"
                        return Response(content=tile_bytes, media_type=m_type, headers=no_cache_headers)
                    except Exception:
                        pass
        except Exception:
            pass

    raise HTTPException(status_code=404, detail="Tile missing")


@router.get("/{case_id}/tiles/{layer}/{z}/{filename}")
def get_tile(
    case_id: uuid.UUID,
    layer: str,
    z: int,
    filename: str,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    slide = db.scalars(select(Slide).where(Slide.case_id == case_id)).first()
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found for case")
    return stream_slide_tile(slide, layer, z, filename, case_id=case_id)


@router.get("/tiles/{slide_id}/{layer}/{z}/{filename}")
def get_tile_direct(
    slide_id: uuid.UUID,
    layer: str,
    z: int,
    filename: str,
    db: Session = Depends(get_db)
):
    slide = db.get(Slide, slide_id)
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")
    return stream_slide_tile(slide, layer, z, filename)

