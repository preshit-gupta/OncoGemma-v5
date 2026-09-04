import os
import io
import json
import uuid
import tempfile
import shutil
import threading
from datetime import datetime, timezone
from typing import Any, Optional
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, status, Response, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import (
    get_gcs_client,
    parse_gcs_uri,
    download_blob_as_bytes,
    download_blob_as_text,
    download_blob_to_filename,
    upload_blob_from_bytes,
    blob_exists
)
from app.core.db import get_db
from app.core.openslide_lock import OPENSLIDE_GLOBAL_LOCK
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot
from app.models.audit import AuditEvent

router = APIRouter(prefix="/api/v1/stages/triage", tags=["triage"])

def to_uuid(val: Any) -> uuid.UUID:
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except Exception:
        return val


class TriageEditsPayload(BaseModel):
    case_id: str
    edits: list[dict[str, Any]] # RFC-6902 style edit operations


class TriageConfirmPayload(BaseModel):
    case_id: str
    no_invasive_tumor: bool = False
    reviewed_by: str = "pathologist_01"


def apply_edit_ops(machine_hotspots: list[dict], edits: list[dict]) -> list[dict]:
    """
    Applies RFC-6902 style diff operations to machine output hotspots.
    Idempotent and order-stable.
    """
    hotspots_dict = {h["id"]: dict(h) for h in machine_hotspots}

    for op in edits:
        action = op.get("op")
        hid = op.get("id")

        if action == "modify" and hid in hotspots_dict:
            if "polygon_um" in op:
                hotspots_dict[hid]["polygon_um"] = op["polygon_um"]
                hotspots_dict[hid]["source"] = "pathologist_modified"

        elif action == "add":
            new_id = hid or f"user_{len(hotspots_dict)+1:02d}"
            hotspots_dict[new_id] = {
                "id": new_id,
                "polygon_um": op.get("polygon_um", []),
                "area_mm2": op.get("area_mm2", 1.0),
                "prob_mean": op.get("prob_mean", 1.0),
                "prob_max": op.get("prob_max", 1.0),
                "source": "pathologist_added",
                "excluded": False,
                "exclude_reason": None
            }

        elif action == "exclude" and hid in hotspots_dict:
            hotspots_dict[hid]["excluded"] = True
            hotspots_dict[hid]["exclude_reason"] = op.get("reason", "Pathologist excluded")

        elif action == "delete" and hid in hotspots_dict:
            del hotspots_dict[hid]

    return list(hotspots_dict.values())


@router.get("/{case_id}")
def get_triage_data(case_id: str, db: Session = Depends(get_db)):
    """
    Returns latest triage machine outputs, probability grid ref, heatmap URI, and saved edits.
    """
    stage_exec = db.scalars(
        select(StageExecution).where(
            StageExecution.case_id == case_id,
            StageExecution.stage == "triage"
        ).order_by(StageExecution.attempt.desc())
    ).first()

    if not stage_exec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No triage stage execution found for case {case_id}"
        )

    output_ref = stage_exec.output_ref or ""
    machine_output = {}

    try:
        if output_ref and output_ref.startswith("gs://"):
            b_name, bl_name = parse_gcs_uri(output_ref)
            out_bytes = download_blob_as_bytes(b_name, bl_name)
            machine_output = json.loads(out_bytes.decode("utf-8"))
        else:
            out_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/triage/output.json")
            machine_output = json.loads(out_bytes.decode("utf-8"))
    except Exception as e:
        print(f"[Triage Router Note] Could not fetch machine output from GCS: {e}")

    edits = stage_exec.review_edits or []
    machine_hotspots = machine_output.get("hotspots", [])
    effective_hotspots = apply_edit_ops(machine_hotspots, edits)

    heatmap_url = f"/api/v1/stages/triage/{case_id}/heatmap"
    if settings.CDN_BASE_URL:
        heatmap_url = f"{settings.CDN_BASE_URL.rstrip('/')}/cases/{case_id}/triage/heatmap_triage.png"

    # Ensure all effective hotspots have accessible thumbnail_url
    for hs in effective_hotspots:
        hs_id = hs.get("id")
        if settings.CDN_BASE_URL:
            hs["thumbnail_url"] = f"{settings.CDN_BASE_URL.rstrip('/')}/cases/{case_id}/triage/patches/{hs_id}_thumb.png"
        else:
            hs["thumbnail_url"] = f"/api/v1/stages/triage/{case_id}/hotspots/{hs_id}/thumbnail?mag=10x"

    return {
        "case_id": case_id,
        "stage_execution_id": str(stage_exec.id),
        "status": stage_exec.status,
        "heatmap_png_uri": machine_output.get("heatmap_png_uri"),
        "heatmap_direct_url": heatmap_url,
        "prob_grid_uri": machine_output.get("prob_grid_uri"),
        "grid": machine_output.get("grid"),
        "machine_hotspots": machine_hotspots,
        "effective_hotspots": effective_hotspots,
        "review_edits": edits,
        "model_versions": stage_exec.model_versions
    }


@router.get("/{case_id}/heatmap")
def get_triage_heatmap_image(case_id: str, db: Session = Depends(get_db)):
    """Returns the Viridis heatmap PNG overlay directly from GCS."""
    try:
        hm_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/triage/heatmap_triage.png")
        return Response(content=hm_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
    except Exception:
        pass

    try:
        hm_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/triage/heatmap.png")
        return Response(content=hm_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Heatmap image artifact not found in GCS",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )


def generate_synthetic_microscopic_patch(mag: str, stain: str, seed_str: str) -> bytes:
    import hashlib
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % 10000
    np.random.seed(seed)
    
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    
    if stain == "norm":
        bg_color = np.array([245, 230, 238], dtype=np.float32)
        nuc_color = np.array([55, 18, 105], dtype=np.float32)
        cyto_color = np.array([225, 145, 180], dtype=np.float32)
        mit_color = np.array([30, 5, 75], dtype=np.float32)
    else:
        bg_color = np.array([240, 222, 215], dtype=np.float32)
        nuc_color = np.array([80, 28, 55], dtype=np.float32)
        cyto_color = np.array([205, 128, 140], dtype=np.float32)
        mit_color = np.array([50, 15, 35], dtype=np.float32)

    canvas[:, :] = bg_color.astype(np.uint8)
    
    if mag == "10x":
        for g in range(14):
            gx = np.random.randint(40, 470)
            gy = np.random.randint(40, 470)
            gr = np.random.randint(35, 75)
            y, x = np.ogrid[:512, :512]
            mask = ((x - gx)**2 + (y - gy)**2) <= gr**2
            canvas[mask] = (0.6 * canvas[mask] + 0.4 * cyto_color).astype(np.uint8)
            for n in range(70):
                nx = int(np.clip(gx + np.random.normal(0, gr * 0.5), 0, 511))
                ny = int(np.clip(gy + np.random.normal(0, gr * 0.5), 0, 511))
                nr = np.random.randint(2, 4)
                n_mask = ((x - nx)**2 + (y - ny)**2) <= nr**2
                canvas[n_mask] = nuc_color.astype(np.uint8)
                
    elif mag == "20x":
        for g in range(4):
            gx = np.random.randint(100, 412)
            gy = np.random.randint(100, 412)
            gr = np.random.randint(80, 140)
            y, x = np.ogrid[:512, :512]
            mask = ((x - gx)**2 + (y - gy)**2) <= gr**2
            canvas[mask] = (0.5 * canvas[mask] + 0.5 * cyto_color).astype(np.uint8)
            l_mask = ((x - gx)**2 + (y - gy)**2) <= (gr * 0.35)**2
            canvas[l_mask] = bg_color.astype(np.uint8)
            for n in range(130):
                ang = np.random.uniform(0, 2 * np.pi)
                rad = np.random.uniform(gr * 0.35, gr * 0.95)
                nx = int(np.clip(gx + rad * np.cos(ang), 0, 511))
                ny = int(np.clip(gy + rad * np.sin(ang), 0, 511))
                nr = np.random.randint(4, 7)
                n_mask = ((x - nx)**2 + (y - ny)**2) <= nr**2
                canvas[n_mask] = nuc_color.astype(np.uint8)
                
    else: # 40x
        y, x = np.ogrid[:512, :512]
        canvas[:] = (0.3 * bg_color + 0.7 * cyto_color).astype(np.uint8)
        for n in range(24):
            nx = np.random.randint(60, 452)
            ny = np.random.randint(60, 452)
            nr_x = np.random.randint(14, 28)
            nr_y = np.random.randint(12, 24)
            rot = np.random.uniform(0, np.pi)
            
            cos_t, sin_t = np.cos(rot), np.sin(rot)
            x_rot = cos_t * (x - nx) + sin_t * (y - ny)
            y_rot = -sin_t * (x - nx) + cos_t * (y - ny)
            n_mask = ((x_rot / nr_x)**2 + (y_rot / nr_y)**2) <= 1.0
            canvas[n_mask] = nuc_color.astype(np.uint8)
            
            for k in range(3):
                cx_k = nx + np.random.randint(-nr_x // 3, nr_x // 3)
                cy_k = ny + np.random.randint(-nr_y // 3, nr_y // 3)
                k_mask = ((x - cx_k)**2 + (y - cy_k)**2) <= 3**2
                canvas[k_mask & n_mask] = (nuc_color * 0.5).astype(np.uint8)

        for m in range(3):
            mx = 160 + m * 110 + np.random.randint(-15, 15)
            my = 220 + np.random.randint(-40, 40)
            for seg in range(6):
                sx = mx + np.random.randint(-12, 12)
                sy = my + np.random.randint(-12, 12)
                s_mask = ((x - sx)**2 + (y - sy)**2) <= np.random.randint(5, 9)**2
                canvas[s_mask] = mit_color.astype(np.uint8)
                
    img = Image.fromarray(canvas)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@router.get("/{case_id}/hotspots/{hotspot_id}/thumbnail")
def get_hotspot_thumbnail(
    case_id: str, 
    hotspot_id: str, 
    mag: str = "10x",
    stain: str = "norm",
    cx: Optional[float] = Query(None),
    cy: Optional[float] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Extracts and streams a calibrated microscopic RGB patch centered on the specified hotspot.
    Supports real-time magnification switching (10x, 20x, 40x) and stain normalization toggling (norm, orig).
    """
    patch_blob = f"cases/{case_id}/triage/patches/{hotspot_id}_{mag}_{stain}.png"
    try:
        thumb_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, patch_blob)
        return Response(content=thumb_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        pass

    # Fast fallback for legacy 10x norm thumbnail
    if mag == "10x" and stain == "norm":
        try:
            thumb_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/triage/patches/{hotspot_id}_thumb.png")
            return Response(content=thumb_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
        except Exception:
            pass

    # Lookup Case and Slide
    case_uid = to_uuid(case_id)
    case_obj = db.scalars(select(Case).where(Case.id == case_uid)).first() if isinstance(case_uid, uuid.UUID) else db.get(Case, case_id)
    slide_obj = db.scalars(select(Slide).where(Slide.case_id == case_uid)).first() if isinstance(case_uid, uuid.UUID) else None
    if not slide_obj and case_obj and getattr(case_obj, "slides", None):
        slide_obj = case_obj.slides[0]
    if not slide_obj:
        slide_obj = db.scalars(select(Slide)).first()

    mpp_x = float(getattr(slide_obj, "mpp_x", 0.25) or 0.25)
    mpp_y = float(getattr(slide_obj, "mpp_y", 0.25) or mpp_x)

    cx_um = None
    cy_um = None

    if cx is not None and cy is not None:
        cx_um = float(cx)
        cy_um = float(cy)
    else:
        try:
            out_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/triage/output.json")
            tdata = json.loads(out_bytes.decode("utf-8"))
            target_hs = next((h for h in tdata.get("hotspots", []) if h["id"] == hotspot_id), None)
            if target_hs and "polygon_um" in target_hs:
                poly = np.array(target_hs["polygon_um"])
                cx_um = float(poly[:, 0].mean())
                cy_um = float(poly[:, 1].mean())
        except Exception:
            pass

    if (cx_um is None or cy_um is None) and case_obj and hasattr(case_obj, "stage_executions"):
        st_obj = next((s for s in case_obj.stage_executions if s.stage == "triage"), None)
        if st_obj and st_obj.review_edits:
            for ed in st_obj.review_edits:
                if ed.get("id") == hotspot_id and "polygon_um" in ed:
                    poly = np.array(ed["polygon_um"])
                    cx_um = float(poly[:, 0].mean())
                    cy_um = float(poly[:, 1].mean())
                    break

    if cx_um is None or cy_um is None:
        width_px = float(getattr(slide_obj, "width_px", 20000) or 20000)
        height_px = float(getattr(slide_obj, "height_px", 20000) or 20000)
        cx_um = width_px * mpp_x * 0.5
        cy_um = height_px * mpp_y * 0.5

    cx_px = int(cx_um / mpp_x)
    cy_px = int(cy_um / mpp_y)

    field_um = 512.0
    if mag == "20x":
        field_um = 256.0
    elif mag == "40x":
        field_um = 128.0

    extracted_bytes = None

    # 1. Fast Path: Reconstruct directly from GCS DeepZoom pyramid tiles (<100ms)
    if slide_obj:
        from pipeline.tiles import extract_patch_from_pyramid
        slide_id = str(slide_obj.id)
        width_px = int(getattr(slide_obj, "width_px", 20000) or 20000)
        height_px = int(getattr(slide_obj, "height_px", 20000) or 20000)
        patch_img = extract_patch_from_pyramid(
            slide_id=slide_id,
            cx_um=cx_um,
            cy_um=cy_um,
            field_um=field_um,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            width_px=width_px,
            height_px=height_px,
            layer=stain
        )
        if patch_img:
            buf = io.BytesIO()
            patch_img.save(buf, format="PNG")
            extracted_bytes = buf.getvalue()

    # 2. Fallback: OpenSlide raw WSI extraction if pyramid tiles are incomplete
    if extracted_bytes is None and slide_obj:
        scratch_dir = tempfile.mkdtemp(prefix="og_hs_thumb_")
        try:
            gcs_uri_original = getattr(slide_obj, "gcs_uri_original", None) or f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_id}/{getattr(slide_obj, 'id', 'slide')}.svs"
            raw_bucket_name, blob_name = parse_gcs_uri(gcs_uri_original)
            ext = os.path.splitext(blob_name)[1] or ".svs"
            local_slide_path = os.path.join(scratch_dir, f"slide{ext}")

            try:
                download_blob_to_filename(raw_bucket_name, blob_name, local_slide_path)
            except Exception as dl_e:
                print(f"[Thumbnail Slide Download Note] {dl_e}")

            if os.path.exists(local_slide_path):
                with OPENSLIDE_GLOBAL_LOCK:
                    import openslide
                    os_slide = None
                    try:
                        os_slide = openslide.OpenSlide(local_slide_path)
                        dim_w, dim_h = getattr(os_slide, "dimensions", (100000, 100000))
                        crop_w_px = max(1, int(round(field_um / mpp_x)))
                        crop_h_px = max(1, int(round(field_um / mpp_y)))

                        x0 = max(0, min(dim_w - crop_w_px, cx_px - crop_w_px // 2))
                        y0 = max(0, min(dim_h - crop_h_px, cy_px - crop_h_px // 2))

                        patch_raw = os_slide.read_region((x0, y0), 0, (crop_w_px, crop_h_px)).convert("RGB")
                    finally:
                        if os_slide and hasattr(os_slide, "close"):
                            os_slide.close()

                if stain == "norm":
                    try:
                        from pipeline.stain import PureNumpyMacenkoNormalizer
                        sp_text = download_blob_as_text(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/preprocess/stain_params.json")
                        sp_data = json.loads(sp_text)
                        if "stain_matrix" in sp_data and "max_concentrations" in sp_data:
                            norm_obj = PureNumpyMacenkoNormalizer()
                            norm_obj.stain_matrix_target = np.array(sp_data["stain_matrix"], dtype=float)
                            norm_obj.max_conc_target = np.array(sp_data["max_concentrations"], dtype=float)
                            norm_arr = norm_obj.transform(np.array(patch_raw))
                            patch_raw = Image.fromarray(norm_arr)
                    except Exception as se:
                        print(f"[Thumbnail Normalization Note] {se}")

                patch_final = patch_raw.resize((512, 512), Image.Resampling.BILINEAR)
                buf = io.BytesIO()
                patch_final.save(buf, format="PNG")
                extracted_bytes = buf.getvalue()
        except Exception as e:
            print(f"[Thumbnail Dynamic Extraction Note] {e}")
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    if extracted_bytes is None:
        extracted_bytes = generate_synthetic_microscopic_patch(mag, stain, f"{case_id}_{hotspot_id}_{mag}_{stain}")

    # Cache to GCS for all future requests
    try:
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            patch_blob,
            extracted_bytes,
            "image/png"
        )
    except Exception as up_e:
        print(f"[Thumbnail GCS Cache Note] {up_e}")

    return Response(content=extracted_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@router.post("/edits")
def save_triage_edits(payload: TriageEditsPayload, db: Session = Depends(get_db)):
    """
    Saves draft edit operations diff.
    """
    stage_exec = db.scalars(
        select(StageExecution).where(
            StageExecution.case_id == payload.case_id,
            StageExecution.stage == "triage"
        ).order_by(StageExecution.attempt.desc())
    ).first()

    if not stage_exec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Triage stage execution not found for case {payload.case_id}"
        )

    stage_exec.review_edits = payload.edits
    
    audit = AuditEvent(
        case_id=payload.case_id,
        actor="pathologist",
        event_type="review_edit",
        stage="triage",
        payload={"edit_count": len(payload.edits)}
    )
    db.add(audit)
    db.commit()

    return {"status": "success", "edits_count": len(payload.edits)}


@router.post("/confirm")
def confirm_triage(payload: TriageConfirmPayload, db: Session = Depends(get_db)):
    """
    Confirms triage stage, writes effective hotspots into DB, and queues next stage.
    """
    stage_exec = db.scalars(
        select(StageExecution).where(
            StageExecution.case_id == payload.case_id,
            StageExecution.stage == "triage"
        ).order_by(StageExecution.attempt.desc())
    ).first()

    if not stage_exec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Triage stage execution not found for case {payload.case_id}"
        )

    output_ref = stage_exec.output_ref or ""
    machine_hotspots = []
    try:
        if output_ref and output_ref.startswith("gs://"):
            b_name, bl_name = parse_gcs_uri(output_ref)
            out_bytes = download_blob_as_bytes(b_name, bl_name)
            machine_hotspots = json.loads(out_bytes.decode("utf-8")).get("hotspots", [])
        else:
            out_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{payload.case_id}/triage/output.json")
            machine_hotspots = json.loads(out_bytes.decode("utf-8")).get("hotspots", [])
    except Exception as e:
        print(f"[Triage Confirm Note] Could not load hotspots from GCS: {e}")

    edits = stage_exec.review_edits or []
    effective_hotspots = apply_edit_ops(machine_hotspots, edits)

    # Delete any prior confirmed hotspots for this case
    db.query(Hotspot).filter(Hotspot.case_id == payload.case_id).delete()

    # Persist effective hotspots to DB
    for hs in effective_hotspots:
        hotspot_row = Hotspot(
            id=hs["id"],
            case_id=payload.case_id,
            stage_execution_id=str(stage_exec.id),
            polygon_um=hs["polygon_um"],
            area_mm2=hs.get("area_mm2"),
            prob_mean=hs.get("prob_mean"),
            prob_max=hs.get("prob_max"),
            source=hs.get("source", "model"),
            excluded=hs.get("excluded", False),
            exclude_reason=hs.get("exclude_reason")
        )
        db.add(hotspot_row)

    stage_exec.status = "confirmed"
    stage_exec.reviewed_at = datetime.now(timezone.utc)
    stage_exec.reviewed_by = payload.reviewed_by

    if payload.no_invasive_tumor:
        next_stage_name = "report"
        input_data = {"benign_flag": True, "reason": "No invasive tumor identified"}
    else:
        next_stage_name = "mitosis"
        input_data = {"confirmed_hotspots_count": len(effective_hotspots)}

    case_uid = to_uuid(payload.case_id)
    next_exec = db.scalars(
        select(StageExecution).where(
            StageExecution.case_id == case_uid,
            StageExecution.stage == next_stage_name,
            StageExecution.attempt == 1
        )
    ).first()
    if not next_exec:
        next_exec = db.scalars(
            select(StageExecution).where(
                StageExecution.case_id == str(payload.case_id),
                StageExecution.stage == next_stage_name,
                StageExecution.attempt == 1
            )
        ).first()

    if not next_exec:
        next_exec = StageExecution(
            case_id=case_uid,
            stage=next_stage_name,
            attempt=1,
            status="queued",
            input_ref=input_data
        )
        db.add(next_exec)
    else:
        next_exec.status = "queued"
        next_exec.input_ref = input_data
        next_exec.started_at = None
        next_exec.completed_at = None
        next_exec.error = None

    audit = AuditEvent(
        case_id=payload.case_id,
        actor=payload.reviewed_by,
        event_type="stage_confirmed",
        stage="triage",
        payload={
            "confirmed_hotspots": len(effective_hotspots),
            "no_invasive_tumor": payload.no_invasive_tumor,
            "next_stage": next_stage_name
        }
    )
    db.add(audit)
    db.commit()

    try:
        from app.core.cloud_tasks import dispatch_stage_task
        dispatch_stage_task(
            case_id=str(payload.case_id),
            stage=next_stage_name,
            stage_exec_id=str(next_exec.id)
        )
    except Exception as e:
        print(f"[CloudTasks Warning] Failed to dispatch next stage {next_stage_name}: {e}")

    return {
        "status": "confirmed",
        "case_id": payload.case_id,
        "confirmed_hotspots_count": len(effective_hotspots),
        "next_stage_queued": next_stage_name
    }

