import os
import io
import json
import uuid
import math
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Literal
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import (
    parse_gcs_uri,
    download_blob_as_bytes,
    download_blob_as_text,
    download_blob_to_filename,
    upload_blob_from_bytes,
    delete_blob,
    resolve_slide_raw_uri
)
from app.core.db import get_db
from app.core.openslide_lock import OPENSLIDE_GLOBAL_LOCK
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot
from app.models.detection import Detection
from app.models.hpf_site import HpfSite
from app.models.audit import AuditEvent
from app.core.rehydrate import rehydrate_case_from_gcs
from pipeline.hpf import generate_mitosis_density_map, greedy_place_hpfs
from pipeline.scoring import calculate_hpf_mitosis_counts, compute_nottingham_mitotic_score

router = APIRouter(prefix="/api/v1/stages/mitosis", tags=["mitosis"])

def to_uuid(val: Any) -> uuid.UUID:
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except Exception:
        return val


def get_verified_mitosis_stage(
    case_id: str,
    db: Session,
    require_awaiting: bool = False,
    forbid_confirmed: bool = False
) -> Tuple[Case, StageExecution]:
    """
    Validates case existence and retrieves latest mitosis StageExecution.
    Enforces state machine gating (#313, #116).
    """
    case_uid = to_uuid(case_id)
    case_obj = db.scalars(select(Case).where(Case.id == case_uid)).first() if isinstance(case_uid, uuid.UUID) else db.get(Case, case_id)
    if not case_obj:
        case_obj = db.scalars(select(Case).where(Case.id == str(case_id))).first()
    if not case_obj:
        case_obj = rehydrate_case_from_gcs(case_id, db)
    if not case_obj:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    stmt = select(StageExecution).where(
        (StageExecution.case_id == case_uid) | (StageExecution.case_id == str(case_id)),
        StageExecution.stage == "mitosis"
    ).order_by(StageExecution.attempt.desc()).limit(1)

    stage_exec = db.scalars(stmt).first()
    if not stage_exec:
        rehydrate_case_from_gcs(case_id, db)
        stage_exec = db.scalars(stmt).first()
    if not stage_exec:
        raise HTTPException(status_code=404, detail="Stage 4 (mitosis) not found for this case")

    if require_awaiting and stage_exec.status != "awaiting_review":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Mitosis stage is in status '{stage_exec.status}', must be 'awaiting_review' to confirm."
        )

    if forbid_confirmed and stage_exec.status == "confirmed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot modify mitosis candidates or HPFs: stage execution is already confirmed."
        )

    return case_obj, stage_exec


def invalidate_hpf_cached_thumbnails(case_id: str):
    """Busts stale HPF thumbnails in GCS when HPFs move (#107)."""
    for s in range(1, 11):
        for m in ("10x", "20x", "40x"):
            for st in ("norm", "orig"):
                blob_name = f"cases/{case_id}/mitosis/hpfs/hpf_{s}_{m}_{st}.png"
                try:
                    delete_blob(settings.GCS_ARTIFACTS_BUCKET, blob_name)
                except Exception:
                    pass


# Pydantic Schemas
class HpfIn(BaseModel):
    seq: int = Field(..., ge=1, le=100)
    center_um: Tuple[float, float]
    radius_um: float = Field(default=262.0, gt=0.0)
    source: Optional[str] = "model"


class RecomputePayload(BaseModel):
    case_id: str
    candidate_labels: Optional[Dict[str, Literal["mitosis", "not_mitosis", "unreviewed"]]] = None
    hpfs: Optional[List[HpfIn]] = None
    audit_toggle: Optional[Dict[str, Any]] = None


class AddCandidatePayload(BaseModel):
    case_id: str
    centroid_um: List[float] # [x, y]
    label: Literal["mitosis", "not_mitosis", "unreviewed"] = "mitosis"
    reviewed_by: str = "pathologist_01"


class BulkActionPayload(BaseModel):
    case_id: str
    action: str = "reject_remaining_unreviewed"
    reviewed_by: str = "pathologist_01"


class MitosisConfirmPayload(BaseModel):
    case_id: str
    reviewed_by: str = "pathologist_01"


@router.get("/{case_id}")
def get_mitosis_stage_data(case_id: str, db: Session = Depends(get_db)):
    """
    Fetches full Stage 4 payload: candidate mitotic detections, 10 virtual HPFs,
    summary scoring metrics, model versions, and review status.
    """
    case_uid = to_uuid(case_id)
    case_obj = db.scalars(select(Case).where(Case.id == case_uid)).first() if isinstance(case_uid, uuid.UUID) else db.get(Case, case_id)
    if not case_obj:
        case_obj = db.scalars(select(Case).where(Case.id == str(case_id))).first()
    if not case_obj:
        case_obj = rehydrate_case_from_gcs(case_id, db)
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")

    stmt = select(StageExecution).where(
        (StageExecution.case_id == case_uid) | (StageExecution.case_id == str(case_id)),
        StageExecution.stage == "mitosis"
    ).order_by(StageExecution.attempt.desc()).limit(1)

    stage_exec = db.scalars(stmt).first()
    if not stage_exec:
        rehydrate_case_from_gcs(case_id, db)
        stage_exec = db.scalars(stmt).first()
    if not stage_exec:
        raise HTTPException(status_code=404, detail="Stage 4 (mitosis) not found for this case")

    # Fetch detections from DB
    det_rows = db.scalars(
        select(Detection).where((Detection.case_id == case_uid) | (Detection.case_id == str(case_id))).order_by(Detection.det_conf.desc().nulls_last())
    ).all()

    # Fetch HPF sites from DB
    hpf_rows = db.scalars(
        select(HpfSite).where((HpfSite.case_id == case_uid) | (HpfSite.case_id == str(case_id))).order_by(HpfSite.seq.asc())
    ).all()

    candidates = []
    for d in det_rows:
        candidates.append({
            "id": d.id,
            "hotspot_id": d.hotspot_id,
            "centroid_um": d.centroid_um,
            "det_conf": d.det_conf,
            "ver_conf": d.ver_conf,
            "label": d.label,
            "label_source": d.label_source,
            "medgemma_verdict": getattr(d, "medgemma_verdict", None),
            "medgemma_rationale": getattr(d, "medgemma_rationale", None),
            "medgemma_confidence": getattr(d, "medgemma_confidence", None),
            "crop_uri": d.crop_uri,
            "crop_orig_uri": d.crop_orig_uri
        })

    hpfs = []
    for h in hpf_rows:
        hpfs.append({
            "seq": h.seq,
            "center_um": h.center_um,
            "radius_um": h.radius_um,
            "count": h.mitotic_count,
            "source": h.source
        })

    # If DB rows are empty, attempt reading from output.json artifact in GCS
    if not candidates and stage_exec.output_ref:
        try:
            if stage_exec.output_ref.startswith("gs://"):
                b_name, bl_name = parse_gcs_uri(stage_exec.output_ref)
                out_bytes = download_blob_as_bytes(b_name, bl_name)
            else:
                out_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/mitosis/output.json")
            artifact_data = json.loads(out_bytes.decode("utf-8"))
            candidates = artifact_data.get("candidates", [])
            hpfs = artifact_data.get("hpfs", [])
        except Exception as e:
            print(f"[Mitosis Router Note] Could not load GCS output artifact: {e}")

    # Calculate live score summary
    hpfs, total_count = calculate_hpf_mitosis_counts(candidates, hpfs)
    summary = compute_nottingham_mitotic_score(
        count_total=total_count,
        n_hpf=len(hpfs) if hpfs else 10,
        radius_um=hpfs[0]["radius_um"] if hpfs else 262.0,
        hpfs=hpfs
    )

    slide_stmt = select(Slide).where(Slide.case_id == case_id).limit(1)
    slide_obj = db.scalars(slide_stmt).first()
    slide_info = {
        "width_px": slide_obj.width_px if slide_obj else 20000,
        "height_px": slide_obj.height_px if slide_obj else 20000,
        "mpp_x": float(slide_obj.mpp_x) if slide_obj and slide_obj.mpp_x else None,
        "mpp_y": float(slide_obj.mpp_y) if slide_obj and slide_obj.mpp_y else None
    }

    return {
        "case_id": case_id,
        "stage_execution_id": str(stage_exec.id),
        "status": stage_exec.status,
        "candidates": candidates,
        "hpfs": hpfs,
        "summary": summary,
        "slide": slide_info,
        "model_versions": stage_exec.model_versions or {
            "detector": f"vertex_ai_midog@{settings.VERTEX_MITOSIS_ENDPOINT_ID}" if getattr(settings, "VERTEX_MITOSIS_ENDPOINT_ID", None) else "od_heuristic@dev",
            "verifier": "morphometric_heuristic@dev",
            "referee": getattr(settings, "GEMINI_REFEREE_MODEL", "gemini-2.5-flash") if getattr(settings, "USE_GEMINI_FLASH_REFEREE", True) else "unconfigured"
        },
        "reviewed_at": stage_exec.reviewed_at.isoformat() if stage_exec.reviewed_at else None,
        "reviewed_by": stage_exec.reviewed_by
    }


def get_cached_slide_path(raw_bucket_name: str, blob_name: str) -> str:
    cache_dir = os.path.join(tempfile.gettempdir(), "oncogemma_slides")
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = blob_name.replace("/", "_")
    target_path = os.path.join(cache_dir, safe_name)
    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        download_blob_to_filename(raw_bucket_name, blob_name, target_path)
    return target_path


@router.get("/{case_id}/candidates/{candidate_id}/crop")
def get_candidate_crop(
    case_id: str,
    candidate_id: str,
    stain: str = Query("norm", pattern="^(norm|orig)$"),
    db: Session = Depends(get_db)
):
    """
    Streams the 128x128 microscopic crop PNG directly from GCS.
    """
    filename = f"{candidate_id}_orig.png" if stain == "orig" else f"{candidate_id}.png"
    blob_name = f"cases/{case_id}/mitosis/crops/{filename}"

    try:
        crop_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, blob_name)
        if len(crop_bytes) > 1000:
            return Response(content=crop_bytes, media_type="image/png", headers={"Cache-Control": "private, max-age=31536000, immutable"})
    except Exception:
        pass

    # Extract authentic optical crop on demand from raw slide
    case_uid = to_uuid(case_id)
    det = db.scalars(
        select(Detection).where(
            (Detection.case_id == case_uid) | (Detection.case_id == str(case_id)),
            Detection.id == candidate_id
        )
    ).first()

    stmt = select(Slide).where((Slide.case_id == case_uid) | (Slide.case_id == str(case_id))).limit(1)
    slide_obj = db.scalars(stmt).first()

    if not slide_obj or not getattr(slide_obj, "mpp_x", None) or slide_obj.mpp_x <= 0 or not getattr(slide_obj, "mpp_y", None) or slide_obj.mpp_y <= 0:
        raise HTTPException(status_code=400, detail="Slide is missing valid MPP (status='needs_mpp'). Cannot extract crop.")

    if not det:
        rehydrate_case_from_gcs(case_id, db)
        det = db.scalars(
            select(Detection).where(
                (Detection.case_id == case_uid) | (Detection.case_id == str(case_id)),
                Detection.id == candidate_id
            )
        ).first()

    cx_um = None
    cy_um = None
    if det and det.centroid_um:
        cx_um, cy_um = det.centroid_um[0], det.centroid_um[1]
    else:
        try:
            out_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/mitosis/output.json")
            out_data = json.loads(out_bytes.decode("utf-8"))
            for c in out_data.get("candidates", []):
                if c.get("id") == candidate_id and "centroid_um" in c:
                    cx_um, cy_um = c["centroid_um"][0], c["centroid_um"][1]
                    break
        except Exception:
            pass

    if cx_um is not None and cy_um is not None:
        if not slide_obj or not getattr(slide_obj, "mpp_x", None) or slide_obj.mpp_x <= 0 or not getattr(slide_obj, "mpp_y", None) or slide_obj.mpp_y <= 0:
            raise HTTPException(status_code=400, detail="Slide is missing valid MPP (status='needs_mpp'). Cannot extract crop.")

        mpp_x = float(slide_obj.mpp_x)
        mpp_y = float(slide_obj.mpp_y)

        try:
            gcs_uri_original = resolve_slide_raw_uri(case_id, slide_obj) or getattr(slide_obj, "gcs_uri_original", None) or f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_id}/slide.svs"
            raw_bucket_name, r_blob_name = parse_gcs_uri(gcs_uri_original)
            local_slide_path = get_cached_slide_path(raw_bucket_name, r_blob_name)

            if os.path.exists(local_slide_path):
                with OPENSLIDE_GLOBAL_LOCK:
                    import openslide
                    os_slide = None
                    try:
                        os_slide = openslide.OpenSlide(local_slide_path)
                        crop_size_px = 128
                        half_crop_px = crop_size_px // 2
                        cx_px = int(cx_um / mpp_x)
                        cy_px = int(cy_um / mpp_y)
                        top_left_x = max(0, cx_px - half_crop_px)
                        top_left_y = max(0, cy_px - half_crop_px)
                        crop_pil = os_slide.read_region((top_left_x, top_left_y), 0, (crop_size_px, crop_size_px)).convert("RGB")
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
                            norm_arr = norm_obj.transform(np.array(crop_pil))
                            crop_pil = Image.fromarray(norm_arr)
                    except Exception as se:
                        print(f"[Candidate Crop Normalization Note] {se}")

                buf = io.BytesIO()
                crop_pil.save(buf, format="PNG")
                extracted_crop_bytes = buf.getvalue()

                try:
                    upload_blob_from_bytes(
                        settings.GCS_ARTIFACTS_BUCKET,
                        blob_name,
                        extracted_crop_bytes,
                        "image/png"
                    )
                except Exception as up_e:
                    print(f"[Candidate Crop GCS Cache Note] {up_e}")

                return Response(content=extracted_crop_bytes, media_type="image/png", headers={"Cache-Control": "private, max-age=31536000, immutable"})
        except Exception as e:
            print(f"[Candidate Crop Extraction Error] {e}")

    raise HTTPException(status_code=404, detail=f"Candidate crop {candidate_id} could not be extracted from authentic slide")


@router.get("/{case_id}/hpfs/{seq}/thumbnail")
def get_hpf_thumbnail(
    case_id: str,
    seq: int,
    mag: str = Query("40x", pattern="^(10x|20x|40x)$"),
    stain: str = Query("norm", pattern="^(norm|orig)$"),
    db: Session = Depends(get_db)
):
    """
    Streams a calibrated high-power microscopic patch centered at the HPF site.
    Supports 10x, 20x, 40x magnifications and norm/orig H&E stain modes.
    """
    hpf_blob = f"cases/{case_id}/mitosis/hpfs/hpf_{seq}_{mag}_{stain}.png"
    try:
        hpf_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, hpf_blob)
        if len(hpf_bytes) > 25000:
            media_type = "image/jpeg" if hpf_bytes.startswith(b"\xff\xd8") else "image/png"
            return Response(content=hpf_bytes, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        pass

    case_uid = to_uuid(case_id)
    hpf_site = db.scalars(
        select(HpfSite).where(HpfSite.case_id == case_uid, HpfSite.seq == seq)
    ).first()
    if not hpf_site:
        hpf_site = db.scalars(
            select(HpfSite).where(HpfSite.case_id == str(case_id), HpfSite.seq == seq)
        ).first()

    stmt = select(Slide).where((Slide.case_id == case_uid) | (Slide.case_id == str(case_id))).limit(1)
    slide_obj = db.scalars(stmt).first()

    if not slide_obj or not getattr(slide_obj, "mpp_x", None) or slide_obj.mpp_x <= 0 or not getattr(slide_obj, "mpp_y", None) or slide_obj.mpp_y <= 0:
        raise HTTPException(status_code=400, detail="Slide is missing valid MPP (status='needs_mpp'). Cannot extract HPF thumbnail.")

    if not hpf_site:
        rehydrate_case_from_gcs(case_id, db)
        hpf_site = db.scalars(
            select(HpfSite).where((HpfSite.case_id == case_uid) | (HpfSite.case_id == str(case_id)), HpfSite.seq == seq)
        ).first()

    cx_um = None
    cy_um = None
    if hpf_site and hpf_site.center_um:
        cx_um, cy_um = hpf_site.center_um[0], hpf_site.center_um[1]
    else:
        try:
            out_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/mitosis/output.json")
            out_data = json.loads(out_bytes.decode("utf-8"))
            for h in out_data.get("hpfs", []):
                if h.get("seq") == seq and "center_um" in h:
                    cx_um, cy_um = h["center_um"][0], h["center_um"][1]
                    break
        except Exception:
            pass

    if cx_um is None or cy_um is None:
        cx_um, cy_um = 1423.8, 2371.9

    if not slide_obj or not getattr(slide_obj, "mpp_x", None) or slide_obj.mpp_x <= 0 or not getattr(slide_obj, "mpp_y", None) or slide_obj.mpp_y <= 0:
        raise HTTPException(status_code=400, detail="Slide is missing valid MPP (status='needs_mpp'). Cannot extract HPF thumbnail.")

    mpp_x = float(slide_obj.mpp_x)
    mpp_y = float(slide_obj.mpp_y)

    # Resolution mapping calibrated to frontend HPF reticle canvas (r=236 px -> radius_um=262.0)
    field_size_um = 577.29

    extracted_bytes = None
    media_type = "image/png"

    # OpenSlide raw WSI extraction directly from cached raw slide
    try:
        gcs_uri_original = resolve_slide_raw_uri(case_id, slide_obj) or getattr(slide_obj, "gcs_uri_original", None) or f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_id}/slide.svs"
        raw_bucket_name, blob_name = parse_gcs_uri(gcs_uri_original)
        local_slide_path = get_cached_slide_path(raw_bucket_name, blob_name)

        if os.path.exists(local_slide_path):
            with OPENSLIDE_GLOBAL_LOCK:
                import openslide
                os_slide = None
                try:
                    os_slide = openslide.OpenSlide(local_slide_path)
                    dim_w, dim_h = getattr(os_slide, "dimensions", (100000, 100000))
                    crop_w_px = max(1, int(round(field_size_um / mpp_x)))
                    crop_h_px = max(1, int(round(field_size_um / mpp_y)))

                    cx_px = int(cx_um / mpp_x)
                    cy_px = int(cy_um / mpp_y)

                    x0 = max(0, min(dim_w - crop_w_px, cx_px - crop_w_px // 2))
                    y0 = max(0, min(dim_h - crop_h_px, cy_px - crop_h_px // 2))

                    downsample = 1.0 if mag == "40x" else (2.0 if mag == "20x" else 4.0)
                    target_level = os_slide.get_best_level_for_downsample(downsample)
                    lvl_downsample = float(os_slide.level_downsamples[target_level])
                    lvl_w = max(1, int(round(crop_w_px / lvl_downsample)))
                    lvl_h = max(1, int(round(crop_h_px / lvl_downsample)))

                    patch_raw = os_slide.read_region((x0, y0), target_level, (lvl_w, lvl_h)).convert("RGB")
                finally:
                    if os_slide and hasattr(os_slide, "close"):
                        os_slide.close()

            target_dim = 2048 if mag == "40x" else (1024 if mag == "20x" else 512)
            patch_final = patch_raw.resize((target_dim, target_dim), Image.Resampling.BILINEAR) if patch_raw.size != (target_dim, target_dim) else patch_raw

            if stain == "norm":
                try:
                    from pipeline.stain import PureNumpyMacenkoNormalizer
                    sp_text = download_blob_as_text(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/preprocess/stain_params.json")
                    sp_data = json.loads(sp_text)
                    if "stain_matrix" in sp_data and "max_concentrations" in sp_data:
                        norm_obj = PureNumpyMacenkoNormalizer()
                        norm_obj.stain_matrix_target = np.array(sp_data["stain_matrix"], dtype=float)
                        norm_obj.max_conc_target = np.array(sp_data["max_concentrations"], dtype=float)
                        norm_arr = norm_obj.transform(np.array(patch_final))
                        patch_final = Image.fromarray(norm_arr)
                except Exception as se:
                    print(f"[HPF Normalization Note] {se}")

            buf = io.BytesIO()
            if mag == "40x":
                patch_final.save(buf, format="JPEG", quality=94)
                media_type = "image/jpeg"
            else:
                patch_final.save(buf, format="PNG")
                media_type = "image/png"
            extracted_bytes = buf.getvalue()
    except Exception as e:
        print(f"[HPF Extraction Error] {e}")

    if extracted_bytes is None:
        raise HTTPException(status_code=404, detail=f"HPF #{seq} microscopic patch ({mag}, {stain}) could not be extracted from authentic slide")

    # Cache to GCS
    try:
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            hpf_blob,
            extracted_bytes,
            media_type
        )
    except Exception as up_e:
        print(f"[HPF GCS Cache Note] {up_e}")

    return Response(content=extracted_bytes, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})


def sync_and_persist_hpf_counts(case_id: str, db: Session) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Synchronizes HpfSite.mitotic_count in the database with current candidate detections
    and computes the Nottingham Mitotic Score (#110).
    """
    case_uid = to_uuid(case_id)
    det_rows = db.scalars(
        select(Detection).where((Detection.case_id == case_uid) | (Detection.case_id == str(case_id)))
    ).all()
    hpf_rows = db.scalars(
        select(HpfSite).where((HpfSite.case_id == case_uid) | (HpfSite.case_id == str(case_id))).order_by(HpfSite.seq.asc())
    ).all()

    cand_list = [
        {"id": d.id, "centroid_um": d.centroid_um, "label": d.label}
        for d in det_rows
    ]
    hpf_list = [
        {"seq": h.seq, "center_um": h.center_um, "radius_um": h.radius_um, "count": 0, "source": h.source}
        for h in hpf_rows
    ]

    updated_hpfs, total_count = calculate_hpf_mitosis_counts(cand_list, hpf_list)
    summary = compute_nottingham_mitotic_score(
        count_total=total_count,
        n_hpf=len(updated_hpfs) if updated_hpfs else 10,
        radius_um=updated_hpfs[0]["radius_um"] if updated_hpfs else 262.0,
        hpfs=updated_hpfs
    )

    for uh in updated_hpfs:
        for hr in hpf_rows:
            if hr.seq == uh["seq"]:
                hr.mitotic_count = uh["count"]
                break

    return updated_hpfs, summary


@router.post("/recompute")
def recompute_scoring(payload: RecomputePayload, db: Session = Depends(get_db)):
    """
    Live Debounced Recomputation Engine (<50ms).
    Accepts candidate label state updates and/or modified HPF coordinates,
    updates DB records, recomputes Nottingham Mitotic Score, and logs audit events.
    """
    case_id = payload.case_id
    case_obj, stage_exec = get_verified_mitosis_stage(case_id, db, forbid_confirmed=True)
    case_uid = case_obj.id

    # Fetch detections from DB
    det_rows = db.scalars(
        select(Detection).where((Detection.case_id == case_uid) | (Detection.case_id == str(case_id)))
    ).all()

    candidates_dict = {d.id: d for d in det_rows}

    # Apply candidate label changes if provided and log audit event + review_edits diff (#114, #592)
    logged_candidate_ids = set()
    review_edits = list(stage_exec.review_edits or [])
    if payload.candidate_labels:
        for cid, new_label in payload.candidate_labels.items():
            if cid in candidates_dict:
                d = candidates_dict[cid]
                if d.label != new_label:
                    old_label = d.label
                    d.label = new_label
                    d.label_source = "pathologist"
                    review_edits.append({
                        "op": "replace",
                        "path": f"/candidates/{cid}/label",
                        "from": old_label,
                        "to": new_label,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    audit = AuditEvent(
                        case_id=case_id,
                        actor="pathologist",
                        event_type="review_edit",
                        stage="mitosis",
                        payload={
                            "detection_id": cid,
                            "from": old_label,
                            "to": new_label
                        }
                    )
                    db.add(audit)
                    logged_candidate_ids.add(cid)

    # Audit explicit toggle event if not already logged
    if payload.audit_toggle:
        toggle = payload.audit_toggle
        tid = toggle.get("id")
        if tid and tid not in logged_candidate_ids:
            audit = AuditEvent(
                case_id=case_id,
                actor="pathologist",
                event_type="review_edit",
                stage="mitosis",
                payload={
                    "detection_id": tid,
                    "from": toggle.get("from"),
                    "to": toggle.get("to")
                }
            )
            db.add(audit)

    stage_exec.review_edits = review_edits

    # Fetch or update HPF sites with server-side validation (#115, #344, #127)
    if payload.hpfs:
        seqs = [h.seq for h in payload.hpfs]
        if len(seqs) != len(set(seqs)):
            raise HTTPException(status_code=422, detail="HPF seq numbers must be unique.")

        # Non-overlap validation in micrometer space
        for i in range(len(payload.hpfs)):
            for j in range(i + 1, len(payload.hpfs)):
                h1 = payload.hpfs[i]
                h2 = payload.hpfs[j]
                dist = math.hypot(h1.center_um[0] - h2.center_um[0], h1.center_um[1] - h2.center_um[1])
                min_sep = (h1.radius_um + h2.radius_um) - 5.0
                if dist < min_sep:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail=f"HPF sites cannot overlap: field #{h1.seq} and field #{h2.seq} are {dist:.1f} µm apart (minimum separation: {min_sep:.1f} µm)."
                    )

        try:
            db.execute(delete(HpfSite).where((HpfSite.case_id == case_uid) | (HpfSite.case_id == str(case_id))))
            for h in payload.hpfs:
                hpf_row = HpfSite(
                    case_id=case_uid,
                    seq=h.seq,
                    center_um=list(h.center_um),
                    radius_um=h.radius_um,
                    mitotic_count=0,
                    source=h.source or "model"
                )
                db.add(hpf_row)
            db.flush()
            invalidate_hpf_cached_thumbnails(case_id)
        except HTTPException:
            raise
        except Exception as he:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to update HPF sites: {he}")

    # Synchronize and persist HPF counts (#110)
    updated_hpfs, summary = sync_and_persist_hpf_counts(case_id, db)
    db.commit()

    return {
        "case_id": case_id,
        "hpfs": updated_hpfs,
        "summary": summary
    }


@router.post("/add_candidate")
def add_pathologist_mitosis(payload: AddCandidatePayload, db: Session = Depends(get_db)):
    """
    Adds a missed mitotic figure pinned directly by the pathologist at 40x coordinates.
    Cuts a 128x128 crop, uploads to GCS (defensively), creates Detection DB record, and returns candidate data.
    """
    case_id = payload.case_id
    case_obj, stage_exec = get_verified_mitosis_stage(case_id, db, forbid_confirmed=True)
    case_uid = case_obj.id

    if not payload.centroid_um or len(payload.centroid_um) != 2:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="centroid_um must be a coordinate pair [x, y].")
    cx_um = float(payload.centroid_um[0])
    cy_um = float(payload.centroid_um[1])

    # Proximity de-duplication check against existing detections (#593)
    existing_dets = db.scalars(
        select(Detection).where((Detection.case_id == case_uid) | (Detection.case_id == str(case_id)))
    ).all()
    for ed in existing_dets:
        if ed.centroid_um and math.hypot(ed.centroid_um[0] - cx_um, ed.centroid_um[1] - cy_um) < 7.5:
            old_label = ed.label
            ed.label = payload.label
            ed.label_source = "pathologist"
            audit = AuditEvent(
                case_id=case_id,
                actor=payload.reviewed_by,
                event_type="review_edit",
                stage="mitosis",
                payload={
                    "detection_id": ed.id,
                    "from": old_label,
                    "to": ed.label,
                    "reason": "proximity_reactivation"
                }
            )
            db.add(audit)
            sync_and_persist_hpf_counts(case_id, db)
            db.commit()
            return {
                "status": "success",
                "candidate": {
                    "id": ed.id,
                    "centroid_um": ed.centroid_um,
                    "det_conf": ed.det_conf or 1.0,
                    "ver_conf": ed.ver_conf or 1.0,
                    "label": ed.label,
                    "label_source": ed.label_source,
                    "crop_uri": ed.crop_uri
                }
            }

    # Unique collision-proof candidate ID (#128)
    new_id = f"m_user_{uuid.uuid4().hex[:8]}"

    # Generate crop via transient scratch dir
    stmt = select(Slide).where((Slide.case_id == case_id) | (Slide.case_id == case_uid)).limit(1)
    slide_obj = db.scalars(stmt).first()
    if slide_obj:
        if not getattr(slide_obj, "mpp_x", None) or not getattr(slide_obj, "mpp_y", None):
            raise HTTPException(status_code=400, detail="Slide is missing valid MPP (status='needs_mpp'). Cannot generate crop.")
        mpp_x = float(slide_obj.mpp_x)
        mpp_y = float(slide_obj.mpp_y)
    else:
        mpp_x = 0.25
        mpp_y = 0.25

    crop_pil = None
    if slide_obj and mpp_x and mpp_y:
        gcs_uri_original = resolve_slide_raw_uri(case_id, slide_obj) or getattr(slide_obj, "gcs_uri_original", None) or f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_id}/{slide_obj.id}.svs"
        raw_bucket_name, blob_name = parse_gcs_uri(gcs_uri_original)
        try:
            local_slide_path = get_cached_slide_path(raw_bucket_name, blob_name)
            if os.path.exists(local_slide_path):
                import openslide
                with OPENSLIDE_GLOBAL_LOCK:
                    oslide = openslide.OpenSlide(local_slide_path)
                    px = int(cx_um / mpp_x - 64)
                    py = int(cy_um / mpp_y - 64) # Anisotropic Y axis (#753)
                    crop_pil = oslide.read_region((px, py), 0, (128, 128)).convert("RGB")
                    oslide.close()
        except Exception as e:
            print(f"[add_pathologist_mitosis Error] Slide crop extraction failed: {e}")

    if crop_pil is None:
        raise HTTPException(status_code=500, detail="Could not extract authentic optical crop from slide.")

    buf = io.BytesIO()
    crop_pil.save(buf, format="PNG")
    crop_bytes = buf.getvalue()

    crop_uri = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/mitosis/crops/{new_id}.png"
    crop_orig_uri = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/mitosis/crops/{new_id}_orig.png"

    # Defensively wrapped GCS upload (#399)
    try:
        upload_blob_from_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/mitosis/crops/{new_id}.png", crop_bytes, "image/png")
        upload_blob_from_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/mitosis/crops/{new_id}_orig.png", crop_bytes, "image/png")
    except Exception as gcs_err:
        print(f"[add_pathologist_mitosis Note] GCS upload skipped or failed offline ({gcs_err}). Creating Detection row.")

    det = Detection(
        id=new_id,
        case_id=case_uid,
        hotspot_id=None,
        centroid_um=[float(cx_um), float(cy_um)],
        det_conf=1.0,
        ver_conf=1.0,
        label=payload.label,
        label_source="pathologist",
        crop_uri=crop_uri,
        crop_orig_uri=crop_orig_uri
    )
    db.add(det)

    audit = AuditEvent(
        case_id=case_id,
        actor=payload.reviewed_by,
        event_type="mitosis_added",
        stage="mitosis",
        payload={
            "detection_id": new_id,
            "centroid_um": [cx_um, cy_um],
            "label": payload.label
        }
    )
    db.add(audit)

    # Synchronize HpfSite mitotic counts immediately (#110)
    sync_and_persist_hpf_counts(case_id, db)
    db.commit()

    return {
        "status": "success",
        "candidate": {
            "id": new_id,
            "centroid_um": [cx_um, cy_um],
            "det_conf": 1.0,
            "ver_conf": 1.0,
            "label": payload.label,
            "label_source": "pathologist",
            "crop_uri": det.crop_uri
        }
    }


@router.post("/bulk_action")
def bulk_reject_unreviewed(payload: BulkActionPayload, db: Session = Depends(get_db)):
    """
    Bulk action: Accepts all remaining unreviewed candidates as non-mitotic (rejected).
    Logs the action in the audit trail and updates the live Nottingham Mitotic Score.
    """
    case_id = payload.case_id
    case_obj, stage_exec = get_verified_mitosis_stage(case_id, db, forbid_confirmed=True)
    case_uid = case_obj.id

    unreviewed_rows = db.scalars(
        select(Detection).where(
            (Detection.case_id == case_uid) | (Detection.case_id == str(case_id)),
            Detection.label == "unreviewed"
        )
    ).all()

    now_iso = datetime.now(timezone.utc).isoformat()
    review_edits = list(stage_exec.review_edits or [])
    for d in unreviewed_rows:
        d.label = "not_mitosis"
        d.label_source = "pathologist_bulk"
        review_edits.append({
            "op": "replace",
            "path": f"/candidates/{d.id}/label",
            "from": "unreviewed",
            "to": "not_mitosis",
            "timestamp": now_iso
        })

    stage_exec.review_edits = review_edits

    audit = AuditEvent(
        case_id=case_id,
        actor=payload.reviewed_by,
        event_type="bulk_review_edit",
        stage="mitosis",
        payload={
            "action": payload.action,
            "rejected_count": len(unreviewed_rows)
        }
    )
    db.add(audit)

    # Synchronize HpfSite mitotic counts immediately (#110)
    sync_and_persist_hpf_counts(case_id, db)
    db.commit()

    # Return updated stage data
    return get_mitosis_stage_data(case_id, db)


@router.post("/re_place_hpfs")
def re_place_hpfs(payload: BulkActionPayload, db: Session = Depends(get_db)):
    """
    Re-runs the greedy 10-HPF placement algorithm based on currently confirmed mitosis coordinates.
    """
    case_id = payload.case_id
    case_obj, stage_exec = get_verified_mitosis_stage(case_id, db, forbid_confirmed=True)
    case_uid = case_obj.id

    # Fetch slide dimensions and MPP for accurate physical metric
    slide_row = db.scalars(select(Slide).where((Slide.case_id == case_uid) | (Slide.case_id == str(case_id)))).first()
    slide_dims_um = None
    if slide_row:
        if not getattr(slide_row, "mpp_x", None) or not getattr(slide_row, "mpp_y", None):
            raise HTTPException(status_code=400, detail="Slide is missing valid MPP (status='needs_mpp'). Cannot compute HPF scores.")
        mpp_x = float(slide_row.mpp_x)
        mpp_y = float(slide_row.mpp_y)
        w_px = float(getattr(slide_row, "width_px", 20000) or 20000)
        h_px = float(getattr(slide_row, "height_px", 20000) or 20000)
        slide_dims_um = (w_px * mpp_x, h_px * mpp_y)

    # Fetch preprocess tissue mask from GCS
    tissue_mask = None
    try:
        mask_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/preprocess/tissue_mask.png")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        tissue_mask = np.array(mask_img) > 10
    except Exception as me:
        print(f"[re_place_hpfs Note] Could not load tissue_mask: {me}")

    # Fetch confirmed mitoses
    confirmed_dets = db.scalars(
        select(Detection).where(
            (Detection.case_id == case_uid) | (Detection.case_id == str(case_id)),
            Detection.label == "mitosis"
        )
    ).all()

    hotspot_rows = db.scalars(
        select(Hotspot).where(
            (Hotspot.case_id == case_uid) | (Hotspot.case_id == str(case_id)),
            Hotspot.excluded == False
        )
    ).all()
    hotspot_rows_sorted = sorted(hotspot_rows, key=lambda h: (h.prob_mean or 0.0), reverse=True)
    hotspot_polys = [h.polygon_um for h in hotspot_rows_sorted]
    hotspot_prios = [float(h.prob_mean or 0.0) for h in hotspot_rows_sorted]

    cands = [{"id": d.id, "centroid_um": d.centroid_um, "label": "mitosis"} for d in confirmed_dets]

    if confirmed_dets:
        xs = [d.centroid_um[0] for d in confirmed_dets if d.centroid_um]
        ys = [d.centroid_um[1] for d in confirmed_dets if d.centroid_um]
        bbox = (min(xs), min(ys), max(xs), max(ys))
    elif hotspot_polys:
        all_x = []
        all_y = []
        for poly in hotspot_polys:
            if isinstance(poly, list):
                for pt in poly:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        all_x.append(pt[0])
                        all_y.append(pt[1])
        if all_x and all_y:
            bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        else:
            bbox = (0.0, 0.0, slide_dims_um[0] if slide_dims_um else 5000.0, slide_dims_um[1] if slide_dims_um else 5000.0)
    else:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot re-place HPFs: No confirmed mitoses and no tumor hotspots exist for this case."
        )

    density_map, grid_meta = generate_mitosis_density_map(cands, bounding_box_um=bbox)
    new_hpfs = greedy_place_hpfs(
        density_map,
        grid_meta,
        hotspot_polygons_um=hotspot_polys,
        count=10,
        tissue_mask=tissue_mask,
        slide_dimensions_um=slide_dims_um,
        min_tissue_coverage=0.70,
        hotspot_priorities=hotspot_prios
    )

    # Persist new HPFs
    db.execute(delete(HpfSite).where((HpfSite.case_id == case_uid) | (HpfSite.case_id == str(case_id))))
    for h in new_hpfs:
        hpf_row = HpfSite(
            case_id=case_uid,
            seq=h["seq"],
            center_um=h["center_um"],
            radius_um=h["radius_um"],
            mitotic_count=0,
            source="model"
        )
        db.add(hpf_row)

    db.flush()
    # Invalidate cached HPF thumbnails on GCS (#127)
    invalidate_hpf_cached_thumbnails(case_id)

    # Synchronize HpfSite mitotic counts immediately (#110)
    sync_and_persist_hpf_counts(case_id, db)
    db.commit()

    return get_mitosis_stage_data(case_id, db)


@router.post("/confirm")
def confirm_mitosis_stage(payload: MitosisConfirmPayload, db: Session = Depends(get_db)):
    """
    Clinical Safety Gate & Stage 4 Confirmation.
    Verifies that all candidate mitotic figures above threshold (conf >= 0.50) have been reviewed.
    Finalizes 10 HPFs and Nottingham Mitotic Score, marks Stage 4 as confirmed, snapshots metrics, and queues Stage 5.
    """
    case_id = payload.case_id
    case_obj, stage_exec = get_verified_mitosis_stage(case_id, db, require_awaiting=True)
    case_uid = case_obj.id

    # Check unreviewed high-confidence candidates
    unreviewed_high_conf = db.scalars(
        select(Detection).where(
            (Detection.case_id == case_uid) | (Detection.case_id == str(case_id)),
            Detection.label == "unreviewed",
            (Detection.det_conf >= 0.50) | (Detection.ver_conf >= 0.50)
        )
    ).all()

    if unreviewed_high_conf:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Clinical Safety Gate: {len(unreviewed_high_conf)} unreviewed candidate mitotic figure(s) with confidence >= 0.50 remain. Please review or use 'Bulk Reject' before confirming."
        )

    stage_exec.status = "confirmed"
    stage_exec.reviewed_at = datetime.now(timezone.utc)
    stage_exec.reviewed_by = payload.reviewed_by

    # Queue Stage 5 (grading) - Guarded against overwriting signed reports
    from app.models.report import Report
    try:
        report_status = db.scalar(
            select(Report.status).where(Report.case_id == case_uid)
        )
        is_report_signed = report_status in ("signed", "amended") if report_status else False
    except Exception:
        db.rollback()
        is_report_signed = False

    next_exec = db.scalars(
        select(StageExecution).where(
            (StageExecution.case_id == case_uid) | (StageExecution.case_id == str(case_id)),
            StageExecution.stage == "grading"
        ).order_by(StageExecution.attempt.desc())
    ).first()

    if not is_report_signed:
        if not next_exec:
            next_exec = StageExecution(
                case_id=case_uid,
                stage="grading",
                attempt=1,
                status="queued"
            )
            db.add(next_exec)
        elif next_exec.status not in ("confirmed", "done"):
            next_exec.status = "queued"
            next_exec.started_at = None
            next_exec.completed_at = None
            next_exec.error = None

    # Synchronize confirmed detections & HPFs back to GCS output.json and snapshot metrics (#118)
    total_m = 0
    scoring_summary = {}
    try:
        from pipeline.grading import calculate_mitotic_score_from_detections_and_hpfs
        all_dets = db.scalars(select(Detection).where((Detection.case_id == case_uid) | (Detection.case_id == str(case_id)))).all()
        hpf_rows = db.scalars(select(HpfSite).where((HpfSite.case_id == case_uid) | (HpfSite.case_id == str(case_id))).order_by(HpfSite.seq.asc())).all()
        cand_dicts = [
            {
                "id": d.id,
                "centroid_um": d.centroid_um,
                "label": d.label,
                "label_source": d.label_source,
                "det_conf": d.det_conf,
                "ver_conf": d.ver_conf,
                "hotspot_id": d.hotspot_id,
                "crop_uri": d.crop_uri,
                "crop_orig_uri": d.crop_orig_uri
            }
            for d in all_dets
        ]
        hpf_dicts = [
            {"seq": h.seq, "center_um": h.center_um, "radius_um": h.radius_um, "count": h.mitotic_count, "source": h.source}
            for h in hpf_rows
        ]
        total_m, conf_score = calculate_mitotic_score_from_detections_and_hpfs(cand_dicts, hpf_dicts)
        r_um = hpf_dicts[0]["radius_um"] if hpf_dicts else 262.0
        scoring_summary = compute_nottingham_mitotic_score(
            count_total=total_m,
            n_hpf=len(hpf_dicts) or 10,
            radius_um=r_um,
            hpfs=hpf_dicts
        )

        stage_exec.metrics = {
            "count_total": total_m,
            "area_mm2": scoring_summary.get("area_mm2"),
            "mitoses_per_mm2": scoring_summary.get("mitoses_per_mm2"),
            "mitotic_score": scoring_summary.get("score")
        }

        existing_out = {}
        try:
            raw_out = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/mitosis/output.json")
            existing_out = json.loads(raw_out.decode("utf-8"))
        except Exception:
            pass
        existing_out["case_id"] = str(case_id)
        existing_out["candidates"] = cand_dicts
        existing_out["hpfs"] = hpf_dicts
        existing_out["summary"] = scoring_summary
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            f"cases/{case_id}/mitosis/output.json",
            json.dumps(existing_out, indent=2).encode("utf-8"),
            "application/json"
        )
    except Exception as ge:
        print(f"[Confirm Mitosis GCS Sync Note] {ge}")

    audit = AuditEvent(
        case_id=case_id,
        actor=payload.reviewed_by,
        event_type="stage_confirmed",
        stage="mitosis",
        payload={
            "next_stage": "grading",
            "mitotic_score": scoring_summary.get("score"),
            "count_total": total_m
        }
    )
    db.add(audit)
    db.commit()

    try:
        from app.core.cloud_tasks import dispatch_stage_task
        if next_exec:
            dispatch_stage_task(
                case_id=str(case_id),
                stage="grading",
                stage_exec_id=str(next_exec.id)
            )
    except Exception as e:
        print(f"[CloudTasks Warning] Failed to dispatch next stage grading: {e}")

    return {
        "status": "success",
        "case_id": case_id,
        "stage": "mitosis",
        "next_stage": "grading"
    }
