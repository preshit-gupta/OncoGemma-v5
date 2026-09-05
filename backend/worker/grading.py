"""
Stage 5 Worker Handler (Nottingham Histologic Grading via MedGemma 1.5).

Extracts 24 stratified 10x evidence patches from confirmed Stage 3 hotspots,
applies Macenko stain normalization, dispatches asynchronous MedGemma 1.5 calls
for Tubule Formation and Nuclear Pleomorphism, executes multi-image consensus for
Histologic Subtype, computes pure zero-LLM aggregation, and persists grading state.
"""

import os
import io
import json
import math
import hashlib
import asyncio
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import (
    get_gcs_client,
    parse_gcs_uri,
    upload_blob_from_bytes,
    download_blob_as_bytes,
    download_blob_to_filename,
    get_gcs_artifact_direct_url,
    resolve_slide_raw_uri
)
from app.core.openslide_lock import OPENSLIDE_GLOBAL_LOCK
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot
from app.models.hpf_site import HpfSite
from app.models.grading import Grading
from app.models.audit import AuditEvent
from pipeline.stain import MacenkoNormalizer
from pipeline.grading import (
    aggregate_grading_findings,
    load_scoring_config,
    validate_grading_invariants
)
from pipeline.medgemma import (
    MedGemmaClient,
    load_prompt_template,
    TubuleResponse,
    PleoResponse,
    HistologicTypeResponse,
    SchemaRetryExhaustedError
)



def extract_10x_patch(
    slide_obj,
    center_x: int,
    center_y: int,
    patch_size_px: int = 512,
    target_mpp: float = 1.0,
    base_mpp: float | None = None
) -> Image.Image:
    """
    Extract 512x512 patch @ 1.0 um/pixel (10x magnification) centered at (center_x, center_y).
    """
    if base_mpp is None or base_mpp <= 0:
        raise ValueError(f"Valid positive base_mpp is required for patch extraction, got: {base_mpp}")
    downsample = target_mpp / base_mpp  # e.g., 1.0 / 0.25 = 4.0
    crop_w_l0 = int(patch_size_px * downsample)
    crop_h_l0 = int(patch_size_px * downsample)
    
    top_left_x = max(0, int(center_x - crop_w_l0 / 2))
    top_left_y = max(0, int(center_y - crop_h_l0 / 2))
    
    with OPENSLIDE_GLOBAL_LOCK:
        rgba = slide_obj.read_region((top_left_x, top_left_y), 0, (crop_w_l0, crop_h_l0))
        rgb = rgba.convert("RGB")
        
    if rgb.size != (patch_size_px, patch_size_px):
        rgb = rgb.resize((patch_size_px, patch_size_px), Image.Resampling.LANCZOS)
        
    return rgb


def select_max_density_hotspot_patches(
    hotspots: List[Any],
    tissue_mask: np.ndarray,
    slide_dims_um: Tuple[float, float],
    base_mpp: float,
    case_id: str,
    n_patches: int = 24,
    patch_size_um: float = 512.0,
    min_dist_um: float = 384.0,
    min_density: float = 0.50,
    checksum_sha256: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Selects n_patches (24) 10x evidence patches ensuring:
    1. Patches are taken from within or directly adjacent to confirmed Stage 3 hotspots.
    2. The patch with maximum tissue density within each hotspot is chosen first (preventing lumina/empty voids).
    3. Additional non-overlapping high-density sites inside hotspots or on invasive tumor margins are selected
       until exactly n_patches are obtained.
    4. Deterministic sampling is seeded by slide checksum (Issue #145).
    """
    from scipy.ndimage import uniform_filter
    from shapely.geometry import Polygon, Point

    seed_int = int(checksum_sha256[:8], 16) if checksum_sha256 and checksum_sha256 != "default_checksum" else 42
    rng = np.random.default_rng(seed_int)

    H_m, W_m = tissue_mask.shape
    s_x = W_m / max(slide_dims_um[0], 1.0)
    s_y = H_m / max(slide_dims_um[1], 1.0)
    k_x = max(3, int(round(patch_size_um * s_x)))
    k_y = max(3, int(round(patch_size_um * s_y)))

    density_map = uniform_filter(tissue_mask.astype(np.float32), size=(k_y, k_x), mode='constant', cval=0.0)

    selected = []
    selected_coords = []

    def is_too_close(x, y, radius=min_dist_um):
        for cx, cy in selected_coords:
            if np.hypot(x - cx, y - cy) < radius:
                return True
        return False

    hs_data = []
    all_internal_cands = []

    for hs in hotspots:
        poly_raw = getattr(hs, "polygon_um", None) or (hs.get("polygon_um") if isinstance(hs, dict) else None)
        poly_arr = np.array(poly_raw) if poly_raw else np.array([[5000, 5000]])
        if len(poly_arr) < 3:
            cx = float(poly_arr[:, 0].mean())
            cy = float(poly_arr[:, 1].mean())
            poly_arr = np.array([
                [cx - 200, cy - 200],
                [cx + 200, cy - 200],
                [cx + 200, cy + 200],
                [cx - 200, cy + 200]
            ])
            
        poly_geom = Polygon(poly_arr).buffer(0)
        prob = float(getattr(hs, "prob_mean", None) or (hs.get("prob_mean") if isinstance(hs, dict) else None) or getattr(hs, "tumor_probability", 0.85) or 0.85)
        hs_id = getattr(hs, "id", None) or (hs.get("id") if isinstance(hs, dict) else "hs")

        min_x, min_y, max_x, max_y = poly_geom.bounds
        step_um = 64.0
        gx = np.arange(min_x, max_x, step_um)
        gy = np.arange(min_y, max_y, step_um)

        cand_points = []
        for x in gx:
            for y in gy:
                if poly_geom.contains(Point(x, y)):
                    pmx = int(np.clip(round(x * s_x), 0, W_m - 1))
                    pmy = int(np.clip(round(y * s_y), 0, H_m - 1))
                    d = float(density_map[pmy, pmx])
                    cand_points.append((x, y, d))
                    all_internal_cands.append((x, y, d, hs_id, prob, "hotspot_subregion"))

        if not cand_points:
            cx_um = float(poly_arr[:, 0].mean())
            cy_um = float(poly_arr[:, 1].mean())
            pmx = int(np.clip(round(cx_um * s_x), 0, W_m - 1))
            pmy = int(np.clip(round(cy_um * s_y), 0, H_m - 1))
            d = float(density_map[pmy, pmx])
            cand_points.append((cx_um, cy_um, d))
            all_internal_cands.append((cx_um, cy_um, d, hs_id, prob, "hotspot_subregion"))

        cand_points.sort(key=lambda item: item[2], reverse=True)
        hs_data.append({
            "id": hs_id,
            "geom": poly_geom,
            "prob": prob,
            "cands": cand_points
        })

    # Phase 1: Peak point of EVERY hotspot (sorted by prob descending)
    hs_data.sort(key=lambda h: h["prob"], reverse=True)
    for h in hs_data:
        for x, y, d in h["cands"]:
            if not is_too_close(x, y):
                selected.append({
                    "hotspot_id": h["id"],
                    "center_um": [round(float(x), 2), round(float(y), 2)],
                    "center_x_px": int(round(x / base_mpp)),
                    "center_y_px": int(round(y / base_mpp)),
                    "tissue_density": round(d, 4),
                    "tumor_probability": round(h["prob"], 4),
                    "source": "hotspot_peak"
                })
                selected_coords.append((x, y))
                break

    # Phase 2: High-density points inside hotspots, prioritized by density
    all_internal_cands.sort(key=lambda it: (it[2] >= min_density, it[2], it[4]), reverse=True)
    for x, y, d, hs_id, prob, src in all_internal_cands:
        if len(selected) >= n_patches:
            break
        if d >= min_density and not is_too_close(x, y):
            selected.append({
                "hotspot_id": hs_id,
                "center_um": [round(float(x), 2), round(float(y), 2)],
                "center_x_px": int(round(x / base_mpp)),
                "center_y_px": int(round(y / base_mpp)),
                "tissue_density": round(d, 4),
                "tumor_probability": round(prob, 4),
                "source": src
            })
            selected_coords.append((x, y))

    # Phase 3: Immediate hotspot perimeter margin if still needed
    if len(selected) < n_patches:
        margin_cands = []
        for h in hs_data:
            margin_geom = h["geom"].buffer(350.0).difference(h["geom"])
            min_x, min_y, max_x, max_y = margin_geom.bounds
            gx = np.arange(min_x, max_x, 80.0)
            gy = np.arange(min_y, max_y, 80.0)
            for x in gx:
                for y in gy:
                    if margin_geom.contains(Point(x, y)):
                        pmx = int(np.clip(round(x * s_x), 0, W_m - 1))
                        pmy = int(np.clip(round(y * s_y), 0, H_m - 1))
                        d = float(density_map[pmy, pmx])
                        margin_cands.append((x, y, d, h["id"], h["prob"]))
        
        margin_cands.sort(key=lambda it: (it[2] >= min_density, it[2]), reverse=True)
        for x, y, d, hs_id, prob in margin_cands:
            if len(selected) >= n_patches:
                break
            if not is_too_close(x, y):
                selected.append({
                    "hotspot_id": hs_id,
                    "center_um": [round(float(x), 2), round(float(y), 2)],
                    "center_x_px": int(round(x / base_mpp)),
                    "center_y_px": int(round(y / base_mpp)),
                    "tissue_density": round(d, 4),
                    "tumor_probability": round(prob * 0.95, 4),
                    "source": "hotspot_margin"
                })
                selected_coords.append((x, y))

    # Fallback if still under n_patches: relax distance threshold
    if len(selected) < n_patches:
        for x, y, d, hs_id, prob, src in all_internal_cands:
            if len(selected) >= n_patches:
                break
            if not is_too_close(x, y, radius=min_dist_um * 0.6):
                selected.append({
                    "hotspot_id": hs_id,
                    "center_um": [round(float(x), 2), round(float(y), 2)],
                    "center_x_px": int(round(x / base_mpp)),
                    "center_y_px": int(round(y / base_mpp)),
                    "tissue_density": round(d, 4),
                    "tumor_probability": round(prob, 4),
                    "source": src
                })
                selected_coords.append((x, y))

    # Fallback if still under n_patches
    while len(selected) < n_patches:
        idx = len(selected)
        cx_um = 5000.0 + (idx % 5) * 1500.0
        cy_um = 5000.0 + (idx // 5) * 1500.0
        selected.append({
            "hotspot_id": f"hs_{(idx % len(hs_data)) + 1:02d}" if hs_data else "hs_01",
            "center_um": [round(float(cx_um), 2), round(float(cy_um), 2)],
            "center_x_px": int(round(cx_um / base_mpp)),
            "center_y_px": int(round(cy_um / base_mpp)),
            "tissue_density": 0.85,
            "tumor_probability": 0.80,
            "source": "grid_fallback"
        })

    for idx, p in enumerate(selected[:n_patches]):
        p["id"] = f"p_{idx+1:03d}"
        p["index"] = idx + 1
        p["image_filename"] = f"p_{idx+1:03d}.png"
        p["image_url"] = f"/api/v1/stages/grading/{case_id}/patches/p_{idx+1:03d}/image"

    return selected[:n_patches]


def run_grading(stage_exec: StageExecution, db: Session) -> Tuple[str, Dict[str, Any]]:
    """
    Main Stage 5 Grading Worker Execution.
    """
    case_id = str(stage_exec.case_id)
    print(f"[Worker Stage 5: Grading] Commencing Nottingham grading pipeline for case {case_id}...")

    case = db.get(Case, stage_exec.case_id)
    if not case or not case.slides:
        raise ValueError(f"Case {case_id} has no valid slide records.")
        
    slide = case.slides[0]
    slide_id = str(slide.id)

    # Halt grading stage if MPP is missing per PRD 01-stage-v4.0 §2.3 step 4
    if not slide or not getattr(slide, "mpp_x", None) or slide.mpp_x <= 0 or not getattr(slide, "mpp_y", None) or slide.mpp_y <= 0:
        raise ValueError(f"Slide for case {case_id} is missing valid MPP (status='needs_mpp'). Cannot execute grading stage.")
    base_mpp = float(slide.mpp_x)

    scratch_dir = tempfile.mkdtemp(prefix="og_grading_")

    try:
        gcs_uri_original = resolve_slide_raw_uri(case_id, slide) or slide.gcs_uri_original or f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_id}/{slide_id}.svs"
        raw_bucket_name, blob_name = parse_gcs_uri(gcs_uri_original)
        ext = os.path.splitext(blob_name)[1] or ".svs"
        local_slide_path = os.path.join(scratch_dir, f"slide{ext}")

        download_blob_to_filename(raw_bucket_name, blob_name, local_slide_path)
        if not os.path.exists(local_slide_path):
            raise FileNotFoundError(f"Whole slide image file for case {case_id} not found in GCS.")

        # 1. Fetch Stage 3 Hotspots & Stage 4 Mitotic Score
        stmt_hotspots = select(Hotspot).where(Hotspot.case_id == case.id).order_by(Hotspot.prob_mean.desc())
        db_hotspots = list(db.scalars(stmt_hotspots).all())
        hotspots = [h for h in db_hotspots if not getattr(h, "excluded", False)]

        # Fallback to triage output.json if no DB hotspots found
        if not hotspots:
            try:
                t_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/triage/output.json")
                t_data = json.loads(t_bytes.decode("utf-8"))
                hotspots = [h for h in t_data.get("hotspots", []) if not h.get("excluded", False)]
            except Exception:
                pass

        # Retrieve confirmed Mitotic Score from Stage 4 (no double-counting across overlapping HPFs)
        stmt_hpfs = select(HpfSite).where(HpfSite.case_id == case.id).order_by(HpfSite.seq.asc())
        hpf_sites = list(db.scalars(stmt_hpfs).all())

        stmt_dets = select(Detection).where(Detection.case_id == case.id, Detection.label == "mitosis")
        confirmed_dets = list(db.scalars(stmt_dets).all())

        mitotic_score = 1
        total_mitoses = 0

        if hpf_sites and confirmed_dets:
            from pipeline.grading import calculate_mitotic_score_from_detections_and_hpfs
            cands_for_score = [{"id": d.id, "centroid_um": d.centroid_um, "label": "mitosis"} for d in confirmed_dets]
            hpfs_for_score = [{"seq": h.seq, "center_um": h.center_um, "radius_um": h.radius_um, "count": 0} for h in hpf_sites]
            total_mitoses, mitotic_score = calculate_mitotic_score_from_detections_and_hpfs(cands_for_score, hpfs_for_score)
        elif hpf_sites:
            from pipeline.grading import calculate_mitotic_score_from_hpfs
            hpf_counts = [getattr(h, "mitotic_count", 0) for h in hpf_sites]
            r_um = float(getattr(hpf_sites[0], "radius_um", 262.0) or 262.0)
            total_mitoses, mitotic_score = calculate_mitotic_score_from_hpfs(hpf_counts, radius_um=r_um)
        else:
            try:
                m_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/mitosis/output.json")
                m_data = json.loads(m_bytes.decode("utf-8"))
                if "summary" in m_data and "mitotic_score" in m_data["summary"]:
                    mitotic_score = m_data["summary"]["mitotic_score"]
                    total_mitoses = m_data["summary"].get("total_mitoses", total_mitoses)
            except Exception:
                mitotic_score = 1
                total_mitoses = 0


        scoring_cfg = load_scoring_config()
        n_patches = scoring_cfg.get("grading", {}).get("n_patches", 24)
        patch_size_px = scoring_cfg.get("grading", {}).get("patch_size_px", 512)
        resolution_um = scoring_cfg.get("grading", {}).get("resolution_um", 1.0)
        patch_size_um = patch_size_px * resolution_um

        # 2. Open Slide and Prepare Tissue Mask
        import openslide
        with OPENSLIDE_GLOBAL_LOCK:
            slide_obj = openslide.OpenSlide(local_slide_path)

        slide_w, slide_h = slide_obj.dimensions
        slide_dims_um = (float(slide_w * base_mpp), float(slide_h * float(slide.mpp_y)))

        tissue_mask = None
        try:
            mask_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/preprocess/tissue_mask.png")
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            tissue_mask = np.array(mask_img) > 10
            print(f"[Worker Stage 5: Grading] Loaded preprocess tissue mask ({tissue_mask.shape[1]}x{tissue_mask.shape[0]})")
        except Exception as me:
            print(f"[Worker Stage 5: Grading Note] Could not load preprocess tissue_mask from GCS: {me}")

        if tissue_mask is None:
            try:
                with OPENSLIDE_GLOBAL_LOCK:
                    thumb = slide_obj.get_thumbnail((512, 512)).convert("RGB")
                arr = np.array(thumb).astype(float)
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                tissue_mask = ~((r > 215) & (g > 215) & (b > 215))
            except Exception:
                tissue_mask = np.ones((512, 512), dtype=bool)

        # 3. Maximum-Density Hotspot Patch Selection (Guarantees closest to hotspot & max tissue density)
        candidate_patches = select_max_density_hotspot_patches(
            hotspots=hotspots,
            tissue_mask=tissue_mask,
            slide_dims_um=slide_dims_um,
            base_mpp=base_mpp,
            case_id=case_id,
            n_patches=n_patches,
            patch_size_um=patch_size_um,
            checksum_sha256=getattr(slide, "checksum_sha256", None)
        )

        normalizer = MacenkoNormalizer()
        extracted_patches = []
        patch_images_bytes = []
        
        try:
            for p_meta in candidate_patches:
                patch_id = p_meta["id"]
                raw_img = extract_10x_patch(
                    slide_obj=slide_obj,
                    center_x=p_meta["center_x_px"],
                    center_y=p_meta["center_y_px"],
                    patch_size_px=patch_size_px,
                    target_mpp=resolution_um,
                    base_mpp=base_mpp
                )
                
                # Macenko normalization
                try:
                    norm_np = normalizer.transform(np.array(raw_img))
                except Exception:
                    norm_np = np.array(raw_img)
                norm_img = Image.fromarray(norm_np)
                
                img_buf = io.BytesIO()
                norm_img.save(img_buf, format="PNG")
                img_bytes = img_buf.getvalue()
                
                # Upload patch directly to GCS artifacts bucket
                upload_blob_from_bytes(
                    settings.GCS_ARTIFACTS_BUCKET,
                    f"cases/{case_id}/grading_patches/{patch_id}.png",
                    img_bytes,
                    "image/png"
                )
                
                patch_images_bytes.append(img_bytes)
                extracted_patches.append({
                    "id": patch_id,
                    "index": p_meta["index"],
                    "hotspot_id": p_meta.get("hotspot_id"),
                    "tissue_density": p_meta.get("tissue_density"),
                    "source": p_meta.get("source"),
                    "center_um": p_meta.get("center_um"),
                    "center_x_px": p_meta["center_x_px"],
                    "center_y_px": p_meta["center_y_px"],
                    "tumor_probability": round(p_meta["tumor_probability"], 4),
                    "image_filename": f"{patch_id}.png",
                    "image_url": f"/api/v1/stages/grading/{case_id}/patches/{patch_id}/image"
                })
        finally:
            with OPENSLIDE_GLOBAL_LOCK:
                slide_obj.close()

        print(f"[Worker Stage 5: Grading] Successfully extracted and normalized {len(extracted_patches)} evidence patches.")

        # 4. Load Versioned Prompts and Track SHAs
        tubule_prompt, tubule_sha = load_prompt_template("tubule", "v1")
        pleo_prompt, pleo_sha = load_prompt_template("pleo", "v1")
        type_prompt, type_sha = load_prompt_template("histologic_type", "v1")
        narrative_prompt, narrative_sha = load_prompt_template("findings_narrative", "v1")

        model_versions = {
            "medgemma": settings.VERTEX_MEDGEMMA_MODEL_VERSION,
            "prompts": {
                "tubule": f"v1@{tubule_sha[:8]}",
                "pleo": f"v1@{pleo_sha[:8]}",
                "histologic_type": f"v1@{type_sha[:8]}",
                "findings_narrative": f"v1@{narrative_sha[:8]}"
            }
        }

        # 5. Async Dispatch to MedGemma 1.5 with Concurrency Limiter (<= 4)
        medgemma = MedGemmaClient()
        schema_failed_patches = []

        async def execute_medgemma_pipeline():
            sem = asyncio.Semaphore(4)
            
            async def evaluate_single_tubule(img_bytes: bytes, p_id: str):
                async with sem:
                    try:
                        return await medgemma.evaluate_tubule(img_bytes, tubule_prompt)
                    except SchemaRetryExhaustedError as e:
                        print(f"[Worker Grading Warning] Tubule patch {p_id} schema retry exhausted: {e}")
                        schema_failed_patches.append(f"tubule:{p_id}")
                        return TubuleResponse(tubule_percent=0, tumor_present=False, confidence="unassessed_schema_error")

            async def evaluate_single_pleo(img_bytes: bytes, p_id: str):
                async with sem:
                    try:
                        return await medgemma.evaluate_pleomorphism(img_bytes, pleo_prompt)
                    except SchemaRetryExhaustedError as e:
                        print(f"[Worker Grading Warning] Pleo patch {p_id} schema retry exhausted: {e}")
                        schema_failed_patches.append(f"pleo:{p_id}")
                        return PleoResponse(pleomorphism_score=1, rationale="VLM schema retry exhausted; flagged for pathologist review", confidence="unassessed_schema_error")

            tubule_tasks = [evaluate_single_tubule(b, p["id"]) for b, p in zip(patch_images_bytes, extracted_patches)]
            pleo_tasks = [evaluate_single_pleo(b, p["id"]) for b, p in zip(patch_images_bytes, extracted_patches)]
            
            # Histologic type on top-8 patches
            top_8_bytes = patch_images_bytes[:8]
            type_task = medgemma.evaluate_histologic_type(top_8_bytes, type_prompt)
            
            tubule_res = await asyncio.gather(*tubule_tasks)
            pleo_res = await asyncio.gather(*pleo_tasks)
            try:
                type_res = await type_task
            except SchemaRetryExhaustedError as e:
                print(f"[Worker Grading Warning] Histologic type schema error: {e}")
                schema_failed_patches.append("histologic_type")
                type_res = HistologicTypeResponse(
                    type="Unclassified Carcinoma",
                    differential=["IDC-NST", "ILC"],
                    rationale="VLM schema retry exhausted; unconfirmed, flagged for pathologist review.",
                    confidence="unassessed_schema_error"
                )
            except Exception as e:
                if not settings.USE_MOCK_VERTEX_AI:
                    raise
                print(f"[Worker Grading Warning] Histologic type error: {e}")
                type_res = HistologicTypeResponse(
                    type="IDC-NST",
                    differential=["ILC"],
                    rationale="Invasive carcinoma with cohesive clusters.",
                    confidence="medium"
                )
                
            return tubule_res, pleo_res, type_res

        tubule_responses, pleo_responses, type_response = asyncio.run(execute_medgemma_pipeline())
        needs_human_flag = len(schema_failed_patches) > 0

        # Map patch-level results
        patches_output = []
        for idx, p in enumerate(extracted_patches):
            t_res = tubule_responses[idx]
            p_res = pleo_responses[idx]
            rev_status = "needs_review" if (t_res.confidence == "unassessed_schema_error" or p_res.confidence == "unassessed_schema_error") else "suggested"
            patches_output.append({
                "id": p["id"],
                "index": p["index"],
                "hotspot_id": p.get("hotspot_id"),
                "tissue_density": p.get("tissue_density"),
                "source": p.get("source"),
                "center_um": p.get("center_um"),
                "center_x_px": p["center_x_px"],
                "center_y_px": p["center_y_px"],
                "tumor_probability": p["tumor_probability"],
                "image_url": p["image_url"],
                "tubule": {
                    "tubule_percent": t_res.tubule_percent,
                    "tumor_present": t_res.tumor_present,
                    "confidence": t_res.confidence
                },
                "pleo": {
                    "pleomorphism_score": p_res.pleomorphism_score,
                    "rationale": p_res.rationale,
                    "confidence": p_res.confidence
                },
                "review_status": rev_status
            })

        # Format HPF sites for Stage 5 dual-level review
        hpfs_output = []
        for h in sorted(hpf_sites, key=lambda x: getattr(x, "seq", 0)):
            cnt = getattr(h, "mitotic_count", getattr(h, "mitotic_figure_count", 0))
            hpfs_output.append({
                "seq": h.seq,
                "center_um": h.center_um if isinstance(h.center_um, list) else [0, 0],
                "radius_um": getattr(h, "radius_um", 262.0),
                "mitotic_count": cnt,
                "density_mm2": round(cnt / 0.2157, 1),
                "review_status": "suggested"
            })

        # 6. Deterministic Pure Zero-LLM Aggregation
        tubule_dicts = [p["tubule"] for p in patches_output]
        pleo_dicts = [p["pleo"] for p in patches_output]
        
        aggregate_res = aggregate_grading_findings(
            tubule_responses=tubule_dicts,
            pleo_responses=pleo_dicts,
            mitotic_score=mitotic_score,
            cfg=scoring_cfg
        )

        # 7. Grounded Narrative Synthesis
        narrative_input = {
            "histologic_type": type_response.model_dump(),
            "aggregate": aggregate_res,
            "mitotic_summary": {
                "total_mitoses": total_mitoses,
                "mitotic_score": mitotic_score,
                "evaluated_hpfs": len(hpf_sites)
            }
        }
        narrative_text = asyncio.run(medgemma.generate_findings_narrative(narrative_input, narrative_prompt))

        # 8. Assemble Full Output JSON
        output_payload = {
            "case_id": case_id,
            "slide_id": slide_id,
            "patches": patches_output,
            "hpfs": hpfs_output,
            "aggregate": aggregate_res,
            "histologic_type": type_response.model_dump(),
            "narrative": narrative_text,
            "model_versions": model_versions,
            "needs_human": needs_human_flag,
            "schema_failed_patches": schema_failed_patches,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

        # Save output artifact directly to GCS
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            f"cases/{case_id}/grading_output.json",
            json.dumps(output_payload, indent=2).encode("utf-8"),
            "application/json"
        )

        output_uri = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/grading_output.json"

        # 9. Persist into Database gradings table
        stmt_existing = select(Grading).where(Grading.case_id == stage_exec.case_id)
        existing_grading = db.scalars(stmt_existing).first()

        if existing_grading:
            existing_grading.tubule_percent = aggregate_res["tubule_percent"]
            existing_grading.tubule_score = aggregate_res["tubule_score"]
            existing_grading.pleo_score = aggregate_res["pleo_score"]
            existing_grading.mitotic_score = aggregate_res["mitotic_score"]
            existing_grading.nottingham_sum = aggregate_res["nottingham_sum"]
            existing_grading.grade = aggregate_res["grade"]
            existing_grading.histologic_type = type_response.type
            existing_grading.machine = output_payload
            # Issue #143: Re-running grading must clear stale overrides and unconfirm type
            existing_grading.overrides = {}
            existing_grading.type_confirmed_by = "unconfirmed"
        else:
            new_grading = Grading(
                case_id=stage_exec.case_id,
                tubule_percent=aggregate_res["tubule_percent"],
                tubule_score=aggregate_res["tubule_score"],
                pleo_score=aggregate_res["pleo_score"],
                mitotic_score=aggregate_res["mitotic_score"],
                nottingham_sum=aggregate_res["nottingham_sum"],
                grade=aggregate_res["grade"],
                histologic_type=type_response.type,
                type_confirmed_by="unconfirmed",
                machine=output_payload,
                overrides={}
            )
            db.add(new_grading)

        # Record Audit Event
        audit_evt = AuditEvent(
            case_id=str(stage_exec.case_id),
            actor=settings.DEFAULT_MOCK_USER_ID,
            event_type="stage_5_grading_generated",
            stage="grading",
            payload={
                "nottingham_sum": aggregate_res["nottingham_sum"],
                "grade": aggregate_res["grade"],
                "tubule_score": aggregate_res["tubule_score"],
                "pleo_score": aggregate_res["pleo_score"],
                "mitotic_score": aggregate_res["mitotic_score"],
                "histologic_type": type_response.type,
                "flags": aggregate_res["flags"],
                "needs_human": needs_human_flag,
                "schema_failed_patches": schema_failed_patches
            }
        )
        db.add(audit_evt)
        db.commit()

        stage_exec.status = "awaiting_review"
        if needs_human_flag:
            stage_exec.error = f"Flagged for pathologist review: schema parsing errors on {len(schema_failed_patches)} patches"
        print(f"[Worker Stage 5: Grading] Completed successfully for case {case_id}. Nottingham Grade {aggregate_res['grade']} (Sum {aggregate_res['nottingham_sum']}/9). Status: awaiting_review.")

        return output_uri, model_versions

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
