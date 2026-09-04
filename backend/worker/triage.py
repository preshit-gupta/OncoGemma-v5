"""
Triage stage worker handler (v4.2 Hotspot Triage).
Extracts 10x patches, retrieves Path Foundation embeddings (with Parquet caching),
runs linear probe, extracts hotspot ROIs, and renders viridis heatmap overlay.
"""
import os
import io
import json
import asyncio
import time
import base64
import tempfile
import shutil
import yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any
import matplotlib
import matplotlib.cm as cm
from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import (
    get_gcs_client,
    parse_gcs_uri,
    upload_blob_from_bytes,
    upload_blob_from_filename,
    download_blob_as_bytes,
    download_blob_as_text,
    download_blob_to_filename,
    get_gcs_artifact_direct_url,
    resolve_slide_raw_uri
)
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.audit import AuditEvent
from pipeline.hotspots import extract_hotspots
from pipeline.probe import ProbeRunner, train_default_probe


class VertexPathFoundationClient:
    """
    Client for Google Cloud Vertex AI Path Foundation Online Prediction Endpoint.
    Requests batched 384-dimensional feature embeddings for 224x224 patch images.
    """
    def __init__(
        self,
        endpoint_id: str,
        location: str = "asia-east1",
        project_id: str = "oncogemma",
        api_endpoint: str | None = None
    ):
        self.endpoint_id = endpoint_id
        self.location = location
        self.project_id = project_id
        self.api_endpoint = api_endpoint

    def predict_embeddings(
        self,
        patch_count: int = 0,
        patches: list[Image.Image] | None = None,
        batch_size: int = 16
    ) -> np.ndarray:
        if patches is not None:
            patch_count = len(patches)

        if settings.USE_MOCK_VERTEX_AI:
            rng = np.random.RandomState(42)
            return rng.randn(patch_count, 384).astype(np.float32)

        if not patches or len(patches) == 0:
            raise ValueError("Real patches are required when calling Vertex AI Path Foundation endpoint. Flat dummy images are prohibited.")

        try:
            from google.cloud import aiplatform
            aiplatform.init(
                project=self.project_id,
                location=self.location
            )

            endpoint = aiplatform.Endpoint(
                endpoint_name=self.endpoint_id,
                project=self.project_id,
                location=self.location
            )

            all_embeddings = []
            for i in range(0, patch_count, batch_size):
                chunk_len = min(batch_size, patch_count - i)
                instances = []
                for j in range(chunk_len):
                    if (i + j) < len(patches):
                        p_img = patches[i + j].convert("RGB").resize((224, 224), Image.BILINEAR)
                        buf = io.BytesIO()
                        p_img.save(buf, format="PNG")
                        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                    else:
                        raise ValueError("Patch count exceeds available patches; dummy padding prohibited.")

                    instances.append({
                        "raw_image_bytes": b64_str,
                        "patch_coordinates": [{"x_origin": 0, "y_origin": 0, "width": 224, "height": 224}]
                    })

                payload = {"instances": instances}
                body = json.dumps(payload).encode("utf-8")
                headers = {"Content-Type": "application/json"}
                
                resp = endpoint.raw_predict(body=body, headers=headers)
                resp_json = resp.json()
                predictions = resp_json.get("predictions", [])
                
                chunk_embs = []
                for p in predictions:
                    patch_list = p.get("result", {}).get("patch_embeddings", [])
                    for pe in patch_list:
                        emb = pe.get("embedding_vector")
                        if emb:
                            chunk_embs.append(emb)
                
                if not chunk_embs:
                    raise RuntimeError(f"Vertex AI Path Foundation returned empty embeddings: {resp_json}")
                
                all_embeddings.append(np.array(chunk_embs, dtype=np.float32))

            return np.vstack(all_embeddings)
        except Exception as e:
            if settings.USE_MOCK_VERTEX_AI:
                print(f"[Vertex AI Path Foundation Note] Endpoint error ({e}). Falling back to deterministic embeddings.")
                rng = np.random.RandomState(42)
                return rng.randn(patch_count, 384).astype(np.float32)
            raise RuntimeError(f"Vertex AI Path Foundation prediction failed: {e}") from e


def load_config(config_dir: str = "configs") -> tuple[dict, dict]:
    base_config_dir = config_dir
    if not os.path.exists(os.path.join(base_config_dir, "triage.yaml")):
        alt = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))
        if os.path.exists(os.path.join(alt, "triage.yaml")):
            base_config_dir = alt

    triage_path = os.path.join(base_config_dir, "triage.yaml")
    pricing_path = os.path.join(base_config_dir, "pricing.yaml")

    with open(triage_path, "r", encoding="utf-8") as f:
        triage_cfg = yaml.safe_load(f)

    pricing_cfg = {}
    if os.path.exists(pricing_path):
        with open(pricing_path, "r", encoding="utf-8") as f:
            pricing_cfg = yaml.safe_load(f)

    return triage_cfg, pricing_cfg


async def mock_vertex_ai_endpoint(patches_count: int) -> np.ndarray:
    """
    Simulates Vertex AI Path Foundation endpoint returning (N, 384) float32 embeddings.
    Used during dev testing mode.
    """
    await asyncio.sleep(0.01)
    np.random.seed(123)
    return np.random.randn(patches_count, 384).astype(np.float32)


def render_viridis_heatmap_png(
    prob_grid: np.ndarray,
    output_path: str,
    scale: float = 1.0
) -> str:
    """
    Renders 2D probability grid as a full-spectrum Viridis color image with alpha channel for OSD overlay.
    """
    ny, nx = prob_grid.shape
    valid_mask = ~np.isnan(prob_grid)

    prob_norm = np.nan_to_num(prob_grid, nan=0.0)
    prob_norm = np.clip(prob_norm, 0.0, 1.0)

    try:
        colormap = matplotlib.colormaps["viridis"]
    except Exception:
        colormap = cm.get_cmap("viridis")

    rgba_mapped = colormap(prob_norm) # Shape (ny, nx, 4)

    # Set alpha channel: 0.0 for non-tissue (NaN), scaled alpha for tissue based on prob
    alpha = np.where(valid_mask, np.clip(0.35 + 0.55 * prob_norm, 0.25, 0.90), 0.0)
    rgba_mapped[..., 3] = alpha

    img_uint8 = (rgba_mapped * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode="RGBA")

    if scale != 1.0:
        new_w = max(1, int(nx * scale))
        new_h = max(1, int(ny * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, format="PNG")
    return output_path


def run_triage(stage_execution: StageExecution, session: Session) -> tuple[str, dict]:
    """
    Triage stage worker handler execution:
    1. Downloads preprocess mask & stain params from GCS.
    2. Downloads raw slide to transient temp file for high-res patch sampling.
    3. Runs Path Foundation embeddings & Linear Probe.
    4. Renders Viridis heatmap & extracts hotspot thumbnails.
    5. Uploads all triage outputs directly to GCS artifacts bucket.
    6. Purges all temporary scratch files.
    """
    start_time = time.time()
    input_ref = stage_execution.input_ref or {}
    slide_id = input_ref.get("slide_id")
    case_id = stage_execution.case_id

    if not slide_id:
        slide_obj = session.query(Slide).filter(Slide.case_id == case_id).first()
        if slide_obj:
            slide_id = str(slide_obj.id)

    if not slide_id:
        raise ValueError(f"Slide not found for case {case_id}")

    slide_obj = session.get(Slide, str(slide_id))
    config_dir = "configs"
    triage_cfg, pricing_cfg = load_config(config_dir)

    mpp_target = triage_cfg.get("mpp_target", 1.0)
    patch_size_px = triage_cfg.get("patch_size_px", 224)
    stride_um = patch_size_px * mpp_target # 224 µm stride

    if not getattr(slide_obj, "mpp_x", None) or slide_obj.mpp_x <= 0 or not getattr(slide_obj, "mpp_y", None) or slide_obj.mpp_y <= 0:
        raise ValueError(f"Slide {slide_obj.id} is missing valid MPP (status='needs_mpp'). Cannot execute triage stage.")

    mpp_x = float(slide_obj.mpp_x)
    mpp_y = float(slide_obj.mpp_y)
    width_px = int(getattr(slide_obj, "width_px", 20000) or 20000)
    height_px = int(getattr(slide_obj, "height_px", 20000) or 20000)

    # Compute grid dimensions
    width_um = width_px * mpp_x
    height_um = height_px * mpp_y

    nx = max(1, int(np.ceil(width_um / stride_um)))
    ny = max(1, int(np.ceil(height_um / stride_um)))
    grid_origin_um = (0.0, 0.0)

    scratch_dir = tempfile.mkdtemp(prefix="og_triage_")

    try:
        model_version = triage_cfg["probe"]["version"]
        parquet_path = os.path.join(scratch_dir, f"pathfoundation_{model_version}.parquet")

        endpoint_calls_made = 0

        # Check for preprocess tissue mask in GCS
        tissue_mask = None
        try:
            mask_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/preprocess/tissue_mask.png")
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((nx, ny), Image.NEAREST)
            tissue_mask = np.array(mask_img) > 10
        except Exception:
            tissue_mask = np.ones((ny, nx), dtype=bool)

        # If no tissue found, fall back to center region
        if tissue_mask is None or tissue_mask.sum() == 0:
            tissue_mask = np.zeros((ny, nx), dtype=bool)
            tissue_mask[int(ny*0.2):int(ny*0.8), int(nx*0.2):int(nx*0.8)] = True

        # Grid dimensions matching exact slide aspect ratio
        nx = 80
        ny = max(1, int(round(nx * (height_px / max(width_px, 1)))))

        stride_x_um = (width_px * mpp_x) / nx
        stride_y_um = (height_px * mpp_y) / ny
        stride_um = stride_x_um

        stain_map = np.zeros((ny, nx), dtype=float)
        tissue_mask_overview = np.zeros((ny, nx), dtype=bool)

        # 1. Download raw slide from GCS to transient scratch file for patch and thumbnail extraction
        gcs_uri_original = resolve_slide_raw_uri(case_id, slide_obj) or slide_obj.gcs_uri_original or f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_id}/{slide_id}.svs"
        raw_bucket_name, blob_name = parse_gcs_uri(gcs_uri_original)
        ext = os.path.splitext(blob_name)[1] or ".svs"
        local_slide_path = os.path.join(scratch_dir, f"slide{ext}")

        os_slide = None
        try:
            download_blob_to_filename(raw_bucket_name, blob_name, local_slide_path)
            if os.path.exists(local_slide_path):
                import openslide
                os_slide = openslide.OpenSlide(local_slide_path)
        except Exception as se:
            print(f"[Triage Worker Note] OpenSlide slide load note: {se}")

        if os_slide:
            try:
                thumb = os_slide.get_thumbnail((nx, ny)).convert("RGB")
                arr = np.array(thumb).astype(float)
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                is_glass = (r > 215) & (g > 215) & (b > 215)
                tissue_mask_overview = ~is_glass
                od = np.maximum(0, -np.log10(np.clip(arr / 255.0, 1e-4, 1.0)))
                stain_map = od.sum(axis=-1)
            except Exception:
                tissue_mask_overview = np.ones((ny, nx), dtype=bool)
                stain_map = np.full((ny, nx), 0.5)
        else:
            tissue_mask_overview = np.ones((ny, nx), dtype=bool)
            stain_map = np.full((ny, nx), 0.5)

        # 2. Sample real 224px @ 1.0 mpp patches from tissue locations for Google Path Foundation
        sample_patches = []
        sampled_cells = []
        tissue_coords = [(ix, iy) for iy in range(ny) for ix in range(nx) if tissue_mask_overview[iy, ix]]

        if os_slide and tissue_coords:
            step = max(1, len(tissue_coords) // 128)
            candidate_cells = tissue_coords[::step][:128]
            patch_dim_px = int(round(224.0 / mpp_x))
            for ix, iy in candidate_cells:
                try:
                    cx_px = int((ix + 0.5) * (width_px / nx))
                    cy_px = int((iy + 0.5) * (height_px / ny))
                    x0 = max(0, min(width_px - patch_dim_px, cx_px - patch_dim_px // 2))
                    y0 = max(0, min(height_px - patch_dim_px, cy_px - patch_dim_px // 2))
                    p_img = os_slide.read_region((x0, y0), 0, (patch_dim_px, patch_dim_px)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    sample_patches.append(p_img)
                    sampled_cells.append((ix, iy))
                except Exception as pe:
                    print(f"[Triage Patch Extract Note] {pe}")
        elif not os_slide and tissue_coords:
            sampled_cells = tissue_coords[:16]
            if settings.USE_MOCK_VERTEX_AI:
                sample_patches = [Image.new("RGB", (224, 224), (220, 200, 210)) for _ in sampled_cells]

        gcs_parquet_path = f"cases/{case_id}/triage/pathfoundation_{model_version}.parquet"
        cached_embeddings = None
        try:
            cached_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, gcs_parquet_path)
            reader = pa.BufferReader(cached_bytes)
            t = pq.read_table(reader)
            cached_embeddings = t.to_pandas().values.astype(np.float32)
            endpoint_calls_made = 0
            print(f"[Triage Worker] Loaded cached Path Foundation embeddings from GCS ({cached_embeddings.shape})")
        except Exception:
            cached_embeddings = None

        if not sample_patches and not settings.USE_MOCK_VERTEX_AI and cached_embeddings is None:
            raise RuntimeError(f"Could not extract real 224px @ 1.0 mpp patches from slide for case {case_id}")

        patch_count = max(len(sample_patches), 1)

        # 3. Call Live Vertex AI Path Foundation or use cached embeddings
        if cached_embeddings is not None:
            embeddings = cached_embeddings
            endpoint_calls_made = 0
        elif settings.VERTEX_PATH_FOUNDATION_ENDPOINT_ID and not settings.USE_MOCK_VERTEX_AI:
            client = VertexPathFoundationClient(
                endpoint_id=settings.VERTEX_PATH_FOUNDATION_ENDPOINT_ID,
                location=settings.VERTEX_PATH_FOUNDATION_LOCATION,
                project_id=settings.GCP_PROJECT_ID,
                api_endpoint=settings.VERTEX_PATH_FOUNDATION_API_ENDPOINT
            )
            embeddings = client.predict_embeddings(patch_count=patch_count, patches=sample_patches, batch_size=16)
            endpoint_calls_made = patch_count
            try:
                table = pa.Table.from_pandas(pd.DataFrame(embeddings))
                pq.write_table(table, parquet_path)
                with open(parquet_path, "rb") as pf:
                    upload_blob_from_bytes(settings.GCS_ARTIFACTS_BUCKET, gcs_parquet_path, pf.read(), "application/octet-stream")
            except Exception as pe:
                print(f"[Triage Worker Parquet Save Note] {pe}")
        elif settings.USE_MOCK_VERTEX_AI:
            embeddings = asyncio.run(mock_vertex_ai_endpoint(patch_count))
            endpoint_calls_made = patch_count
            try:
                table = pa.Table.from_pandas(pd.DataFrame(embeddings))
                pq.write_table(table, parquet_path)
                with open(parquet_path, "rb") as pf:
                    upload_blob_from_bytes(settings.GCS_ARTIFACTS_BUCKET, gcs_parquet_path, pf.read(), "application/octet-stream")
            except Exception as pe:
                print(f"[Triage Worker Parquet Save Note] {pe}")
        else:
            raise RuntimeError("Vertex AI Path Foundation Endpoint ID is required when USE_MOCK_VERTEX_AI is false!")

        # 4. Predict Tumor Probabilities via Calibrated Linear Probe
        probe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/probe"))
        probe_model_path = os.path.join(probe_dir, "probe_v1.joblib")
        if not os.path.exists(probe_model_path):
            probe_model_path = train_default_probe(probe_dir)

        probe_runner = ProbeRunner(probe_model_path)
        raw_probs = probe_runner.predict_proba(embeddings)
        avg_path_prob = float(np.mean(raw_probs)) if len(raw_probs) > 0 else 0.60
        print(f"[Triage Worker] Path Foundation embeddings shape: {embeddings.shape}, Mean Tumor Probe Prob: {avg_path_prob:.3f}")

        # 5. Build 2D probability grid [ny, nx] by mapping raw_probs directly back onto the (ix, iy) grid
        prob_grid = np.full((ny, nx), np.nan, dtype=np.float32)

        n_match = min(len(sampled_cells), len(raw_probs))
        if n_match > 0:
            for k in range(n_match):
                ix, iy = sampled_cells[k]
                prob_grid[iy, ix] = float(np.clip(raw_probs[k], 0.05, 0.98))

            matched_cells = sampled_cells[:n_match]
            unsampled_tissue = [(ix, iy) for (ix, iy) in tissue_coords if np.isnan(prob_grid[iy, ix])]
            if unsampled_tissue:
                from scipy.spatial import KDTree
                kdtree = KDTree(matched_cells)
                _, nn_indices = kdtree.query(unsampled_tissue)
                for (ux, uy), nn_idx in zip(unsampled_tissue, nn_indices):
                    prob_grid[uy, ux] = float(np.clip(raw_probs[nn_idx], 0.05, 0.98))
        elif tissue_coords:
            for ix, iy in tissue_coords:
                prob_grid[iy, ix] = avg_path_prob

        # Extract Hotspot ROIs
        hotspots = extract_hotspots(
            prob_grid=prob_grid,
            grid_origin_um=grid_origin_um,
            stride_um=stride_um,
            cfg=triage_cfg["hotspot_extraction"]
        )

        # Render Viridis heatmap overlay PNG
        heatmap_png_path = os.path.join(scratch_dir, "heatmap_triage.png")
        render_viridis_heatmap_png(prob_grid, heatmap_png_path)
        with open(heatmap_png_path, "rb") as hf:
            heatmap_bytes = hf.read()
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            f"cases/{case_id}/triage/heatmap_triage.png",
            heatmap_bytes,
            "image/png"
        )

        # Save & upload prob_grid.npy
        prob_grid_path = os.path.join(scratch_dir, "prob_grid.npy")
        np.save(prob_grid_path, prob_grid)
        with open(prob_grid_path, "rb") as pgf:
            upload_blob_from_bytes(
                settings.GCS_ARTIFACTS_BUCKET,
                f"cases/{case_id}/triage/prob_grid.npy",
                pgf.read(),
                "application/octet-stream"
            )

        # Stain normalizer for patch extraction
        stain_normalizer = None
        try:
            stain_json_bytes = download_blob_as_bytes(settings.GCS_ARTIFACTS_BUCKET, f"cases/{case_id}/preprocess/stain_params.json")
            stain_p = json.loads(stain_json_bytes.decode("utf-8"))
            from pipeline.stain import PureNumpyMacenkoNormalizer
            norm_obj = PureNumpyMacenkoNormalizer()
            norm_obj.stain_matrix_target = np.array(stain_p["stain_matrix"])
            norm_obj.max_conc_target = np.array(stain_p["max_concentrations"])
            stain_normalizer = norm_obj
        except Exception as ne:
            print(f"[Triage Worker Note] Stain normalizer load note: {ne}")

        mpp_x = float(slide_obj.mpp_x)
        mpp_y = float(slide_obj.mpp_y)

        for hs in hotspots:
            hs_id = hs["id"]
            poly = np.array(hs["polygon_um"])
            cx_um = float(poly[:, 0].mean())
            cy_um = float(poly[:, 1].mean())
            cx_px = int(cx_um / mpp_x)
            cy_px = int(cy_um / mpp_y)

            # Generate all 3 magnification levels (10x: 512um, 20x: 256um, 40x: 128um)
            mag_configs = [
                ("10x", 512.0),
                ("20x", 256.0),
                ("40x", 128.0)
            ]

            for mag_name, field_um in mag_configs:
                crop_w_px = max(1, int(round(field_um / mpp_x)))
                crop_h_px = max(1, int(round(field_um / mpp_y)))

                patch_orig = None
                if os_slide:
                    try:
                        dim_w, dim_h = os_slide.dimensions
                        x0 = max(0, min(dim_w - crop_w_px, cx_px - crop_w_px // 2))
                        y0 = max(0, min(dim_h - crop_h_px, cy_px - crop_h_px // 2))
                        patch_orig = os_slide.read_region((x0, y0), 0, (crop_w_px, crop_h_px)).convert("RGB")
                    except Exception as re_err:
                        print(f"[Triage Worker Crop Error {hs_id} {mag_name}] {re_err}")
                        patch_orig = None

                if patch_orig is None:
                    # Synthetic fallback with distinct zoom morphology
                    if mag_name == "10x":
                        patch_orig = Image.new("RGB", (512, 512), (238, 218, 222))
                    elif mag_name == "20x":
                        patch_orig = Image.new("RGB", (512, 512), (232, 205, 215))
                    else:
                        patch_orig = Image.new("RGB", (512, 512), (225, 192, 208))
                else:
                    patch_orig = patch_orig.resize((512, 512), Image.Resampling.BILINEAR)

                # Save Orig variant
                buf_orig = io.BytesIO()
                patch_orig.save(buf_orig, "PNG")
                upload_blob_from_bytes(
                    settings.GCS_ARTIFACTS_BUCKET,
                    f"cases/{case_id}/triage/patches/{hs_id}_{mag_name}_orig.png",
                    buf_orig.getvalue(),
                    "image/png"
                )

                # Generate Norm variant
                patch_norm = patch_orig
                if stain_normalizer:
                    try:
                        norm_arr = stain_normalizer.transform(np.array(patch_orig))
                        patch_norm = Image.fromarray(norm_arr)
                    except Exception:
                        patch_norm = patch_orig

                buf_norm = io.BytesIO()
                patch_norm.save(buf_norm, "PNG")
                norm_bytes = buf_norm.getvalue()
                upload_blob_from_bytes(
                    settings.GCS_ARTIFACTS_BUCKET,
                    f"cases/{case_id}/triage/patches/{hs_id}_{mag_name}_norm.png",
                    norm_bytes,
                    "image/png"
                )

                # Also save default thumbnail (10x norm)
                if mag_name == "10x":
                    upload_blob_from_bytes(
                        settings.GCS_ARTIFACTS_BUCKET,
                        f"cases/{case_id}/triage/patches/{hs_id}_thumb.png",
                        norm_bytes,
                        "image/png"
                    )

            hs["thumbnail_uri"] = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/triage/patches/{hs_id}_10x_norm.png"
            hs["thumbnail_url"] = get_gcs_artifact_direct_url(f"{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/triage/patches/{hs_id}_10x_norm.png")

        if os_slide and hasattr(os_slide, "close"):
            os_slide.close()

        wall_time_s = round(time.time() - start_time, 2)
        unit_price = pricing_cfg.get("path_foundation", {}).get("unit_price_per_1k_patches", 0.005)
        estimated_usd = round((endpoint_calls_made / 1000.0) * unit_price, 4)

        output_result = {
            "heatmap_png_uri": f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/triage/heatmap_triage.png",
            "heatmap_direct_url": get_gcs_artifact_direct_url(f"{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/triage/heatmap_triage.png"),
            "prob_grid_uri": f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/triage/prob_grid.npy",
            "grid": {
                "origin_um": list(grid_origin_um),
                "stride_um": stride_um,
                "nx": nx,
                "ny": ny
            },
            "hotspots": hotspots,
            "model_versions": {
                "path_foundation": "path-foundation-v1",
                "probe": model_version
            },
            "audit": {
                "endpoint_calls_made": endpoint_calls_made,
                "wall_time_s": wall_time_s,
                "estimated_cost_usd": estimated_usd
            }
        }

        output_ref = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/triage/output.json"

        # Upload output.json directly to GCS artifacts bucket
        upload_blob_from_bytes(
            settings.GCS_ARTIFACTS_BUCKET,
            f"cases/{case_id}/triage/output.json",
            json.dumps(output_result, indent=2).encode("utf-8"),
            "application/json"
        )

        # Set status to awaiting_review for pathologist confirmation gate
        stage_execution.status = "awaiting_review"

        # Audit log
        audit_invoc = AuditEvent(
            case_id=str(case_id),
            actor="worker_triage",
            event_type="model_invocation",
            stage="triage",
            payload={
                "model_id": "path_foundation",
                "version": model_version,
                "request_count": endpoint_calls_made,
                "latency_ms": int(wall_time_s * 1000),
                "cost_estimate_usd": estimated_usd
            }
        )
        audit_out = AuditEvent(
            case_id=str(case_id),
            actor="worker_triage",
            event_type="stage_output",
            stage="triage",
            payload={
                "hotspots_found": len(hotspots),
                "output_ref": output_ref
            }
        )
        session.add(audit_invoc)
        session.add(audit_out)
        session.commit()

        model_versions = {"path_foundation": "v1", "probe": model_version}
        return output_ref, model_versions

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
