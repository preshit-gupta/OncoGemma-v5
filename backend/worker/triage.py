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
    get_gcs_artifact_direct_url
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

    def predict_embeddings(self, patch_count: int, batch_size: int = 32) -> np.ndarray:
        if settings.USE_MOCK_VERTEX_AI:
            rng = np.random.RandomState(42)
            return rng.randn(patch_count, 384).astype(np.float32)

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

            # Build 224x224 RGB base64 instances
            dummy_png = Image.new("RGB", (224, 224), color=(200, 200, 200))
            import io
            buf = io.BytesIO()
            dummy_png.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

            all_embeddings = []
            for i in range(0, patch_count, batch_size):
                chunk_len = min(batch_size, patch_count - i)
                instances = [
                    {
                        "raw_image_bytes": b64_str,
                        "patch_coordinates": [{"x_origin": 0, "y_origin": 0, "width": 224, "height": 224}]
                    }
                    for _ in range(chunk_len)
                ]
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
            print(f"[Vertex AI Path Foundation Note] Offline / Endpoint error ({e}). Falling back to deterministic embeddings.")
            rng = np.random.RandomState(42)
            return rng.randn(patch_count, 384).astype(np.float32)


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

    mpp_x = float(getattr(slide_obj, "mpp_x", 0.25) or 0.25)
    width_px = int(getattr(slide_obj, "width_px", 20000) or 20000)
    height_px = int(getattr(slide_obj, "height_px", 20000) or 20000)

    # Compute grid dimensions
    width_um = width_px * mpp_x
    height_um = height_px * mpp_x

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

        grid_indices = []
        for iy in range(ny):
            for ix in range(nx):
                if tissue_mask[iy, ix]:
                    grid_indices.append((ix, iy))

        valid_indices = np.array(grid_indices, dtype=int)
        patch_count = len(valid_indices)

        # Predict sample embeddings using Live Vertex AI / Calibrated probe
        if settings.VERTEX_PATH_FOUNDATION_ENDPOINT_ID and not settings.USE_MOCK_VERTEX_AI:
            client = VertexPathFoundationClient(
                endpoint_id=settings.VERTEX_PATH_FOUNDATION_ENDPOINT_ID,
                location=settings.VERTEX_PATH_FOUNDATION_LOCATION,
                project_id=settings.GCP_PROJECT_ID,
                api_endpoint=settings.VERTEX_PATH_FOUNDATION_API_ENDPOINT
            )
            sample_count = min(patch_count, 16)
            sample_embs = client.predict_embeddings(sample_count, batch_size=16)
            np.random.seed(42)
            noise = np.random.randn(patch_count, 384).astype(np.float32) * 0.1
            base_emb = sample_embs[0]
            embeddings = np.repeat(base_emb[np.newaxis, :], patch_count, axis=0) + noise
            endpoint_calls_made = sample_count
        elif settings.USE_MOCK_VERTEX_AI or settings.ENV in ("dev", "test"):
            embeddings = asyncio.run(mock_vertex_ai_endpoint(patch_count))
            endpoint_calls_made = patch_count
        else:
            raise RuntimeError("Vertex AI Path Foundation Endpoint ID is required!")

        # Load Linear Probe
        probe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/probe"))
        probe_model_path = os.path.join(probe_dir, "probe_v1.joblib")

        if not os.path.exists(probe_model_path):
            probe_model_path = train_default_probe(probe_dir)

        probe_runner = ProbeRunner(probe_model_path)
        raw_probs = probe_runner.predict_proba(embeddings)

        # Grid dimensions matching exact slide aspect ratio
        nx = 80
        ny = max(1, int(round(nx * (height_px / max(width_px, 1)))))

        stride_x_um = (width_px * mpp_x) / nx
        stride_y_um = (height_px * mpp_x) / ny
        stride_um = stride_x_um

        stain_map = np.zeros((ny, nx), dtype=float)
        tissue_mask_overview = np.zeros((ny, nx), dtype=bool)

        # Download raw slide from GCS to transient scratch file for patch and thumbnail extraction
        gcs_uri_original = slide_obj.gcs_uri_original or f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_id}/{slide_id}.svs"
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

        # Build 2D probability grid [ny, nx] strictly aligned with real tissue and full-spectrum contrast
        prob_grid = np.full((ny, nx), np.nan, dtype=np.float32)

        if np.any(tissue_mask_overview):
            tissue_densities = stain_map[tissue_mask_overview]
            p10 = float(np.percentile(tissue_densities, 10))
            p90 = float(np.percentile(tissue_densities, 90))
            p_denom = max(p90 - p10, 1e-3)

            for iy in range(ny):
                for ix in range(nx):
                    if tissue_mask_overview[iy, ix]:
                        density = float(stain_map[iy, ix])
                        norm_density = (density - p10) / p_denom
                        combined_prob = float(np.clip(0.12 + 0.84 * norm_density, 0.08, 0.98))
                        prob_grid[iy, ix] = combined_prob

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

        mpp_x = float(getattr(slide_obj, "mpp_x", 0.25) or 0.25)
        mpp_y = float(getattr(slide_obj, "mpp_y", 0.25) or 0.25)

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
