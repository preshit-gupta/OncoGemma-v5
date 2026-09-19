"""
OncoGemma Stage v4.3 - Mitosis Detector & Tiling Engine.
Performs 40x high-magnification tile extraction, Macenko stain normalization,
YOLO candidate sweeping, and physical micrometer cross-tile NMS.
"""
import os
import io
import math
import base64
from typing import Protocol, List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    from app.core.config import settings
except ImportError:
    try:
        from backend.app.core.config import settings
    except ImportError:
        settings = None

class MitosisDetector(Protocol):
    def detect(self, tile_rgb: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Runs mitosis detection on a single 40x RGB tile (H, W, 3).
        Returns list of (cx_px, cy_px, confidence) relative to tile top-left.
        """
        ...


class YoloMitosisDetector:
    """
    YOLO-family Mitosis Object Detector (trained on MIDOG / MIDOG++).
    Provides high-recall sweeping for dark, dense hyperchromatic nuclear structures.
    Supports:
      1. Remote Vertex AI custom prediction endpoint (e.g. YOLOv8-MIDOG container)
      2. Local PyTorch / Ultralytics weights checkpoint
      3. First-principles optical density fallback (od_heuristic@dev)
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        conf_threshold: float = 0.35,
        device: str = "cpu",
        max_candidates_per_tile: int = 12,
        batch_size: int = 16,
        fp16: bool = False
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.weights_path = weights_path
        self.endpoint_id = endpoint_id or (getattr(settings, "VERTEX_MITOSIS_ENDPOINT_ID", None) if settings else None)
        self.endpoint_location = getattr(settings, "VERTEX_MITOSIS_LOCATION", "us-central1") if settings else "us-central1"
        self.project_id = getattr(settings, "GCP_PROJECT_ID", "oncogemma-dev") if settings else "oncogemma-dev"
        self.max_candidates_per_tile = max_candidates_per_tile
        self.batch_size = batch_size
        self.fp16 = fp16
        self.model = None
        self.vertex_endpoint = None
        self.model_version = "od_heuristic@dev"

        # 1. Connect to remote Vertex AI Endpoint if configured
        if self.endpoint_id:
            try:
                from google.cloud import aiplatform
                aiplatform.init(project=self.project_id, location=self.endpoint_location)
                self.vertex_endpoint = aiplatform.Endpoint(
                    endpoint_name=self.endpoint_id,
                    project=self.project_id,
                    location=self.endpoint_location
                )
                self.model_version = f"vertex_ai_midog@{self.endpoint_id}"
                print(f"[MitosisDetector] Connected to Vertex AI Mitosis Endpoint: {self.endpoint_id}")
            except Exception as e:
                print(f"[MitosisDetector Warning] Failed to connect to Vertex AI Endpoint {self.endpoint_id}: {e}")
                self.vertex_endpoint = None

        # 2. If not using remote endpoint, load local weights if present
        if self.vertex_endpoint is None and weights_path and os.path.exists(weights_path):
            try:
                # Try loading via Ultralytics YOLO first if available
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(weights_path)
                    self.model_version = "midog22_yolov8x_sweep@v1.0"
                    print(f"[MitosisDetector] Loaded Ultralytics weights from {weights_path}")
                except Exception:
                    import torch
                    loaded = torch.load(weights_path, map_location=device)
                    if isinstance(loaded, dict) and "model" in loaded:
                        self.model = loaded["model"]
                    else:
                        self.model = loaded
                    if hasattr(self.model, "eval"):
                        self.model.eval()
                    self.model_version = "midog22_yolov8x_sweep@v1.0"
                    print(f"[MitosisDetector] Loaded PyTorch weights from {weights_path}")
            except Exception as e:
                print(f"[MitosisDetector Warning] Failed to load {weights_path}: {e}. Running in algorithmic fallback mode.")
                self.model = None
                self.model_version = "od_heuristic@dev"

    def _detect_vertex_ai(self, tile_rgb: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """Queries remote Google Cloud Vertex AI endpoint for mitosis predictions.

        Dynamically sub-patches large tiles (e.g. 1024x1024) into 512x512 sub-patches
        conforming to the MIDOG KongNet detector contract, batches them in a single predict call,
        remaps detected bounding-box coordinates to the tile reference frame, and checks for
        endpoint error responses to enable graceful fallback.
        """
        if self.vertex_endpoint is None:
            return None
        try:
            h, w, _ = tile_rgb.shape
            target_patch_size = 512
            instances = []
            offsets = []  # List of tuples: (ox, oy, ph, pw)

            if h <= target_patch_size and w <= target_patch_size:
                if h == target_patch_size and w == target_patch_size:
                    patch = tile_rgb
                else:
                    padded = np.full((target_patch_size, target_patch_size, 3), 255, dtype=tile_rgb.dtype)
                    padded[:h, :w] = tile_rgb
                    patch = padded
                img = Image.fromarray(patch)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                instances.append({
                    "image_bytes": b64_str,
                    "confidence_threshold": self.conf_threshold
                })
                offsets.append((0, 0, h, w))
            else:
                for y in range(0, h, target_patch_size):
                    for x in range(0, w, target_patch_size):
                        sub = tile_rgb[y:min(y + target_patch_size, h), x:min(x + target_patch_size, w)]
                        ph, pw, _ = sub.shape
                        if ph != target_patch_size or pw != target_patch_size:
                            padded = np.full((target_patch_size, target_patch_size, 3), 255, dtype=tile_rgb.dtype)
                            padded[:ph, :pw] = sub
                            sub = padded
                        img = Image.fromarray(sub)
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=90)
                        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                        instances.append({
                            "image_bytes": b64_str,
                            "confidence_threshold": self.conf_threshold
                        })
                        offsets.append((x, y, ph, pw))

            response = self.vertex_endpoint.predict(instances=instances)
            if not response or not hasattr(response, "predictions") or not response.predictions:
                print("[MitosisDetector Vertex AI] Empty response from endpoint. Falling back to feature extractor.")
                return None

            detections = []
            for idx, (preds, (ox, oy, ph, pw)) in enumerate(zip(response.predictions, offsets)):
                if isinstance(preds, dict) and preds.get("error"):
                    print(f"[MitosisDetector Vertex AI Warning] Endpoint error on patch #{idx}: {preds.get('error')}. Falling back.")
                    return None

                boxes = preds.get("boxes", preds) if isinstance(preds, dict) else preds
                if not isinstance(boxes, (list, tuple)):
                    continue
                for b in boxes:
                    if isinstance(b, dict):
                        cx = float(b.get("cx", (b.get("x1", 0) + b.get("x2", 0)) / 2.0))
                        cy = float(b.get("cy", (b.get("y1", 0) + b.get("y2", 0)) / 2.0))
                        conf = float(b.get("confidence", b.get("conf", 0.5)))
                    elif isinstance(b, (list, tuple)) and len(b) >= 5:
                        cx = float((b[0] + b[2]) / 2.0)
                        cy = float((b[1] + b[3]) / 2.0)
                        conf = float(b[4])
                    else:
                        continue

                    # Discard any candidate located in the padded margin
                    if pw < target_patch_size and cx >= pw:
                        continue
                    if ph < target_patch_size and cy >= ph:
                        continue

                    if conf >= self.conf_threshold:
                        detections.append((cx + ox, cy + oy, conf))

            return detections[:self.max_candidates_per_tile]
        except Exception as e:
            print(f"[MitosisDetector Vertex AI Error] {e}. Falling back to visual feature extractor.")
            return None

    def detect(self, tile_rgb: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Sweeps 40x tile (1024x1024 px) for mitotic candidates.
        Uses two-stage high-recall sweeping:
          - Queries Vertex AI Endpoint (MIDOG/KongNet/YOLO) if configured
          - Seamlessly augments or falls back to first-principles optical density (OD)
            hyperchromatic feature extraction to guarantee high recall for Gemini referee.
        """
        # 1. Try Vertex AI Endpoint first if configured
        vertex_results = None
        if self.vertex_endpoint is not None:
            vertex_results = self._detect_vertex_ai(tile_rgb)

        # 2. Try Local YOLO Model if loaded and Vertex AI was not used
        local_results = None
        if vertex_results is None and self.model is not None:
            try:
                # If Ultralytics YOLO predict interface exists
                if hasattr(self.model, "predict"):
                    results = self.model.predict(
                        tile_rgb,
                        conf=self.conf_threshold,
                        device=self.device,
                        verbose=False,
                        half=self.fp16
                    )
                    local_results = []
                    for r in results:
                        if hasattr(r, "boxes") and r.boxes is not None:
                            for box in r.boxes:
                                coords = box.xyxy[0].tolist()
                                conf = float(box.conf[0].item())
                                cx = float((coords[0] + coords[2]) / 2.0)
                                cy = float((coords[1] + coords[3]) / 2.0)
                                local_results.append((cx, cy, conf))
                else:
                    # Direct PyTorch module inference
                    import torch
                    img_t = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
                    if self.fp16 and self.device != "cpu":
                        img_t = img_t.half()
                    img_t = img_t.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        preds = self.model(img_t)
                    local_results = []
                    if isinstance(preds, (list, tuple)) and len(preds) > 0:
                        raw_boxes = preds[0]
                        if hasattr(raw_boxes, "shape") and len(raw_boxes.shape) >= 2 and raw_boxes.shape[-1] >= 5:
                            for box in raw_boxes:
                                conf = float(box[4])
                                if conf >= self.conf_threshold:
                                    cx = float((box[0] + box[2]) / 2.0)
                                    cy = float((box[1] + box[3]) / 2.0)
                                    local_results.append((cx, cy, conf))
            except Exception as e:
                print(f"[MitosisDetector Runtime Error] {e}. Falling back to visual feature extractor.")

        primary_results = vertex_results if vertex_results is not None else local_results

        # If primary detector (Vertex AI MIDOG or local model) ran successfully,
        # its predictions are the authoritative candidate set.
        # Even if empty [] (meaning 0 mitoses on this tile), respect this negative result!
        # NEVER pollute deep-learning predictions with uncalibrated optical density blobs.
        if primary_results is not None:
            return primary_results[:self.max_candidates_per_tile]

        # 3. Only if primary model is unavailable or encountered an unrecoverable failure (None),
        # gracefully fall back to first-principles OD hyperchromatic candidate extraction.
        self.model_version = "od_heuristic@dev"
        heuristic_candidates = self._detect_hyperchromatic_features(tile_rgb)
        return heuristic_candidates[:self.max_candidates_per_tile]

    def _detect_hyperchromatic_features(self, tile_rgb: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        First-principles hematoxylin optical density & morphological candidate sweep.
        Sweeps 40x tile for dense condensed chromatin clusters using absolute OD thresholding
        and connected component analysis. Applies van Diest morphometric gating to reject
        small round apoptotic fragments and lymphocytes.
        """
        h, w, _ = tile_rgb.shape
        if h < 32 or w < 32:
            return []

        # Convert to optical density
        rgb_norm = np.maximum(tile_rgb.astype(np.float32), 1.0) / 255.0
        od = -np.log(rgb_norm)
        # Hematoxylin OD component
        h_od = od[:, :, 0] - 0.15 * od[:, :, 1] - 0.15 * od[:, :, 2]

        # Absolute threshold for condensed chromatin (mitotic chromosomes exhibit H_OD >= 0.85)
        chromatin_thresh = 0.85
        dense_mask = (h_od > chromatin_thresh).astype(np.uint8) * 255

        try:
            import cv2
            # Find connected components of dense chromatin
            cnts, _ = cv2.findContours(dense_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for cnt in cnts:
                area = float(cv2.contourArea(cnt))
                # Mitotic chromatin clusters typically occupy 350 to 4200 pixels (equivalent diameter ~21-73 px / 5.25-18.25 um)
                if 350 <= area <= 4200:
                    perim = float(cv2.arcLength(cnt, True))
                    if perim <= 0:
                        continue
                    circ = float((4.0 * np.pi * area) / (perim * perim))
                    equiv_diam = float(np.sqrt(4.0 * area / np.pi))

                    # Exclude small round bodies (lymphocytes and apoptotic fragments)
                    if equiv_diam < 28.0 and circ > 0.62:
                        continue

                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])

                        # Calculate local peak OD within contour using bounding-box sub-mask
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        sub_mask = np.zeros((bh, bw), dtype=np.uint8)
                        cnt_shifted = cnt - [bx, by]
                        cv2.drawContours(sub_mask, [cnt_shifted], -1, 255, -1)
                        sub_od = h_od[by:by + bh, bx:bx + bw]
                        pixels_inside = sub_od[sub_mask > 0]
                        if len(pixels_inside) > 0:
                            p95_od = float(np.percentile(pixels_inside, 95))
                        else:
                            p95_od = float(np.max(sub_od))

                        # Calibrate heuristic confidence to conservative screening range (0.10 - 0.65)
                        # Heuristic candidates NEVER bypass the verifier / referee as definite mitoses
                        raw_conf = 0.25 + min(0.20, max(0.0, (p95_od - 0.75) * 0.40)) + min(0.20, (area / 1500.0) * 0.20)
                        conf = float(np.clip(raw_conf, 0.10, 0.65))
                        if conf >= self.conf_threshold:
                            candidates.append((cx, cy, conf))

            # Apply intra-tile NMS (radius 80 px = 20 um at 0.25 um/px) to avoid multi-contour fragments of same cell
            candidates.sort(key=lambda c: c[2], reverse=True)
            suppressed = []
            for c in candidates:
                if not any(math.hypot(c[0] - s[0], c[1] - s[1]) < 80.0 for s in suppressed):
                    suppressed.append(c)
            if len(suppressed) > self.max_candidates_per_tile:
                print(f"[MitosisDetector] Capping {len(suppressed)} tile candidates to max limit {self.max_candidates_per_tile}")
            return suppressed[:self.max_candidates_per_tile]
        except ImportError:
            # Fallback if OpenCV is not available
            stride = 48
            candidates = []
            for y in range(stride // 2, h - stride // 2, stride):
                for x in range(stride // 2, w - stride // 2, stride):
                    patch = h_od[y - stride // 2 : y + stride // 2, x - stride // 2 : x + stride // 2]
                    max_val = float(np.max(patch))
                    if max_val > chromatin_thresh:
                        py, px = np.unravel_index(np.argmax(patch), patch.shape)
                        actual_x = float(x - stride // 2 + px)
                        actual_y = float(y - stride // 2 + py)
                        raw_conf = 0.25 + (max_val - chromatin_thresh) * 0.50
                        conf = float(np.clip(raw_conf, 0.05, 0.95))
                        if conf >= self.conf_threshold:
                            candidates.append((actual_x, actual_y, conf))
            candidates.sort(key=lambda c: c[2], reverse=True)
            suppressed = []
            for c in candidates:
                if not any(math.hypot(c[0] - s[0], c[1] - s[1]) < 80.0 for s in suppressed):
                    suppressed.append(c)
            return suppressed[:12]


def apply_global_nms(
    candidates: List[Dict[str, Any]],
    nms_radius_um: float = 20.0
) -> List[Dict[str, Any]]:
    """
    Applies greedy Non-Maximum Suppression across candidate mitotic figures in physical micrometer space.
    Suppresses lower-confidence detections within nms_radius_um (MIDOG challenge standard: 15-20 um cell diameter).
    """
    if not candidates:
        return []

    def _cand_priority(c: Dict[str, Any]) -> Tuple[int, float]:
        lbl = c.get("label", "unreviewed")
        rank = 2 if lbl == "mitosis" else (1 if lbl == "unreviewed" else 0)
        conf = float(c.get("ver_conf") if c.get("ver_conf") is not None else c.get("det_conf", 0.0))
        return (rank, conf)

    sorted_cands = sorted(candidates, key=_cand_priority, reverse=True)

    kept: List[Dict[str, Any]] = []
    kept_coords: List[Tuple[float, float]] = []

    for cand in sorted_cands:
        cx, cy = cand["centroid_um"]
        suppress = False
        for kx, ky in kept_coords:
            dist = math.hypot(cx - kx, cy - ky)
            if dist < nms_radius_um:
                suppress = True
                break
        if not suppress:
            kept.append(cand)
            kept_coords.append((cx, cy))

    return kept


def enumerate_hotspot_tiles(
    hotspot_polygon_um: List[List[float]],
    tile_size_px: int = 1024,
    mpp: float | None = None,
    stride_px: int = 960,
    tissue_mask: Optional[np.ndarray] = None,
    slide_dimensions_um: Optional[Tuple[float, float]] = None,
    min_tissue_ratio: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Generates 40x tile coordinates covering a hotspot polygon.
    Filters out tiles with less than min_tissue_ratio tissue coverage when tissue_mask is provided.
    Returns list of dicts with tile bounding box in pixels and base micrometers.
    """
    if not hotspot_polygon_um:
        return []

    if mpp is None or mpp <= 0:
        raise ValueError(f"Valid positive MPP is required for tile enumeration, got: {mpp}")

    xs = [p[0] for p in hotspot_polygon_um]
    ys = [p[1] for p in hotspot_polygon_um]
    min_x_um, max_x_um = min(xs), max(xs)
    min_y_um, max_y_um = min(ys), max(ys)

    tile_size_um = tile_size_px * mpp
    stride_um = stride_px * mpp

    mh, mw = (tissue_mask.shape if tissue_mask is not None else (0, 0))
    slide_w_um, slide_h_um = (slide_dimensions_um if slide_dimensions_um is not None else (float("inf"), float("inf")))

    # Prepare polygon geometry for intersection testing (#121)
    poly_geom = None
    if len(hotspot_polygon_um) >= 3:
        try:
            from shapely.geometry import Polygon, box
            poly_geom = Polygon(hotspot_polygon_um)
            if not poly_geom.is_valid:
                poly_geom = poly_geom.buffer(0)
        except Exception:
            poly_geom = None

    tiles = []
    curr_y = max(0.0, min_y_um)
    while curr_y <= max_y_um and curr_y < slide_h_um:
        curr_x = max(0.0, min_x_um)
        while curr_x <= max_x_um and curr_x < slide_w_um:
            # Check tile intersection with hotspot polygon (#121)
            if poly_geom is not None:
                from shapely.geometry import box
                tile_box = box(curr_x, curr_y, curr_x + tile_size_um, curr_y + tile_size_um)
                if not poly_geom.intersects(tile_box):
                    curr_x += stride_um
                    continue

            # Check tissue coverage if tissue_mask is available
            include_tile = True
            if tissue_mask is not None and mh > 0 and mw > 0 and slide_dimensions_um is not None:
                mx0 = max(0, min(mw - 1, int(round(curr_x / max(slide_w_um, 1.0) * (mw - 1)))))
                mx1 = max(0, min(mw - 1, int(round((curr_x + tile_size_um) / max(slide_w_um, 1.0) * (mw - 1)))))
                my0 = max(0, min(mh - 1, int(round(curr_y / max(slide_h_um, 1.0) * (mh - 1)))))
                my1 = max(0, min(mh - 1, int(round((curr_y + tile_size_um) / max(slide_h_um, 1.0) * (mh - 1)))))
                sub = tissue_mask[min(my0, my1):max(my0, my1) + 1, min(mx0, mx1):max(mx0, mx1) + 1]
                if sub.size > 0:
                    cov = (sub > 0).sum() / sub.size
                    if cov < min_tissue_ratio:
                        include_tile = False

            if include_tile:
                tiles.append({
                    "origin_um": [float(curr_x), float(curr_y)],
                    "size_um": [float(tile_size_um), float(tile_size_um)],
                    "origin_px": [int(curr_x / mpp), int(curr_y / mpp)],
                    "size_px": [tile_size_px, tile_size_px],
                })
            curr_x += stride_um
        curr_y += stride_um

    return tiles

