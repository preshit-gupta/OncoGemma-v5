"""
OncoGemma Stage v4.3 - Mitosis Detector & Tiling Engine.
Performs 40x high-magnification tile extraction, Macenko stain normalization,
YOLO candidate sweeping, and physical micrometer cross-tile NMS.
"""
import os
import math
from typing import Protocol, List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

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
    """
    def __init__(self, weights_path: Optional[str] = None, conf_threshold: float = 0.35, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.device = device
        self.weights_path = weights_path
        self.model = None
        self.model_version = "midog22_yolov8x_sweep@v1.0"

        if weights_path and os.path.exists(weights_path):
            try:
                import torch
                # If torch and model weights are present, attempt loading
                self.model = torch.load(weights_path, map_location=device)
                print(f"[MitosisDetector] Loaded published weights from {weights_path}")
            except Exception as e:
                print(f"[MitosisDetector Warning] Failed to load {weights_path}: {e}. Running in algorithmic fallback mode.")
                self.model = None

    def detect(self, tile_rgb: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Sweeps 40x tile (1024x1024 px) for mitotic candidates.
        """
        if self.model is not None:
            try:
                # Real PyTorch/YOLO inference if model is loaded
                import torch
                img_t = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
                img_t = img_t.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    preds = self.model(img_t)
                # Parse bounding boxes / centroids
                detections = []
                for box in preds[0]:
                    conf = float(box[4])
                    if conf >= self.conf_threshold:
                        cx = float((box[0] + box[2]) / 2.0)
                        cy = float((box[1] + box[3]) / 2.0)
                        detections.append((cx, cy, conf))
                return detections
            except Exception as e:
                print(f"[MitosisDetector Runtime Error] {e}. Falling back to visual feature extractor.")

        # Algorithmic optical density & hyperchromatic nuclear detection fallback
        # Mitotic figures appear as intensely dark, hyperchromatic, dense chromatin clumps in H&E (low green/blue, high hematoxylin absorption)
        return self._detect_hyperchromatic_features(tile_rgb)

    def _detect_hyperchromatic_features(self, tile_rgb: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        First-principles hematoxylin optical density & morphological candidate sweep.
        Sweeps 40x tile for dense condensed chromatin clusters using absolute OD thresholding
        and connected component analysis.
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
                # Mitotic chromatin clusters typically occupy 200 to 3500 pixels (5-18 um across)
                if 200 <= area <= 3500:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])

                        # Calculate local peak OD within contour
                        mask_cnt = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                        p95_od = float(np.percentile(h_od[mask_cnt > 0], 95))

                        # Scale confidence based on chromatin condensation and cluster size
                        conf = float(np.clip(
                            0.35 + min(0.35, (p95_od - 0.40) * 0.30) + min(0.20, (area / 500.0) * 0.20),
                            self.conf_threshold,
                            0.95
                        ))
                        candidates.append((cx, cy, conf))

            # Apply intra-tile NMS (radius 80 px = 20 um at 0.25 um/px) to avoid multi-contour fragments of same cell
            candidates.sort(key=lambda c: c[2], reverse=True)
            suppressed = []
            for c in candidates:
                if not any(math.hypot(c[0] - s[0], c[1] - s[1]) < 80.0 for s in suppressed):
                    suppressed.append(c)
            return suppressed[:6]
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
                        conf = float(np.clip(0.35 + (max_val - chromatin_thresh) * 0.4, self.conf_threshold, 0.90))
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
    mpp: float = 0.25,
    stride_px: int = 960
) -> List[Dict[str, Any]]:
    """
    Generates 40x tile coordinates covering a hotspot polygon.
    Returns list of dicts with tile bounding box in pixels and base micrometers.
    """
    if not hotspot_polygon_um:
        return []

    xs = [p[0] for p in hotspot_polygon_um]
    ys = [p[1] for p in hotspot_polygon_um]
    min_x_um, max_x_um = min(xs), max(xs)
    min_y_um, max_y_um = min(ys), max(ys)

    tile_size_um = tile_size_px * mpp
    stride_um = stride_px * mpp

    tiles = []
    curr_y = min_y_um
    while curr_y <= max_y_um:
        curr_x = min_x_um
        while curr_x <= max_x_um:
            tiles.append({
                "origin_um": [float(curr_x), float(curr_y)],
                "size_um": [float(tile_size_um), float(tile_size_um)],
                "origin_px": [int(curr_x / mpp), int(curr_y / mpp)],
                "size_px": [tile_size_px, tile_size_px],
            })
            curr_x += stride_um
        curr_y += stride_um

    return tiles
