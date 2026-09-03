"""
OncoGemma Stage v4.3 - HoVer-Net Mitosis Verifier & Nuclear Morphometry Engine.
Performs second-pass instance segmentation and classification on 128x128 candidate crops.
Filters out apoptotic bodies, lymphocytes, and pyknotic debris from true mitotic figures.
"""
import os
import math
from typing import Protocol, Tuple, List, Optional
import numpy as np


class MitosisVerifier(Protocol):
    def verify(self, crop_rgb: np.ndarray) -> Tuple[float, Optional[List[List[int]]]]:
        """
        Evaluates a 128x128 crop centered at candidate centroid.
        Returns:
            p_mitosis: Probability (0.0 to 1.0) that crop contains a genuine mitotic figure.
            contour: Approximate boundary coordinates [[x1, y1], [x2, y2], ...] of the central nucleus.
        """
        ...


class HoVerNetMitosisVerifier:
    """
    HoVer-Net Architecture & Morphological Nuclear Instance Verifier.
    Differentiates true mitotic figures (metaphase plates, anaphase spindles, telophase clusters)
    from resting tumor nuclei, lymphocytes, apoptotic fragments, and debris.
    """
    def __init__(self, weights_path: Optional[str] = None, threshold: float = 0.50, device: str = "cpu"):
        self.weights_path = weights_path
        self.threshold = threshold
        self.device = device
        self.model = None
        self.model_version = "hovernet_fast_mitosis@v1.2"

        if weights_path and os.path.exists(weights_path):
            try:
                import torch
                self.model = torch.load(weights_path, map_location=device)
                print(f"[HoVerNetVerifier] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[HoVerNetVerifier Warning] Failed to load {weights_path}: {e}. Using morphological verification engine.")
                self.model = None

    def verify(self, crop_rgb: np.ndarray) -> Tuple[float, Optional[List[List[int]]]]:
        """
        Evaluates a 128x128 crop at 0.25 um/px.
        """
        h, w, _ = crop_rgb.shape
        if h < 32 or w < 32:
            return 0.0, None

        if self.model is not None:
            try:
                import torch
                img_t = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
                img_t = img_t.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(img_t)
                # Output has tp_map (nuclear type prediction) and np_map (nuclear pixel map)
                p_mitosis = float(output.get("p_mitosis", 0.5))
                return p_mitosis, None
            except Exception as e:
                print(f"[HoVerNet Runtime Error] {e}. Falling back to morphometric classifier.")

        # Morphometric nuclear instance analysis on 128x128 patch
        return self._morphometric_nuclear_analysis(crop_rgb)

    def _morphometric_nuclear_analysis(self, crop_rgb: np.ndarray) -> Tuple[float, Optional[List[List[int]]]]:
        """
        First-principles cellular morphometry:
        1. Analyzes central region for chromatin condensation and nuclear morphology.
        2. Measures boundary irregularity / spiculation (spindle protrusions vs smooth lymphocyte/nuclear envelope).
        3. Detects absence of intact nuclear membrane (classic hallmark of active mitosis).
        4. Explicitly filters out non-mitotic mimickers:
           - Apoptotic bodies (small, dense, pyknotic, high circularity, haloed)
           - Lymphocytes (small, smooth continuous unbroken envelope, high circularity/solidity)
           - Resting / interphase nuclei (intact membrane, vesicular chromatin, lower OD)
           - Background stroma / debris / dust specks
        """
        try:
            import cv2
        except ImportError:
            cv2 = None

        h, w, _ = crop_rgb.shape
        cy, cx = h // 2, w // 2
        r_px = min(24, min(h, w) // 4) # ~12 um radius region

        # Optical density transformation
        rgb_f = np.maximum(crop_rgb.astype(np.float32), 1.0) / 255.0
        od = -np.log(rgb_f)
        # Hematoxylin absorption component
        h_od = od[:, :, 0] - 0.15 * od[:, :, 1] - 0.15 * od[:, :, 2]

        center_h_od = h_od[cy - r_px : cy + r_px, cx - r_px : cx + r_px]
        mean_h_od = float(np.mean(center_h_od))

        # Reject empty background / stroma
        if mean_h_od < 0.20:
            return 0.05, None

        # Robust 95th percentile OD (avoids single-pixel outlier blowout)
        p95_od = float(np.percentile(center_h_od, 95))
        std_od = float(np.std(center_h_od))

        # Segment central chromatin clump
        thresh = max(0.35, float(np.median(h_od) + 1.2 * np.std(h_od)))
        chromatin_mask = (center_h_od > thresh).astype(np.uint8) * 255

        contour_pts = None

        if cv2 is not None:
            cnts, _ = cv2.findContours(chromatin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                return 0.10, None

            # Filter to contours close to center
            # Extract the central connected component located at (r_px, r_px)
            central_cnt = None
            for cnt in cnts:
                if cv2.pointPolygonTest(cnt, (r_px, r_px), False) >= -2.0:
                    central_cnt = cnt
                    break

            if central_cnt is None:
                # Fallback to closest contour within central radius
                central_cnt = min(cnts, key=lambda c: cv2.pointPolygonTest(c, (r_px, r_px), True) ** 2)

            area = float(cv2.contourArea(central_cnt))
            perim = float(cv2.arcLength(central_cnt, True))

            if area < 80 or perim <= 0:
                # Tiny debris / noise
                return 0.12, None

            equiv_diam = float(np.sqrt(4.0 * area / np.pi)) # diameter in pixels
            circ = float((4.0 * np.pi * area) / (perim * perim))
            hull = cv2.convexHull(central_cnt)
            hull_area = max(1.0, float(cv2.contourArea(hull)))
            solidity = float(area / hull_area)

            equiv_perim = np.pi * equiv_diam
            spiculation = float((perim - equiv_perim) / max(1.0, equiv_perim))

            # Approximate nuclear contour in crop coordinates
            approx_cnt = cv2.approxPolyDP(central_cnt, 1.5, True)
            contour_pts = []
            for pt in approx_cnt:
                px_crop = int(pt[0][0] + (cx - r_px))
                py_crop = int(pt[0][1] + (cy - r_px))
                contour_pts.append([px_crop, py_crop])

            # Measure halo contrast ratio to detect apoptotic retraction space
            mask_cnt_inner = np.zeros_like(center_h_od, dtype=np.uint8)
            cv2.drawContours(mask_cnt_inner, [central_cnt], -1, 255, -1)
            kernel = np.ones((5, 5), np.uint8)
            mask_cnt_outer = cv2.dilate(mask_cnt_inner, kernel, iterations=2)
            halo_mask = (mask_cnt_outer > 0) & (mask_cnt_inner == 0)
            halo_od = float(np.mean(center_h_od[halo_mask])) if np.any(halo_mask) else 0.5
            core_od = float(np.mean(center_h_od[mask_cnt_inner > 0])) if np.any(mask_cnt_inner > 0) else 0.5

            # 1. Reject Apoptotic Bodies (Van Diest Criteria):
            # Apoptotic bodies feature small, smooth, compact globular pyknotic fragments
            # (equiv_diam < 24 px, circ > 0.58, spiculation < 0.20) surrounded by a clear retraction halo.
            if equiv_diam < 24.0 and circ > 0.58 and spiculation < 0.20 and halo_od < 0.18:
                return 0.08, contour_pts

            # 2. Reject Lymphocyte / Inflammatory Cell:
            # Small (diam 16-30 px ~ 4-7.5 um), high circularity (>0.64), high solidity (>0.86), low spiculation (<0.18)
            if 16.0 <= equiv_diam <= 30.0 and circ > 0.64 and solidity > 0.86 and spiculation < 0.18:
                return 0.12, contour_pts

            # 3. Reject Resting Interphase Nuclei:
            # True mitoses REQUIRE dissolved nuclear envelope. An intact, continuous oval/circular membrane
            # with smooth contour (spiculation < 0.18, circ > 0.55, solidity > 0.84) represents interphase.
            if spiculation < 0.18 and solidity > 0.84 and (circ > 0.55 or p95_od < 0.90):
                return 0.15, contour_pts

            # 4. Reject Tiny Debris / Giant Tissue Folds:
            if area < 300.0 or equiv_diam < 20.0 or area > 3200.0 or equiv_diam > 62.0:
                return 0.12, contour_pts

            # 5. Mitotic Figure Scoring (Van Diest Classic Metaphase / Anaphase / Telophase Criteria):
            # True dividing cell requires:
            # - High spiculation / chromosome arms protruding into cytoplasm (spiculation >= 0.18)
            # - Intensely condensed basophilic chromatin (p95_od >= 0.75)
            # - High texture variance from individual chromosomes (std_od >= 0.20)
            # - Irregular, jagged contour from envelope dissolution (solidity < 0.82)
            size_score = float(np.clip((equiv_diam - 20.0) / 22.0, 0.0, 1.0))
            spic_score = float(np.clip((spiculation - 0.18) / 0.40, 0.0, 1.0))
            od_score = float(np.clip((p95_od - 0.75) / 0.75, 0.0, 1.0))
            texture_score = float(np.clip((std_od - 0.20) / 0.35, 0.0, 1.0))
            irregularity_score = float(np.clip((0.85 - solidity) / 0.25, 0.0, 1.0))

            if spic_score < 0.10 or od_score < 0.20 or texture_score < 0.15:
                return 0.22, contour_pts

            p_mitosis = 0.15 + (
                0.30 * spic_score +
                0.25 * od_score +
                0.25 * texture_score +
                0.10 * size_score +
                0.10 * irregularity_score
            )

            return float(np.clip(p_mitosis, 0.05, 0.95)), contour_pts
        else:
            # Fallback when OpenCV is not installed
            dense_pixels = int(np.sum(chromatin_mask > 0))
            if dense_pixels < 30:
                return 0.12, None
            score = 0.20 + min(0.70, (dense_pixels / 600.0) * 0.35 + (p95_od / 2.0) * 0.35)
            return float(np.clip(score, 0.05, 0.95)), None


def create_dual_magnification_composite(
    slide_obj,
    center_x: int,
    center_y: int,
    mpp_x: float = 0.25
) -> tuple[bytes, bytes]:
    """
    Extract dual-magnification views for MedGemma multimodal refereeing:
    1. Focus Crop (40x, 128x128 px): Target cell with subtle circular reticle.
    2. Context Patch (10x, 512x512 px): Surrounding tumor bed architecture.
    """
    from PIL import Image, ImageDraw
    import io
    from app.core.openslide_lock import OPENSLIDE_GLOBAL_LOCK

    # 1. 40x Focus Crop (128x128 px @ level 0)
    crop_size = 128
    x0 = max(0, int(center_x - crop_size // 2))
    y0 = max(0, int(center_y - crop_size // 2))

    with OPENSLIDE_GLOBAL_LOCK:
        rgba_focus = slide_obj.read_region((x0, y0), 0, (crop_size, crop_size))
        rgb_focus = rgba_focus.convert("RGB")

    buf_focus = io.BytesIO()
    rgb_focus.save(buf_focus, format="PNG")
    crop_bytes = buf_focus.getvalue()

    # 2. 10x Context Crop (512x512 px @ 1.0 um/px)
    downsample = 1.0 / max(mpp_x, 0.1) # e.g. 4.0
    l0_w = int(512 * downsample)
    l0_h = int(512 * downsample)
    cx0 = max(0, int(center_x - l0_w // 2))
    cy0 = max(0, int(center_y - l0_h // 2))

    with OPENSLIDE_GLOBAL_LOCK:
        rgba_ctx = slide_obj.read_region((cx0, cy0), 0, (l0_w, l0_h))
        rgb_ctx = rgba_ctx.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)

    buf_ctx = io.BytesIO()
    rgb_ctx.save(buf_ctx, format="JPEG", quality=85)
    context_bytes = buf_ctx.getvalue()

    return crop_bytes, context_bytes
