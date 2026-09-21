"""
Unit tests for HoVerNetMitosisVerifier cellular morphometry and non-mitotic discrimination.
"""
import numpy as np
import pytest
from pipeline.verify import HoVerNetMitosisVerifier
from pipeline.detect import YoloMitosisDetector, apply_global_nms


def test_morphometric_rejects_empty_background():
    verifier = HoVerNetMitosisVerifier()
    # Pure white or light stroma crop
    crop = np.full((128, 128, 3), 245, dtype=np.uint8)
    p_mitosis, contour = verifier.verify(crop)
    assert p_mitosis <= 0.15


def test_morphometric_rejects_smooth_lymphocyte():
    verifier = HoVerNetMitosisVerifier()
    # Synthetic lymphocyte: small, smooth, circular, dense nucleus (~20px diameter)
    crop = np.full((128, 128, 3), (230, 210, 225), dtype=np.uint8)
    cy, cx = 64, 64
    y, x = np.ogrid[:128, :128]
    mask = ((x - cx)**2 + (y - cy)**2) <= (10**2) # circle r=10 (20px diam)
    crop[mask] = (60, 20, 95)

    p_mitosis, contour = verifier.verify(crop)
    # Lymphocytes should be rejected as not_mitosis (p < 0.35)
    assert p_mitosis < 0.35


def test_morphometric_rejects_apoptotic_body():
    verifier = HoVerNetMitosisVerifier()
    # Synthetic apoptotic body: tiny pyknotic sphere (~10px diameter) with retraction halo
    crop = np.full((128, 128, 3), (230, 210, 225), dtype=np.uint8)
    cy, cx = 64, 64
    y, x = np.ogrid[:128, :128]
    # Retraction halo (clear unstained rim)
    halo_mask = ((x - cx)**2 + (y - cy)**2) <= (12**2)
    crop[halo_mask] = (250, 245, 250)
    # Pyknotic core
    core_mask = ((x - cx)**2 + (y - cy)**2) <= (5**2)
    crop[core_mask] = (30, 5, 50)

    p_mitosis, contour = verifier.verify(crop)
    # Apoptotic bodies should be rejected as not_mitosis (p < 0.30)
    assert p_mitosis < 0.30


def test_morphometric_identifies_true_mitotic_figure():
    verifier = HoVerNetMitosisVerifier()
    # Synthetic metaphase mitotic figure: asymmetric, dense, spiculated chromatin plate
    crop = np.full((128, 128, 3), (230, 210, 225), dtype=np.uint8)
    cy, cx = 64, 64
    # Central chromatin plate (36x18 px) with irregular jagged arms
    crop[cy - 18 : cy + 18, cx - 8 : cx + 8] = (40, 10, 80)
    crop[cy - 12 : cy + 12, cx - 14 : cx + 14] = (45, 12, 85)
    crop[cy - 6 : cy + 6, cx - 18 : cx + 18] = (50, 15, 90)
    # Add chromosome arm projections
    crop[cy - 16 : cy - 10, cx + 8 : cx + 15] = (42, 10, 82)
    crop[cy + 10 : cy + 16, cx - 15 : cx - 8] = (42, 10, 82)

    p_mitosis, contour = verifier.verify(crop)
    # True mitotic figure should receive high confidence (>= 0.65)
    assert p_mitosis >= 0.65
    assert contour is not None
    assert len(contour) >= 3


def test_detector_chromatin_sweep():
    detector = YoloMitosisDetector(endpoint_id="")
    tile = np.full((1024, 1024, 3), (235, 215, 230), dtype=np.uint8)
    # Scatter 2 simulated mitotic figures
    tile[200:230, 200:230] = (50, 15, 80)
    tile[600:630, 600:630] = (50, 15, 80)

    preds = detector.detect(tile)
    assert len(preds) == 2
    for cx, cy, conf in preds:
        assert conf >= 0.35
