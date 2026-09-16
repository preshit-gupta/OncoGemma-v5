import numpy as np
import pytest
from pipeline.hotspots import extract_hotspots


def test_extract_hotspots_standardized_candidates():
    # Create 100x100 grid (stride = 224 µm => total grid size 22.4 mm x 22.4 mm)
    ny, nx = 100, 100
    prob_grid = np.zeros((ny, nx), dtype=np.float32)

    # Blob 1: Center around (30, 30), radius ~10 cells (~5 mm²)
    y, x = np.ogrid[:ny, :nx]
    dist1 = np.sqrt((y - 30)**2 + (x - 30)**2)
    prob_grid[dist1 <= 10] = 0.95

    # Blob 2: Center around (70, 70), radius ~6 cells (~1.8 mm²)
    dist2 = np.sqrt((y - 70)**2 + (x - 70)**2)
    prob_grid[dist2 <= 6] = 0.85

    # Set some cells to NaN (no tissue)
    prob_grid[0:10, 0:10] = np.nan

    cfg = {
        "sigma": 2.0,
        "prob_threshold": 0.5,
        "max_hotspots": 8,
        "hpf_half_size_um": 300.0
    }

    hotspots = extract_hotspots(
        prob_grid=prob_grid,
        grid_origin_um=(0.0, 0.0),
        stride_um=224.0,
        cfg=cfg
    )

    assert len(hotspots) == 8
    assert hotspots[0]["id"] == "hs_01"
    assert hotspots[1]["id"] == "hs_02"
    assert hotspots[0]["prob_mean"] >= 0.8
    assert hotspots[0]["area_mm2"] == 0.36  # Standard 600 µm x 600 µm HPF candidate
    assert len(hotspots[0]["polygon_um"]) == 5


def test_extract_hotspots_empty_and_nan():
    prob_grid = np.full((50, 50), np.nan, dtype=np.float32)
    cfg = {
        "sigma": 2.0,
        "prob_threshold": 0.5,
        "max_hotspots": 8,
        "hpf_half_size_um": 300.0
    }

    hotspots = extract_hotspots(
        prob_grid=prob_grid,
        grid_origin_um=(0.0, 0.0),
        stride_um=224.0,
        cfg=cfg
    )

    assert hotspots == []


def test_extract_hotspots_spatial_separation():
    ny, nx = 50, 50
    prob_grid = np.zeros((ny, nx), dtype=np.float32)

    y, x = np.ogrid[:ny, :nx]
    for cy, cx, val in [(25, 25, 0.99), (10, 10, 0.95), (10, 40, 0.92), (40, 10, 0.89), (40, 40, 0.85)]:
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        prob_grid[dist <= 2] = val

    cfg = {
        "sigma": 1.0,
        "prob_threshold": 0.5,
        "max_hotspots": 5,
        "hpf_half_size_um": 300.0
    }

    hotspots = extract_hotspots(
        prob_grid=prob_grid,
        grid_origin_um=(0.0, 0.0),
        stride_um=224.0,
        cfg=cfg
    )

    assert len(hotspots) == 5
    assert hotspots[0]["id"] == "hs_01"
    assert hotspots[0]["area_mm2"] == 0.36
