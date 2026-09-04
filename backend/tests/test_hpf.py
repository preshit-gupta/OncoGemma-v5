"""
Unit tests for HPF Placement and Spatial Density Engine (v4.3).
"""
import math
import numpy as np
import pytest
from pipeline.hpf import (
    create_circular_disk_mask,
    generate_mitosis_density_map,
    greedy_place_hpfs,
    is_point_in_polygon
)


def test_circular_disk_mask():
    radius_cells = 3.0
    mask = create_circular_disk_mask(radius_cells)
    # Dimensions should be 2*ceil(r) + 1 = 7x7
    assert mask.shape == (7, 7)
    # Center cell must be 1.0
    assert mask[3, 3] == 1.0
    # Far corner must be 0.0
    assert mask[0, 0] == 0.0
    # Boundary points
    assert mask[3, 0] == 1.0 # distance = 3
    assert mask[0, 3] == 1.0


def test_generate_mitosis_density_map():
    # Synthetic candidates clustered near (1000, 1000)
    candidates = [
        {"id": "m_01", "centroid_um": [1000.0, 1000.0], "label": "mitosis"},
        {"id": "m_02", "centroid_um": [1050.0, 1020.0], "label": "mitosis"},
        {"id": "m_03", "centroid_um": [980.0, 1010.0], "label": "mitosis"},
        {"id": "m_04", "centroid_um": [2000.0, 2000.0], "label": "not_mitosis"}, # should be ignored
    ]
    bbox_um = (800.0, 800.0, 1200.0, 1200.0)
    density_map, grid_meta = generate_mitosis_density_map(
        candidates,
        bounding_box_um=bbox_um,
        grid_res_um=16.0,
        radius_um=262.0
    )

    assert density_map.ndim == 2
    assert grid_meta["stride_um"] == 16.0
    assert np.max(density_map) >= 3.0 # At least 3 mitoses convolved in cluster


def test_greedy_place_hpfs_non_overlap_invariant():
    # Create synthetic density map with several high peaks
    ny, nx = 100, 100
    stride = 16.0
    density_map = np.zeros((ny, nx), dtype=np.float32)
    # Add multiple peaks separated by >= 524 um
    density_map[20, 20] = 10.0
    density_map[20, 60] = 8.0
    density_map[60, 20] = 7.0
    density_map[60, 60] = 9.0
    density_map[95, 20] = 6.0

    grid_meta = {
        "origin_um": [0.0, 0.0],
        "stride_um": stride,
        "nx": nx,
        "ny": ny
    }

    radius_um = 262.0
    min_sep_um = 524.0 # 2 * radius

    hpfs = greedy_place_hpfs(
        density_map,
        grid_meta,
        count=5,
        radius_um=radius_um,
        min_separation_um=min_sep_um
    )

    assert len(hpfs) == 5
    # Verify non-overlap distance invariant between all placed pairs
    for i in range(len(hpfs)):
        for j in range(i + 1, len(hpfs)):
            c1 = hpfs[i]["center_um"]
            c2 = hpfs[j]["center_um"]
            dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
            assert dist >= min_sep_um - 1e-2, f"HPF {i} and {j} overlap: dist={dist} < {min_sep_um}"


def test_greedy_place_hpfs_overlap_relaxation_fallback():
    # Small area where 10 strictly non-overlapping circles cannot fit
    ny, nx = 40, 40
    stride = 16.0
    density_map = np.ones((ny, nx), dtype=np.float32) * 5.0
    grid_meta = {
        "origin_um": [0.0, 0.0],
        "stride_um": stride,
        "nx": nx,
        "ny": ny
    }

    hpfs = greedy_place_hpfs(
        density_map,
        grid_meta,
        count=10,
        radius_um=262.0,
        min_separation_um=524.0,
        relaxed_min_separation_um=393.0
    )

    # Issue #718: Must return ONLY fields that actually fit (no spiral padding to 10)
    assert 1 <= len(hpfs) < 10
    for i in range(len(hpfs)):
        for j in range(i + 1, len(hpfs)):
            c1 = hpfs[i]["center_um"]
            c2 = hpfs[j]["center_um"]
            dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
            assert dist >= 393.0 - 1e-2

    # Verify area-normalized scoring uses actual counted area per PRD 04 §4.2
    from pipeline.scoring import compute_nottingham_mitotic_score
    score_res = compute_nottingham_mitotic_score(count_total=5, n_hpf=len(hpfs), radius_um=262.0)
    single_hpf_area = math.pi * (0.262 ** 2)
    expected_area = round(len(hpfs) * single_hpf_area, 3)
    assert score_res["n_hpf"] == len(hpfs)
    assert score_res["area_mm2"] == expected_area


def test_greedy_place_hpfs_rejects_empty_glass():
    # Grid where left half (x < 1000 um) is dense tissue and right half (x >= 1000 um) is empty glass
    ny, nx = 100, 100
    stride = 16.0
    slide_w_um = nx * stride # 1600 um
    slide_h_um = ny * stride # 1600 um

    # Tissue mask: 1 for left half, 0 for right half
    tissue_mask = np.zeros((100, 100), dtype=np.uint8)
    tissue_mask[:, :50] = 255 # Left half is tissue

    # Put high density candidates on the right half (glass) and lower on the left half (tissue)
    density_map = np.zeros((ny, nx), dtype=np.float32)
    density_map[30, 20] = 5.0  # Tissue
    density_map[70, 20] = 5.0  # Tissue
    density_map[30, 80] = 100.0 # Glass (should be strictly rejected!)
    density_map[70, 80] = 100.0 # Glass (should be strictly rejected!)

    grid_meta = {
        "origin_um": [0.0, 0.0],
        "stride_um": stride,
        "nx": nx,
        "ny": ny
    }

    hpfs = greedy_place_hpfs(
        density_map,
        grid_meta,
        count=2,
        radius_um=262.0,
        min_separation_um=524.0,
        tissue_mask=tissue_mask,
        slide_dimensions_um=(slide_w_um, slide_h_um),
        min_tissue_coverage=0.70
    )

    assert len(hpfs) == 2
    for h in hpfs:
        cx, cy = h["center_um"]
        # Must be on left half (tissue)
        assert cx < 800.0, f"HPF placed in glass area: center_x = {cx}"
        assert h["tissue_coverage"] >= 0.70, f"Insufficient tissue coverage: {h['tissue_coverage']}"


def test_greedy_place_hpfs_prioritizes_dense_hotspots():
    ny, nx = 100, 100
    stride = 16.0
    slide_w_um = nx * stride
    slide_h_um = ny * stride

    tissue_mask = np.ones((100, 100), dtype=np.uint8) * 255
    density_map = np.ones((ny, nx), dtype=np.float32)

    grid_meta = {
        "origin_um": [0.0, 0.0],
        "stride_um": stride,
        "nx": nx,
        "ny": ny
    }

    # Hotspot A (lower priority 0.50): centered at (300, 300)
    hs_a = [[100.0, 100.0], [500.0, 100.0], [500.0, 500.0], [100.0, 500.0]]
    # Hotspot B (higher priority 0.95): centered at (1200, 1200)
    hs_b = [[1000.0, 1000.0], [1400.0, 1000.0], [1400.0, 1400.0], [1000.0, 1400.0]]

    hpfs = greedy_place_hpfs(
        density_map,
        grid_meta,
        hotspot_polygons_um=[hs_a, hs_b],
        hotspot_priorities=[0.50, 0.95],
        count=2,
        radius_um=262.0,
        min_separation_um=524.0,
        tissue_mask=tissue_mask,
        slide_dimensions_um=(slide_w_um, slide_h_um)
    )

    assert len(hpfs) == 2
    # HPF 1 must be from the higher priority hotspot (Hotspot B, cx >= 1000)
    assert hpfs[0]["center_um"][0] >= 1000.0, f"Expected HPF 1 in Hotspot B (>= 1000 um), got {hpfs[0]['center_um']}"

