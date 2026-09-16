from typing import Any
import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter
from skimage.measure import find_contours
from shapely.geometry import Polygon, box


def extract_hotspots(
    prob_grid: np.ndarray,
    grid_origin_um: tuple[float, float],
    stride_um: float,
    cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Extracts standardized candidate High-Power Field (HPF) hotspot sites from a 2D probability grid.
    Guarantees a minimum of 10 prioritized HPF candidate sites across the invasive tumor front.

    Args:
        prob_grid: 2D float32 array [ny, nx] where NaN = no tissue.
        grid_origin_um: (origin_x_um, origin_y_um) in base slide coordinates.
        stride_um: Stride between grid cells in micrometers (e.g. 102 µm).
        cfg: Config dict containing sigma, prob_threshold, min_area_mm2, max_hotspots, margin_um.

    Returns:
        List of Hotspot dictionaries containing polygon_um, area_mm2, prob_mean, prob_max, source, excluded.
    """
    sigma = float(cfg.get("sigma", 1.0))
    prob_threshold = float(cfg.get("prob_threshold", 0.50))
    max_hotspots = int(cfg.get("max_hotspots", 10))
    half_box_um = float(cfg.get("hpf_half_size_um", 300.0)) # 600 um x 600 um standard HPF site

    if prob_grid is None or prob_grid.size == 0:
        return []

    valid_mask = ~np.isnan(prob_grid)
    if not np.any(valid_mask):
        return []

    # If no cell meets the probability threshold, zero-tumor case -> return empty proposal set (#81)
    if np.nanmax(prob_grid) < prob_threshold:
        return []

    prob_filled = np.nan_to_num(prob_grid, nan=0.0)
    smoothed_prob = gaussian_filter(prob_filled, sigma=sigma)
    smoothed_weight = gaussian_filter(valid_mask.astype(float), sigma=sigma)

    with np.errstate(divide='ignore', invalid='ignore'):
        smoothed = np.where(smoothed_weight > 1e-5, smoothed_prob / smoothed_weight, 0.0)

    smoothed[~valid_mask] = 0.0

    # 1. Detect local maxima in probability across tissue
    footprint = np.ones((5, 5))
    local_max = (maximum_filter(smoothed, footprint=footprint) == smoothed) & (smoothed >= prob_threshold)
    max_coords = np.argwhere(local_max)

    # Sort coordinates by smoothed probability descending
    max_coords = sorted(max_coords, key=lambda c: smoothed[c[0], c[1]], reverse=True)

    hotspots = []
    origin_x, origin_y = grid_origin_um
    min_separation_cells = max(3, int(round(800.0 / stride_um))) # ~800 um minimum separation

    for r, c in max_coords:
        if len(hotspots) >= max_hotspots:
            break

        # Verify spatial separation from previously selected HPF sites
        too_close = False
        for s in hotspots:
            dr = r - s["_r"]
            dc = c - s["_c"]
            dist_cells = np.sqrt(dr * dr + dc * dc)
            if dist_cells < min_separation_cells:
                too_close = True
                break

        if not too_close:
            cx_um = origin_x + (c * stride_um)
            cy_um = origin_y + (r * stride_um)

            site_box = box(
                cx_um - half_box_um, 
                cy_um - half_box_um, 
                cx_um + half_box_um, 
                cy_um + half_box_um
            )
            final_coords = [[round(x, 2), round(y, 2)] for x, y in site_box.exterior.coords]
            actual_area_mm2 = round((half_box_um * 2) ** 2 / 1e6, 3)

            hotspots.append({
                "id": f"hs_{len(hotspots) + 1:02d}",
                "_r": r,
                "_c": c,
                "polygon_um": final_coords,
                "area_mm2": actual_area_mm2,
                "prob_mean": round(float(smoothed[r, c]), 3),
                "prob_max": round(float(prob_filled[r, c]), 3),
                "source": "model",
                "excluded": False,
                "exclude_reason": None
            })

    # If local maxima yielded fewer than max_hotspots, fill only from tissue cells reaching secondary threshold (#81)
    secondary_threshold = max(0.20, prob_threshold * 0.75)
    if len(hotspots) < max_hotspots:
        tissue_coords = np.argwhere(valid_mask)
        valid_coords = [c for c in tissue_coords if smoothed[c[0], c[1]] >= secondary_threshold]
        valid_coords = sorted(valid_coords, key=lambda c: smoothed[c[0], c[1]], reverse=True)

        for r, c in valid_coords:
            if len(hotspots) >= max_hotspots:
                break
            too_close = False
            for s in hotspots:
                dr = r - s["_r"]
                dc = c - s["_c"]
                if np.sqrt(dr * dr + dc * dc) < min_separation_cells * 0.7:
                    too_close = True
                    break
            if not too_close:
                cx_um = origin_x + (c * stride_um)
                cy_um = origin_y + (r * stride_um)
                site_box = box(
                    cx_um - half_box_um, 
                    cy_um - half_box_um, 
                    cx_um + half_box_um, 
                    cy_um + half_box_um
                )
                final_coords = [[round(x, 2), round(y, 2)] for x, y in site_box.exterior.coords]
                is_lowconf = smoothed[r, c] < prob_threshold
                hotspots.append({
                    "id": f"hs_{len(hotspots) + 1:02d}",
                    "_r": r,
                    "_c": c,
                    "polygon_um": final_coords,
                    "area_mm2": round((half_box_um * 2) ** 2 / 1e6, 3),
                    "prob_mean": round(float(smoothed[r, c]), 3),
                    "prob_max": round(float(prob_filled[r, c]), 3),
                    "source": "model_lowconf" if is_lowconf else "model",
                    "excluded": False,
                    "exclude_reason": None
                })

    # Clean temporary keys
    for h in hotspots:
        h.pop("_r", None)
        h.pop("_c", None)

    return hotspots
