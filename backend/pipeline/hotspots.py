from typing import Any
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from shapely.geometry import box


def extract_hotspots(
    prob_grid: np.ndarray,
    grid_origin_um: tuple[float, float],
    stride_um: float | tuple[float, float] | list[float],
    cfg: dict[str, Any],
    slide_dimensions_um: tuple[float, float] | None = None
) -> list[dict[str, Any]]:
    """
    Extracts standardized candidate High-Power Field (HPF) hotspot sites from a 2D probability grid.
    Guarantees prioritized HPF candidate sites across the invasive tumor front.

    Args:
        prob_grid: 2D float32 array [ny, nx] where NaN = no tissue.
        grid_origin_um: (origin_x_um, origin_y_um) in base slide coordinates.
        stride_um: Stride between grid cells in micrometers (scalar or (stride_x, stride_y)).
        cfg: Config dict containing sigma, prob_threshold, max_hotspots, hpf_half_size_um.
        slide_dimensions_um: Optional (width_um, height_um) to clip boundary coordinates.

    Returns:
        List of Hotspot dictionaries containing polygon_um, area_mm2, prob_mean, prob_max, source, excluded.
    """
    sigma = float(cfg.get("sigma", 1.0))
    prob_threshold = float(cfg.get("prob_threshold", 0.50))
    max_hotspots = int(cfg.get("max_hotspots", 10))
    half_box_um = float(cfg.get("hpf_half_size_um", 300.0))  # Standard 600 µm x 600 µm HPF site

    if prob_grid is None or prob_grid.size == 0:
        return []

    ny, nx = prob_grid.shape

    # Support anisotropic strides (#82)
    if isinstance(stride_um, (tuple, list)):
        stride_x = float(stride_um[0])
        stride_y = float(stride_um[1]) if len(stride_um) > 1 else stride_x
    else:
        stride_x = float(stride_um)
        stride_y = float(stride_um)

    valid_mask = ~np.isnan(prob_grid)
    if not np.any(valid_mask):
        return []

    # If no cell meets the probability threshold, zero-tumor case -> return empty proposal set (#81)
    if np.nanmax(prob_grid) < prob_threshold:
        return []

    prob_filled = np.nan_to_num(prob_grid, nan=0.0)
    smoothed_prob = gaussian_filter(prob_filled, sigma=sigma)
    smoothed_weight = gaussian_filter(valid_mask.astype(float), sigma=sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
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
    mean_stride = (stride_x + stride_y) / 2.0
    min_separation_cells = max(3, int(round(800.0 / mean_stride)))  # ~800 µm minimum separation

    def _build_candidate(r: int, c: int, is_lowconf: bool = False) -> dict[str, Any]:
        # Authentic cell center alignment: (c + 0.5) * stride (#82)
        cx_um = origin_x + ((c + 0.5) * stride_x)
        cy_um = origin_y + ((r + 0.5) * stride_y)

        # Coordinate clipping to non-negative bounds and slide dimensions (#572)
        x_min = max(0.0, cx_um - half_box_um)
        y_min = max(0.0, cy_um - half_box_um)

        if slide_dimensions_um is not None:
            max_w_um, max_h_um = slide_dimensions_um
            x_max = min(max_w_um, cx_um + half_box_um)
            y_max = min(max_h_um, cy_um + half_box_um)
        else:
            x_max = min(origin_x + nx * stride_x, cx_um + half_box_um)
            y_max = min(origin_y + ny * stride_y, cy_um + half_box_um)

        site_box = box(x_min, y_min, x_max, y_max)
        final_coords = [[round(x, 2), round(y, 2)] for x, y in site_box.exterior.coords]
        actual_area_mm2 = round(((x_max - x_min) * (y_max - y_min)) / 1e6, 3)

        # Authentic regional statistics over hotspot footprint (#728)
        c_min = max(0, int(np.floor((x_min - origin_x) / max(stride_x, 1e-4))))
        c_max = min(nx, max(c_min + 1, int(np.ceil((x_max - origin_x) / max(stride_x, 1e-4)))))
        r_min = max(0, int(np.floor((y_min - origin_y) / max(stride_y, 1e-4))))
        r_max = min(ny, max(r_min + 1, int(np.ceil((y_max - origin_y) / max(stride_y, 1e-4)))))

        reg_smoothed = smoothed[r_min:r_max, c_min:c_max]
        reg_raw = prob_filled[r_min:r_max, c_min:c_max]
        reg_valid = valid_mask[r_min:r_max, c_min:c_max]

        if np.any(reg_valid):
            p_mean = round(float(np.mean(reg_smoothed[reg_valid])), 3)
            p_max = round(float(np.max(reg_raw[reg_valid])), 3)
        else:
            p_mean = round(float(smoothed[r, c]), 3)
            p_max = round(float(prob_filled[r, c]), 3)

        return {
            "id": f"hs_{len(hotspots) + 1:02d}",
            "_r": r,
            "_c": c,
            "polygon_um": final_coords,
            "area_mm2": actual_area_mm2,
            "prob_mean": p_mean,
            "prob_max": p_max,
            "source": "model_lowconf" if is_lowconf else "model",
            "excluded": False,
            "exclude_reason": None
        }

    for r, c in max_coords:
        if len(hotspots) >= max_hotspots:
            break

        too_close = False
        for s in hotspots:
            dr = r - s["_r"]
            dc = c - s["_c"]
            dist_cells = np.sqrt(dr * dr + dc * dc)
            if dist_cells < min_separation_cells:
                too_close = True
                break

        if not too_close:
            hotspots.append(_build_candidate(r, c, is_lowconf=False))

    # If local maxima yielded fewer than max_hotspots, fill from secondary threshold (#81)
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
                is_lowconf = smoothed[r, c] < prob_threshold
                hotspots.append(_build_candidate(r, c, is_lowconf=is_lowconf))

    # Clean internal coordinate tracking keys
    for h in hotspots:
        h.pop("_r", None)
        h.pop("_c", None)

    return hotspots
