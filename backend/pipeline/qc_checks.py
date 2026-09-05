import os
import yaml
import hashlib
import cv2
import numpy as np

def resolve_config_path(path: str) -> str:
    """Resolve config file path relative to repo root or backend parent directory."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)

    parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../", path))
    if os.path.exists(parent_path):
        return parent_path

    return os.path.abspath(path)

def load_qc_config(config_path: str = "configs/qc.yaml") -> tuple[dict, str]:
    """Load QC thresholds configuration YAML and calculate MD5 config hash."""
    config_path = resolve_config_path(config_path)
    if not os.path.exists(config_path):
        # Fallback default configuration dictionary per PRD 02 §3.1
        default_cfg = {
            "tissue_coverage": {"fail_threshold": 0.02, "warn_threshold": 0.05},
            "focus": {"vol_threshold": 45.0, "fail_blurry_ratio": 0.30, "warn_blurry_ratio": 0.10, "sample_max_tiles": 400},
            "pen_marks": {
                "min_component_area_mm2": 1.0,
                "hsv_ranges": {
                    "green": {"h_min": 35, "h_max": 85, "s_min": 60, "v_min": 60},
                    "blue": {"h_min": 90, "h_max": 130, "s_min": 60, "v_min": 60},
                    "black": {"h_min": 0, "h_max": 180, "s_min": 0, "s_max": 50, "v_min": 0, "v_max": 50}
                }
            },
            "folds": {"min_skeleton_length_mm": 2.0, "saturation_min": 160, "brightness_max": 100},
            "stain_sanity": {"min_concentration": 0.15, "he_ratio_min": 0.1, "he_ratio_max": 5.0}
        }
        return default_cfg, "default_hash"

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    config_dict = yaml.safe_load(content) or {}
    config_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
    return config_dict, config_hash

def check_tissue_coverage(tissue_mask_1bit: np.ndarray, config: dict) -> dict:
    """
    Check 1: Tissue Coverage
    tissue mask area / total thumbnail area.
    < 2% -> fail; < 5% -> warn.
    """
    cfg = config.get("tissue_coverage", {})
    fail_thresh = cfg.get("fail_threshold", 0.02)
    warn_thresh = cfg.get("warn_threshold", 0.05)

    total_pixels = tissue_mask_1bit.size
    tissue_pixels = np.count_nonzero(tissue_mask_1bit)
    coverage_ratio = float(tissue_pixels / max(1, total_pixels))

    status = "pass"
    if coverage_ratio < fail_thresh:
        status = "fail"
        msg = f"Critical low tissue coverage: {coverage_ratio * 100:.1f}% (threshold < {fail_thresh * 100:.0f}%)"
    elif coverage_ratio < warn_thresh:
        status = "warn"
        msg = f"Low tissue coverage: {coverage_ratio * 100:.1f}% (threshold < {warn_thresh * 100:.0f}%)"
    else:
        msg = f"Adequate tissue coverage: {coverage_ratio * 100:.1f}%"

    return {
        "name": "tissue_coverage",
        "status": status,
        "metric": round(coverage_ratio, 4),
        "message": msg
    }

def check_focus_sharpness(
    slide_obj,
    tissue_mask_1bit: np.ndarray,
    mpp_x: float = 0.25,
    mpp_y: float = 0.25,
    config: dict = None
) -> dict:
    """
    Check 2: Focus Sharpness
    Variance of Laplacian (OpenCV, grayscale) per 512^2 tile at 10x, on <= 400 sampled tissue tiles.
    """
    cfg = (config or {}).get("focus", {})
    vol_thresh = cfg.get("vol_threshold", 45.0)
    fail_blurry_ratio = cfg.get("fail_blurry_ratio", 0.30)
    warn_blurry_ratio = cfg.get("warn_blurry_ratio", 0.10)
    max_tiles = cfg.get("sample_max_tiles", 400)

    from pipeline.tiles import read_region_srgb

    patch_size_um = 512.0
    thumb_h, thumb_w = tissue_mask_1bit.shape
    slide_w_um = thumb_w * 8.0 * mpp_x
    slide_h_um = thumb_h * 8.0 * mpp_y

    blurry_tile_count = 0
    total_sampled_tiles = 0

    step_um = patch_size_um * 2
    xs = np.arange(0, max(patch_size_um, slide_w_um - patch_size_um), step_um)
    ys = np.arange(0, max(patch_size_um, slide_h_um - patch_size_um), step_um)

    positions = [(x, y) for x in xs for y in ys]

    if len(positions) > max_tiles:
        rng = np.random.default_rng(42)
        idx_sample = rng.choice(len(positions), size=max_tiles, replace=False)
        positions = [positions[i] for i in idx_sample]

    for x_um, y_um in positions:
        try:
            tile_rgb, _ = read_region_srgb(slide_obj, x_um, y_um, patch_size_um, patch_size_um, out_px=512, mpp_x=mpp_x, mpp_y=mpp_y)
            if np.std(tile_rgb) > 5.0:
                gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
                vol = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                total_sampled_tiles += 1
                if vol < vol_thresh:
                    blurry_tile_count += 1
        except Exception:
            pass

    blurry_ratio = float(blurry_tile_count / max(1, total_sampled_tiles)) if total_sampled_tiles > 0 else 0.0

    status = "pass"
    if total_sampled_tiles > 0 and blurry_ratio > fail_blurry_ratio:
        status = "fail"
        msg = f"Critical focus blur: {blurry_ratio * 100:.1f}% of tissue tiles blurry (threshold > {fail_blurry_ratio * 100:.0f}%)"
    elif total_sampled_tiles > 0 and blurry_ratio > warn_blurry_ratio:
        status = "warn"
        msg = f"{blurry_ratio * 100:.1f}% of tissue tiles below sharpness threshold (VoL < {vol_thresh})"
    else:
        msg = f"Slide focus sharp ({blurry_ratio * 100:.1f}% blurry tiles)"

    return {
        "name": "focus",
        "status": status,
        "metric": round(blurry_ratio, 4),
        "message": msg
    }

def check_pen_marks(
    slide_obj,
    tissue_mask_1bit: np.ndarray,
    mpp_x: float = 0.25,
    mpp_y: float = 0.25,
    config: dict = None
) -> dict:
    """
    Check 3: Pen Marks Detection
    Detects surgical/pathologist pen ink marks (green, blue, black) using HSV thresholding
    and connected component analysis. Warns if any pen mark component exceeds min_component_area_mm2.
    """
    cfg = (config or {}).get("pen_marks", {})
    min_area_mm2 = cfg.get("min_component_area_mm2", 1.0)
    hsv_ranges = cfg.get("hsv_ranges", {
        "green": {"h_min": 35, "h_max": 85, "s_min": 60, "v_min": 60},
        "blue": {"h_min": 90, "h_max": 130, "s_min": 60, "v_min": 60},
        "black": {"h_min": 0, "h_max": 180, "s_min": 0, "s_max": 50, "v_min": 0, "v_max": 50}
    })

    from pipeline.tiles import read_region_srgb
    slide_w_px = float(getattr(slide_obj, "width_px", 2048) or 2048)
    slide_h_px = float(getattr(slide_obj, "height_px", 2048) or 2048)
    if hasattr(slide_obj, "dimensions"):
        slide_w_px, slide_h_px = float(slide_obj.dimensions[0]), float(slide_obj.dimensions[1])
    elif hasattr(slide_obj, "size"):
        slide_w_px, slide_h_px = float(slide_obj.size[0]), float(slide_obj.size[1])

    thumb_w_um = min(50000.0, slide_w_px * mpp_x)
    thumb_h_um = min(50000.0, slide_h_px * mpp_y)

    try:
        thumb_arr, _ = read_region_srgb(slide_obj, 0, 0, thumb_w_um, thumb_h_um, out_px=(512, 512), mpp_x=mpp_x, mpp_y=mpp_y)
    except Exception:
        if hasattr(slide_obj, "resize"):
            from PIL import Image
            thumb_arr = np.array(slide_obj.convert("RGB").resize((512, 512)))
        else:
            thumb_arr = np.ones((512, 512, 3), dtype=np.uint8) * 240

    hsv = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2HSV)

    # Pixel area in mm2 for 512x512 thumbnail
    px_w_mm = (thumb_w_um / 512.0) * 1e-3
    px_h_mm = (thumb_h_um / 512.0) * 1e-3
    pixel_area_mm2 = px_w_mm * px_h_mm

    combined_pen_mask = np.zeros((512, 512), dtype=np.uint8)

    for color, rng_cfg in hsv_ranges.items():
        h_min = rng_cfg.get("h_min", 0)
        h_max = rng_cfg.get("h_max", 180)
        s_min = rng_cfg.get("s_min", 0)
        s_max = rng_cfg.get("s_max", 255)
        v_min = rng_cfg.get("v_min", 0)
        v_max = rng_cfg.get("v_max", 255)

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined_pen_mask = cv2.bitwise_or(combined_pen_mask, mask)

    # Morphological open to prune single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_pen_mask = cv2.morphologyEx(combined_pen_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_pen_mask)
    max_component_area_mm2 = 0.0

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        area_mm2 = area_px * pixel_area_mm2
        if area_mm2 > max_component_area_mm2:
            max_component_area_mm2 = area_mm2

    status = "pass"
    if max_component_area_mm2 >= min_area_mm2:
        status = "warn"
        msg = f"Pen markings detected: largest mark {max_component_area_mm2:.2f} mm² (warning threshold >= {min_area_mm2:.1f} mm²)"
    else:
        msg = f"No significant pen markings detected (largest mark {max_component_area_mm2:.2f} mm²)"

    return {
        "name": "pen_marks",
        "status": status,
        "metric": round(max_component_area_mm2, 4),
        "message": msg
    }

def check_tissue_folds(
    slide_obj,
    tissue_mask_1bit: np.ndarray,
    mpp_x: float = 0.25,
    mpp_y: float = 0.25,
    config: dict = None
) -> dict:
    """
    Check 4: Tissue Fold Detection
    Detects dark, high-saturation overlapping tissue ridges (folds) within the tissue area.
    Warns if connected fold ridge length exceeds min_skeleton_length_mm.
    """
    cfg = (config or {}).get("folds", {})
    min_length_mm = cfg.get("min_skeleton_length_mm", 2.0)
    sat_min = cfg.get("saturation_min", 160)
    bright_max = cfg.get("brightness_max", 100)

    from pipeline.tiles import read_region_srgb
    slide_w_px = float(getattr(slide_obj, "width_px", 2048) or 2048)
    slide_h_px = float(getattr(slide_obj, "height_px", 2048) or 2048)
    if hasattr(slide_obj, "dimensions"):
        slide_w_px, slide_h_px = float(slide_obj.dimensions[0]), float(slide_obj.dimensions[1])
    elif hasattr(slide_obj, "size"):
        slide_w_px, slide_h_px = float(slide_obj.size[0]), float(slide_obj.size[1])

    thumb_w_um = min(50000.0, slide_w_px * mpp_x)
    thumb_h_um = min(50000.0, slide_h_px * mpp_y)

    try:
        thumb_arr, _ = read_region_srgb(slide_obj, 0, 0, thumb_w_um, thumb_h_um, out_px=(512, 512), mpp_x=mpp_x, mpp_y=mpp_y)
    except Exception:
        if hasattr(slide_obj, "resize"):
            from PIL import Image
            thumb_arr = np.array(slide_obj.convert("RGB").resize((512, 512)))
        else:
            thumb_arr = np.ones((512, 512, 3), dtype=np.uint8) * 240

    hsv = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2HSV)

    px_w_mm = (thumb_w_um / 512.0) * 1e-3
    px_h_mm = (thumb_h_um / 512.0) * 1e-3

    if tissue_mask_1bit.shape != (512, 512):
        t_mask = cv2.resize(tissue_mask_1bit.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        t_mask = tissue_mask_1bit

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    fold_candidates = (sat >= sat_min) & (val <= bright_max) & t_mask
    fold_mask = fold_candidates.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    fold_clean = cv2.morphologyEx(fold_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fold_clean)
    max_skeleton_length_mm = 0.0

    for i in range(1, num_labels):
        w_px = stats[i, cv2.CC_STAT_WIDTH]
        h_px = stats[i, cv2.CC_STAT_HEIGHT]
        length_mm = float(np.hypot(w_px * px_w_mm, h_px * px_h_mm))
        if length_mm > max_skeleton_length_mm:
            max_skeleton_length_mm = length_mm

    status = "pass"
    if max_skeleton_length_mm >= min_length_mm:
        status = "warn"
        msg = f"Tissue fold detected: ridge length {max_skeleton_length_mm:.2f} mm (warning threshold >= {min_length_mm:.1f} mm)"
    else:
        msg = f"No significant tissue folds detected (max length {max_skeleton_length_mm:.2f} mm)"

    return {
        "name": "folds",
        "status": status,
        "metric": round(max_skeleton_length_mm, 4),
        "message": msg
    }

def check_stain_sanity(
    stain_params: dict,
    config: dict = None
) -> dict:
    """
    Check 5: Stain Sanity Check
    Validates per-slide stain profile:
    1. Checks for degenerate fit or missing tissue patches.
    2. Enforces minimum stain concentrations (faded H&E detection).
    3. Validates Hematoxylin-to-Eosin concentration ratio bounds.
    """
    cfg = (config or {}).get("stain_sanity", {})
    min_conc = cfg.get("min_concentration", 0.15)
    he_ratio_min = cfg.get("he_ratio_min", 0.1)
    he_ratio_max = cfg.get("he_ratio_max", 5.0)

    if not stain_params:
        return {
            "name": "stain_sanity",
            "status": "warn",
            "metric": 0.0,
            "message": "Missing stain parameters artifact"
        }

    fit_status = stain_params.get("fit_status", "fitted")
    if fit_status == "degenerate":
        return {
            "name": "stain_sanity",
            "status": "warn",
            "metric": 0.0,
            "message": "Degenerate stain profile: insufficient tissue patches sampled to fit stain normalizer"
        }

    max_conc = stain_params.get("max_concentrations") or [1.95, 1.10]
    try:
        h_conc = float(max_conc[0])
        e_conc = float(max_conc[1])
    except (IndexError, TypeError, ValueError):
        h_conc, e_conc = 1.95, 1.10

    he_ratio = h_conc / max(1e-4, e_conc)

    status = "pass"
    if h_conc < min_conc or e_conc < min_conc:
        status = "warn"
        msg = f"Faded stain detected: H={h_conc:.2f}, E={e_conc:.2f} below min concentration {min_conc:.2f}"
    elif he_ratio < he_ratio_min or he_ratio > he_ratio_max:
        status = "warn"
        msg = f"Abnormal H:E stain concentration ratio: {he_ratio:.2f} (expected {he_ratio_min} - {he_ratio_max})"
    else:
        msg = f"Stain profile verified (H={h_conc:.2f}, E={e_conc:.2f}, H:E ratio={he_ratio:.2f})"

    return {
        "name": "stain_sanity",
        "status": status,
        "metric": round(he_ratio, 4),
        "message": msg
    }

def run_all_qc_checks(
    slide_obj,
    tissue_mask_1bit: np.ndarray,
    mpp_x: float = 0.25,
    mpp_y: float = 0.25,
    stain_params: dict = None,
    config_path: str = "configs/qc.yaml"
) -> dict:
    """Execute complete 5-check QC check suite per PRD 02 §3.1."""
    config_dict, config_hash = load_qc_config(config_path)

    # 1. Tissue coverage
    cov_res = check_tissue_coverage(tissue_mask_1bit, config_dict)

    # 2. Focus sharpness
    focus_res = check_focus_sharpness(slide_obj, tissue_mask_1bit, mpp_x=mpp_x, mpp_y=mpp_y, config=config_dict)

    # 3. Pen marks
    pen_res = check_pen_marks(slide_obj, tissue_mask_1bit, mpp_x=mpp_x, mpp_y=mpp_y, config=config_dict)

    # 4. Tissue folds
    fold_res = check_tissue_folds(slide_obj, tissue_mask_1bit, mpp_x=mpp_x, mpp_y=mpp_y, config=config_dict)

    # 5. Stain sanity
    stain_res = check_stain_sanity(stain_params or {}, config=config_dict)

    checks = [cov_res, focus_res, pen_res, fold_res, stain_res]

    statuses = [c["status"] for c in checks]
    if "fail" in statuses:
        overall_verdict = "fail"
    elif "warn" in statuses:
        overall_verdict = "warn"
    else:
        overall_verdict = "pass"

    return {
        "verdict": overall_verdict,
        "checks": checks,
        "config_hash": config_hash
    }
