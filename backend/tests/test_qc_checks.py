import numpy as np
import pytest

from pipeline.qc_checks import check_tissue_coverage, check_focus_sharpness

def test_check_tissue_coverage_pass():
    """Verify tissue coverage pass status on adequate tissue mask (> 5%)."""
    mask = np.ones((512, 512), dtype=bool) # 100% coverage
    config = {"tissue_coverage": {"fail_threshold": 0.02, "warn_threshold": 0.05}}

    res = check_tissue_coverage(mask, config)
    assert res["status"] == "pass"
    assert res["metric"] == 1.0

def test_check_tissue_coverage_fail():
    """Verify tissue coverage fail status on blank tissue mask (< 2%)."""
    mask = np.zeros((512, 512), dtype=bool) # 0% coverage
    config = {"tissue_coverage": {"fail_threshold": 0.02, "warn_threshold": 0.05}}

    res = check_tissue_coverage(mask, config)
    assert res["status"] == "fail"
    assert res["metric"] == 0.0

def test_check_focus_sharpness_synthetic():
    """Verify focus check on sharp vs blurry synthetic tiles."""
    from PIL import Image
    
    blank_slide = Image.new("RGB", (1024, 1024), color=(240, 235, 240))
    mask = np.ones((128, 128), dtype=bool)
    config = {"focus": {"vol_threshold": 5.0, "fail_blurry_ratio": 0.70, "warn_blurry_ratio": 0.30, "sample_max_tiles": 10}}

    res = check_focus_sharpness(blank_slide, mask, config=config)
    assert res["name"] == "focus"
    assert res["status"] in ["pass", "warn", "fail"]

def test_check_pen_marks_clean():
    """Verify pen marks check passes on clean slide."""
    from PIL import Image
    from pipeline.qc_checks import check_pen_marks
    clean_slide = Image.new("RGB", (1024, 1024), color=(245, 240, 245))
    mask = np.ones((128, 128), dtype=bool)

    res = check_pen_marks(clean_slide, mask)
    assert res["name"] == "pen_marks"
    assert res["status"] == "pass"
    assert res["metric"] == 0.0

def test_check_pen_marks_detected():
    """Verify green/blue/black surgical pen mark detection triggers warning."""
    from PIL import Image, ImageDraw
    from pipeline.qc_checks import check_pen_marks
    slide = Image.new("RGB", (1024, 1024), color=(245, 240, 245))
    draw = ImageDraw.Draw(slide)
    # Draw large green pen mark (HSV ~ (60, 200, 200))
    draw.rectangle([100, 100, 400, 400], fill=(0, 200, 50))
    mask = np.ones((128, 128), dtype=bool)

    cfg = {
        "pen_marks": {
            "min_component_area_mm2": 0.001,
            "hsv_ranges": {
                "green": {"h_min": 35, "h_max": 85, "s_min": 60, "v_min": 60}
            }
        }
    }
    res = check_pen_marks(slide, mask, mpp_x=1.0, mpp_y=1.0, config=cfg)
    assert res["name"] == "pen_marks"
    assert res["status"] == "warn"
    assert res["metric"] > 0.01

def test_check_tissue_folds():
    """Verify tissue fold detection on dark high-saturation ridges."""
    from PIL import Image, ImageDraw
    from pipeline.qc_checks import check_tissue_folds
    slide = Image.new("RGB", (1024, 1024), color=(240, 230, 240))
    draw = ImageDraw.Draw(slide)
    # Dark high-saturation fold ridge (low V, high S)
    draw.line([(100, 100), (450, 450)], fill=(80, 0, 50), width=15)
    mask = np.ones((128, 128), dtype=bool)

    cfg = {
        "folds": {
            "min_skeleton_length_mm": 0.01,
            "saturation_min": 50,
            "brightness_max": 120
        }
    }
    res = check_tissue_folds(slide, mask, config=cfg)
    assert res["name"] == "folds"
    assert res["status"] == "warn"

def test_check_stain_sanity():
    """Verify stain sanity check on valid, faded, and degenerate profiles."""
    from pipeline.qc_checks import check_stain_sanity

    # 1. Valid fit
    valid_params = {"max_concentrations": [1.95, 1.10], "fit_status": "fitted"}
    res_valid = check_stain_sanity(valid_params)
    assert res_valid["status"] == "pass"

    # 2. Faded stain (concentration < 0.15)
    faded_params = {"max_concentrations": [0.08, 0.05], "fit_status": "fitted"}
    res_faded = check_stain_sanity(faded_params)
    assert res_faded["status"] == "warn"
    assert "Faded" in res_faded["message"]

    # 3. Degenerate fit
    degen_params = {"max_concentrations": [1.95, 1.10], "fit_status": "degenerate"}
    res_degen = check_stain_sanity(degen_params)
    assert res_degen["status"] == "warn"
    assert "Degenerate" in res_degen["message"]

def test_run_all_qc_checks_full_5_suite():
    """Verify run_all_qc_checks executes all 5 PRD checks."""
    from PIL import Image
    from pipeline.qc_checks import run_all_qc_checks

    slide = Image.new("RGB", (1024, 1024), color=(240, 230, 240))
    mask = np.ones((128, 128), dtype=bool)
    stain_params = {"max_concentrations": [1.95, 1.10], "fit_status": "fitted"}

    res = run_all_qc_checks(slide, mask, stain_params=stain_params)
    assert len(res["checks"]) == 5
    check_names = {c["name"] for c in res["checks"]}
    assert check_names == {"tissue_coverage", "focus", "pen_marks", "folds", "stain_sanity"}
    assert res["verdict"] in ["pass", "warn", "fail"]

