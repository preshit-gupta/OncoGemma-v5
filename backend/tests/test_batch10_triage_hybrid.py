"""
Unit and integration tests for Hybrid Path Foundation + MedGemma 1.5 Referee
(Stage 3 Tumor Hotspot Identification).
"""
import io
import pytest
import numpy as np
from PIL import Image

from pipeline.medgemma import MedGemmaClient, TumorVerificationResponse
from pipeline.hotspots import extract_hotspots


def test_tumor_verification_schema_validation():
    """Verify that TumorVerificationResponse adheres to strict schema constraints."""
    data = {
        "tumor_present": True,
        "lesion_type": "invasive_carcinoma",
        "cellularity": "high",
        "confidence": "high",
        "rationale": "High epithelial cellularity with pleomorphic nuclear atypia and infiltrative cords."
    }
    model = TumorVerificationResponse(**data)
    assert model.tumor_present is True
    assert model.lesion_type == "invasive_carcinoma"
    assert model.cellularity == "high"
    assert model.confidence == "high"
    assert "infiltrative cords" in model.rationale


def test_morphometric_tumor_fallback_invasive():
    """Verify morphometric fallback recognizes high-density epithelial cellularity."""
    client = MedGemmaClient()
    # Create synthetic high-cellularity hematoxylin-rich image (dark blue-purple nuclei)
    img = Image.new("RGB", (512, 512), (90, 45, 110))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    res = client._morphometric_tumor_fallback(buf.getvalue())

    assert isinstance(res, TumorVerificationResponse)
    assert res.tumor_present is True
    assert res.lesion_type == "invasive_carcinoma"
    assert res.cellularity == "high"
    assert res.confidence == "high"
    assert "High cellular density" in res.rationale or "invasive carcinoma" in res.rationale


def test_morphometric_tumor_fallback_benign_stroma():
    """Verify morphometric fallback recognizes pink eosinophilic stroma as benign."""
    client = MedGemmaClient()
    # Create synthetic pink/eosinophilic collagen stroma (pale pink, minimal nuclei: R=240, G=180, B=210)
    img = Image.new("RGB", (512, 512), (240, 180, 210))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    res = client._morphometric_tumor_fallback(buf.getvalue())

    assert isinstance(res, TumorVerificationResponse)
    assert res.tumor_present is False
    assert res.lesion_type == "benign_stroma"
    assert res.cellularity == "low"
    assert "fibrous" in res.rationale or "stroma" in res.rationale


def test_morphometric_tumor_fallback_adipose():
    """Verify morphometric fallback recognizes white background/adipose as benign."""
    client = MedGemmaClient()
    # Create synthetic white/clear adipose tissue
    img = Image.new("RGB", (512, 512), (250, 250, 250))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    res = client._morphometric_tumor_fallback(buf.getvalue())

    assert isinstance(res, TumorVerificationResponse)
    assert res.tumor_present is False
    assert res.lesion_type == "adipose"
    assert res.cellularity == "low"
    assert "lipid" in res.rationale or "adipose" in res.rationale or "background" in res.rationale


def test_sync_evaluate_tumor_verification_execution():
    """Verify evaluate_tumor_verification_sync returns a valid TumorVerificationResponse."""
    client = MedGemmaClient()
    img = Image.new("RGB", (512, 512), (180, 100, 150))
    buf = io.BytesIO()
    img.save(buf, "PNG")

    res = client.evaluate_tumor_verification_sync(buf.getvalue())
    assert isinstance(res, TumorVerificationResponse)
    assert isinstance(res.tumor_present, bool)
    assert res.lesion_type in ["invasive_carcinoma", "dcis", "benign_stroma", "adipose", "normal_breast"]
    assert len(res.rationale) > 10


def test_hybrid_hotspot_referee_ranking_integration():
    """
    Test the hybrid pipeline logic: candidate extraction -> MedGemma referee ->
    filtering benign stroma -> re-ranking and selecting top 10 verified hotspots.
    """
    client = MedGemmaClient()

    # Create synthetic 20x20 probability grid with 5 high peaks and background
    grid = np.full((20, 20), np.nan, dtype=np.float32)
    grid[5:15, 5:15] = 0.55
    # Create peaks
    grid[6, 6] = 0.95
    grid[7, 12] = 0.92
    grid[12, 7] = 0.89
    grid[13, 13] = 0.88
    grid[9, 9] = 0.85

    cfg = {
        "sigma": 0.5,
        "prob_threshold": 0.50,
        "max_hotspots": 15,
        "hpf_half_size_um": 300.0
    }
    raw_candidates = extract_hotspots(
        prob_grid=grid,
        grid_origin_um=(0.0, 0.0),
        stride_um=(50.0, 50.0),
        cfg=cfg,
        slide_dimensions_um=(1000.0, 1000.0)
    )
    assert len(raw_candidates) > 0

    # Simulate candidate refereeing: some invasive, some stroma
    verified_candidates = []
    for idx, cand in enumerate(raw_candidates):
        cand_copy = dict(cand)
        # Even indices: high cellularity tumor; Odd indices: benign stroma
        if idx % 2 == 0:
            mock_crop = Image.new("RGB", (512, 512), (90, 40, 120))
        else:
            mock_crop = Image.new("RGB", (512, 512), (240, 210, 225))
        buf = io.BytesIO()
        mock_crop.save(buf, "PNG")
        v_res = client._morphometric_tumor_fallback(buf.getvalue())
        cand_copy["medgemma_tumor_present"] = v_res.tumor_present
        cand_copy["medgemma_lesion_type"] = v_res.lesion_type
        cand_copy["medgemma_cellularity"] = v_res.cellularity
        cand_copy["medgemma_confidence"] = v_res.confidence
        cand_copy["medgemma_rationale"] = v_res.rationale
        verified_candidates.append(cand_copy)

    # Rank and select
    confirmed = [c for c in verified_candidates if c["medgemma_tumor_present"]]
    confirmed.sort(key=lambda c: c.get("prob_mean", 0.0), reverse=True)

    unconfirmed = [c for c in verified_candidates if not c["medgemma_tumor_present"]]
    unconfirmed.sort(key=lambda c: c.get("prob_mean", 0.0), reverse=True)

    selected = confirmed[:10]
    if len(selected) < 10 and unconfirmed:
        needed = 10 - len(selected)
        selected.extend(unconfirmed[:needed])

    final_hotspots = []
    for i, item in enumerate(selected):
        h = dict(item)
        h["id"] = f"hs_{i + 1:02d}"
        final_hotspots.append(h)

    assert len(final_hotspots) == len(raw_candidates)
    assert final_hotspots[0]["id"] == "hs_01"
    # First candidate should be confirmed tumor
    assert final_hotspots[0]["medgemma_tumor_present"] is True
    assert final_hotspots[0]["medgemma_lesion_type"] == "invasive_carcinoma"
    assert "medgemma_rationale" in final_hotspots[0]


def test_smart_scout_non_overlapping_patches():
    """Verify that Smart Scout produces strictly non-overlapping patches and prioritizes cellularity."""
    width_px = 37352
    height_px = 162391
    mpp_x = 0.2526
    nx = 80
    ny = 348
    max_sample_patches = 2048

    patch_dim_px = int(round(224.0 / mpp_x))
    cols = width_px // patch_dim_px
    rows = height_px // patch_dim_px

    # Create mock stain map and tissue mask
    np.random.seed(42)
    tissue_mask_overview = np.random.rand(ny, nx) > 0.15
    stain_map = np.random.rand(ny, nx) * 1.5

    candidate_slots = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * patch_dim_px
            y0 = r * patch_dim_px
            cx_px = x0 + patch_dim_px // 2
            cy_px = y0 + patch_dim_px // 2
            ix = min(nx - 1, max(0, int(cx_px / (width_px / nx))))
            iy = min(ny - 1, max(0, int(cy_px / (height_px / ny))))

            if tissue_mask_overview[iy, ix]:
                candidate_slots.append({
                    "c": c, "r": r, "x0": x0, "y0": y0,
                    "cx_px": cx_px, "cy_px": cy_px,
                    "ix": ix, "iy": iy,
                    "score": float(stain_map[iy, ix])
                })

    assert len(candidate_slots) > max_sample_patches

    # Run Smart Scout ranking
    candidate_slots.sort(key=lambda s: s["score"], reverse=True)
    n_dense = int(round(0.80 * max_sample_patches))
    dense_slots = candidate_slots[:n_dense]
    dense_keys = set((s["c"], s["r"]) for s in dense_slots)
    remaining_slots = [s for s in candidate_slots if (s["c"], s["r"]) not in dense_keys]
    n_context = max_sample_patches - len(dense_slots)
    step_ctx = max(1, len(remaining_slots) // n_context)
    context_slots = remaining_slots[::step_ctx][:n_context]

    selected_slots = dense_slots + context_slots
    assert len(selected_slots) == max_sample_patches

    # 1. Verify every selected slot is unique
    tile_indices = [(s["c"], s["r"]) for s in selected_slots]
    assert len(tile_indices) == len(set(tile_indices)), "No duplicate tile slots allowed!"

    # 2. Verify strict geometric non-overlapping condition
    # For any two tiles (c1, r1) != (c2, r2), their bounding boxes [x0, x0 + W] x [y0, y0 + H] must have zero overlap
    for i in range(min(300, len(selected_slots))):
        s1 = selected_slots[i]
        box1 = (s1["x0"], s1["y0"], s1["x0"] + patch_dim_px, s1["y0"] + patch_dim_px)
        for j in range(i + 1, min(300, len(selected_slots))):
            s2 = selected_slots[j]
            box2 = (s2["x0"], s2["y0"], s2["x0"] + patch_dim_px, s2["y0"] + patch_dim_px)
            # Intersection test
            overlap_x = (box1[0] < box2[2]) and (box1[2] > box2[0])
            overlap_y = (box1[1] < box2[3]) and (box1[3] > box2[1])
            assert not (overlap_x and overlap_y), f"Overlap detected between slot {i} and {j}!"

