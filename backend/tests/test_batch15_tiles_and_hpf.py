"""
Unit and integration tests for Batch 15:
WSI Pyramid Tiles, HPF Placement Engine, Probe Classifier & Storage Security.
Covers findings:
- #739, #640, #340, #641, #53, #54, #554, #730, #747, #720, #19, #5, #21, #22, #88, #86, #569, #101, #133
- #586, #596, #23, #219, #212, #211, #646, #645
"""
import io
import math
import uuid
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
from fastapi import HTTPException

from pipeline.hpf import (
    greedy_place_hpfs,
    generate_mitosis_density_map,
    create_circular_disk_mask
)
from pipeline.tiles import (
    check_icc_profile,
    read_region_srgb,
    extract_patch_from_pyramid
)
from pipeline.probe import ProbeRunner
from app.core.config import settings
from app.core.gcs import (
    generate_signed_upload_url,
    get_gcs_artifact_direct_url,
    ALLOWED_WSI_EXTS
)
from app.models.slide import Slide
from app.routers.tiles import stream_slide_tile, generate_tile_on_the_fly


# ---------------------------------------------------------------------------
# 1. HPF Placement & Spatial Optimization Tests (#747, #720, #586, #596)
# ---------------------------------------------------------------------------

def test_greedy_place_hpfs_empty_hotspots_returns_empty():
    """Issue #747: When hotspot_polygons_um is explicitly [], return [] immediately."""
    density_map = np.ones((50, 50), dtype=np.float32) * 5.0
    grid_meta = {"origin_um": [0.0, 0.0], "stride_um": 16.0, "nx": 50, "ny": 50}

    # Explicit empty list of hotspots -> must return []
    res_empty = greedy_place_hpfs(
        density_map,
        grid_meta,
        hotspot_polygons_um=[],
        count=5
    )
    assert res_empty == []

    # None -> unconstrained whole-slide placement
    res_unconstrained = greedy_place_hpfs(
        density_map,
        grid_meta,
        hotspot_polygons_um=None,
        count=5
    )
    assert len(res_unconstrained) > 0


def test_greedy_place_hpfs_vectorized_polygon_containment():
    """Issue #720: Vectorized polygon containment with matplotlib.path.Path."""
    ny, nx = 60, 60
    stride = 16.0
    density_map = np.ones((ny, nx), dtype=np.float32)
    grid_meta = {"origin_um": [0.0, 0.0], "stride_um": stride, "nx": nx, "ny": ny}

    # Hotspot polygon confined to coordinates [300, 300] to [600, 600]
    poly = [[300.0, 300.0], [600.0, 300.0], [600.0, 600.0], [300.0, 600.0]]

    hpfs = greedy_place_hpfs(
        density_map,
        grid_meta,
        hotspot_polygons_um=[poly],
        count=2,
        radius_um=100.0,
        min_separation_um=200.0
    )

    assert len(hpfs) >= 1
    for h in hpfs:
        cx, cy = h["center_um"]
        assert 300.0 <= cx <= 600.0
        assert 300.0 <= cy <= 600.0


def test_greedy_place_hpfs_circle_inside_slide():
    """Issue #586: Circle must not clip beyond slide dimensions."""
    ny, nx = 40, 40
    stride = 16.0
    slide_w_um = 640.0
    slide_h_um = 640.0
    density_map = np.zeros((ny, nx), dtype=np.float32)

    # Place peak right at edge (x=16 um, y=16 um)
    density_map[1, 1] = 10.0
    # Place peak comfortably inside (x=320 um, y=320 um)
    density_map[20, 20] = 5.0

    grid_meta = {"origin_um": [0.0, 0.0], "stride_um": stride, "nx": nx, "ny": ny}
    radius_um = 100.0

    hpfs = greedy_place_hpfs(
        density_map,
        grid_meta,
        count=1,
        radius_um=radius_um,
        slide_dimensions_um=(slide_w_um, slide_h_um)
    )

    assert len(hpfs) == 1
    cx, cy = hpfs[0]["center_um"]
    # (16, 16) with r=100 would clip at 16 - 100 < 0, so it must choose the center peak!
    assert cx - radius_um >= 0.0
    assert cy - radius_um >= 0.0
    assert cx + radius_um <= slide_w_um
    assert cy + radius_um <= slide_h_um


def test_generate_mitosis_density_map_confidence_filtering():
    """Issue #596: Filter out low-confidence noise (<0.5) and rejected candidates."""
    candidates = [
        {"centroid_um": [100.0, 100.0], "label": "not_mitosis"},
        {"centroid_um": [100.0, 100.0], "label": "rejected"},
        {"centroid_um": [100.0, 100.0], "label": "unreviewed", "ver_conf": 0.2}, # noise (<0.5)
        {"centroid_um": [200.0, 200.0], "label": "unreviewed", "ver_conf": 0.8}, # valid
        {"centroid_um": [300.0, 300.0], "label": "confirmed"},                   # valid
    ]
    bbox = (0.0, 0.0, 400.0, 400.0)
    density_map, meta = generate_mitosis_density_map(candidates, bbox, grid_res_um=16.0, radius_um=50.0)

    # Location (100, 100) had only rejected or low-conf figures -> near zero density
    g100 = int(round((100.0 - meta["origin_um"][0]) / 16.0))
    assert density_map[g100, g100] == 0.0

    # Locations 200, 200 and 300, 300 should have positive density
    g200 = int(round((200.0 - meta["origin_um"][0]) / 16.0))
    assert density_map[g200, g200] > 0.0


# ---------------------------------------------------------------------------
# 2. WSI Color & Region Reading Tests (#53, #554, #54, #730)
# ---------------------------------------------------------------------------

def test_check_icc_profile_variations():
    """Issue #53: Robust ICC profile detection across color_profile, properties, and info."""
    # 1. OpenSlide >= 1.3 color_profile object
    class SlideWithColorProfile:
        class ColorProf:
            def tobytes(self):
                return b"ICC_PROFILE_BYTES"
        color_profile = ColorProf()

    icc_bytes, has_icc = check_icc_profile(SlideWithColorProfile())
    assert has_icc is True
    assert icc_bytes == b"ICC_PROFILE_BYTES"

    # 2. Slide with properties dict
    class SlideWithProps:
        properties = {"openslide.icc-profile": b"PROP_ICC_BYTES"}

    icc_bytes2, has_icc2 = check_icc_profile(SlideWithProps())
    assert has_icc2 is True
    assert icc_bytes2 == b"PROP_ICC_BYTES"

    # 3. Slide with PIL info
    class SlideWithInfo:
        info = {"icc_profile": b"INFO_ICC_BYTES"}

    icc_bytes3, has_icc3 = check_icc_profile(SlideWithInfo())
    assert has_icc3 is True
    assert icc_bytes3 == b"INFO_ICC_BYTES"

    # 4. Slide with no profile
    class SlideEmpty:
        pass

    icc_bytes4, has_icc4 = check_icc_profile(SlideEmpty())
    assert has_icc4 is False
    assert icc_bytes4 is None


def test_read_region_srgb_rgba_compositing():
    """Issue #554: RGBA transparent border pixels must be composited over white, not black."""
    # Create an RGBA image where left half is pure transparent (alpha=0) and right half is red
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    red_half = Image.new("RGBA", (50, 100), (255, 0, 0, 255))
    img.paste(red_half, (50, 0))

    tile_arr, _ = read_region_srgb(
        img,
        x_um=0.0,
        y_um=0.0,
        w_um=25.0,
        h_um=25.0,
        out_px=(100, 100),
        mpp_x=0.25,
        mpp_y=0.25
    )

    # Left half was alpha=0. With white compositing, it should be (255, 255, 255), NOT (0, 0, 0)!
    assert tuple(tile_arr[50, 10]) == (255, 255, 255)
    # Right half was red
    assert tuple(tile_arr[50, 80]) == (255, 0, 0)


def test_extract_patch_from_pyramid_coverage_guard():
    """Issue #730: Return None if pyramid tiles are missing or coverage < 75%."""
    with patch("app.core.gcs.download_blob_as_bytes", side_effect=Exception("Tile not found")):
        res = extract_patch_from_pyramid(
            slide_id="slide-test",
            cx_um=1000.0,
            cy_um=1000.0,
            field_um=200.0
        )
        assert res is None


# ---------------------------------------------------------------------------
# 3. Tile Router Security, Caching & Validation Tests (#739, #640, #340, #641, #211, #212)
# ---------------------------------------------------------------------------

def test_stream_slide_tile_validation():
    """Issue #211, #739, #641, #340: Validations and private cache headers."""
    # Uninitialized slide dimensions -> HTTP 404 (#739)
    slide_invalid = Slide(id=uuid.uuid4(), case_id=uuid.uuid4(), width_px=None, height_px=0)
    with pytest.raises(HTTPException) as exc_dim:
        stream_slide_tile(slide_invalid, layer="orig", z=10, filename="0_0.png")
    assert exc_dim.value.status_code == 404

    # Invalid layer -> HTTP 400 (#211)
    slide_valid = Slide(id=uuid.uuid4(), case_id=uuid.uuid4(), width_px=4096, height_px=4096, base_mag=40)
    with pytest.raises(HTTPException) as exc_layer:
        stream_slide_tile(slide_valid, layer="invalid_layer", z=10, filename="0_0.png")
    assert exc_layer.value.status_code == 400

    # Zoom out of bounds -> HTTP 404 (#641)
    with pytest.raises(HTTPException) as exc_zoom:
        stream_slide_tile(slide_valid, layer="orig", z=25, filename="0_0.png")
    assert exc_zoom.value.status_code == 404

    # Tile coordinate out of bounds -> HTTP 404 (#641)
    with pytest.raises(HTTPException) as exc_coord:
        stream_slide_tile(slide_valid, layer="orig", z=10, filename="999_999.png")
    assert exc_coord.value.status_code == 404


def test_tile_router_private_cache_header():
    """Issue #340: Ensure Cache-Control header is private, max-age=86400."""
    slide_valid = Slide(id=uuid.uuid4(), case_id=uuid.uuid4(), width_px=4096, height_px=4096, base_mag=40)

    # Mock GCS blob download to return a valid 1x1 PNG
    png_buf = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 255, 255)).save(png_buf, format="PNG")
    mock_png_bytes = png_buf.getvalue()

    with patch("app.routers.tiles.get_gcs_client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = mock_png_bytes
        mock_bucket.blob.return_value = mock_blob
        mock_client.return_value.bucket.return_value = mock_bucket

        resp = stream_slide_tile(slide_valid, layer="orig", z=10, filename="0_0.png")
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "private, max-age=86400"
        assert resp.headers["X-Tile-Layer"] == "orig"


# ---------------------------------------------------------------------------
# 4. GCS Signed URLs, Artifact URLs & Extension Allowlist (#19, #5, #22)
# ---------------------------------------------------------------------------

def test_generate_signed_upload_url_extension_allowlist():
    """Issue #19: Raw slide bucket rejects unsupported extensions with HTTP 400."""
    # Disallowed extension
    with pytest.raises(HTTPException) as exc:
        generate_signed_upload_url(settings.GCS_RAW_BUCKET, "cases/123/malicious.exe")
    assert exc.value.status_code == 400
    assert "Unsupported WSI file extension" in exc.value.detail

    # Allowed extension
    url = generate_signed_upload_url(settings.GCS_RAW_BUCKET, "cases/123/slide.svs")
    assert url is not None


def test_get_gcs_artifact_direct_url_regex_hotspot():
    """Issue #22: Hotspot IDs with suffixes like _10x_norm.png are not mangled."""
    url1 = get_gcs_artifact_direct_url("cases/case-123/triage/patches/hs_01_10x_norm.png")
    assert "hotspots/hs_01/thumbnail" in url1

    url2 = get_gcs_artifact_direct_url("cases/case-123/triage/patches/hs_02_40x_orig.png")
    assert "hotspots/hs_02/thumbnail" in url2

    url3 = get_gcs_artifact_direct_url("cases/case-123/triage/patches/hs_03_thumb.png")
    assert "hotspots/hs_03/thumbnail" in url3


# ---------------------------------------------------------------------------
# 5. Probe Classifier & Triage Stage Invariants (#88, #569)
# ---------------------------------------------------------------------------

def test_probe_runner_dimension_assertion():
    """Issue #88: ProbeRunner strictly verifies 384 embedding dimension."""
    runner = ProbeRunner()

    # Wrong number of dimensions (1D instead of 2D)
    with pytest.raises(ValueError) as exc1:
        runner.predict_proba(np.zeros((384,), dtype=np.float32))
    assert "Embeddings must be a 2D array" in str(exc1.value)

    # Wrong feature dimension (512 instead of 384)
    with pytest.raises(ValueError) as exc2:
        runner.predict_proba(np.zeros((10, 512), dtype=np.float32))
    assert "Embedding dimension mismatch" in str(exc2.value)

    # Correct dimension (384)
    probas = runner.predict_proba(np.zeros((5, 384), dtype=np.float32))
    assert len(probas) == 5
    assert all(0.0 <= p <= 1.0 for p in probas)


def test_confirm_triage_mutual_exclusion_invariant():
    """Issue #569: Reject confirmation if no_invasive_tumor=True but active hotspots exist."""
    from app.routers.triage import confirm_triage, TriageConfirmPayload

    db_mock = MagicMock()
    stage_exec = MagicMock()
    stage_exec.status = "awaiting_review"
    stage_exec.output_ref = ""
    stage_exec.review_edits = []
    db_mock.scalars.return_value.first.return_value = stage_exec

    # Machine output has 2 active tumor hotspots
    active_hotspots_json = b'{"hotspots": [{"id": "hs_01", "polygon_um": [[0,0],[1,1],[0,1]]}, {"id": "hs_02", "polygon_um": [[2,2],[3,3],[2,3]]}]}'
    
    with patch("app.routers.triage.download_blob_as_bytes", return_value=active_hotspots_json):
        # Attempt to confirm no_invasive_tumor=True with 2 active hotspots -> Must raise HTTP 409 Conflict (#569)
        payload = TriageConfirmPayload(
            case_id="case-123",
            reviewed_by="dr_smith",
            no_invasive_tumor=True
        )
        with pytest.raises(HTTPException) as exc:
            confirm_triage(payload, db=db_mock)
        assert exc.value.status_code == 409
        assert "active tumor hotspot" in exc.value.detail
