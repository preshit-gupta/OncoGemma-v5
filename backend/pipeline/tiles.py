import io
import numpy as np
from PIL import Image, ImageCms

# Global cache for built ImageCms ICC transforms
_CMS_TRANSFORM_CACHE = {}

def get_icc_transform(icc_bytes: bytes):
    """Build and cache PIL ImageCms transform from embedded raw ICC profile bytes to sRGB."""
    if not icc_bytes:
        return None
    cache_key = hash(icc_bytes)
    if cache_key in _CMS_TRANSFORM_CACHE:
        return _CMS_TRANSFORM_CACHE[cache_key]

    try:
        in_profile = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
        srgb_profile = ImageCms.createProfile("sRGB")
        transform = ImageCms.buildTransform(in_profile, srgb_profile, "RGB", "RGB")
        _CMS_TRANSFORM_CACHE[cache_key] = transform
        return transform
    except Exception as e:
        print(f"[ICC Profile Warning] Failed to parse ICC profile transform: {e}")
        _CMS_TRANSFORM_CACHE[cache_key] = None
        return None

def check_icc_profile(slide) -> tuple[bytes | None, bool]:
    """Inspect slide object or properties for embedded ICC color profile."""
    icc_bytes = None
    
    # 1. OpenSlide properties
    if hasattr(slide, "properties"):
        icc_bytes = slide.properties.get("openslide.color-profile")
        if isinstance(icc_bytes, str):
            icc_bytes = icc_bytes.encode("utf-8")

    # 2. PIL Image info fallback
    if not icc_bytes and hasattr(slide, "info"):
        icc_bytes = slide.info.get("icc_profile")

    has_icc = bool(icc_bytes and len(icc_bytes) > 0)
    return (icc_bytes if has_icc else None), has_icc

def read_region_srgb(
    slide,
    x_um: float,
    y_um: float,
    w_um: float,
    h_um: float,
    out_px: int | tuple[int, int],
    mpp_x: float = 0.25,
    mpp_y: float = 0.25
) -> tuple[np.ndarray, bool]:
    """
    Authoritative single entry point for reading slide tile regions in sRGB color space.
    
    :param slide: OpenSlide object or PIL Image object.
    :param x_um: Top-left X coordinate in micrometers (base level 0).
    :param y_um: Top-left Y coordinate in micrometers (base level 0).
    :param w_um: Width of region in micrometers.
    :param h_um: Height of region in micrometers.
    :param out_px: Target pixel dimension (int or (width, height) tuple).
    :param mpp_x: Micrometers per pixel X at level 0.
    :param mpp_y: Micrometers per pixel Y at level 0.
    :return: (RGB uint8 numpy array of shape (H, W, 3), icc_applied boolean)
    """
    if isinstance(out_px, int):
        target_w_px, target_h_px = out_px, out_px
    else:
        target_w_px, target_h_px = out_px

    x_px_0 = int(round(x_um / mpp_x))
    y_px_0 = int(round(y_um / mpp_y))
    w_px_0 = int(round(w_um / mpp_x))
    h_px_0 = int(round(h_um / mpp_y))

    icc_bytes, has_icc = check_icc_profile(slide)
    icc_applied = False

    # 1. OpenSlide Slide object
    if hasattr(slide, "read_region"):
        dim_w, dim_h = getattr(slide, "dimensions", (100000, 100000))
        if x_px_0 >= dim_w or y_px_0 >= dim_h or (x_px_0 + w_px_0) <= 0 or (y_px_0 + h_px_0) <= 0:
            pil_tile = Image.new("RGB", (max(1, target_w_px), max(1, target_h_px)), color=(245, 240, 245))
        else:
            best_level = 0
            if hasattr(slide, "get_best_level_for_downsample"):
                target_downsample = w_px_0 / max(1, target_w_px)
                best_level = slide.get_best_level_for_downsample(target_downsample)

            level_ds = slide.level_downsamples[best_level] if hasattr(slide, "level_downsamples") else 1.0
            w_px_lvl = max(1, int(round(w_px_0 / level_ds)))
            h_px_lvl = max(1, int(round(h_px_0 / level_ds)))

            safe_x = max(0, min(dim_w - 1, x_px_0))
            safe_y = max(0, min(dim_h - 1, y_px_0))
            pil_tile = slide.read_region((safe_x, safe_y), best_level, (w_px_lvl, h_px_lvl))
            if pil_tile.mode != "RGB":
                pil_tile = pil_tile.convert("RGB")

    # 2. PIL Image object fallback
    elif hasattr(slide, "crop"):
        img_w, img_h = slide.size
        box_x1 = max(0, min(img_w, x_px_0))
        box_y1 = max(0, min(img_h, y_px_0))
        box_x2 = max(box_x1, min(img_w, x_px_0 + w_px_0))
        box_y2 = max(box_y1, min(img_h, y_px_0 + h_px_0))

        if box_x2 > box_x1 and box_y2 > box_y1:
            pil_tile = slide.crop((box_x1, box_y1, box_x2, box_y2))
        else:
            pil_tile = Image.new("RGB", (max(1, target_w_px), max(1, target_h_px)), color=(240, 235, 240))

        if pil_tile.mode != "RGB":
            pil_tile = pil_tile.convert("RGB")

    else:
        raise TypeError(f"Unsupported slide object type: {type(slide)}")

    # Apply ICC transform if embedded profile exists
    if has_icc and icc_bytes:
        transform = get_icc_transform(icc_bytes)
        if transform:
            try:
                pil_tile = ImageCms.applyTransform(pil_tile, transform)
                icc_applied = True
            except Exception as pe:
                print(f"[ICC Transform Note] {pe}")

    # Ensure tile has valid dimensions
    if pil_tile.width == 0 or pil_tile.height == 0:
        pil_tile = Image.new("RGB", (max(1, target_w_px), max(1, target_h_px)), color=(240, 235, 240))

    # Resize to exact target output dimensions if needed
    if pil_tile.size != (target_w_px, target_h_px):
        pil_tile = pil_tile.resize((max(1, target_w_px), max(1, target_h_px)), Image.Resampling.BILINEAR)

    tile_arr = np.array(pil_tile, dtype=np.uint8)
    return tile_arr, icc_applied


def extract_patch_from_pyramid(
    slide_id: str,
    cx_um: float,
    cy_um: float,
    field_um: float,
    mpp_x: float = 0.25,
    mpp_y: float = 0.25,
    width_px: int = 20000,
    height_px: int = 20000,
    layer: str = "norm"
) -> Image.Image | None:
    """
    Rapidly reconstructs a high-resolution microscopic field directly from GCS DeepZoom pyramid tiles.
    Avoids downloading massive raw WSI files over the network.
    """
    import math
    from app.core.config import settings
    from app.core.gcs import download_blob_as_bytes

    try:
        max_dim = max(width_px, height_px)
        max_level = int(math.ceil(math.log2(max_dim))) if max_dim > 0 else 18
        
        out_size = 512
        crop_mpp = field_um / float(out_size)
        
        scale_ratio = (mpp_x / crop_mpp)
        level_offset = int(round(math.log2(scale_ratio))) if scale_ratio > 0 else 0
        z = max(0, min(max_level, max_level + level_offset))
        
        level_scale = 2.0 ** (z - max_level)
        
        cx_z = (cx_um / mpp_x) * level_scale
        cy_z = (cy_um / mpp_y) * level_scale
        w_z = (field_um / mpp_x) * level_scale
        h_z = (field_um / mpp_y) * level_scale
        
        x0_z = cx_z - w_z / 2.0
        y0_z = cy_z - h_z / 2.0
        x1_z = x0_z + w_z
        y1_z = y0_z + h_z
        
        c_min = max(0, int(math.floor(x0_z / 256.0)))
        c_max = int(math.floor(x1_z / 256.0))
        r_min = max(0, int(math.floor(y0_z / 256.0)))
        r_max = int(math.floor(y1_z / 256.0))
        
        total_tiles = (c_max - c_min + 1) * (r_max - r_min + 1)
        if total_tiles > 16 or total_tiles <= 0:
            return None
            
        stitch_w = (c_max - c_min + 1) * 256
        stitch_h = (r_max - r_min + 1) * 256
        if stitch_w <= 0 or stitch_h <= 0 or stitch_w > 4096 or stitch_h > 4096:
            return None
            
        mosaic = Image.new("RGB", (stitch_w, stitch_h), (240, 230, 235))
        found_any = False
        
        for c in range(c_min, c_max + 1):
            for r in range(r_min, r_max + 1):
                tile_bytes = None
                for ext in (".png", ".jpg"):
                    tile_blob = f"{slide_id}/{layer}/{z}/{c}_{r}{ext}"
                    try:
                        tile_bytes = download_blob_as_bytes(settings.GCS_PYRAMIDS_BUCKET, tile_blob)
                        break
                    except Exception:
                        pass
                if tile_bytes:
                    try:
                        t_img = Image.open(io.BytesIO(tile_bytes)).convert("RGB")
                        paste_x = (c - c_min) * 256
                        paste_y = (r - r_min) * 256
                        mosaic.paste(t_img, (paste_x, paste_y))
                        found_any = True
                    except Exception:
                        pass
                    
        if not found_any:
            return None
            
        crop_x0 = int(round(x0_z - c_min * 256.0))
        crop_y0 = int(round(y0_z - r_min * 256.0))
        crop_x1 = int(round(x1_z - c_min * 256.0))
        crop_y1 = int(round(y1_z - r_min * 256.0))
        
        cropped = mosaic.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        return cropped.resize((out_size, out_size), Image.Resampling.BILINEAR)
    except Exception as e:
        print(f"[Tile Patch Stitching Error] {e}")
        return None
