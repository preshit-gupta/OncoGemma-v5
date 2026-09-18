import os
import sys
import uuid
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.db import SessionLocal
from app.core.config import settings
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution

def diagnose_case(case_id_str: str):
    """
    Diagnostic tool to inspect end-to-end pipeline execution, disk pyramid tiles,
    pixel intensity statistics, and tile router responses for a given case ID.
    """
    print(f"\n==================================================")
    print(f"   ONCOGEMMA v4.1 DIAGNOSTIC TOOL FOR CASE: {case_id_str}")
    print(f"==================================================\n")

    db = SessionLocal()
    try:
        case_uuid = uuid.UUID(case_id_str)
    except Exception as e:
        print(f"[Error] Invalid UUID string '{case_id_str}': {e}")
        return

    # 1. Inspect DB Case
    case_obj = db.get(Case, case_uuid)
    if not case_obj:
        print(f"[DB Error] Case {case_id_str} not found in database!")
        return

    print(f"[1. Database Case Info]")
    print(f"   - Case ID: {case_obj.id}")
    print(f"   - Status:  {case_obj.status}")
    print(f"   - Created: {case_obj.created_at}")

    # 2. Inspect DB Slide
    slide = db.scalars(__import__("sqlalchemy").select(Slide).where(Slide.case_id == case_uuid)).first()
    if not slide:
        print(f"[DB Error] No slide record found for case {case_id_str}!")
        return

    print(f"\n[2. Database Slide Info]")
    print(f"   - Slide ID:      {slide.id}")
    print(f"   - Original GCS:  {slide.gcs_uri_original}")
    print(f"   - Pyramid GCS:   {slide.gcs_uri_pyramid}")
    print(f"   - Dimensions:   {slide.width_px} x {slide.height_px}")
    print(f"   - MPP (x, y):    {slide.mpp_x}, {slide.mpp_y}")

    # 3. Inspect Stage Executions
    stages = db.scalars(__import__("sqlalchemy").select(StageExecution).where(StageExecution.case_id == case_uuid)).all()
    print(f"\n[3. Pipeline Stage Executions]")
    for s in stages:
        print(f"   - Stage: {s.stage:12s} | Status: {s.status:12s} | Attempt: {s.attempt}")

    # 4. Inspect Disk Pyramid Directories
    base_fake_gcs = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fake_gcs"))
    slide_pyramid_dir = os.path.join(base_fake_gcs, settings.GCS_PYRAMIDS_BUCKET, str(slide.id))
    print(f"\n[4. Disk Pyramid Storage]")
    print(f"   - Pyramid Directory: {slide_pyramid_dir}")
    print(f"   - Directory Exists:  {os.path.exists(slide_pyramid_dir)}")

    if os.path.exists(slide_pyramid_dir):
        for layer in ["orig", "norm"]:
            layer_dir = os.path.join(slide_pyramid_dir, layer)
            if os.path.exists(layer_dir):
                levels = sorted([int(d) for d in os.listdir(layer_dir) if d.isdigit()])
                print(f"   - Layer '{layer:4s}' levels: {levels}")
                for z in levels:
                    z_path = os.path.join(layer_dir, str(z))
                    files = [f for f in os.listdir(z_path) if f.endswith(".png") or f.endswith(".jpg")]
                    if files:
                        sample_file = files[0]
                        sample_img = np.array(Image.open(os.path.join(z_path, sample_file)))
                        print(f"     - Level {z:2d}: {len(files):3d} tiles | Sample tile {sample_file:8s} -> shape: {sample_img.shape}, min: {sample_img.min():3d}, max: {sample_img.max():3d}, mean: {sample_img.mean():.1f}")
            else:
                print(f"   - Layer '{layer:4s}': MISSING on disk!")

    # 5. Generate Diagnostic Visual Patch Composite
    scratch_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scratch"))
    os.makedirs(scratch_dir, exist_ok=True)
    diag_img_path = os.path.join(scratch_dir, f"diag_{case_id_str}.png")

    orig_sample_tile = None
    norm_sample_tile = None

    orig_dir = os.path.join(slide_pyramid_dir, "orig")
    if os.path.exists(orig_dir):
        levels = sorted([int(d) for d in os.listdir(orig_dir) if d.isdigit()])
        if levels:
            max_lvl = max(levels)
            z_path = os.path.join(orig_dir, str(max_lvl))
            files = [f for f in os.listdir(z_path) if f.endswith(".jpg") or f.endswith(".png")]
            if files:
                orig_sample_tile = Image.open(os.path.join(z_path, files[0])).convert("RGB")

    norm_dir = os.path.join(slide_pyramid_dir, "norm")
    if os.path.exists(norm_dir):
        levels = sorted([int(d) for d in os.listdir(norm_dir) if d.isdigit()])
        if levels:
            max_lvl = max(levels)
            z_path = os.path.join(norm_dir, str(max_lvl))
            files = [f for f in os.listdir(z_path) if f.endswith(".png") or f.endswith(".jpg")]
            if files:
                norm_sample_tile = Image.open(os.path.join(z_path, files[0])).convert("RGB")

    if orig_sample_tile and norm_sample_tile:
        comp = Image.new("RGB", (512, 256))
        comp.paste(orig_sample_tile.resize((256, 256)), (0, 0))
        comp.paste(norm_sample_tile.resize((256, 256)), (256, 0))
        comp.save(diag_img_path)
        print(f"\n[5. Visual Composite Generated]")
        print(f"   - Saved side-by-side (Original | Normalized) tile composite to: {diag_img_path}")

    # 6. Test HTTP API Tile Endpoint
    import urllib.request
    print(f"\n[6. HTTP API Endpoint Test (http://127.0.0.1:8000)]")
    for layer in ["orig", "norm"]:
        for z in [0, 8, 9, 10, 11]:
            url = f"http://127.0.0.1:8000/api/v1/cases/{case_id_str}/tiles/{layer}/{z}/0_0.png"
            try:
                req = urllib.request.Request(url, headers={"Authorization": "Bearer dev-token"})
                with urllib.request.urlopen(req) as resp:
                    print(f"   - GET /{layer}/{z}/0_0.png -> HTTP {resp.status} | Layer: {resp.headers.get('X-Tile-Layer'):20s} | Zoom: {resp.headers.get('X-Tile-Zoom')}")
            except Exception as e:
                print(f"   - GET /{layer}/{z}/0_0.png -> ERROR ({e})")

    print(f"\n==================================================\n")

if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "7b10eb26-3733-4382-bbb0-424fa0fc1e12"
    diagnose_case(cid)
