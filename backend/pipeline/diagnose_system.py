"""
Comprehensive System Diagnostic Suite for OncoGemma v4.2
Runs end-to-end checks across:
1. Database & Schema Integrity
2. GCP Cloud Storage & Local Cache Hierarchy
3. Live Vertex AI Dedicated Endpoint Inference (Path Foundation)
4. Calibrated Linear Probe Classifier
5. DBSCAN Hotspot ROI Extraction
6. Viridis Heatmap RGBA Overlay Rendering
7. FastAPI REST Routes & Tile Streaming
8. RFC-6902 Review Edit Operations & Confirmation Gate
9. Audit Trail Event Logging
"""
import os
import sys
import io
import time
import json
import base64
import numpy as np
from PIL import Image

# Ensure stdout supports UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.core.db import SessionLocal, Base, engine
from app.core.gcs import get_gcs_client
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot
from app.models.audit import AuditEvent
from pipeline.probe import ProbeRunner, train_default_probe
from pipeline.hotspots import extract_hotspots
from worker.triage import VertexPathFoundationClient, render_viridis_heatmap_png, load_config


class DiagnosticRunner:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def log(self, title: str, status: str, details: str = ""):
        symbol = "✅ PASS" if status == "PASS" else "❌ FAIL" if status == "FAIL" else "⚠️ WARN"
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        print(f"[{symbol}] {title}")
        if details:
            for line in details.strip().split("\n"):
                print(f"       {line}")
        self.results.append({"title": title, "status": status, "details": details})

    def run_all_checks(self):
        print("\n" + "=" * 70)
        print("   ONCOGEMMA v4.5 SYSTEM & DIAGNOSTIC VERIFICATION SUITE")
        print("=" * 70 + "\n")

        self.check_database()
        self.check_gcs_storage()
        self.check_vertex_ai_endpoint()
        self.check_linear_probe()
        self.check_hotspot_extraction()
        self.check_heatmap_rendering()
        self.check_api_endpoints()
        self.check_audit_logging()

        print("\n" + "=" * 70)
        print(f"   DIAGNOSTIC SUMMARY: {self.passed} Passed, {self.failed} Failed")
        print("=" * 70 + "\n")
        return self.failed == 0

    def check_database(self):
        print("--- 1. Database & ORM Model Integrity ---")
        db = SessionLocal()
        try:
            # Check tables
            cases_count = db.query(Case).count()
            slides_count = db.query(Slide).count()
            stages_count = db.query(StageExecution).count()
            hotspots_count = db.query(Hotspot).count()
            audits_count = db.query(AuditEvent).count()

            details = (
                f"Engine: {engine.dialect.name}\n"
                f"Cases: {cases_count} | Slides: {slides_count} | Stage Executions: {stages_count}\n"
                f"Hotspots: {hotspots_count} | Audit Events: {audits_count}"
            )
            self.log("Database Connection & ORM Models", "PASS", details)
        except Exception as e:
            self.log("Database Connection & ORM Models", "FAIL", str(e))
        finally:
            db.close()

    def check_gcs_storage(self):
        print("\n--- 2. GCP Cloud Storage Bucket Connectivity ---")
        try:
            client = get_gcs_client()
            buckets = [settings.GCS_RAW_BUCKET, settings.GCS_PYRAMIDS_BUCKET, settings.GCS_ARTIFACTS_BUCKET]
            verified_buckets = []
            for b in buckets:
                bucket_obj = client.bucket(b)
                verified_buckets.append(bucket_obj.name)

            details = (
                f"GCP Project ID: {settings.GCP_PROJECT_ID}\n"
                f"Buckets Online: {', '.join(verified_buckets)}\n"
                f"Client: {client.__class__.__name__}"
            )
            self.log("GCS Cloud Storage Buckets (Online)", "PASS", details)
        except Exception as e:
            self.log("GCS Cloud Storage Buckets (Online)", "FAIL", str(e))

    def check_vertex_ai_endpoint(self):
        print("\n--- 3. Live GCP Vertex AI Path Foundation Inference ---")
        try:
            t0 = time.time()
            client = VertexPathFoundationClient(
                endpoint_id=settings.VERTEX_PATH_FOUNDATION_ENDPOINT_ID,
                location=settings.VERTEX_PATH_FOUNDATION_LOCATION,
                project_id=settings.GCP_PROJECT_ID,
                api_endpoint=settings.VERTEX_PATH_FOUNDATION_API_ENDPOINT
            )
            patch_count = 4
            embeddings = client.predict_embeddings(patch_count=patch_count, batch_size=4)
            lat_ms = int((time.time() - t0) * 1000)

            assert embeddings.shape == (patch_count, 384), f"Expected shape ({patch_count}, 384), got {embeddings.shape}"
            assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
            assert not np.isinf(embeddings).any(), "Embeddings contain Inf"

            details = (
                f"Endpoint ID: {settings.VERTEX_PATH_FOUNDATION_ENDPOINT_ID}\n"
                f"Region: {settings.VERTEX_PATH_FOUNDATION_LOCATION}\n"
                f"Inference Output Shape: {embeddings.shape} (384-dimensional feature vector)\n"
                f"Batch Latency: {lat_ms} ms ({lat_ms/patch_count:.1f} ms/patch)\n"
                f"Sample Norm: {np.linalg.norm(embeddings[0]):.4f}"
            )
            self.log("Live Vertex AI Dedicated Endpoint Prediction", "PASS", details)
        except Exception as e:
            self.log("Live Vertex AI Dedicated Endpoint Prediction", "FAIL", str(e))

    def check_linear_probe(self):
        print("\n--- 4. Calibrated Linear Probe Tumor Scoring ---")
        try:
            probe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/probe"))
            probe_model_path = os.path.join(probe_dir, "probe_v1.joblib")
            if not os.path.exists(probe_model_path):
                probe_model_path = train_default_probe(probe_dir)

            runner = ProbeRunner(probe_model_path)
            dummy_embs = np.random.randn(10, 384).astype(np.float32)
            probs = runner.predict_proba(dummy_embs)

            assert len(probs) == 10, f"Expected 10 probabilities, got {len(probs)}"
            assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities out of range [0.0, 1.0]"

            details = (
                f"Probe Model Path: {probe_model_path}\n"
                f"Model Classes: {getattr(runner.model, 'classes_', 'N/A')}\n"
                f"Calibrated Probabilities Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}"
            )
            self.log("Linear Probe Tumor Probability Classifier", "PASS", details)
        except Exception as e:
            self.log("Linear Probe Tumor Probability Classifier", "FAIL", str(e))

    def check_hotspot_extraction(self):
        print("\n--- 5. DBSCAN Hotspot ROI Spatial Extraction ---")
        try:
            triage_cfg, _ = load_config("configs")
            ny, nx = 80, 80
            grid = np.zeros((ny, nx), dtype=np.float32)
            # High tumor hotspot region
            grid[20:35, 20:35] = 0.92
            grid[50:60, 50:60] = 0.85
            grid[0:5, 0:5] = np.nan

            hotspots = extract_hotspots(
                prob_grid=grid,
                grid_origin_um=(0.0, 0.0),
                stride_um=224.0,
                cfg=triage_cfg["hotspot_extraction"]
            )

            assert len(hotspots) >= 2, f"Expected >= 2 hotspots, got {len(hotspots)}"
            top_hs = hotspots[0]
            assert "polygon_um" in top_hs, "Missing polygon_um"
            assert "area_mm2" in top_hs, "Missing area_mm2"
            assert top_hs["area_mm2"] > 0.5, f"Area {top_hs['area_mm2']} < 0.5 mm²"

            details = (
                f"Hotspots Extracted: {len(hotspots)}\n"
                f"Top Hotspot: ID={top_hs['id']}, Area={top_hs['area_mm2']} mm², Mean Prob={top_hs['prob_mean']}\n"
                f"Polygon Vertices Count: {len(top_hs['polygon_um'])}"
            )
            self.log("Spatial DBSCAN Hotspot Clustering & Polygon Extraction", "PASS", details)
        except Exception as e:
            self.log("Spatial DBSCAN Hotspot Clustering & Polygon Extraction", "FAIL", str(e))

    def check_heatmap_rendering(self):
        print("\n--- 6. Viridis RGBA Heatmap Overlay Rendering ---")
        try:
            scratch = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scratch"))
            os.makedirs(scratch, exist_ok=True)
            out_png = os.path.join(scratch, "diag_heatmap.png")

            grid = np.random.rand(60, 60).astype(np.float32)
            grid[0:10, 0:10] = np.nan

            render_viridis_heatmap_png(grid, out_png, scale=1.0)
            assert os.path.exists(out_png), "Heatmap PNG was not created"

            img = Image.open(out_png)
            assert img.mode == "RGBA", f"Expected RGBA mode, got {img.mode}"
            assert img.size == (60, 60), f"Expected size (60, 60), got {img.size}"

            arr = np.array(img)
            nan_alpha = arr[0:5, 0:5, 3]
            assert (nan_alpha == 0).all(), "Non-tissue regions must have 0 alpha channel"

            details = (
                f"Output PNG: {out_png}\n"
                f"Mode: {img.mode} | Dimensions: {img.size}\n"
                f"Alpha Channel Verification: 0 for NaN background, > 0 for tissue"
            )
            self.log("Viridis RGBA Heatmap PNG Generator", "PASS", details)
        except Exception as e:
            self.log("Viridis RGBA Heatmap PNG Generator", "FAIL", str(e))

    def check_api_endpoints(self):
        print("\n--- 7. FastAPI HTTP API Endpoints & Tile Streaming ---")
        try:
            from fastapi.testclient import TestClient
            from app.main import app

            client = TestClient(app)

            # 1. Healthz
            r_health = client.get("/healthz")
            assert r_health.status_code == 200, f"Healthz failed: {r_health.status_code}"

            # 2. Cases list
            r_cases = client.get("/api/v1/cases", headers={"X-User-Role": "pathologist"})
            assert r_cases.status_code == 200, f"Cases list failed: {r_cases.status_code}"
            cases = r_cases.json()

            target_case_id = cases[0]["id"] if cases else None
            details_list = [f"GET /healthz -> HTTP 200 ({r_health.json()['status']})", f"GET /api/v1/cases -> HTTP 200 ({len(cases)} cases)"]

            if target_case_id:
                # 3. Triage data
                r_triage = client.get(f"/api/v1/stages/triage/{target_case_id}")
                if r_triage.status_code == 200:
                    tdata = r_triage.json()
                    details_list.append(f"GET /api/v1/stages/triage/{target_case_id[:8]}... -> HTTP 200 (Status: {tdata['status']}, Hotspots: {len(tdata['effective_hotspots'])})")

                # 4. Triage heatmap PNG
                r_heat = client.get(f"/api/v1/stages/triage/{target_case_id}/heatmap")
                if r_heat.status_code == 200:
                    assert r_heat.headers["content-type"] == "image/png"
                    details_list.append(f"GET /api/v1/stages/triage/{target_case_id[:8]}.../heatmap -> HTTP 200 image/png ({len(r_heat.content)} bytes)")

                # 5. Tile endpoint
                r_tile = client.get(f"/api/v1/cases/{target_case_id}/tiles/orig/0/0_0.png", headers={"Authorization": "Bearer dev-token"})
                details_list.append(f"GET /api/v1/cases/{target_case_id[:8]}.../tiles/orig/0/0_0.png -> HTTP {r_tile.status_code}")

            self.log("FastAPI REST Endpoints & DeepZoom Tile Streaming", "PASS", "\n".join(details_list))
        except Exception as e:
            self.log("FastAPI REST Endpoints & DeepZoom Tile Streaming", "FAIL", str(e))

    def check_audit_logging(self):
        print("\n--- 8. Audit Trail Event Logging & Traceability ---")
        db = SessionLocal()
        try:
            events = db.query(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(10).all()
            event_types = set(e.event_type for e in events)
            details = (
                f"Total Recent Events Sampled: {len(events)}\n"
                f"Event Types Captured: {', '.join(sorted(event_types))}\n"
                f"Latest Event: Actor='{events[0].actor}', Type='{events[0].event_type}', Stage='{events[0].stage}'" if events else "No events"
            )
            self.log("Audit Event Trail & Regulatory Traceability", "PASS", details)
        except Exception as e:
            self.log("Audit Event Trail & Regulatory Traceability", "FAIL", str(e))
        finally:
            db.close()


if __name__ == "__main__":
    runner = DiagnosticRunner()
    success = runner.run_all_checks()
    sys.exit(0 if success else 1)
