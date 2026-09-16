"""
Unit and Integration Tests for Batch 12:
- #501 [HIGH]: Jinja2/WeasyPrint CAP PDF generation, print CSS & client-visible HTML preview endpoint
- #504 [HIGH]: Authentic evidence geometry burn-in (hotspots & HPF reticles) and unavailable fallback banner
- #512 [HIGH]: Batch retrospective validation CLI harness (resumability, concurrency, QC exclusions, Cohen's kappa)
"""

import os
import io
import csv
import json
import uuid
import tempfile
import shutil
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.main import app
from app.core.config import settings
from app.core.db import SessionLocal, Base, engine
from app.models.case import Case
from app.models.slide import Slide
from app.models.grading import Grading
from app.models.report import Report
from app.models.hotspot import Hotspot
from app.models.hpf_site import HpfSite

from pipeline.report_pdf import (
    render_report_html,
    generate_evidence_thumbnail,
    burn_hotspot_polygons_on_overview,
    burn_hpf_reticle_on_patch,
    generate_clinical_cap_pdf
)
from cli.validate import (
    compute_cohen_kappa,
    compute_confusion_matrix,
    calculate_validation_metrics,
    execute_validation_run,
    run_case_headless
)

client = TestClient(app)


# ==============================================================================
# Finding #501: Jinja2 HTML Report & Preview Endpoint
# ==============================================================================

def test_render_report_html_draft_watermark():
    """Verify render_report_html renders Jinja2 template and includes DRAFT watermark when draft=True."""
    sample_data = {
        "case_id": "case-test-501-draft",
        "procedure": "Core Needle Biopsy",
        "laterality": "Left",
        "tumor_site": "Upper Outer Quadrant",
        "histologic_type": "Invasive Carcinoma of No Special Type (NST)",
        "tumor_size_mm": 18.5,
        "lvi_status": "absent",
        "dcis_present": False,
        "margins": {"status": "negative", "closest_margin_mm": 4.2, "closest_margin_name": "Posterior"},
        "lymph_nodes": {"examined_count": 2, "positive_count": 0},
        "biomarkers": {
            "er": {"status": "positive", "percent": 95},
            "pr": {"status": "positive", "percent": 80},
            "her2": {"ihc_score": "1+", "result": "negative"},
            "ki67": {"percent": 15}
        },
        "staging": {"pt_stage": "pT1c", "pn_stage": "pN0", "stage_group": "IA"},
        "nottingham_grade": {
            "grade": 2,
            "tubule_score": 2,
            "tubule_percent": 45.0,
            "pleo_score": 2,
            "mitotic_score": 2,
            "nottingham_sum": 6
        },
        "narrative": {
            "diagnosis_line": "LEFT BREAST, BIOPSY: INVASIVE CARCINOMA OF NO SPECIAL TYPE, GRADE 2.",
            "microscopic_findings": "Invasive carcinoma forming moderate tubules with intermediate nuclear pleomorphism.",
            "clinical_correlation": "Correlates with clinical imaging mass."
        },
        "status": "draft",
        "integrity_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }

    # 1. Draft render must contain DRAFT watermark (#501)
    html_draft = render_report_html(sample_data, is_draft=True)
    assert "CAP SYNOPTIC REPORT" in html_draft
    assert "DRAFT" in html_draft
    assert "watermark" in html_draft
    assert "case-test-501-draft" in html_draft
    assert "Invasive Carcinoma of No Special Type" in html_draft
    assert "Grade 2" in html_draft

    # 2. Signed/Final render must NOT contain DRAFT watermark overlay
    sample_data["status"] = "signed"
    sample_data["signed_by"] = "Dr. Jane Doe, MD"
    sample_data["npi"] = "1928374650"
    html_signed = render_report_html(sample_data, is_draft=False)
    assert 'class="watermark"' not in html_signed
    assert "Dr. Jane Doe, MD" in html_signed


def test_get_report_html_preview_endpoint():
    """Verify GET /api/v1/stages/report/{case_id}/html endpoint returns valid HTML preview (#501)."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    case_uid = uuid.uuid4()
    try:
        case = Case(id=case_uid, created_by="test_user", status="open")
        db.add(case)
        grading = Grading(
            case_id=case_uid,
            grade=2,
            tubule_score=2,
            tubule_percent=40.0,
            pleo_score=2,
            mitotic_score=2,
            nottingham_sum=6,
            histologic_type="Invasive Carcinoma of No Special Type"
        )
        db.add(grading)
        report = Report(
            case_id=case_uid,
            version=1,
            procedure="Core Needle Biopsy",
            laterality="right",
            tumor_site="upper_outer_quadrant",
            histologic_type="Invasive Carcinoma of No Special Type",
            tumor_size_mm=15.0,
            status="draft",
            narrative={
                "diagnosis_line": "RIGHT BREAST: INVASIVE CARCINOMA GRADE 2",
                "microscopic_findings": "Moderately differentiated ductal carcinoma.",
                "clinical_correlation": "Aligns with BIRADS 5."
            }
        )
        db.add(report)
        db.commit()

        resp = client.get(f"/api/v1/stages/report/{case_uid}/html")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "CAP SYNOPTIC REPORT" in body
        assert "RIGHT BREAST" in body
        assert "DRAFT" in body
    finally:
        db.close()


# ==============================================================================
# Finding #504: Authentic Evidence Geometry Burn-In & Unavailable Graphic
# ==============================================================================

def test_evidence_burn_in_hotspots_and_hpf():
    """Verify hotspot polygons and HPF calibrated reticles burn into thumbnails (#504)."""
    # 1. Test hotspot polygon burn-in
    overview_img = Image.new("RGB", (800, 600), color=(240, 230, 230))
    geo_hotspots = {
        "hotspots": [
            {
                "seq": 1,
                "polygon_coords_um": [[1000, 1000], [2000, 1000], [2000, 2000], [1000, 2000]],
                "center_um": [1500, 1500]
            },
            {
                "seq": 2,
                "polygon_coords_um": [[3000, 3000], [4000, 3000], [4000, 4000], [3000, 4000]],
                "center_um": [3500, 3500]
            }
        ]
    }
    burned_overview = burn_hotspot_polygons_on_overview(overview_img, geo_hotspots)
    assert burned_overview.size == (800, 600)
    # Ensure pixels modified (not identical to original)
    diff = np.sum(np.abs(np.array(burned_overview) - np.array(overview_img)))
    assert diff > 0, "Hotspot polygon overlay should modify pixels on the image"

    # 2. Test HPF reticle burn-in
    patch_img = Image.new("RGB", (600, 600), color=(235, 220, 225))
    geo_hpf = {
        "top_hpf": {
            "seq": 1,
            "mitotic_count": 8,
            "center_um": [1500, 1500],
            "radius_um": 262.0
        }
    }
    burned_hpf = burn_hpf_reticle_on_patch(patch_img, geo_hpf)
    assert burned_hpf.size == (600, 600)
    diff_hpf = np.sum(np.abs(np.array(burned_hpf) - np.array(patch_img)))
    assert diff_hpf > 0, "HPF reticle overlay should modify pixels on the image"


def test_generate_evidence_thumbnail_missing_shows_unavailable_banner():
    """Verify missing artifacts produce [EVIDENCE UNAVAILABLE] banner without synthetic fabrication (#504)."""
    buf = generate_evidence_thumbnail(
        image_path="/non/existent/path/heatmap.png",
        fallback_text="WSI Triage Heatmap",
        size_px=(260, 160)
    )
    assert isinstance(buf, io.BytesIO)
    im = Image.open(buf)
    assert im.size == (260, 160)
    # Verify image is created and not empty
    arr = np.array(im)
    assert arr.shape == (160, 260, 3)
    # The background is slate-50 (248, 250, 252) with outline and text; not synthetic tissue
    assert np.mean(arr) > 200


# ==============================================================================
# Finding #512: Batch Validation CLI & Statistical Metrics
# ==============================================================================

def test_cohen_kappa_exact_computations():
    """Verify Cohen's kappa unweighted, quadratic, and linear implementations (#512)."""
    # 1. Perfect agreement -> kappa = 1.0
    y_true = [1, 2, 3, 1, 2, 3, 2, 2, 1, 3]
    y_pred = [1, 2, 3, 1, 2, 3, 2, 2, 1, 3]
    assert compute_cohen_kappa(y_true, y_pred) == 1.0
    assert compute_cohen_kappa(y_true, y_pred, weights="quadratic") == 1.0
    assert compute_cohen_kappa(y_true, y_pred, weights="linear") == 1.0

    # 2. Moderate agreement with off-by-one errors
    y_pred_near = [1, 2, 2, 1, 2, 3, 2, 3, 1, 2]
    k_unweighted = compute_cohen_kappa(y_true, y_pred_near)
    k_quad = compute_cohen_kappa(y_true, y_pred_near, weights="quadratic")
    assert 0.0 < k_unweighted <= 1.0
    # Quadratic weighted kappa is generally higher than unweighted when errors are off by only 1 grade
    assert k_quad >= k_unweighted

    # 3. Empty inputs
    assert compute_cohen_kappa([], []) == 0.0


def test_confusion_matrix_and_metrics_calculation():
    """Verify 3x3 confusion matrix and metrics calculation (#512)."""
    records = [
        {"status": "completed", "signout_grade": 1, "predicted_grade": 1, "signout_tubule": 1, "predicted_tubule": 1},
        {"status": "completed", "signout_grade": 2, "predicted_grade": 2, "signout_tubule": 2, "predicted_tubule": 2},
        {"status": "completed", "signout_grade": 3, "predicted_grade": 3, "signout_tubule": 3, "predicted_tubule": 3},
        {"status": "completed", "signout_grade": 2, "predicted_grade": 3, "signout_tubule": 2, "predicted_tubule": 3},
        {"status": "excluded", "qc_status": "fail", "qc_exclusion_reason": "qc: focus sharpness below threshold (0.18 < 0.35)"}
    ]

    metrics = calculate_validation_metrics(records)
    assert metrics["total_cases"] == 5
    assert metrics["completed_cases"] == 4
    assert metrics["excluded_qc_cases"] == 1
    assert metrics["qc_exclusion_rate"] == 0.2
    assert "focus sharpness" in list(metrics["qc_exclusion_reasons"].keys())[0]

    # 3 out of 4 matching grades -> 75% accuracy
    assert metrics["nottingham_grade_accuracy"] == 0.75
    cm = metrics["confusion_matrix_3x3"]
    assert cm["1"]["1"] == 1
    assert cm["2"]["2"] == 1
    assert cm["2"]["3"] == 1
    assert cm["3"]["3"] == 1


def test_batch_validation_cli_harness_run_and_resumability():
    """Verify execute_validation_run executes end-to-end and resumes from checkpoint (#512)."""
    scratch = tempfile.mkdtemp(prefix="test_val_cli_")
    manifest_csv = os.path.join(scratch, "archive_manifest.csv")
    out_dir = os.path.join(scratch, "run_results")

    try:
        # Create synthetic manifest
        rows = [
            {"slide_path": "slide_001.svs", "checksum": "chk001", "signout_grade": "2", "signout_tubule": "2", "signout_pleo": "2", "signout_mitotic": "2"},
            {"slide_path": "slide_002.svs", "checksum": "chk002", "signout_grade": "3", "signout_tubule": "3", "signout_pleo": "3", "signout_mitotic": "3"},
            {"slide_path": "slide_003.svs", "checksum": "chk003", "signout_grade": "1", "signout_tubule": "1", "signout_pleo": "1", "signout_mitotic": "1"}
        ]
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # 1. First execution
        metrics1 = execute_validation_run(
            manifest_path=manifest_csv,
            out_dir=out_dir,
            mode="auto",
            concurrency=2
        )

        assert metrics1["total_cases"] == 3
        assert os.path.exists(os.path.join(out_dir, "summary.json"))
        assert os.path.exists(os.path.join(out_dir, "results.csv"))
        assert os.path.exists(os.path.join(out_dir, "checkpoint.json"))

        # Verify checkpoint contents
        with open(os.path.join(out_dir, "checkpoint.json"), "r", encoding="utf-8") as f:
            chk = json.load(f)
        assert len(chk) == 3
        assert "chk001" in chk
        assert "chk002" in chk

        # 2. Resumability: re-run should resume and reuse completed checkpoint
        metrics2 = execute_validation_run(
            manifest_path=manifest_csv,
            out_dir=out_dir,
            mode="auto",
            concurrency=2
        )
        assert metrics2["total_cases"] == 3
        assert metrics2["completed_cases"] == metrics1["completed_cases"]

    finally:
        shutil.rmtree(scratch, ignore_errors=True)
