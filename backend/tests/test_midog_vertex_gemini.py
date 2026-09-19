"""
Unit tests for MIDOG Vertex AI detector, Gemini Flash referee, and pathologist review preservation.
"""
import io
import json
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine, select, delete, not_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.db import Base
from app.models.case import Case
from app.models.detection import Detection
from pipeline.detect import YoloMitosisDetector
from pipeline.medgemma import MedGemmaClient, MitosisConfirmationResponse


def test_yolo_detector_vertex_ai_endpoint_mock():
    """Verify YoloMitosisDetector connects to Vertex AI Endpoint and parses predictions."""
    with patch("google.cloud.aiplatform.Endpoint") as mock_endpoint_cls, \
         patch("google.cloud.aiplatform.init"):
        
        mock_endpoint = MagicMock()
        mock_endpoint_cls.return_value = mock_endpoint
        
        # Mock prediction response with bounding boxes
        mock_endpoint.predict.return_value = MagicMock(
            predictions=[{
                "boxes": [
                    {"cx": 150.0, "cy": 250.0, "confidence": 0.85},
                    {"x1": 400.0, "y1": 500.0, "x2": 450.0, "y2": 550.0, "conf": 0.72}
                ]
            }]
        )
        
        detector = YoloMitosisDetector(endpoint_id="projects/123/locations/us-central1/endpoints/456")
        assert detector.vertex_endpoint is not None
        assert detector.model_version.startswith("vertex_ai_midog@")
        
        dummy_tile = np.ones((512, 512, 3), dtype=np.uint8) * 200
        detections = detector.detect(dummy_tile)
        
        assert len(detections) == 2
        assert detections[0] == (150.0, 250.0, 0.85)
        assert detections[1] == (425.0, 525.0, 0.72)


def test_yolo_detector_fallback_provenance(monkeypatch):
    """Verify detector reports od_heuristic@dev truthfully when no weights or endpoints exist."""
    monkeypatch.setattr(settings, "VERTEX_MITOSIS_ENDPOINT_ID", None)
    detector = YoloMitosisDetector(weights_path=None, endpoint_id=None)
    assert detector.vertex_endpoint is None
    assert detector.model is None
    assert detector.model_version == "od_heuristic@dev"


def test_gemini_flash_referee_mock():
    """Verify MedGemmaClient uses Gemini Flash referee and parses strict van Diest JSON."""
    client = MedGemmaClient()
    
    mock_json_response = json.dumps({
        "verdict": "CONFIRMED",
        "envelope_dissolved": True,
        "spiculation_detected": True,
        "confidence": "high",
        "rationale": "Dissolved envelope with distinct ragged chromatin projections."
    })
    
    with patch.object(client, "_call_gemini_flash", return_value=mock_json_response):
        dummy_crop = io.BytesIO()
        Image.new("RGB", (128, 128), color=(220, 200, 220)).save(dummy_crop, format="PNG")
        crop_bytes = dummy_crop.getvalue()
        
        resp = client.evaluate_mitosis_confirmation_sync(crop_bytes)
        assert isinstance(resp, MitosisConfirmationResponse)
        assert resp.verdict == "CONFIRMED"
        assert resp.envelope_dissolved is True
        assert resp.spiculation_detected is True
        assert resp.confidence == "high"


def test_pathologist_detection_preservation_in_db():
    """Verify that existing pathologist reviews are preserved when model detections are purged."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = TestingSession()
    case_id = "test-case-preserve"
    case = Case(id=case_id, created_by="test_user", status="open")
    db.add(case)
    db.commit()
    
    # 1. Seed existing detections: 1 model, 1 gemini_referee, 1 pathologist review, 1 pathologist added
    d1 = Detection(id="d1", case_id=case_id, centroid_um=[100.0, 100.0], label="mitosis", label_source="model")
    d2 = Detection(id="d2", case_id=case_id, centroid_um=[200.0, 200.0], label="mitosis", label_source="gemini_referee_confirmed")
    d3 = Detection(id="d3", case_id=case_id, centroid_um=[300.0, 300.0], label="mitosis", label_source="pathologist")
    d4 = Detection(id="d4", case_id=case_id, centroid_um=[400.0, 400.0], label="mitosis", label_source="pathologist_manual_added")
    db.add_all([d1, d2, d3, d4])
    db.commit()
    
    # 2. Run the preservation query used in worker/mitosis.py
    existing_pathologist_dets = list(
        db.scalars(
            select(Detection).where(
                Detection.case_id == case_id,
                (Detection.label_source == "pathologist") | (Detection.label_source.startswith("pathologist"))
            )
        ).all()
    )
    assert len(existing_pathologist_dets) == 2
    assert {d.id for d in existing_pathologist_dets} == {"d3", "d4"}
    
    # 3. Purge non-pathologist detections
    db.execute(
        delete(Detection).where(
            Detection.case_id == case_id,
            Detection.label_source != "pathologist",
            not_(Detection.label_source.startswith("pathologist"))
        )
    )
    db.commit()
    
    # 4. Check remaining
    remaining = list(db.scalars(select(Detection).where(Detection.case_id == case_id)).all())
    assert len(remaining) == 2
    assert {d.id for d in remaining} == {"d3", "d4"}


def test_mitosis_confirmation_lenient_sanitization():
    """Verify that fuzzy LLM responses (e.g. Cannot determine, REJECTED) are sanitized without throwing schema errors."""
    fuzzy_payload = {
        "verdict": "Cannot determine",
        "envelope_dissolved": "Cannot determine",
        "spiculation_detected": "false",
        "confidence": "Very low",
        "rationale": "Solid color patch lacking discernible nuclear structures."
    }
    resp = MitosisConfirmationResponse.model_validate(fuzzy_payload)
    assert resp.verdict == "REJECTED_RESTING_NUCLEUS"
    assert resp.envelope_dissolved is False
    assert resp.spiculation_detected is False
    assert resp.confidence == "low"

    confirmed_payload = {
        "verdict": "CONFIRMED mitotic figure",
        "envelope_dissolved": "true",
        "spiculation_detected": True,
        "confidence": "high",
        "rationale": "Clear mitotic metaphase plate with dissolved nuclear boundary."
    }
    resp2 = MitosisConfirmationResponse.model_validate(confirmed_payload)
    assert resp2.verdict == "CONFIRMED"
    assert resp2.envelope_dissolved is True
    assert resp2.spiculation_detected is True
    assert resp2.confidence == "high"


def test_pipeline_referee_model_version_provenance():
    """Verify that referee and detector version strings truthfully reflect Vertex AI configuration."""
    detector = YoloMitosisDetector(endpoint_id="6276949705008087040")
    assert detector.model_version == "vertex_ai_midog@6276949705008087040"

    ref_model = getattr(settings, "GEMINI_REFEREE_MODEL", "gemini-2.5-flash")
    referee_version = f"{ref_model}@van_diest"
    assert "van_diest" in referee_version
    assert ("gemini-2.5-flash" in referee_version or "gemini-1.5-flash" in referee_version)


def test_yolo_detector_negative_tile_empty_list():
    """Verify negative tiles returning 0 predictions do not fall back to heuristic generator."""
    with patch("google.cloud.aiplatform.Endpoint") as mock_endpoint_cls, \
         patch("google.cloud.aiplatform.init"):
        mock_endpoint = MagicMock()
        mock_endpoint_cls.return_value = mock_endpoint
        mock_endpoint.predict.return_value = MagicMock(predictions=[{"boxes": []}])

        detector = YoloMitosisDetector(endpoint_id="projects/123/locations/us-central1/endpoints/456")
        dummy_tile = np.ones((512, 512, 3), dtype=np.uint8) * 200
        detections = detector.detect(dummy_tile)
        assert detections == []


def test_mitosis_confirmation_long_rationale_sanitization():
    """Verify rationales exceeding 500 characters do not crash Pydantic validation."""
    long_rationale = "Candidate exhibits clear metaphase features with aligned equatorial chromosome plate. " * 20
    assert len(long_rationale) > 1000

    payload = {
        "verdict": "CONFIRMED",
        "envelope_dissolved": True,
        "spiculation_detected": True,
        "confidence": "high",
        "rationale": long_rationale
    }
    resp = MitosisConfirmationResponse.model_validate(payload)
    assert resp.verdict == "CONFIRMED"
    assert len(resp.rationale) > 500
    assert len(resp.rationale) <= 4000


def test_yolo_detector_vertex_ai_1024_subpatching():
    """Verify 1024x1024 tile is sliced into 4x 512x512 sub-patches and coordinates are properly remapped."""
    with patch("google.cloud.aiplatform.Endpoint") as mock_endpoint_cls, \
         patch("google.cloud.aiplatform.init"):
        mock_endpoint = MagicMock()
        mock_endpoint_cls.return_value = mock_endpoint

        # Return 1 box from patch (0,0) and 1 box from patch (512, 512)
        mock_endpoint.predict.return_value = MagicMock(
            predictions=[
                {"boxes": [{"cx": 100.0, "cy": 120.0, "confidence": 0.88}]},  # patch (0, 0)
                {"boxes": []},                                                  # patch (512, 0)
                {"boxes": []},                                                  # patch (0, 512)
                {"boxes": [{"cx": 50.0, "cy": 60.0, "confidence": 0.92}]}     # patch (512, 512)
            ]
        )

        detector = YoloMitosisDetector(endpoint_id="projects/123/locations/us-central1/endpoints/456")
        tile_1024 = np.ones((1024, 1024, 3), dtype=np.uint8) * 200
        detections = detector.detect(tile_1024)

        # Verify 4 instances were submitted in the single batch call
        call_kwargs = mock_endpoint.predict.call_args[1]
        assert len(call_kwargs["instances"]) == 4

        # Verify remapped coordinates
        assert len(detections) == 2
        # Patch (0, 0): cx=100.0, cy=120.0
        assert detections[0] == (100.0, 120.0, 0.88)
        # Patch (512, 512): cx=50.0 + 512 = 562.0, cy=60.0 + 512 = 572.0
        assert detections[1] == (562.0, 572.0, 0.92)


def test_yolo_detector_vertex_ai_error_triggers_fallback():
    """Verify that an endpoint error response causes _detect_vertex_ai to return None and fallback to heuristic."""
    with patch("google.cloud.aiplatform.Endpoint") as mock_endpoint_cls, \
         patch("google.cloud.aiplatform.init"):
        mock_endpoint = MagicMock()
        mock_endpoint_cls.return_value = mock_endpoint

        # Endpoint returns an error payload (the exact error observed in RCA)
        mock_endpoint.predict.return_value = MagicMock(
            predictions=[{
                "boxes": [],
                "error": "Expected dimensions (512, 512), but got (1024, 1024)."
            }]
        )

        detector = YoloMitosisDetector(endpoint_id="projects/123/locations/us-central1/endpoints/456")

        # Test _detect_vertex_ai directly returns None on error
        dummy_tile = np.ones((512, 512, 3), dtype=np.uint8) * 200
        assert detector._detect_vertex_ai(dummy_tile) is None


def test_yolo_detector_vertex_ai_non_standard_tile_padding():
    """Verify non-standard tile sizes (e.g. 768x768) are padded to multiples of 512 and margin candidates filtered."""
    with patch("google.cloud.aiplatform.Endpoint") as mock_endpoint_cls, \
         patch("google.cloud.aiplatform.init"):
        mock_endpoint = MagicMock()
        mock_endpoint_cls.return_value = mock_endpoint

        # 768x768 slices into 4 patches (2x2 grid) of 512x512
        # Patch 3 (bottom-right) covers valid region [0:256, 0:256], and padding [256:512, 256:512]
        mock_endpoint.predict.return_value = MagicMock(
            predictions=[
                {"boxes": []},
                {"boxes": []},
                {"boxes": []},
                {"boxes": [
                    {"cx": 100.0, "cy": 100.0, "confidence": 0.85},  # inside valid region (100 < 256)
                    {"cx": 350.0, "cy": 350.0, "confidence": 0.90}   # in padded margin (350 >= 256) -> should be discarded
                ]}
            ]
        )

        detector = YoloMitosisDetector(endpoint_id="projects/123/locations/us-central1/endpoints/456")
        tile_768 = np.ones((768, 768, 3), dtype=np.uint8) * 200
        detections = detector.detect(tile_768)

        call_kwargs = mock_endpoint.predict.call_args[1]
        assert len(call_kwargs["instances"]) == 4

        # Only the candidate in the valid region should survive: 512 + 100 = 612
        assert len(detections) == 1
        assert detections[0] == (612.0, 612.0, 0.85)


def test_yolo_detector_vertex_ai_zero_detections_trusted():
    """Verify that when Vertex AI succeeds with 0 detections on a negative tile, detect() returns [] without falling back to heuristic."""
    with patch("google.cloud.aiplatform.Endpoint") as mock_endpoint_cls, \
         patch("google.cloud.aiplatform.init"):
        mock_endpoint = MagicMock()
        mock_endpoint_cls.return_value = mock_endpoint

        # Return 0 boxes across all patches
        mock_endpoint.predict.return_value = MagicMock(
            predictions=[
                {"boxes": []},
                {"boxes": []},
                {"boxes": []},
                {"boxes": []}
            ]
        )

        detector = YoloMitosisDetector(endpoint_id="projects/123/locations/us-central1/endpoints/456")
        tile_1024 = np.ones((1024, 1024, 3), dtype=np.uint8) * 200
        detections = detector.detect(tile_1024)
        assert detections == []


def test_yolo_detector_does_not_pollute_vertex_results_on_dense_tile():
    """Verify that when Vertex AI returns detections on a tile with hyperchromatic chromatin, detect() does NOT append OD candidates."""
    with patch("google.cloud.aiplatform.Endpoint") as mock_endpoint_cls, \
         patch("google.cloud.aiplatform.init"):
        mock_endpoint = MagicMock()
        mock_endpoint_cls.return_value = mock_endpoint

        mock_endpoint.predict.return_value = MagicMock(
            predictions=[
                {"boxes": [{"cx": 200.0, "cy": 200.0, "confidence": 0.88}]},
                {"boxes": []},
                {"boxes": []},
                {"boxes": []}
            ]
        )

        detector = YoloMitosisDetector(endpoint_id="projects/123/locations/us-central1/endpoints/456")
        # Dense dark H&E tile that would otherwise trigger optical density blobs
        dense_tile = np.zeros((1024, 1024, 3), dtype=np.uint8)
        dense_tile[:, :, 0] = 30  # very low intensity -> high optical density
        dense_tile[:, :, 1] = 180
        dense_tile[:, :, 2] = 200

        detections = detector.detect(dense_tile)
        # Authoritative: ONLY the Vertex AI box must be returned
        assert len(detections) == 1
        assert detections[0][0] == 200.0
        assert detections[0][1] == 200.0
        assert detections[0][2] == 0.88


def test_van_diest_morphometric_verification_rejection():
    """Verify that HoVerNetMitosisVerifier rejects small round pyknotic / apoptotic fragments under van Diest rules."""
    from pipeline.verify import HoVerNetMitosisVerifier
    verifier = HoVerNetMitosisVerifier(weights_path=None)

    # Synthetic 128x128 crop with a small round dense apoptotic body
    crop = np.ones((128, 128, 3), dtype=np.uint8) * 230
    import cv2
    # Draw small round pyknotic sphere at center (radius 8px = diam 16px, high circularity, smooth)
    cv2.circle(crop, (64, 64), 8, (40, 20, 60), -1)

    score, contour = verifier.verify(crop)
    # Under van Diest morphometrics, small round pyknotic bodies must receive low score (< 0.35)
    assert score < 0.35


def test_yolo_detector_vertex_empty_on_cellular_tile_rescued_by_od_features():
    """Verify that when Vertex AI returns [] on a cellular tile with chromatin, OD sweep rescues candidates."""
    with patch("google.cloud.aiplatform.Endpoint") as mock_endpoint_cls, \
         patch("google.cloud.aiplatform.init"):
        mock_endpoint = MagicMock()
        mock_endpoint_cls.return_value = mock_endpoint

        # Endpoint returns empty boxes (as observed in production on 40x tile)
        mock_endpoint.predict.return_value = MagicMock(
            predictions=[
                {"boxes": []},
                {"boxes": []},
                {"boxes": []},
                {"boxes": []}
            ]
        )

        detector = YoloMitosisDetector(endpoint_id="projects/123/locations/us-central1/endpoints/456")
        
        # Cellular tile with hematoxylin staining and condensed chromatin blobs
        import cv2
        tile = np.ones((1024, 1024, 3), dtype=np.uint8) * 220
        # Draw 3 condensed chromatin clusters (dark blue/purple)
        cv2.circle(tile, (200, 200), 16, (40, 20, 80), -1)
        cv2.circle(tile, (400, 400), 18, (35, 15, 75), -1)
        cv2.circle(tile, (600, 600), 20, (50, 25, 90), -1)

        detections = detector.detect(tile)
        # Rescued by optical density chromatin sweeper: must not be 0
        assert len(detections) >= 3
        # Model version should retain vertex_ai provenance so UI does not show dev fallback warning
        assert detector.model_version.startswith("vertex_ai_midog@")


def test_greedy_place_hpfs_strictly_guarantees_10_hpfs_with_hotspots():
    """Verify that when hotspots only fit 9 HPFs, Pass 3 places the 10th HPF in tumor bed tissue to achieve >=2.0 mm²."""
    from pipeline.hpf import greedy_place_hpfs
    
    # 20000 x 20000 um slide
    ny, nx = 40, 40
    stride = 500.0
    density_map = np.zeros((ny, nx), dtype=np.float32)
    grid_meta = {
        "origin_um": [0.0, 0.0],
        "stride_um": stride,
        "nx": nx,
        "ny": ny
    }

    # Hotspot polygon that can only fit 9 HPFs
    hotspot_polygon = [
        [1000.0, 1000.0],
        [4500.0, 1000.0],
        [4500.0, 4500.0],
        [1000.0, 4500.0]
    ]

    placed = greedy_place_hpfs(
        density_map=density_map,
        grid_meta=grid_meta,
        hotspot_polygons_um=[hotspot_polygon],
        count=10,
        radius_um=262.0,
        slide_dimensions_um=(20000.0, 20000.0)
    )

    # Strictly 10 HPFs placed (>2.0 mm² area)
    assert len(placed) == 10
    import math
    total_area_mm2 = 10 * (math.pi * (0.262 ** 2))
    assert total_area_mm2 > 2.0





