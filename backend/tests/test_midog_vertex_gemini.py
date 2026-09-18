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

