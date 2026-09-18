import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Case, Slide, StageExecution, AuditEvent, Hotspot
from app.core.db import Base
from worker.triage import run_triage


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def synthetic_triage_env(monkeypatch):
    """
    Ensure run_triage uses mock GCS and mock Vertex AI by default,
    guaranteeing completely offline execution.
    """
    from app.core.config import settings
    monkeypatch.setattr(settings, "USE_REAL_GCS", False)
    monkeypatch.setattr(settings, "USE_MOCK_VERTEX_AI", True)
    return settings


def test_run_triage_stage_e2e(db_session, tmp_path, synthetic_triage_env, monkeypatch):
    case_id = "test_case_triage_123"
    slide_id = "test_slide_triage_456"

    # Seed Case & Slide with created_by
    case = Case(id=case_id, created_by="test_user", status="processing")
    slide = Slide(
        id=slide_id,
        case_id=case_id,
        gcs_uri_original="gs://raw/test.svs",
        mpp_x=0.25,
        mpp_y=0.25,
        width_px=10000,
        height_px=10000
    )
    stage_exec = StageExecution(
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="running",
        input_ref={"slide_id": slide_id}
    )
    db_session.add(case)
    db_session.add(slide)
    db_session.add(stage_exec)
    db_session.commit()

    # Run triage worker handler
    output_ref, model_versions = run_triage(stage_exec, db_session)

    assert "triage/output.json" in output_ref
    assert model_versions["path_foundation"] == "v1"
    assert stage_exec.status == "awaiting_review"

    # Verify second run uses cached parquet embeddings (0 new endpoint calls) (#459)
    def fail_if_endpoint_called(*args, **kwargs):
        raise AssertionError("Vertex AI endpoint should NOT be invoked when embeddings are cached in parquet!")

    monkeypatch.setattr("worker.triage.mock_vertex_ai_endpoint", fail_if_endpoint_called)

    stage_exec.status = "running"
    db_session.commit()

    output_ref_2, _ = run_triage(stage_exec, db_session)
    assert output_ref_2 == output_ref


def test_triage_worker_raises_cleanly_when_mock_disabled(db_session, monkeypatch):
    """
    Issue #71 & #72:
    On Vertex AI failure, only use mock fallback if settings.USE_MOCK_VERTEX_AI is true;
    otherwise raise/fail the stage cleanly instead of substituting random noise.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "USE_MOCK_VERTEX_AI", False)
    monkeypatch.setattr(settings, "VERTEX_PATH_FOUNDATION_ENDPOINT_ID", "")

    case_id = "test_case_triage_fail"
    slide_id = "test_slide_triage_fail"
    case = Case(id=case_id, created_by="test_user", status="processing")
    slide = Slide(
        id=slide_id,
        case_id=case_id,
        gcs_uri_original="gs://raw/test.svs",
        mpp_x=0.25,
        mpp_y=0.25,
        width_px=1000,
        height_px=1000
    )
    stage_exec = StageExecution(
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="running",
        input_ref={"slide_id": slide_id}
    )
    db_session.add(case)
    db_session.add(slide)
    db_session.add(stage_exec)
    db_session.commit()

    with pytest.raises(RuntimeError, match="(Could not extract real|Endpoint ID is required)"):
        run_triage(stage_exec, db_session)


def test_vertex_path_foundation_client_ignores_dedicated_prediction_dns_in_init(monkeypatch):
    """
    Ensure dedicated prediction endpoints (*.prediction.vertexai.goog) are never
    passed to aiplatform.init() as control plane api_endpoint (which causes 501 UNIMPLEMENTED).
    """
    from worker.triage import VertexPathFoundationClient
    from app.core.config import settings
    from PIL import Image

    monkeypatch.setattr(settings, "USE_MOCK_VERTEX_AI", False)

    captured_init_kwargs = {}

    class MockEndpoint:
        def __init__(self, endpoint_name, project, location):
            self.endpoint_name = endpoint_name
            self.project = project
            self.location = location

        def raw_predict(self, body, headers):
            class MockResponse:
                def json(self):
                    return {
                        "predictions": [
                            {"result": {"patch_embeddings": [{"embedding_vector": [0.1] * 384}]}}
                        ]
                    }
            return MockResponse()

    def mock_aiplatform_init(**kwargs):
        nonlocal captured_init_kwargs
        captured_init_kwargs = kwargs

    import google.cloud.aiplatform as mock_aiplatform
    monkeypatch.setattr(mock_aiplatform, "init", mock_aiplatform_init)
    monkeypatch.setattr(mock_aiplatform, "Endpoint", MockEndpoint)

    # 1. Test with dedicated prediction endpoint DNS -> must NOT be in init kwargs
    client = VertexPathFoundationClient(
        endpoint_id="mg-endpoint-test",
        location="asia-south1",
        project_id="oncogemma",
        api_endpoint="mg-endpoint-test.asia-south1-962838713357.prediction.vertexai.goog"
    )
    patches = [Image.new("RGB", (224, 224), (200, 200, 200))]
    embs = client.predict_embeddings(patches=patches)
    assert embs.shape == (1, 384)
    assert "api_endpoint" not in captured_init_kwargs
    assert captured_init_kwargs["project"] == "oncogemma"
    assert captured_init_kwargs["location"] == "asia-south1"

    # 2. Test with control plane endpoint -> allowed in init kwargs
    client2 = VertexPathFoundationClient(
        endpoint_id="mg-endpoint-test",
        location="asia-south1",
        project_id="oncogemma",
        api_endpoint="asia-south1-aiplatform.googleapis.com"
    )
    client2.predict_embeddings(patches=patches)
    assert captured_init_kwargs.get("api_endpoint") == "asia-south1-aiplatform.googleapis.com"

