"""
Test Suite for Batches 19–21: System-Wide Hardening, Pydantic v2 Migration & Code Polish.

Validates:
1. Pydantic v2 migration: SettingsConfigDict, ConfigDict(from_attributes=True), zero PydanticDeprecatedSince20 warnings.
2. CORS security: Strict whitelist parsing, Cloud Run regex matching, rejection of unauthorized origins.
3. Health check endpoints: Multi-route availability (/health, /api/health, /healthz, /api/v1/health) and database ping.
4. Router hygiene & 404 integrity: Clean error structures and consistent status code responses.
"""

import os
import uuid
import warnings
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import Settings, settings
from app.schemas.case import CaseResponse, CaseDetailResponse
from app.schemas.audit import AuditEventResponse


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_pydantic_v2_settings_model_config():
    """Verify Settings uses modern SettingsConfigDict without PydanticDeprecatedSince20 warnings."""
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        s = Settings()
        assert s.CORS_ORIGINS is not None
        assert "http://localhost:3000" in s.CORS_ORIGINS
    
    # Assert no Pydantic deprecation warnings
    pydantic_warnings = [
        w for w in recorded_warnings 
        if "PydanticDeprecatedSince20" in getattr(w.category, "__name__", "")
    ]
    assert len(pydantic_warnings) == 0, f"Found unexpected Pydantic v2 deprecation warnings: {pydantic_warnings}"


def test_pydantic_v2_schema_from_attributes():
    """Verify schemas serialize objects with from_attributes=True cleanly."""
    class DummyCase:
        id = uuid.uuid4()
        created_by = "dr_smith"
        case_number = "ONC-2026-TEST"
        patient_id_hash = "hash_xyz"
        clinical_data = {"age": 52, "procedure": "lumpectomy"}
        status = "open"
        current_stage = "triage"
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)
        slides = []
        stages = []
        hotspots = []
        grading = None
        report = None

    class DummyAudit:
        id = 1
        case_id = str(uuid.uuid4())
        actor = "dr_smith"
        event_type = "stage_approved"
        stage = "triage"
        payload = {"next_stage": "mitosis"}
        created_at = datetime.now(timezone.utc)

    # Validate CaseResponse from attributes
    case_resp = CaseResponse.model_validate(DummyCase())
    assert case_resp.created_by == "dr_smith"
    assert case_resp.status == "open"

    # Validate CaseDetailResponse from attributes
    detail_resp = CaseDetailResponse.model_validate(DummyCase())
    assert detail_resp.created_by == "dr_smith"
    assert detail_resp.status == "open"
    assert detail_resp.slides == []

    # Validate AuditEventResponse from attributes
    audit_resp = AuditEventResponse.model_validate(DummyAudit())
    assert audit_resp.id == 1
    assert audit_resp.actor == "dr_smith"
    assert audit_resp.event_type == "stage_approved"
    assert audit_resp.payload["next_stage"] == "mitosis"


def test_cors_whitelist_allowed_origin(client):
    """Verify CORS middleware responds with correct headers for whitelisted localhost origin."""
    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:3000"}
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_regex_allowed_cloud_run_origin(client):
    """Verify CORS middleware allows Cloud Run subdomains matching regex pattern."""
    cr_origin = "https://oncogemma-frontend-522209116839.us-central1.run.app"
    response = client.get(
        "/health",
        headers={"Origin": cr_origin}
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == cr_origin
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_rejected_unauthorized_origin(client):
    """Verify CORS middleware rejects unauthorized external origins (#3)."""
    malicious_origin = "https://evil-unauthorized-domain.com"
    response = client.get(
        "/health",
        headers={"Origin": malicious_origin}
    )
    assert response.status_code == 200
    # Access-Control-Allow-Origin header must NOT reflect unauthorized origins
    assert response.headers.get("access-control-allow-origin") != malicious_origin


def test_all_health_endpoints(client):
    """Verify all health endpoints return 200 with consistent structure."""
    endpoints = ["/health", "/api/health", "/healthz", "/api/healthz", "/api/v1/health", "/api/v1/healthz"]
    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 200, f"Failed on endpoint {ep}"
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "4.5.0"
        assert "env" in data


def test_cases_router_not_found_handling(client):
    """Verify requesting a non-existent case returns 404 Not Found."""
    fake_case_id = str(uuid.uuid4())
    resp = client.get(f"/api/v1/cases/{fake_case_id}")
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()
