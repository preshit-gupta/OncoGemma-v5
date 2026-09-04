"""
Root Pytest Configuration and Global Isolation Fixtures.

Finding #405: Guarantees that the test suite runs 100% offline and isolated by default:
- Sets USE_REAL_GCS="false" to prevent hitting live Google Cloud Storage.
- Sets USE_MOCK_VERTEX_AI="true" to prevent hitting live Vertex AI endpoints.
- Sets ENV="test" to ensure test runtime configuration.
"""

import os
import pytest

# Configure environment variables before any application modules are imported
os.environ["USE_REAL_GCS"] = "false"
os.environ["USE_MOCK_VERTEX_AI"] = "true"
os.environ["ENV"] = "test"
os.environ["ENVIRONMENT"] = "test"

# Update the singleton settings instance
from app.core.config import settings

settings.USE_REAL_GCS = False
settings.USE_MOCK_VERTEX_AI = True
settings.ENV = "test"
settings.ENVIRONMENT = "test"


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch):
    """
    Guarantees every test runs with offline and isolated settings by default.
    Individual tests may explicitly monkeypatch these settings if testing failure modes.
    """
    monkeypatch.setenv("USE_REAL_GCS", "false")
    monkeypatch.setenv("USE_MOCK_VERTEX_AI", "true")
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setattr(settings, "USE_REAL_GCS", False)
    monkeypatch.setattr(settings, "USE_MOCK_VERTEX_AI", True)
    monkeypatch.setattr(settings, "ENV", "test")
    monkeypatch.setattr(settings, "ENVIRONMENT", "test", raising=False)

    import app.core.gcs as gcs
    monkeypatch.setattr(gcs, "_gcs_client", None)

