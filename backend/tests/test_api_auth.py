import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.db import Base, get_db

# In-memory SQLite DB for fast isolated unit testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def setup_auth_test_db():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)

client = TestClient(app)

def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_mock_auth_headers():
    # Valid default header
    response = client.get("/api/v1/cases", headers={"X-User-Role": "pathologist"})
    assert response.status_code == 200

    # Invalid role header -> 403 Forbidden
    response_invalid = client.get("/api/v1/cases", headers={"X-User-Role": "unauthorized_role"})
    assert response_invalid.status_code == 403

def test_create_and_get_case():
    # Create case
    res = client.post("/api/v1/cases", headers={"X-User-Role": "pathologist", "X-User-Id": "path_001"})
    assert res.status_code == 201
    case_data = res.json()
    case_id = case_data["id"]
    assert case_data["created_by"] == "path_001"

    # Get case details
    res_detail = client.get(f"/api/v1/cases/{case_id}", headers={"X-User-Role": "pathologist"})
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["id"] == case_id


def test_rbac_case_permissions():
    # 1. Viewer cannot create a case -> 403
    res_v_create = client.post("/api/v1/cases", headers={"X-User-Role": "viewer", "X-User-Id": "viewer_1"})
    assert res_v_create.status_code == 403

    # 2. Admin can create a case -> 201
    res_a_create = client.post("/api/v1/cases", headers={"X-User-Role": "admin", "X-User-Id": "admin_1"})
    assert res_a_create.status_code == 201
    case_id = res_a_create.json()["id"]

    # 3. Viewer cannot delete a case -> 403
    res_v_del = client.delete(f"/api/v1/cases/{case_id}", headers={"X-User-Role": "viewer"})
    assert res_v_del.status_code == 403

    # 4. Pathologist cannot clear all cases -> 403
    res_p_clear = client.delete("/api/v1/cases", headers={"X-User-Role": "pathologist"})
    assert res_p_clear.status_code == 403

    # 5. Admin can clear all cases -> 200
    res_a_clear = client.delete("/api/v1/cases", headers={"X-User-Role": "admin"})
    assert res_a_clear.status_code == 200
    assert res_a_clear.json()["status"] == "cleared"
