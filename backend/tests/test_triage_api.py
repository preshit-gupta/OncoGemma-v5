import os
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models # Load all ORM models into Base.metadata
from app.main import app
from app.core.db import Base, get_db
from app.models.case import Case
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot


@pytest.fixture
def client_and_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    db = TestingSessionLocal()
    yield client, db
    db.close()
    app.dependency_overrides.clear()


def test_triage_api_workflow(client_and_db):
    client, db = client_and_db
    case_id = "test_case_api_789"
    exec_id = "00000000-0000-0000-0000-000000000001"

    # Seed Case and StageExecution
    c = Case(id=case_id, created_by="test_user", status="processing")
    se = StageExecution(
        id=exec_id,
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="awaiting_review",
        input_ref={},
        output_ref=f"gs://og-artifacts-local/cases/{case_id}/triage/output.json"
    )
    db.add(c)
    db.add(se)
    db.commit()

    mock_output = {
        "heatmap_png_uri": "/artifacts/heatmap.png",
        "prob_grid_uri": "/artifacts/probs.npy",
        "grid": {"origin_um": [0, 0], "stride_um": 224, "nx": 10, "ny": 10},
        "hotspots": [
            {
                "id": "hs_01",
                "polygon_um": [[0, 0], [224, 0], [224, 224], [0, 224]],
                "area_mm2": 0.05,
                "prob_mean": 0.85,
                "prob_max": 0.95,
                "source": "model",
                "excluded": False
            }
        ]
    }
    mock_bytes = json.dumps(mock_output).encode("utf-8")

    from unittest.mock import patch
    with patch("app.routers.triage.download_blob_as_bytes", return_value=mock_bytes):
        # 1. GET triage data
        res_get = client.get(f"/api/v1/stages/triage/{case_id}")
        assert res_get.status_code == 200
        data = res_get.json()
        assert data["case_id"] == case_id
        assert len(data["machine_hotspots"]) == 1

        # 2. POST edits (add user hotspot & exclude hs_01)
        edits = [
            {"op": "exclude", "id": "hs_01", "reason": "DCIS only"},
            {"op": "add", "id": "user_01", "polygon_um": [[500, 500], [700, 500], [700, 700], [500, 700]], "area_mm2": 0.04}
        ]
        res_edits = client.post("/api/v1/stages/triage/edits", json={"case_id": case_id, "edits": edits})
        assert res_edits.status_code == 200
        assert res_edits.json()["edits_count"] == 2

        # 3. POST confirm
        res_confirm = client.post("/api/v1/stages/triage/confirm", json={"case_id": case_id, "no_invasive_tumor": False})
        assert res_confirm.status_code == 200
        confirm_data = res_confirm.json()
        assert confirm_data["status"] == "confirmed"
        assert confirm_data["next_stage_queued"] == "mitosis"

    # Verify DB hotspots records
    db_hotspots = db.query(Hotspot).filter(Hotspot.case_id == case_id).all()
    assert len(db_hotspots) == 2
