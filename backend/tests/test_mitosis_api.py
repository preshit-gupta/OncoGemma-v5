"""
Integration tests for Stage 4 Mitosis REST API endpoints, live recompute, and safety gate (v4.3).
"""
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.db import Base, get_db
from app.models.case import Case
from app.models.stage_execution import StageExecution
from app.models.detection import Detection
from app.models.hpf_site import HpfSite
from app.models.report import Report

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
def setup_mitosis_test_db():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)

client = TestClient(app)


@pytest.fixture
def setup_test_case():
    db = TestingSessionLocal()
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="pathologist_test", status="open")
    db.add(case)

    stage_exec = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="mitosis",
        attempt=1,
        status="awaiting_review"
    )
    db.add(stage_exec)

    # Add 10 HPFs
    for i in range(1, 11):
        hpf = HpfSite(
            case_id=case_id,
            seq=i,
            center_um=[1000.0 * i, 1000.0 * i],
            radius_um=262.0,
            mitotic_count=1 if i <= 3 else 0,
            source="model"
        )
        db.add(hpf)

    # Add candidates
    d1 = Detection(
        id="m_0001",
        case_id=case_id,
        centroid_um=[1010.0, 1010.0], # Inside HPF 1
        det_conf=0.92,
        ver_conf=0.88,
        label="mitosis",
        label_source="model"
    )
    d2 = Detection(
        id="m_0002",
        case_id=case_id,
        centroid_um=[2020.0, 2020.0], # Inside HPF 2
        det_conf=0.85,
        ver_conf=0.75,
        label="mitosis",
        label_source="model"
    )
    d3 = Detection(
        id="m_0003",
        case_id=case_id,
        centroid_um=[3010.0, 3010.0], # Inside HPF 3
        det_conf=0.65,
        ver_conf=0.60,
        label="unreviewed",
        label_source="model"
    )
    db.add_all([d1, d2, d3])
    db.commit()
    db.close()

    return str(case_id)


def test_get_mitosis_stage_data(setup_test_case):
    case_id = setup_test_case
    res = client.get(f"/api/v1/stages/mitosis/{case_id}", headers={"X-User-Role": "pathologist"})
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == case_id
    assert len(data["candidates"]) == 3
    assert len(data["hpfs"]) == 10
    assert "summary" in data
    assert data["summary"]["count_total"] == 2 # m_0001 and m_0002 confirmed mitoses
    assert data["summary"]["mitotic_score"] == 1


def test_recompute_endpoint(setup_test_case):
    case_id = setup_test_case
    # Toggle m_0003 from unreviewed to mitosis
    payload = {
        "case_id": case_id,
        "candidate_labels": {
            "m_0001": "mitosis",
            "m_0002": "mitosis",
            "m_0003": "mitosis"
        },
        "audit_toggle": {
            "id": "m_0003",
            "from": "unreviewed",
            "to": "mitosis"
        }
    }
    res = client.post("/api/v1/stages/mitosis/recompute", json=payload, headers={"X-User-Role": "pathologist"})
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["count_total"] == 3 # Now 3 confirmed mitoses


def test_add_candidate_endpoint(setup_test_case):
    case_id = setup_test_case
    payload = {
        "case_id": case_id,
        "centroid_um": [1005.0, 1005.0],
        "label": "mitosis",
        "reviewed_by": "pathologist_01"
    }
    res = client.post("/api/v1/stages/mitosis/add_candidate", json=payload, headers={"X-User-Role": "pathologist"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["candidate"]["label_source"] == "pathologist"


def test_bulk_action_endpoint(setup_test_case):
    case_id = setup_test_case
    payload = {
        "case_id": case_id,
        "action": "reject_remaining_unreviewed",
        "reviewed_by": "pathologist_01"
    }
    res = client.post("/api/v1/stages/mitosis/bulk_action", json=payload, headers={"X-User-Role": "pathologist"})
    assert res.status_code == 200
    data = res.json()
    # All unreviewed should now be not_mitosis
    unreviewed = [c for c in data["candidates"] if c["label"] == "unreviewed"]
    assert len(unreviewed) == 0


def test_confirm_safety_gate_blocking_and_success(setup_test_case):
    case_id = setup_test_case
    
    # Attempt confirm while unreviewed high-conf candidates exist
    db = TestingSessionLocal()
    # Re-insert high confidence unreviewed candidate
    db.add(Detection(
        id="m_high_conf",
        case_id=uuid.UUID(case_id),
        centroid_um=[1500.0, 1500.0],
        det_conf=0.88,
        ver_conf=0.82,
        label="unreviewed",
        label_source="model"
    ))
    db.commit()
    db.close()

    # Should fail 400 Bad Request
    res_fail = client.post(
        "/api/v1/stages/mitosis/confirm",
        json={"case_id": case_id, "reviewed_by": "pathologist_01"},
        headers={"X-User-Role": "pathologist"}
    )
    assert res_fail.status_code == 400
    assert "Clinical Safety Gate" in res_fail.json()["detail"]

    # Clear unreviewed via bulk reject
    client.post(
        "/api/v1/stages/mitosis/bulk_action",
        json={"case_id": case_id, "action": "reject_remaining_unreviewed", "reviewed_by": "pathologist_01"},
        headers={"X-User-Role": "pathologist"}
    )

    # Now confirm should succeed 200 OK and queue Stage 5 (grading)
    res_success = client.post(
        "/api/v1/stages/mitosis/confirm",
        json={"case_id": case_id, "reviewed_by": "pathologist_01"},
        headers={"X-User-Role": "pathologist"}
    )
    assert res_success.status_code == 200
    assert res_success.json()["next_stage"] == "grading"


def test_confirm_mitosis_guards_signed_report(setup_test_case):
    case_id = setup_test_case
    case_uid = uuid.UUID(case_id)

    db = TestingSessionLocal()
    # Add a signed report
    signed_rep = Report(
        case_id=case_uid,
        status="signed",
        signed_by="Dr. Attending Pathologist",
        signed_at=None,
        integrity_hash="abcdef123456"
    )
    db.add(signed_rep)

    # Set grading stage execution to done
    grading_exec = StageExecution(
        case_id=case_uid,
        stage="grading",
        attempt=1,
        status="done"
    )
    db.add(grading_exec)
    db.commit()
    db.close()

    # Clear unreviewed detections
    client.post(
        "/api/v1/stages/mitosis/bulk_action",
        json={"case_id": case_id, "action": "reject_remaining_unreviewed", "reviewed_by": "pathologist_01"},
        headers={"X-User-Role": "pathologist"}
    )

    # Calling confirm should succeed but NOT reset grading stage to queued
    res_confirm = client.post(
        "/api/v1/stages/mitosis/confirm",
        json={"case_id": case_id, "reviewed_by": "pathologist_01"},
        headers={"X-User-Role": "pathologist"}
    )
    assert res_confirm.status_code == 200

    db = TestingSessionLocal()
    g_exec = db.scalars(select(StageExecution).where(StageExecution.case_id == case_uid, StageExecution.stage == "grading")).first()
    assert g_exec.status == "done"  # Preserved, not clobbered to queued
    db.close()
