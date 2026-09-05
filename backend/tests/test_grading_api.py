import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.db import Base, get_db
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.grading import Grading
from app.models.hpf_site import HpfSite

# Shared in-memory test database
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
def setup_db_override():
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


def test_grading_api_full_workflow():
    client = TestClient(app)
    db = TestingSessionLocal()
    case_id = uuid.uuid4()
    case = Case(id=case_id, created_by="test_pathologist", status="open")
    slide = Slide(id=uuid.uuid4(), case_id=case_id, gcs_uri_original="gs://oncogemma-dev-raw/test.svs", mpp_x=0.25, mpp_y=0.25)
    
    stage_exec = StageExecution(
        case_id=case_id,
        stage="grading",
        attempt=1,
        status="awaiting_review"
    )
    
    machine_payload = {
        "case_id": str(case_id),
        "patches": [
            {
                "id": "p_001",
                "index": 1,
                "tumor_probability": 0.95,
                "tubule": {"tubule_percent": 25, "tumor_present": True, "confidence": "high"},
                "pleo": {"pleomorphism_score": 2, "rationale": "Moderate atypia", "confidence": "medium"},
                "image_url": f"/api/v1/stages/grading/{case_id}/patches/p_001/image"
            }
        ],
        "aggregate": {
            "tubule_percent": 25.0,
            "tubule_score": 2,
            "pleo_score": 2,
            "mitotic_score": 3,
            "nottingham_sum": 7,
            "grade": 2,
            "flags": []
        },
        "histologic_type": {
            "type": "IDC-NST",
            "differential": ["ILC"],
            "rationale": "Invasive carcinoma with glandular formation",
            "confidence": "high"
        },
        "narrative": "Nottingham Histological Grade 2 (Score 7/9).",
        "model_versions": {"medgemma": "1.5@2026.08"}
    }

    grading = Grading(
        case_id=case_id,
        tubule_percent=25.0,
        tubule_score=2,
        pleo_score=2,
        mitotic_score=3,
        nottingham_sum=7,
        grade=2,
        histologic_type="IDC-NST",
        type_confirmed_by="unconfirmed",
        machine=machine_payload,
        overrides={}
    )

    db.add(case)
    db.add(slide)
    db.add(stage_exec)
    db.add(grading)
    for i in range(10):
        db.add(HpfSite(
            case_id=case_id,
            seq=i + 1,
            center_um=[1000.0 * (i + 1), 1000.0 * (i + 1)],
            mitotic_count=2,
            radius_um=262.0,
            source="model"
        ))
    db.commit()
    db.close()

    # 1. GET /api/v1/stages/grading/{case_id}
    res = client.get(f"/api/v1/stages/grading/{case_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == str(case_id)
    assert data["machine"]["nottingham_sum"] == 7
    assert data["machine"]["grade"] == 2
    assert data["histologic_type"]["proposed_type"] == "IDC-NST"
    assert data["histologic_type"]["is_confirmed"] is False
    assert data["review_summary"]["all_patches_reviewed"] is False
    assert data["review_summary"]["all_hpfs_reviewed"] is False
    assert data["review_summary"]["can_confirm"] is False

    # 2. POST /api/v1/stages/grading/recompute (override Pleo to 3)
    recompute_payload = {
        "case_id": str(case_id),
        "tubule_score": 2,
        "pleo_score": 3,
        "mitotic_score": 3
    }
    res_rec = client.post("/api/v1/stages/grading/recompute", json=recompute_payload)
    assert res_rec.status_code == 200
    rec_data = res_rec.json()
    assert rec_data["nottingham_sum"] == 8  # 2 + 3 + 3 = 8
    assert rec_data["grade"] == 3          # Grade 3
    assert rec_data["is_overridden"] is True

    # 3. Patch Review: Update single patch and then Approve All
    patch_update_payload = {
        "case_id": str(case_id),
        "reviewed_by": "Dr. Smith",
        "action": "update",
        "reviews": [
            {
                "patch_id": "p_001",
                "tubule_percent": 30.0,
                "tumor_present": True,
                "pleomorphism_score": 3,
                "status": "modified",
                "notes": "Higher atypia in this field"
            }
        ]
    }
    res_patch = client.post("/api/v1/stages/grading/patches/review", json=patch_update_payload)
    assert res_patch.status_code == 200
    patch_data = res_patch.json()
    assert patch_data["patches"][0]["review_status"] == "modified"
    assert patch_data["patches"][0]["user_tubule_percent"] == 30.0
    assert patch_data["patches"][0]["user_pleo_score"] == 3
    assert patch_data["review_summary"]["all_patches_reviewed"] is True

    # 4. HPF Review: Approve All HPFs
    hpf_approve_payload = {
        "case_id": str(case_id),
        "reviewed_by": "Dr. Smith",
        "action": "approve_all"
    }
    res_hpf = client.post("/api/v1/stages/grading/hpfs/review", json=hpf_approve_payload)
    assert res_hpf.status_code == 200
    hpf_data = res_hpf.json()
    assert hpf_data["review_summary"]["all_hpfs_reviewed"] is True

    # 5. POST /api/v1/stages/grading/confirm - Blocked if type_confirmed is False
    blocked_payload = {
        "case_id": str(case_id),
        "reviewed_by": "Dr. Smith",
        "histologic_type": "IDC-NST",
        "type_confirmed": False, # NOT CONFIRMED
        "overrides": {},
        "tubule_score": 2,
        "pleo_score": 2,
        "mitotic_score": 3,
        "nottingham_sum": 7,
        "grade": 2
    }
    res_block = client.post("/api/v1/stages/grading/confirm", json=blocked_payload)
    assert res_block.status_code == 400
    assert "Histologic Type must be explicitly confirmed" in res_block.json()["detail"]

    # 6. POST /api/v1/stages/grading/confirm - Blocked if override justification < 10 chars
    short_just_payload = {
        "case_id": str(case_id),
        "reviewed_by": "Dr. Smith",
        "histologic_type": "IDC-NST",
        "type_confirmed": True,
        "overrides": {
            "pleo": {
                "score": 3,
                "original_score": 2,
                "justification": "short" # Only 5 chars < 10
            }
        },
        "tubule_score": 2,
        "pleo_score": 3,
        "mitotic_score": 3,
        "nottingham_sum": 8,
        "grade": 3
    }
    res_short = client.post("/api/v1/stages/grading/confirm", json=short_just_payload)
    assert res_short.status_code == 400
    assert "minimum 10-character justification" in res_short.json()["detail"]

    # 7. POST /api/v1/stages/grading/confirm - Success with valid sign-off and justification
    valid_confirm_payload = {
        "case_id": str(case_id),
        "reviewed_by": "Dr. Smith",
        "histologic_type": "IDC-NST",
        "type_confirmed": True,
        "overrides": {
            "pleo": {
                "score": 3,
                "original_score": 2,
                "justification": "Severe vesicular change and prominent nucleoli observed across peripheral invasive tumor nests."
            }
        },
        "tubule_score": 2,
        "pleo_score": 3,
        "mitotic_score": 3,
        "nottingham_sum": 8,
        "grade": 3
    }
    res_confirm = client.post("/api/v1/stages/grading/confirm", json=valid_confirm_payload)
    assert res_confirm.status_code == 200
    conf_data = res_confirm.json()
    assert conf_data["status"] == "success"
    assert conf_data["grade"] == 3
    assert conf_data["next_stage"] == "report"

    # Verify persisted row in DB
    db2 = TestingSessionLocal()
    updated_grading = db2.scalars(select(Grading).where(Grading.case_id == case_id)).first()
    assert updated_grading.grade == 3
    assert updated_grading.nottingham_sum == 8
    assert updated_grading.pleo_score == 3
    assert updated_grading.type_confirmed_by == "Dr. Smith"
    db2.close()

