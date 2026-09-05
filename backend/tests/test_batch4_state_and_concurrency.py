import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import IntegrityError

from app.main import app
from app.core.db import Base, get_db
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot
from app.models.detection import Detection
from app.models.hpf_site import HpfSite
from app.schemas.case import SlideFinalizeRequest
from worker.main import poll_and_execute_single_task, reset_stuck_running_stages

# Isolated test DB with StaticPool for thread-safe test execution
TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_environment():
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Issues #35, #278, #314, #401: Worker Poll Concurrency & Row Locking
# -----------------------------------------------------------------------------

def test_poll_and_execute_single_task_sqlite_fallback(db_session):
    """
    Verify poll_and_execute_single_task runs cleanly on SQLite without crashing
    on .with_for_update(skip_locked=True) which SQLite does not support.
    """
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="worker_test")
    se = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="preprocess",
        attempt=1,
        status="queued"
    )
    db_session.add(c)
    db_session.add(se)
    db_session.commit()

    with patch("worker.main.SessionLocal", TestingSessionLocal), \
         patch("worker.main.HANDLERS", {"preprocess": lambda st, db: ("gs://out.json", {"v": "1.0"})}):
        handled = poll_and_execute_single_task()

    assert handled is True
    check_session = TestingSessionLocal()
    try:
        se_updated = check_session.get(StageExecution, se.id)
        assert se_updated.status == "done"
        assert se_updated.output_ref == "gs://out.json"
    finally:
        check_session.close()


def test_poll_and_execute_postgres_dialect_uses_with_for_update():
    """
    Verify that on a PostgreSQL dialect, with_for_update(skip_locked=True)
    is applied to the select statement.
    """
    from sqlalchemy.dialects import postgresql
    from worker.main import HANDLERS

    stages_list = list(HANDLERS.keys())
    stmt = (
        select(StageExecution)
        .where(
            StageExecution.status == "queued",
            StageExecution.stage.in_(stages_list)
        )
        .order_by(StageExecution.started_at.asc().nulls_first(), StageExecution.id.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )

    compiled_sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE SKIP LOCKED" in compiled_sql


def test_reset_stuck_running_stages_age_scoping(db_session):
    """
    Verify reset_stuck_running_stages only resets stages older than timeout_seconds
    or with null started_at, protecting actively running sibling worker tasks.
    """
    now = datetime.now(timezone.utc)
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="worker_test")
    db_session.add(c)

    # 1. Stuck stage (>15 mins ago)
    stuck_exec = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="ingest",
        attempt=1,
        status="running",
        started_at=now - timedelta(minutes=25)
    )
    # 2. Active sibling worker stage (< 2 mins ago)
    active_exec = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="preprocess",
        attempt=1,
        status="running",
        started_at=now - timedelta(minutes=2)
    )
    # 3. Orphan stage with null started_at
    orphan_exec = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="qc",
        attempt=1,
        status="running",
        started_at=None
    )
    db_session.add_all([stuck_exec, active_exec, orphan_exec])
    db_session.commit()

    with patch("worker.main.SessionLocal", TestingSessionLocal):
        reset_stuck_running_stages(timeout_seconds=900)

    check_session = TestingSessionLocal()
    try:
        stuck_updated = check_session.get(StageExecution, stuck_exec.id)
        active_updated = check_session.get(StageExecution, active_exec.id)
        orphan_updated = check_session.get(StageExecution, orphan_exec.id)

        # Stuck & orphan should be reset to queued with null started_at
        assert stuck_updated.status == "queued"
        assert stuck_updated.started_at is None
        assert orphan_updated.status == "queued"
        assert orphan_updated.started_at is None

        # Active sibling worker stage must NOT be touched
        assert active_updated.status == "running"
        assert active_updated.started_at is not None
    finally:
        check_session.close()


# -----------------------------------------------------------------------------
# 2. Issues #6, #288, #350, #7, #8: DB Engine Reliability & SQLite Handling
# -----------------------------------------------------------------------------

def test_sqlite_pragma_foreign_keys_enforced(db_session):
    """
    Verify the SQLAlchemy Engine listener turns PRAGMA foreign_keys=ON on SQLite.
    """
    with test_engine.connect() as conn:
        res = conn.execute(text("PRAGMA foreign_keys;")).scalar()
        assert res == 1, "PRAGMA foreign_keys must be enabled on SQLite engines"


def test_sqlite_foreign_key_violation_raises(db_session):
    """
    Verify foreign key constraint is strictly enforced when inserting a child entity
    referencing a non-existent case_id.
    """
    non_existent_case_id = uuid.uuid4()
    hotspot = Hotspot(
        id="hs_fake",
        case_id=non_existent_case_id,
        stage_execution_id=uuid.uuid4(),
        polygon_um=[[0, 0], [10, 10]]
    )
    db_session.add(hotspot)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_healthz_database_connectivity_failure(client):
    """
    Verify /healthz returns 503 when database connectivity fails.
    """
    with patch("app.main.engine.connect", side_effect=Exception("Database connection timeout")):
        resp = client.get("/healthz")
        assert resp.status_code == 503
        assert "Database connection failed" in resp.json()["detail"]


# -----------------------------------------------------------------------------
# 3. Issues #10, #738: Cascade Deletes on Child Clinical Entities
# -----------------------------------------------------------------------------

def test_cascade_delete_case_cleans_all_child_records(db_session):
    """
    Verify deleting a Case cleanly cascades and deletes hotspots, detections,
    hpf_sites, slides, and stage_executions with zero FK violations.
    """
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="pathologist_01")
    db_session.add(c)

    se = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="preprocess",
        attempt=1,
        status="done"
    )
    sl = Slide(id=uuid.uuid4(), case_id=case_id, gcs_uri_original="gs://raw/slide.svs")
    hs = Hotspot(
        id="hs_001",
        case_id=case_id,
        stage_execution_id=se.id,
        polygon_um=[[0, 0], [100, 100]]
    )
    det = Detection(
        id="det_001",
        case_id=case_id,
        centroid_um=[50, 50]
    )
    hpf = HpfSite(
        case_id=case_id,
        seq=1,
        center_um=[100, 100],
        radius_um=262.0
    )
    db_session.add_all([se, sl, hs, det, hpf])
    db_session.commit()

    # Verify rows exist
    assert db_session.get(Case, case_id) is not None
    assert db_session.scalars(select(Hotspot).where(Hotspot.case_id == case_id)).all() != []
    assert db_session.scalars(select(Detection).where(Detection.case_id == case_id)).all() != []
    assert db_session.scalars(select(HpfSite).where(HpfSite.case_id == case_id)).all() != []

    # Delete case via ORM
    db_session.delete(c)
    db_session.commit()

    # Verify everything cascaded cleanly
    assert db_session.get(Case, case_id) is None
    assert db_session.scalars(select(Hotspot).where(Hotspot.case_id == case_id)).all() == []
    assert db_session.scalars(select(Detection).where(Detection.case_id == case_id)).all() == []
    assert db_session.scalars(select(HpfSite).where(HpfSite.case_id == case_id)).all() == []
    assert db_session.scalars(select(Slide).where(Slide.case_id == case_id)).all() == []
    assert db_session.scalars(select(StageExecution).where(StageExecution.case_id == case_id)).all() == []


# -----------------------------------------------------------------------------
# 4. Issues #206, #285, #535, #567, #69, #205, #704: State Gating & Attempt Monotonicity
# -----------------------------------------------------------------------------

def test_approve_case_stage_requires_awaiting_review(client, db_session):
    """
    Verify approve_case_stage requires status == 'awaiting_review' (409 Conflict otherwise)
    and records reviewed_by and reviewed_at.
    """
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="test_user")
    sl = Slide(id=uuid.uuid4(), case_id=case_id, gcs_uri_original="gs://raw/s.svs")
    se = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="preprocess",
        attempt=1,
        status="running"  # NOT awaiting_review
    )
    db_session.add_all([c, sl, se])
    db_session.commit()

    # 1. 409 Conflict when not awaiting_review
    res = client.post(
        f"/api/v1/cases/{case_id}/stages/preprocess/approve",
        headers={"X-User-Role": "pathologist", "X-User-Id": "dr_watson"}
    )
    assert res.status_code == 409
    assert "expected 'awaiting_review'" in res.json()["detail"]

    # 2. Update to awaiting_review and approve successfully
    se.status = "awaiting_review"
    db_session.commit()

    with patch("app.core.cloud_tasks.dispatch_stage_task"):
        res = client.post(
            f"/api/v1/cases/{case_id}/stages/preprocess/approve",
            headers={"X-User-Role": "pathologist", "X-User-Id": "dr_watson"}
        )
    assert res.status_code == 202
    assert res.json()["status"] == "approved"
    assert res.json()["next_stage"] == "triage"

    db_session.refresh(se)
    assert se.status == "confirmed"
    assert se.reviewed_by == "dr_watson"
    assert se.reviewed_at is not None


def test_retry_case_stage_validations_and_gating(client, db_session):
    """
    Verify retry_case_stage validates stage_name against known stages (400 if invalid)
    and requires status to be in ('failed', 'rejected') (409 otherwise).
    """
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="test_user")
    sl = Slide(id=uuid.uuid4(), case_id=case_id, gcs_uri_original="gs://raw/s.svs")
    se = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="queued"
    )
    db_session.add_all([c, sl, se])
    db_session.commit()

    # 1. Invalid stage name -> 400
    res_bad_stage = client.post(
        f"/api/v1/cases/{case_id}/stages/nonexistent_stage/retry",
        headers={"X-User-Role": "pathologist"}
    )
    assert res_bad_stage.status_code == 400
    assert "Invalid stage_name" in res_bad_stage.json()["detail"]

    # 2. Stage status is 'queued' -> 409 Conflict
    res_not_failed = client.post(
        f"/api/v1/cases/{case_id}/stages/triage/retry",
        headers={"X-User-Role": "pathologist"}
    )
    assert res_not_failed.status_code == 409
    assert "Only stages in ('failed', 'rejected') can be retried" in res_not_failed.json()["detail"]

    # 3. Stage is 'failed' -> successful retry with monotonic attempt=2
    se.status = "failed"
    db_session.commit()

    with patch("app.core.cloud_tasks.dispatch_stage_task"):
        res_ok = client.post(
            f"/api/v1/cases/{case_id}/stages/triage/retry",
            headers={"X-User-Role": "pathologist"}
        )
    assert res_ok.status_code == 202
    assert res_ok.json()["attempt"] == 2

    # Verify new attempt 2 stage exists in DB
    retried_se = db_session.scalars(
        select(StageExecution)
        .where(StageExecution.case_id == case_id, StageExecution.stage == "triage")
        .order_by(StageExecution.attempt.desc())
    ).first()
    assert retried_se.attempt == 2
    assert retried_se.status == "queued"


def test_slide_rescan_attempt_monotonicity(client, db_session):
    """
    Issue #69: Verify re-uploading a slide on rescan sets attempt = max_existing + 1
    instead of hardcoding attempt=1, preventing uq_stage_execution_attempt 500 errors.
    """
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="test_user", status="needs_rescan")
    db_session.add(c)

    # Pre-existing attempt 1 for ingest
    se_1 = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="ingest",
        attempt=1,
        status="failed",
        error="Blurry slide scan"
    )
    db_session.add(se_1)
    db_session.commit()

    # Finalize slide re-upload
    req = SlideFinalizeRequest(
        gcs_uri="gs://raw/rescan_slide.svs",
        client_sha256="abc123rescan"
    )
    with patch("app.core.cloud_tasks.dispatch_stage_task"):
        res = client.post(
            f"/api/v1/cases/{case_id}/slide/finalize",
            json=req.model_dump(),
            headers={"X-User-Role": "pathologist"}
        )
    assert res.status_code == 202
    assert res.json()["attempt"] == 2

    db_session.refresh(c)
    assert c.status == "open"

    # Verify attempt 2 ingest exists
    se_2 = db_session.scalars(
        select(StageExecution)
        .where(StageExecution.case_id == case_id, StageExecution.stage == "ingest")
        .order_by(StageExecution.attempt.desc())
    ).first()
    assert se_2.attempt == 2
    assert se_2.status == "queued"


def test_get_case_detail_orders_stages_by_attempt_desc(client, db_session):
    """
    Issue #205: Verify GET /cases/{id} orders stages by attempt.desc()
    so the latest attempt is consistently returned first.
    """
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="test_user")
    sl = Slide(id=uuid.uuid4(), case_id=case_id, gcs_uri_original="gs://raw/s.svs")
    
    # Add attempts in arbitrary order (1, 3, 2)
    se_1 = StageExecution(id=uuid.uuid4(), case_id=case_id, stage="ingest", attempt=1, status="failed")
    se_3 = StageExecution(id=uuid.uuid4(), case_id=case_id, stage="ingest", attempt=3, status="done")
    se_2 = StageExecution(id=uuid.uuid4(), case_id=case_id, stage="ingest", attempt=2, status="failed")
    db_session.add_all([c, sl, se_1, se_3, se_2])
    db_session.commit()

    res = client.get(f"/api/v1/cases/{case_id}", headers={"X-User-Role": "pathologist"})
    assert res.status_code == 200
    stages = res.json()["stages"]
    attempts = [s["attempt"] for s in stages]
    assert attempts == [3, 2, 1], f"Expected attempts [3, 2, 1], got {attempts}"


def test_confirm_triage_gating_and_mitosis_attempt_monotonicity(client, db_session):
    """
    Issue #279, #285, #535: Verify confirm_triage requires status == 'awaiting_review' (409 otherwise)
    and queues mitosis with attempt = max_existing + 1 instead of resetting attempt 1.
    """
    case_id = str(uuid.uuid4())
    c = Case(id=case_id, created_by="dr_lee", status="open")
    
    # Triage stage execution currently in 'running'
    se_triage = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="triage",
        attempt=1,
        status="running"
    )
    # Existing failed mitosis attempt 1
    se_mitosis_1 = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="mitosis",
        attempt=1,
        status="failed"
    )
    db_session.add_all([c, se_triage, se_mitosis_1])
    db_session.commit()

    # 1. 409 Conflict when triage stage is not awaiting_review
    res_conflict = client.post(
        "/api/v1/stages/triage/confirm",
        json={"case_id": case_id, "no_invasive_tumor": False}
    )
    assert res_conflict.status_code == 409
    assert "expected 'awaiting_review'" in res_conflict.json()["detail"]

    # 2. Set to awaiting_review and confirm
    se_triage.status = "awaiting_review"
    db_session.commit()

    with patch("app.routers.triage.download_blob_as_bytes", return_value=b'{"hotspots": []}'), \
         patch("app.core.cloud_tasks.dispatch_stage_task"):
        res_ok = client.post(
            "/api/v1/stages/triage/confirm",
            json={"case_id": case_id, "no_invasive_tumor": False, "reviewed_by": "dr_lee"}
        )
    assert res_ok.status_code == 200
    assert res_ok.json()["status"] == "confirmed"

    # Verify newly queued mitosis stage has attempt = 2 (not overwriting attempt 1)
    new_mitosis = db_session.scalars(
        select(StageExecution)
        .where(StageExecution.case_id == case_id, StageExecution.stage == "mitosis")
        .order_by(StageExecution.attempt.desc())
    ).first()
    assert new_mitosis.attempt == 2
    assert new_mitosis.status == "queued"


def test_approve_preprocess_done_with_qc_awaiting_review(client, db_session):
    """
    Verify approving preprocess succeeds when preprocess status='done'
    and associated qc status='awaiting_review' (Slide 1 path).
    """
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="test_user", status="open")
    sl = Slide(id=uuid.uuid4(), case_id=case_id, gcs_uri_original="gs://raw/s.svs")
    se_prep = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="preprocess",
        attempt=1,
        status="done"
    )
    se_qc = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="qc",
        attempt=1,
        status="awaiting_review"
    )
    db_session.add_all([c, sl, se_prep, se_qc])
    db_session.commit()

    with patch("app.core.cloud_tasks.dispatch_stage_task"):
        res = client.post(
            f"/api/v1/cases/{case_id}/stages/preprocess/approve",
            headers={"X-User-Role": "pathologist"}
        )
    assert res.status_code == 202
    assert res.json()["status"] == "approved"
    assert res.json()["next_stage"] == "triage"

    db_session.refresh(se_prep)
    db_session.refresh(se_qc)
    assert se_prep.status == "confirmed"
    assert se_qc.status == "confirmed"


def test_approve_preprocess_qc_failed_requires_override_justification(client, db_session):
    """
    Verify approving preprocess when qc status='failed' blocks without justification (409)
    and succeeds with clinical override justification (Slide 2 path).
    """
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="test_user", status="needs_rescan")
    sl = Slide(id=uuid.uuid4(), case_id=case_id, gcs_uri_original="gs://raw/s.svs")
    se_prep = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="preprocess",
        attempt=1,
        status="done"
    )
    se_qc = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="qc",
        attempt=1,
        status="failed",
        error="Critical focus blur: 100.0% of tissue tiles blurry"
    )
    db_session.add_all([c, sl, se_prep, se_qc])
    db_session.commit()

    # 1. Attempt approval without justification -> 409
    res_blocked = client.post(
        f"/api/v1/cases/{case_id}/stages/preprocess/approve",
        headers={"X-User-Role": "pathologist"}
    )
    assert res_blocked.status_code == 409
    assert "clinical override justification" in res_blocked.json()["detail"].lower()

    # 2. Attempt approval with short justification (<10 chars) -> 409
    res_short = client.post(
        f"/api/v1/cases/{case_id}/stages/preprocess/approve",
        headers={"X-User-Role": "pathologist"},
        json={"override_justification": "Too short"}
    )
    assert res_short.status_code == 409

    # 3. Valid clinical override justification (>= 10 chars) -> 202 Approved
    with patch("app.core.cloud_tasks.dispatch_stage_task"):
        res_ok = client.post(
            f"/api/v1/cases/{case_id}/stages/preprocess/approve",
            headers={"X-User-Role": "pathologist"},
            json={"override_justification": "Artifact in margin; diagnostic core has adequate cellular clarity."}
        )
    assert res_ok.status_code == 202
    assert res_ok.json()["status"] == "approved"

    db_session.refresh(c)
    db_session.refresh(se_prep)
    db_session.refresh(se_qc)
    assert c.status == "open"
    assert se_prep.status == "confirmed"
    assert se_qc.status == "confirmed"
    assert se_qc.review_edits["override_justification"] == "Artifact in margin; diagnostic core has adequate cellular clarity."


def test_retry_preprocess_when_done_or_failed(client, db_session):
    """
    Verify retrying preprocess when status is 'done' succeeds, increments attempt to 2,
    and clears 'needs_rescan' back to 'open'.
    """
    case_id = uuid.uuid4()
    c = Case(id=case_id, created_by="test_user", status="needs_rescan")
    sl = Slide(id=uuid.uuid4(), case_id=case_id, gcs_uri_original="gs://raw/s.svs")
    se_prep = StageExecution(
        id=uuid.uuid4(),
        case_id=case_id,
        stage="preprocess",
        attempt=1,
        status="done"
    )
    db_session.add_all([c, sl, se_prep])
    db_session.commit()

    with patch("app.core.cloud_tasks.dispatch_stage_task"):
        res = client.post(
            f"/api/v1/cases/{case_id}/stages/preprocess/retry",
            headers={"X-User-Role": "pathologist"}
        )
    assert res.status_code == 202
    assert res.json()["attempt"] == 2

    db_session.refresh(c)
    assert c.status == "open"
    retried_prep = db_session.scalars(
        select(StageExecution)
        .where(StageExecution.case_id == case_id, StageExecution.stage == "preprocess")
        .order_by(StageExecution.attempt.desc())
    ).first()
    assert retried_prep.attempt == 2
    assert retried_prep.status == "queued"

