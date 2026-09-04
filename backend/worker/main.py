import os
import sys
import time
import uuid
import traceback
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.db import SessionLocal, engine
from app.models.stage_execution import StageExecution
from worker.ingest import run_ingest
from worker.preprocess import run_preprocess
from worker.qc import run_qc
from worker.triage import run_triage
from worker.mitosis import run_mitosis
from worker.grading import run_grading
from worker.report import run_report

HANDLERS = {
    "ingest": run_ingest,
    "preprocess": run_preprocess,
    "qc": run_qc,
    "triage": run_triage,
    "mitosis": run_mitosis,
    "grading": run_grading,
    "report": run_report
}

def reset_stuck_running_stages(timeout_seconds: int = 300):
    """
    Reset orphan stages left in 'running' state exceeding timeout_seconds (default: 5 minutes / Cloud Run timeout).
    Prevents resetting actively executing sibling worker tasks on worker startup.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
    db = SessionLocal()
    try:
        stmt = (
            update(StageExecution)
            .where(
                StageExecution.status == "running",
                (StageExecution.started_at <= cutoff) | (StageExecution.started_at.is_(None))
            )
            .values(status="queued", started_at=None)
        )
        res = db.execute(stmt)
        db.commit()
        if res.rowcount > 0:
            print(f"[Worker Reset] Reset {res.rowcount} stuck 'running' stages (> {timeout_seconds}s) back to 'queued'...")
    except Exception as e:
        print(f"[Worker Reset Note] {e}")
    finally:
        db.close()

def poll_and_execute_single_task():
    """
    Executes a single queued task using SQLAlchemy ORM queue fetch with row locking.
    Uses .with_for_update(skip_locked=True) on PostgreSQL and cleanly falls back on SQLite.
    """
    db: Session = SessionLocal()
    try:
        stages_list = list(HANDLERS.keys())
        stmt = (
            select(StageExecution)
            .where(
                StageExecution.status == "queued",
                StageExecution.stage.in_(stages_list)
            )
            .order_by(StageExecution.started_at.asc().nulls_first(), StageExecution.id.asc())
            .limit(1)
        )

        is_postgres = False
        try:
            bind = db.get_bind()
            if bind and bind.dialect.name == "postgresql":
                is_postgres = True
        except Exception:
            pass

        if is_postgres:
            stmt = stmt.with_for_update(skip_locked=True)

        try:
            stage_exec = db.scalars(stmt).first()
        except Exception as exc:
            if is_postgres and ("for update" in str(exc).lower() or "skip locked" in str(exc).lower()):
                stmt_fallback = (
                    select(StageExecution)
                    .where(
                        StageExecution.status == "queued",
                        StageExecution.stage.in_(stages_list)
                    )
                    .order_by(StageExecution.started_at.asc().nulls_first(), StageExecution.id.asc())
                    .limit(1)
                )
                stage_exec = db.scalars(stmt_fallback).first()
            else:
                raise

        if not stage_exec:
            return False

        # Mark as running
        stage_exec.status = "running"
        stage_exec.started_at = datetime.now(timezone.utc)
        db.commit()

        print(f"[Worker] Processing stage '{stage_exec.stage}' for case {stage_exec.case_id} (attempt {stage_exec.attempt})...")

        try:
            handler = HANDLERS[stage_exec.stage]
            out_uri, model_versions = handler(stage_exec, db)

            if stage_exec.status == "running":
                stage_exec.status = "done"
            stage_exec.output_ref = out_uri
            stage_exec.model_versions = model_versions
            stage_exec.completed_at = datetime.now(timezone.utc)
            db.commit()
            print(f"[Worker] Successfully completed stage '{stage_exec.stage}' for case {stage_exec.case_id} (Status: {stage_exec.status}).")

        except Exception as e:
            db.rollback()
            err_msg = traceback.format_exc()
            print(f"[Worker ERROR] Stage '{stage_exec.stage}' failed for case {stage_exec.case_id}: {e}")
            try:
                stage_exec_curr = db.get(StageExecution, stage_exec.id)
                if stage_exec_curr:
                    stage_exec_curr.status = "failed"
                    stage_exec_curr.error = err_msg
                    stage_exec_curr.completed_at = datetime.now(timezone.utc)
                    db.commit()
            except Exception as e2:
                print(f"[Worker Fail State Error] {e2}")

        return True

    finally:
        db.close()

def run_worker_loop():
    print(f"[Worker] Starting OncoGemma stage worker poll loop. Engine: {engine.dialect.name}. Handlers: {list(HANDLERS.keys())}")
    reset_stuck_running_stages(timeout_seconds=300)
    last_reset_check = time.time()
    while True:
        try:
            if time.time() - last_reset_check > 60.0:
                reset_stuck_running_stages(timeout_seconds=300)
                last_reset_check = time.time()

            executed = poll_and_execute_single_task()
            if not executed:
                time.sleep(1.0)
        except Exception as e:
            print(f"[Worker Loop Exception] {e}")
            time.sleep(3.0)

if __name__ == "__main__":
    run_worker_loop()
