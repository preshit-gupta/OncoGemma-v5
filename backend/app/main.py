import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from app.core.config import settings
from app.core.gcs import ensure_buckets_exist
from app.core.db import Base, engine
from app.routers import (
    cases_router, tiles_router, audit_router, triage_router, mitosis_router, grading_router, report_router, worker_webhook_router
)
from app.routers.admin import router as admin_router

import logging

logger = logging.getLogger("oncogemma.daemon")

async def background_pipeline_worker():
    """
    Self-healing, always-on background worker daemon running inside Cloud Run.
    Continuously monitors the database for any 'queued' pipeline stages and executes
    them immediately without relying on external task queues.
    """
    logger.info("[Always-On Worker] Initializing in-process pipeline worker daemon...")
    from worker.main import poll_and_execute_single_task, reset_stuck_running_stages
    try:
        await asyncio.to_thread(reset_stuck_running_stages)
    except Exception as e:
        logger.warning(f"[Always-On Worker Reset Note] {e}")

    while True:
        try:
            had_work = await asyncio.to_thread(poll_and_execute_single_task)
            if not had_work:
                await asyncio.sleep(1.5)
            else:
                # Chained stages execute immediately with minimal latency
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            logger.info("[Always-On Worker] Background worker cancelled gracefully.")
            break
        except Exception as exc:
            logger.error(f"[Always-On Worker Exception] {exc}")
            await asyncio.sleep(2.0)


def ensure_schema_up_to_date():
    """Ensure newly added columns and cascade foreign keys exist in production tables."""
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            if conn.dialect.name == "postgresql":
                conn.execute(text("ALTER TABLE detections ADD COLUMN IF NOT EXISTS medgemma_verdict VARCHAR;"))
                conn.execute(text("ALTER TABLE detections ADD COLUMN IF NOT EXISTS medgemma_rationale TEXT;"))
                conn.execute(text("ALTER TABLE detections ADD COLUMN IF NOT EXISTS medgemma_confidence VARCHAR;"))
                conn.execute(text("ALTER TABLE detections ALTER COLUMN medgemma_confidence TYPE VARCHAR USING medgemma_confidence::VARCHAR;"))
                conn.execute(text("ALTER TABLE slides ADD COLUMN IF NOT EXISTS status VARCHAR DEFAULT 'ready';"))
            elif conn.dialect.name == "sqlite":
                cols = [c[1] for c in conn.execute(text("PRAGMA table_info(slides);")).fetchall()]
                if cols and "status" not in cols:
                    conn.execute(text("ALTER TABLE slides ADD COLUMN status VARCHAR DEFAULT 'ready';"))
            logger.info("[Database Schema] Verified all columns exist on 'detections' and 'slides' tables.")

            # On PostgreSQL: ensure foreign key constraints on child tables have ON DELETE CASCADE
            if conn.dialect.name == "postgresql":
                fk_updates = [
                    ("hotspots", "hotspots_stage_execution_id_fkey", "stage_execution_id", "stage_executions(id)"),
                    ("hotspots", "hotspots_case_id_fkey", "case_id", "cases(id)"),
                    ("detections", "detections_case_id_fkey", "case_id", "cases(id)"),
                    ("hpf_sites", "hpf_sites_case_id_fkey", "case_id", "cases(id)"),
                    ("slides", "slides_case_id_fkey", "case_id", "cases(id)"),
                    ("stage_executions", "stage_executions_case_id_fkey", "case_id", "cases(id)"),
                    ("gradings", "gradings_case_id_fkey", "case_id", "cases(id)"),
                    ("reports", "reports_case_id_fkey", "case_id", "cases(id)"),
                ]
                for tbl, cname, col, target in fk_updates:
                    try:
                        conn.execute(text(f"""
                            DO $$
                            BEGIN
                                IF EXISTS (
                                    SELECT 1 FROM information_schema.table_constraints 
                                    WHERE constraint_name = '{cname}' AND table_name = '{tbl}'
                                ) THEN
                                    ALTER TABLE {tbl} DROP CONSTRAINT {cname};
                                END IF;
                                ALTER TABLE {tbl} ADD CONSTRAINT {cname} 
                                    FOREIGN KEY ({col}) REFERENCES {target} ON DELETE CASCADE;
                            END $$;
                        """))
                    except Exception as fk_e:
                        logger.warning(f"[Schema FK Migration Note] Could not update FK {cname} on {tbl}: {fk_e}")
                logger.info("[Database Schema] Ensured ON DELETE CASCADE on all case and stage_execution foreign keys.")
    except Exception as e:
        logger.warning(f"[Database Schema Migration Note] {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure database tables exist for dev
    Base.metadata.create_all(bind=engine)
    # Ensure newly added columns exist in existing tables
    ensure_schema_up_to_date()
    # Ensure GCS buckets exist for local dev
    ensure_buckets_exist()
    # Launch persistent background worker task inside the Cloud Run process
    worker_task = asyncio.create_task(background_pipeline_worker())
    try:
        yield
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title="OncoGemma v4.5 API",
    description="Breast Cancer Diagnostic Copilot API — Nottingham Grading & CAP-Compliant Synoptic Reporting",
    version="4.5.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases_router)
app.include_router(tiles_router)
app.include_router(audit_router)
app.include_router(triage_router)
app.include_router(mitosis_router)
app.include_router(grading_router)
app.include_router(report_router)
app.include_router(worker_webhook_router)
app.include_router(admin_router)

@app.get("/health")
@app.get("/api/health")
@app.get("/healthz")
def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {e}"
        )
    return {
        "status": "healthy",
        "version": app.version,
        "env": settings.ENV
    }
