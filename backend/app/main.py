import sys
import time
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
    Periodically checks for and recovers any orphaned 'running' stages (> 300s).
    """
    logger.info("[Always-On Worker] Initializing in-process pipeline worker daemon...")
    from worker.main import poll_and_execute_single_task, reset_stuck_running_stages
    try:
        await asyncio.to_thread(reset_stuck_running_stages, 300)
    except Exception as e:
        logger.warning(f"[Always-On Worker Reset Note] {e}")

    last_reset_check = time.time()
    while True:
        try:
            # Self-healing watchdog: recover stages stuck in 'running' after container restarts or crashes
            if time.time() - last_reset_check > 60.0:
                try:
                    await asyncio.to_thread(reset_stuck_running_stages, 300)
                except Exception as r_err:
                    logger.warning(f"[Always-On Worker Periodic Reset Note] {r_err}")
                last_reset_check = time.time()

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
    if engine.dialect.name == "postgresql":
        # 1. Ensure columns exist with strict 1s lock timeout per statement
        col_statements = [
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS medgemma_verdict VARCHAR;",
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS medgemma_rationale TEXT;",
            "ALTER TABLE detections ADD COLUMN IF NOT EXISTS medgemma_confidence VARCHAR;",
            "ALTER TABLE slides ADD COLUMN IF NOT EXISTS status VARCHAR DEFAULT 'ready';",
            "ALTER TABLE slides ADD COLUMN IF NOT EXISTS gcs_uri_pyramid_norm VARCHAR;",
            "ALTER TABLE reports ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;",
            "ALTER TABLE reports ADD COLUMN IF NOT EXISTS pdf_sha256 VARCHAR;",
            "ALTER TABLE reports ADD COLUMN IF NOT EXISTS narrative_edited BOOLEAN NOT NULL DEFAULT FALSE;",
            "ALTER TABLE gradings ADD COLUMN IF NOT EXISTS type_confirmed_by VARCHAR DEFAULT 'unconfirmed';",
        ]
        for stmt in col_statements:
            try:
                with engine.begin() as conn:
                    conn.execute(text("SET LOCAL lock_timeout = '1s';"))
                    conn.execute(text(stmt))
            except Exception as e:
                logger.warning(f"[Schema DDL Note] {stmt[:40]}...: {e}")

        # 2. Ensure reports primary key is composite (case_id, version)
        try:
            with engine.begin() as conn:
                conn.execute(text("SET LOCAL lock_timeout = '1s';"))
                conn.execute(text("""
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.table_constraints 
                            WHERE constraint_name = 'reports_pkey' AND table_name = 'reports'
                        ) THEN
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.key_column_usage 
                                WHERE table_name = 'reports' AND column_name = 'version'
                            ) THEN
                                ALTER TABLE reports DROP CONSTRAINT reports_pkey;
                                ALTER TABLE reports ADD PRIMARY KEY (case_id, version);
                            END IF;
                        END IF;
                    END $$;
                """))
        except Exception as pk_err:
            logger.warning(f"[Schema PK Update Note] {pk_err}")

        # 3. Ensure foreign key constraints on child tables have ON DELETE CASCADE (only if not already CASCADE)
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
                with engine.begin() as conn:
                    conn.execute(text("SET LOCAL lock_timeout = '1s';"))
                    conn.execute(text(f"""
                        DO $$
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.referential_constraints 
                                WHERE constraint_name = '{cname}' AND delete_rule = 'CASCADE'
                            ) THEN
                                IF EXISTS (
                                    SELECT 1 FROM information_schema.table_constraints 
                                    WHERE constraint_name = '{cname}' AND table_name = '{tbl}'
                                ) THEN
                                    ALTER TABLE {tbl} DROP CONSTRAINT {cname};
                                END IF;
                                ALTER TABLE {tbl} ADD CONSTRAINT {cname} 
                                    FOREIGN KEY ({col}) REFERENCES {target} ON DELETE CASCADE;
                            END IF;
                        END $$;
                    """))
            except Exception as fk_e:
                logger.warning(f"[Schema FK Migration Note] Could not update FK {cname} on {tbl}: {fk_e}")
        logger.info("[Database Schema] Ensured ON DELETE CASCADE on foreign keys.")

    elif engine.dialect.name == "sqlite":
        try:
            with engine.begin() as conn:
                cols = [c[1] for c in conn.execute(text("PRAGMA table_info(slides);")).fetchall()]
                if cols and "status" not in cols:
                    conn.execute(text("ALTER TABLE slides ADD COLUMN status VARCHAR DEFAULT 'ready';"))
                if cols and "gcs_uri_pyramid_norm" not in cols:
                    conn.execute(text("ALTER TABLE slides ADD COLUMN gcs_uri_pyramid_norm VARCHAR;"))
                rep_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(reports);")).fetchall()]
                if rep_cols and "version" not in rep_cols:
                    conn.execute(text("ALTER TABLE reports ADD COLUMN version INTEGER DEFAULT 1;"))
                if rep_cols and "pdf_sha256" not in rep_cols:
                    conn.execute(text("ALTER TABLE reports ADD COLUMN pdf_sha256 VARCHAR;"))
                if rep_cols and "narrative_edited" not in rep_cols:
                    conn.execute(text("ALTER TABLE reports ADD COLUMN narrative_edited BOOLEAN DEFAULT 0;"))
                grad_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(gradings);")).fetchall()]
                if grad_cols and "type_confirmed_by" not in grad_cols:
                    conn.execute(text("ALTER TABLE gradings ADD COLUMN type_confirmed_by VARCHAR DEFAULT 'unconfirmed';"))
        except Exception as sq_err:
            logger.warning(f"[SQLite Schema Note] {sq_err}")

async def _async_init_and_worker():
    """Perform database checks, schema updates, and start background worker non-blockingly."""
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, Base.metadata.create_all, engine)
    except Exception as e:
        logger.warning(f"[DB Create All Note] {e}")
    try:
        await loop.run_in_executor(None, ensure_schema_up_to_date)
    except Exception as e:
        logger.warning(f"[Schema Migration Note] {e}")
    try:
        await loop.run_in_executor(None, ensure_buckets_exist)
    except Exception as e:
        logger.warning(f"[GCS Bucket Check Note] {e}")
    await background_pipeline_worker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Launch startup initialization and worker daemon in background task.
    # This guarantees the ASGI server binds port 8080 instantly and passes Cloud Run startup probes in <1s.
    init_task = asyncio.create_task(_async_init_and_worker())
    try:
        yield
    finally:
        init_task.cancel()
        try:
            await init_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="OncoGemma v4.5 API",
    description="Breast Cancer Diagnostic Copilot API — Nottingham Grading & CAP-Compliant Synoptic Reporting",
    version="4.5.0",
    lifespan=lifespan
)

cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"^https://.*\.run\.app$",
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
@app.get("/api/healthz")
@app.get("/api/v1/health")
@app.get("/api/v1/healthz")
async def health_check():
    from starlette.concurrency import run_in_threadpool
    def _ping():
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    try:
        await run_in_threadpool(_ping)
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
