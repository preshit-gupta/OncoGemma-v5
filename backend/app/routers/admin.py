"""
admin.py — Internal admin utilities for OncoGemma.
IMPORTANT: These endpoints are for development/testing use only.
They are guarded by a secret token defined in the ADMIN_SECRET env var.
"""
import os
from fastapi import APIRouter, HTTPException, Header
from sqlalchemy import text
from app.core.db import engine

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

def _check_secret(x_admin_secret: str | None):
    expected = os.environ.get("ADMIN_SECRET", "")
    if not expected or x_admin_secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin secret")


@router.post("/reset-database")
def reset_database(x_admin_secret: str | None = Header(default=None)):
    """
    TRUNCATE all clinical data tables in dependency order.
    Retains schema (tables/columns). Does NOT drop the database.
    Requires X-Admin-Secret header matching the ADMIN_SECRET env var.
    """
    _check_secret(x_admin_secret)

    tables = [
        "audit_events",
        "stage_executions",
        "reports",
        "gradings",
        "hpf_sites",
        "detections",
        "hotspots",
        "slides",
        "cases",
    ]

    try:
        with engine.begin() as conn:
            # Disable FK checks for SQLite, use CASCADE-aware delete order for Postgres
            dialect = conn.dialect.name
            if dialect == "sqlite":
                conn.execute(text("PRAGMA foreign_keys = OFF;"))
                for tbl in tables:
                    conn.execute(text(f"DELETE FROM {tbl};"))
                conn.execute(text("PRAGMA foreign_keys = ON;"))
            else:
                # PostgreSQL — TRUNCATE with CASCADE and restart identity
                conn.execute(text(
                    f"TRUNCATE TABLE {', '.join(tables)} RESTART IDENTITY CASCADE;"
                ))
        return {
            "status": "ok",
            "message": f"Cleared {len(tables)} tables: {', '.join(tables)}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
