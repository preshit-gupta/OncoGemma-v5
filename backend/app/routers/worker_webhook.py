"""
OncoGemma v5 - Cloud Tasks Internal Stage Execution Webhook.
Targeted by Google Cloud Tasks HTTP tasks to execute pipeline stages
inside a Cloud Run instance without requiring any background polling loop.
"""

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.config import settings
from app.models.stage_execution import StageExecution
from app.models.case import Case
from worker.ingest import run_ingest
from worker.preprocess import run_preprocess
from worker.qc import run_qc
from worker.triage import run_triage
from worker.mitosis import run_mitosis
from worker.grading import run_grading
from worker.report import run_report

logger = logging.getLogger("oncogemma.worker_webhook")

router = APIRouter(prefix="/api/v1/internal", tags=["internal"])

STAGE_HANDLERS = {
    "ingest": run_ingest,
    "preprocess": run_preprocess,
    "qc": run_qc,
    "triage": run_triage,
    "mitosis": run_mitosis,
    "grading": run_grading,
    "report": run_report,
}


class ExecuteStagePayload(BaseModel):
    case_id: str
    stage: str
    stage_exec_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


@router.post("/execute-stage", status_code=status.HTTP_200_OK)
def execute_stage_webhook(
    body: ExecuteStagePayload,
    db: Session = Depends(get_db),
    x_cloudtasks_taskname: Optional[str] = Header(None)
):
    """
    HTTP handler invoked by Google Cloud Tasks to execute a stage asynchronously.
    """
    case_id_str = body.case_id
    stage_name = body.stage.lower()

    if stage_name not in STAGE_HANDLERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown stage '{stage_name}'. Valid stages: {list(STAGE_HANDLERS.keys())}"
        )

    logger.info(f"[CloudTasks Webhook] Received task '{x_cloudtasks_taskname}' for stage '{stage_name}' (case: {case_id_str})")

    # Locate or create StageExecution
    stage_exec: Optional[StageExecution] = None
    if body.stage_exec_id:
        try:
            stage_exec = db.get(StageExecution, UUID(body.stage_exec_id))
        except Exception:
            stage_exec = db.get(StageExecution, body.stage_exec_id)

    if not stage_exec:
        try:
            c_uuid = UUID(case_id_str)
        except Exception:
            c_uuid = case_id_str
            
        stage_exec = (
            db.query(StageExecution)
            .filter(StageExecution.case_id == c_uuid, StageExecution.stage == stage_name)
            .order_by(StageExecution.attempt.desc())
            .first()
        )

    if not stage_exec:
        stage_exec = StageExecution(
            case_id=UUID(case_id_str) if isinstance(case_id_str, str) else case_id_str,
            stage=stage_name,
            attempt=1,
            status="running",
            started_at=datetime.now(timezone.utc),
            input_ref=body.payload or {}
        )
        db.add(stage_exec)
        db.commit()
        db.refresh(stage_exec)
    else:
        stage_exec.status = "running"
        stage_exec.started_at = datetime.now(timezone.utc)
        db.commit()

    handler = STAGE_HANDLERS[stage_name]

    try:
        out_uri, model_versions = handler(stage_exec, db)
        
        if stage_exec.status == "running":
            stage_exec.status = "done"
        stage_exec.output_ref = out_uri
        stage_exec.model_versions = model_versions
        stage_exec.completed_at = datetime.now(timezone.utc)
        db.commit()

        logger.info(f"[CloudTasks Webhook] Successfully executed stage '{stage_name}' for case {case_id_str} (status: {stage_exec.status}).")
        return {
            "status": "success",
            "stage": stage_name,
            "stage_execution_id": str(stage_exec.id),
            "output_ref": out_uri
        }

    except Exception as exc:
        db.rollback()
        err_msg = traceback.format_exc()
        logger.error(f"[CloudTasks Webhook Error] Stage '{stage_name}' failed for case {case_id_str}: {exc}\n{err_msg}")

        try:
            curr = db.get(StageExecution, stage_exec.id)
            if curr:
                curr.status = "failed"
                curr.error = str(exc)
                curr.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass

        # Return 500 so Cloud Tasks can retry the task according to queue retry policy
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stage execution failed: {str(exc)}"
        )
