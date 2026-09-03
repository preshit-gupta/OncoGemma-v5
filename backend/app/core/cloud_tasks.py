"""
OncoGemma v5 - Google Cloud Tasks Stage Dispatcher.
Enqueues stage execution tasks to an asynchronous Cloud Tasks queue,
which dispatches HTTP POST events to Cloud Run Jobs or worker webhooks.
"""

import json
import logging
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger("oncogemma.cloud_tasks")


def dispatch_stage_task(
    case_id: str,
    stage: str,
    stage_exec_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    delay_seconds: int = 0
) -> Optional[str]:
    """
    Dispatches a pipeline stage execution request via Google Cloud Tasks.
    
    Args:
        case_id: The UUID string of the Case.
        stage: The stage name ('ingest', 'preprocess', 'qc', 'triage', 'mitosis', 'grading', 'report').
        stage_exec_id: Optional UUID string of the StageExecution row.
        payload: Optional additional dictionary parameters.
        delay_seconds: Delay before Cloud Tasks dispatches the task.
        
    Returns:
        The Cloud Tasks task name if successfully dispatched, or None.
    """
    body = {
        "case_id": str(case_id),
        "stage": stage,
        "stage_exec_id": str(stage_exec_id) if stage_exec_id else None,
        "payload": payload or {}
    }
    
    if not settings.USE_CLOUD_TASKS:
        logger.info(
            f"[CloudTasks Disabled] Skipping Cloud Tasks dispatch for stage '{stage}' (case: {case_id}). "
            f"Execution will rely on worker polling loop."
        )
        return None

    try:
        from google.cloud import tasks_v2
        from google.protobuf import timestamp_pb2
        import time

        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(
            settings.GCP_PROJECT_ID,
            settings.CLOUD_TASKS_LOCATION,
            settings.CLOUD_TASKS_QUEUE
        )

        target_url = f"{settings.WORKER_SERVICE_URL.rstrip('/')}/api/v1/internal/execute-stage"
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": target_url,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(body).encode("utf-8")
            }
        }

        # Add OIDC authentication if running in authenticated Cloud Run environment
        if settings.CLOUD_TASKS_SERVICE_ACCOUNT:
            task["http_request"]["oidc_token"] = {
                "service_account_email": settings.CLOUD_TASKS_SERVICE_ACCOUNT,
                "audience": settings.WORKER_SERVICE_URL
            }

        # Handle scheduled delay
        if delay_seconds > 0:
            d = time.time() + delay_seconds
            timestamp = timestamp_pb2.Timestamp()
            timestamp.seconds = int(d)
            timestamp.nanos = int((d - int(d)) * 1e9)
            task["schedule_time"] = timestamp

        response = client.create_task(request={"parent": parent, "task": task})
        logger.info(f"[CloudTasks] Successfully dispatched task '{response.name}' for stage '{stage}' on case {case_id}.")
        return response.name

    except Exception as exc:
        logger.warning(
            f"[CloudTasks Warning] Failed to dispatch Cloud Tasks event for stage '{stage}' ({exc}). "
            f"Falling back to database-backed execution."
        )
        return None
