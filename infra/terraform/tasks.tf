# ==============================================================================
# Google Cloud Tasks Stage Orchestration Queue
# ==============================================================================

resource "google_cloud_tasks_queue" "stage_queue" {
  name     = "oncogemma-stage-queue"
  location = var.region

  rate_limits {
    max_dispatches_per_second = 50
    max_concurrent_dispatches = 100
  }

  retry_config {
    max_attempts       = 5
    min_backoff        = "5s"
    max_backoff        = "300s"
    max_doublings      = 4
    max_retry_duration = "3600s"
  }

  depends_on = [google_project_service.enabled_apis]
}
