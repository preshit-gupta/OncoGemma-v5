# ==============================================================================
# Google Cloud IAM Service Accounts & Role Bindings
# ==============================================================================

# 1. Cloud Run Workload Service Account
resource "google_service_account" "cloud_run_sa" {
  account_id   = "oncogemma-cloudrun-sa"
  display_name = "OncoGemma Cloud Run Service Account"
}

# Grant GCS Admin
resource "google_project_iam_member" "sa_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Grant Cloud Tasks Enqueuer
resource "google_project_iam_member" "sa_tasks_enqueuer" {
  project = var.project_id
  role    = "roles/cloudtasks.enqueuer"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Grant Vertex AI User
resource "google_project_iam_member" "sa_vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Grant Cloud SQL Client
resource "google_project_iam_member" "sa_cloud_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Grant Cloud Run Invoker to Cloud Tasks
resource "google_project_iam_member" "sa_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}
