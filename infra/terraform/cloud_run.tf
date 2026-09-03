# ==============================================================================
# Artifact Registry, Cloud Run Services & Cloud Run Jobs
# ==============================================================================

# 1. Artifact Registry Docker Repository
resource "google_artifact_registry_repository" "oncogemma_repo" {
  location      = var.region
  repository_id = "oncogemma-repo"
  description   = "Docker container images for OncoGemma v5"
  format        = "DOCKER"

  depends_on = [google_project_service.enabled_apis]
}

# 2. Cloud Run Service: FastAPI Control Plane
resource "google_cloud_run_v2_service" "api_service" {
  name     = "oncogemma-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello"

      resources {
        limits = {
          cpu    = "4"
          memory = "8Gi"
        }
      }

      env {
        name  = "ENV"
        value = var.environment
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_REGION"
        value = var.region
      }
      env {
        name  = "USE_CLOUD_TASKS"
        value = "true"
      }
      env {
        name  = "CLOUD_TASKS_QUEUE"
        value = google_cloud_tasks_queue.stage_queue.name
      }
      env {
        name  = "CLOUD_TASKS_LOCATION"
        value = var.region
      }
      env {
        name  = "GCS_RAW_BUCKET"
        value = google_storage_bucket.raw_slides.name
      }
      env {
        name  = "GCS_PYRAMIDS_BUCKET"
        value = google_storage_bucket.pyramids.name
      }
      env {
        name  = "GCS_ARTIFACTS_BUCKET"
        value = google_storage_bucket.artifacts.name
      }
      env {
        name  = "DATABASE_URL"
        value = "postgresql://${var.db_user}:${var.db_password}@/${var.db_name}?host=/cloudsql/${google_sql_database_instance.oncogemma_db_instance.connection_name}"
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.oncogemma_db_instance.connection_name]
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
    ]
  }

  depends_on = [
    google_project_service.enabled_apis,
    google_sql_database_instance.oncogemma_db_instance
  ]
}

# Allow public access to FastAPI control plane
resource "google_cloud_run_v2_service_iam_member" "api_public_access" {
  location = google_cloud_run_v2_service.api_service.location
  name     = google_cloud_run_v2_service.api_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# 3. Cloud Run Service: Next.js Frontend
resource "google_cloud_run_v2_service" "frontend_service" {
  name     = "oncogemma-frontend"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello"

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      env {
        name  = "API_BASE_URL"
        value = google_cloud_run_v2_service.api_service.uri
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
    ]
  }

  depends_on = [
    google_cloud_run_v2_service.api_service
  ]
}

resource "google_cloud_run_v2_service_iam_member" "frontend_public_access" {
  location = google_cloud_run_v2_service.frontend_service.location
  name     = google_cloud_run_v2_service.frontend_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# 4. Cloud Run Job: Asynchronous Serverless Worker (Heavy Ingestion & Macenko Engine)
resource "google_cloud_run_v2_job" "worker_job" {
  name     = "oncogemma-worker-job"
  location = var.region

  template {
    template {
      service_account = google_service_account.cloud_run_sa.email
      timeout         = "3600s" # 1 hour max runtime per job

      containers {
        image = "us-docker.pkg.dev/cloudrun/container/hello"

        resources {
          limits = {
            cpu    = "8"     # 8 vCPUs for SIMD PyVips DZI generation
            memory = "32Gi"  # 32 GB RAM for gigapixel slide deconvolution
          }
        }

        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }
        env {
          name  = "GCP_REGION"
          value = var.region
        }
        env {
          name  = "GCS_RAW_BUCKET"
          value = google_storage_bucket.raw_slides.name
        }
        env {
          name  = "GCS_PYRAMIDS_BUCKET"
          value = google_storage_bucket.pyramids.name
        }
        env {
          name  = "GCS_ARTIFACTS_BUCKET"
          value = google_storage_bucket.artifacts.name
        }
        env {
          name  = "DATABASE_URL"
          value = "postgresql://${var.db_user}:${var.db_password}@/${var.db_name}?host=/cloudsql/${google_sql_database_instance.oncogemma_db_instance.connection_name}"
        }

        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.oncogemma_db_instance.connection_name]
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].image,
    ]
  }

  depends_on = [
    google_project_service.enabled_apis,
    google_sql_database_instance.oncogemma_db_instance
  ]
}
