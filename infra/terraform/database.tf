# ==============================================================================
# Managed Google Cloud SQL (PostgreSQL 16)
# ==============================================================================

resource "google_sql_database_instance" "oncogemma_db_instance" {
  name             = "oncogemma-${var.environment}-psql"
  database_version = "POSTGRES_16"
  region           = var.region

  deletion_protection = false

  settings {
    tier = var.db_instance_tier

    ip_configuration {
      ipv4_enabled = true
      # In production, private_network (VPC peering) is recommended
    }

    backup_configuration {
      enabled            = true
      point_in_time_recovery_enabled = true
    }

    database_flags {
      name  = "max_connections"
      value = "200"
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_sql_database" "oncogemma_db" {
  name     = var.db_name
  instance = google_sql_database_instance.oncogemma_db_instance.name
}

resource "google_sql_user" "oncogemma_user" {
  name     = var.db_user
  instance = google_sql_database_instance.oncogemma_db_instance.name
  password = var.db_password
}
