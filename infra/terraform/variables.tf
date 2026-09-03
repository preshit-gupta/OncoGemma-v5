variable "project_id" {
  type        = string
  description = "Google Cloud Project ID"
  default     = "oncogemma"
}

variable "region" {
  type        = string
  description = "Primary GCP Region for Cloud Run, Cloud Tasks, and Cloud SQL"
  default     = "us-central1"
}

variable "zone" {
  type        = string
  description = "Primary GCP Zone"
  default     = "us-central1-a"
}

variable "environment" {
  type        = string
  description = "Deployment Environment (dev / staging / prod)"
  default     = "dev"
}

variable "db_instance_tier" {
  type        = string
  description = "Machine tier for Cloud SQL PostgreSQL instance"
  default     = "db-custom-2-7680" # 2 vCPU, 7.5 GB RAM (or db-f1-micro for dev testing)
}

variable "db_name" {
  type        = string
  description = "PostgreSQL database name"
  default     = "oncogemma_db"
}

variable "db_user" {
  type        = string
  description = "PostgreSQL user"
  default     = "oncogemma"
}

variable "db_password" {
  type        = string
  description = "PostgreSQL password"
  sensitive   = true
  default     = "oncogemma_secure_cloud_password"
}
