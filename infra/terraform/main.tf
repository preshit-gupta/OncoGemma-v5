terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.20.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Required GCP APIs
locals {
  services = [
    "run.googleapis.com",
    "cloudtasks.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "compute.googleapis.com",
    "cloudbuild.googleapis.com"
  ]
}

resource "google_project_service" "enabled_apis" {
  for_each           = toset(local.services)
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}
