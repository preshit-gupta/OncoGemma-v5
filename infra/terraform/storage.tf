# ==============================================================================
# Google Cloud Storage Buckets & Cloud CDN Edge Caching
# ==============================================================================

# 1. Raw Slides Bucket (Direct Resumable Upload from Browser)
resource "google_storage_bucket" "raw_slides" {
  name          = "oncogemma-${var.environment}-raw"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST", "OPTIONS"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  depends_on = [google_project_service.enabled_apis]
}

# 2. Pyramids Bucket (OpenSeadragon High-DPI DZI Tile Streaming)
resource "google_storage_bucket" "pyramids" {
  name          = "oncogemma-${var.environment}-pyramids"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "OPTIONS"]
    response_header = ["*"]
    max_age_seconds = 86400
  }

  depends_on = [google_project_service.enabled_apis]
}

# 3. Artifacts Bucket (PDF Reports, Heatmaps, Mitosis Overlays, Embeddings)
resource "google_storage_bucket" "artifacts" {
  name          = "oncogemma-${var.environment}-artifacts"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "OPTIONS"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  depends_on = [google_project_service.enabled_apis]
}

# 4. Cloud CDN Backend Bucket for Sub-25ms OpenSeadragon Tile Delivery
resource "google_compute_backend_bucket" "pyramid_cdn_backend" {
  name        = "oncogemma-${var.environment}-pyramid-cdn"
  bucket_name = google_storage_bucket.pyramids.name
  enable_cdn  = true

  cdn_policy {
    cache_mode                   = "CACHE_ALL_STATIC"
    default_ttl                  = 86400  # 24 hours
    max_ttl                      = 604800 # 7 days
    client_ttl                   = 86400
    negative_caching             = true
    serve_while_stale            = 86400
    request_coalescing           = true
  }

  depends_on = [google_project_service.enabled_apis]
}

# Public read access on pyramids bucket for CDN caching
resource "google_storage_bucket_iam_member" "pyramids_public_read" {
  bucket = google_storage_bucket.pyramids.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}
