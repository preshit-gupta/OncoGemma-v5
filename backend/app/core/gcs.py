import os
import io
import shutil
import glob
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from app.core.config import settings

_gcs_client: storage.Client | None = None

def get_gcs_client() -> storage.Client:
    """
    Authoritative Google Cloud Storage client provider.
    Initializes and returns a singleton storage.Client instance connected to Google Cloud Storage.
    """
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client(project=settings.GCP_PROJECT_ID)
    return _gcs_client

def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """
    Parses a gs://bucket_name/blob_name URI into (bucket_name, blob_name).
    """
    if gcs_uri.startswith("gs://"):
        parts = gcs_uri[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        return bucket_name, blob_name
    parts = gcs_uri.lstrip("/").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    return bucket_name, blob_name

def get_bucket(bucket_name: str) -> storage.Bucket:
    """Retrieves a GCS Bucket object."""
    client = get_gcs_client()
    return client.bucket(bucket_name)

def upload_blob_from_file(bucket_name: str, blob_name: str, file_obj, content_type: str | None = None) -> str:
    """Uploads a file-like object directly to a GCS bucket."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file_obj, content_type=content_type)
    return f"gs://{bucket_name}/{blob_name}"

def upload_blob_from_bytes(bucket_name: str, blob_name: str, data: bytes, content_type: str | None = None) -> str:
    """Uploads raw bytes directly to a GCS bucket."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{blob_name}"

def upload_blob_from_filename(bucket_name: str, blob_name: str, local_path: str, content_type: str | None = None) -> str:
    """Uploads a local file directly to a GCS bucket."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type=content_type)
    return f"gs://{bucket_name}/{blob_name}"

def download_blob_as_bytes(bucket_name: str, blob_name: str) -> bytes:
    """Downloads a blob directly from GCS as bytes."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

def download_blob_as_text(bucket_name: str, blob_name: str, encoding: str = "utf-8") -> str:
    """Downloads a blob directly from GCS as text."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text(encoding=encoding)

def download_blob_to_filename(bucket_name: str, blob_name: str, destination_filename: str):
    """Downloads a blob directly from GCS to a local temporary destination filename."""
    os.makedirs(os.path.dirname(destination_filename), exist_ok=True)
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_filename)

def blob_exists(bucket_name: str, blob_name: str) -> bool:
    """Checks whether a blob exists in GCS."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()

def delete_blob(bucket_name: str, blob_name: str):
    """Deletes a blob from GCS if it exists."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exists():
        blob.delete()

def generate_signed_upload_url(bucket_name: str, blob_name: str, expiration_minutes: int = 60) -> str:
    """Generates a signed upload URL for direct browser-to-bucket upload."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="PUT",
            content_type="application/octet-stream"
        )
        return url
    except Exception as e:
        # Fallback to direct GCS endpoint if signed URL generation credentials lack private key (e.g. ADC user token)
        return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"

def upload_directory_to_gcs_and_purge(local_dir: str, bucket_name: str, dest_prefix: str, max_workers: int = 16):
    """
    Concurrently uploads an entire directory tree directly to Google Cloud Storage,
    and immediately purges the local temporary directory with zero local caching.
    """
    bucket = get_bucket(bucket_name)
    all_files = glob.glob(os.path.join(local_dir, "**", "*.*"), recursive=True)

    def _upload_one(local_file: str):
        try:
            rel_path = os.path.relpath(local_file, local_dir).replace("\\", "/")
            blob_path = f"{dest_prefix.strip('/')}/{rel_path}"
            blob = bucket.blob(blob_path)
            content_type = "image/png" if local_file.endswith(".png") else "image/jpeg" if local_file.endswith((".jpg", ".jpeg")) else "application/json"
            blob.upload_from_filename(local_file, content_type=content_type, timeout=30)
        except Exception as err:
            print(f"[GCS Directory Upload Warning] {local_file} -> {blob_path}: {err}")

    if all_files:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_upload_one, all_files))

    # Purge scratch directory
    try:
        shutil.rmtree(local_dir, ignore_errors=True)
    except Exception:
        pass

def ensure_buckets_exist():
    """Ensures configured GCS buckets exist in the GCP project."""
    client = get_gcs_client()
    for bucket_name in [settings.GCS_RAW_BUCKET, settings.GCS_PYRAMIDS_BUCKET, settings.GCS_ARTIFACTS_BUCKET]:
        try:
            bucket = client.bucket(bucket_name)
            if not bucket.exists(timeout=5.0):
                client.create_bucket(bucket_name, location=settings.GCP_REGION)
                print(f"[GCS] Created bucket: {bucket_name}")
        except Exception as e:
            # Bucket exists or already accessible
            pass

def get_gcs_tile_template_url(slide_id: str, layer: str = "{layer}") -> str:
    """
    Returns high-speed streaming tile URL template for OpenSeadragon.
    Format: https://cdn.oncogemma.com/{slide_id}/{layer}/{z}/{x}_{y}.png
    Or API endpoint: /api/v1/cases/tiles/{slide_id}/{layer}/{z}/{x}_{y}.png
    """
    if settings.CDN_BASE_URL:
        return f"{settings.CDN_BASE_URL.rstrip('/')}/{slide_id}/{layer}/{{z}}/{{x}}_{{y}}.png"
    return f"/api/v1/cases/tiles/{slide_id}/{layer}/{{z}}/{{x}}_{{y}}.png"

def get_gcs_artifact_direct_url(relative_gcs_path: str) -> str:
    """
    Resolves a gs:// or relative artifact path to a direct Cloud CDN / GCS URL or API endpoint.
    """
    path = relative_gcs_path.replace("gs://", "").lstrip("/")
    if settings.CDN_BASE_URL:
        return f"{settings.CDN_BASE_URL.rstrip('/')}/{path}"
    
    parts = path.split("/")
    if "cases" in parts:
        c_idx = parts.index("cases")
        if len(parts) > c_idx + 3 and parts[c_idx+2] == "triage":
            case_id = parts[c_idx+1]
            if "patches" in parts:
                hs_file = parts[-1]
                hs_id = hs_file.replace("_thumb.png", "").replace(".png", "")
                return f"/api/v1/stages/triage/{case_id}/hotspots/{hs_id}/thumbnail?mag=10x"
            elif "heatmap" in parts[-1]:
                return f"/api/v1/stages/triage/{case_id}/heatmap"

    return f"https://storage.googleapis.com/{path}"

