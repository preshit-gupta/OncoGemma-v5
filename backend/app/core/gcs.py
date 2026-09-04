import os
import io
import shutil
import glob
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
import google.oauth2.service_account
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

def resolve_slide_raw_uri(case_id: str, slide_obj=None) -> str | None:
    """
    Authoritative resolution of the raw Whole-Slide Image (SVS/NDPI/TIFF) URI in GCS.
    Resolves dynamically even when the uploaded file has a UUID or arbitrary filename.
    """
    if slide_obj:
        uri = getattr(slide_obj, "gcs_uri_original", None)
        if uri and uri.startswith("gs://"):
            b_name, o_name = parse_gcs_uri(uri)
            try:
                if blob_exists(b_name, o_name):
                    return uri
            except Exception:
                pass

    try:
        client = get_gcs_client()
        bucket = client.bucket(settings.GCS_RAW_BUCKET)
        blobs = list(bucket.list_blobs(prefix=f"cases/{case_id}/"))
        
        for b in blobs:
            if b.name.lower().endswith((".svs", ".ndpi", ".tiff", ".tif", ".mrxs", ".bif", ".vms")):
                return f"gs://{settings.GCS_RAW_BUCKET}/{b.name}"
                
        if blobs:
            blobs.sort(key=lambda x: getattr(x, "size", 0) or 0, reverse=True)
            return f"gs://{settings.GCS_RAW_BUCKET}/{blobs[0].name}"
    except Exception as e:
        print(f"[Resolve Raw Slide URI Error] {e}")

    return None

def get_service_account_email() -> str:
    """
    Resolves the canonical service account email for signing.
    Avoids using 'default' which is returned by compute_engine credentials.
    """
    sa_override = os.getenv("SERVICE_ACCOUNT_EMAIL")
    if sa_override and "@" in sa_override:
        return sa_override

    try:
        import urllib.request
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            email = resp.read().decode("utf-8").strip()
            if "@" in email:
                return email
    except Exception:
        pass

    return "oncogemma-cloudrun-sa@oncogemma.iam.gserviceaccount.com"


def generate_signed_upload_url(bucket_name: str, blob_name: str, expiration_minutes: int = 60) -> str:
    """
    Generates a V4 signed upload URL for direct browser-to-GCS upload.
    Uses IAM Credentials API with explicit cloud-platform scope for Cloud Run compatibility.
    """
    import google.auth
    from google.auth.transport.requests import Request
    from google.auth.iam import Signer as IAMSigner

    expiration = timedelta(minutes=expiration_minutes)

    # 1. Strategy 1: Explicit private key signer (local dev with service account JSON file)
    try:
        client = get_gcs_client()
        creds = client._credentials
        sa_email = getattr(creds, "service_account_email", "")
        if hasattr(creds, "signer") and "@" in sa_email:
            bucket = get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="PUT",
                content_type="application/octet-stream",
            )
    except Exception as e:
        print(f"[Signed URL] Strategy 1 (SA private key) not available: {e}")

    # 2. Strategy 2: IAM Credentials API Signer (Cloud Run / GCE / Workload Identity)
    try:
        sa_email = get_service_account_email()
        iam_auth_creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        req = Request()
        if not iam_auth_creds.valid:
            iam_auth_creds.refresh(req)

        iam_signer = IAMSigner(req, iam_auth_creds, sa_email)
        signing_creds = google.oauth2.service_account.Credentials(
            signer=iam_signer,
            service_account_email=sa_email,
            token_uri="https://oauth2.googleapis.com/token",
        )

        iam_client = storage.Client(project=settings.GCP_PROJECT_ID, credentials=signing_creds)
        iam_bucket = iam_client.bucket(bucket_name)
        iam_blob = iam_bucket.blob(blob_name)

        return iam_blob.generate_signed_url(
            version="v4",
            expiration=expiration,
            method="PUT",
            content_type="application/octet-stream",
        )
    except Exception as e2:
        print(f"[Signed URL] Strategy 2 (IAM Signer) failed: {e2}")
        raise RuntimeError(
            f"Could not generate signed upload URL via IAM Credentials API: {e2}\n"
            f"Ensure {sa_email} has roles/iam.serviceAccountTokenCreator on project {settings.GCP_PROJECT_ID}."
        )

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

