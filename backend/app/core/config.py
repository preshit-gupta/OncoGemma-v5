import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = os.getenv("ENV", "dev")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "dev")
    DEBUG: bool = True
    
    # GCP
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "oncogemma")
    GCP_REGION: str = os.getenv("GCP_REGION", "us-central1")
    USE_REAL_GCS: bool = os.getenv("USE_REAL_GCS", "true").lower() in ("true", "1")
    
    # Vertex AI Endpoint Configuration - Path Foundation (Stage 3)
    VERTEX_PATH_FOUNDATION_ENDPOINT_ID: str = os.getenv(
        "VERTEX_PATH_FOUNDATION_ENDPOINT_ID",
        "mg-endpoint-25e5ee92-10b3-41b5-9da7-bccbd2b255f8"
    )
    VERTEX_PATH_FOUNDATION_LOCATION: str = os.getenv(
        "VERTEX_PATH_FOUNDATION_LOCATION",
        "asia-south1"
    )
    VERTEX_PATH_FOUNDATION_API_ENDPOINT: str = os.getenv(
        "VERTEX_PATH_FOUNDATION_API_ENDPOINT",
        "mg-endpoint-25e5ee92-10b3-41b5-9da7-bccbd2b255f8.asia-south1-962838713357.prediction.vertexai.goog"
    )

    # Vertex AI Endpoint Configuration - MedGemma 1.5 (Stage 5 Grading)
    VERTEX_MEDGEMMA_ENDPOINT_ID: str = os.getenv(
        "VERTEX_MEDGEMMA_ENDPOINT_ID",
        "medgemma-1-5-endpoint"
    )
    VERTEX_MEDGEMMA_LOCATION: str = os.getenv(
        "VERTEX_MEDGEMMA_LOCATION",
        "us-central1"
    )
    VERTEX_MEDGEMMA_MODEL_VERSION: str = os.getenv(
        "VERTEX_MEDGEMMA_MODEL_VERSION",
        "1.5@2026.08"
    )
    MEDGEMMA_TEMPERATURE: float = float(os.getenv("MEDGEMMA_TEMPERATURE", "0.0"))
    MEDGEMMA_MAX_RETRIES: int = int(os.getenv("MEDGEMMA_MAX_RETRIES", "2"))
    USE_MOCK_VERTEX_AI: bool = os.getenv("USE_MOCK_VERTEX_AI", "false").lower() in ("true", "1")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://oncogemma:oncogemma_dev_password@localhost:5432/oncogemma_db"
    )
    
    # GCS Configuration
    GCS_RAW_BUCKET: str = os.getenv("GCS_RAW_BUCKET", "oncogemma-dev-raw")
    GCS_PYRAMIDS_BUCKET: str = os.getenv("GCS_PYRAMIDS_BUCKET", "oncogemma-dev-pyramids")
    GCS_ARTIFACTS_BUCKET: str = os.getenv("GCS_ARTIFACTS_BUCKET", "oncogemma-dev-artifacts")
    CDN_BASE_URL: str | None = os.getenv("CDN_BASE_URL", None)
    STORAGE_EMULATOR_HOST: str | None = os.getenv("STORAGE_EMULATOR_HOST", None)

    # Cloud Tasks & Asynchronous Cloud Workers
    USE_CLOUD_TASKS: bool = os.getenv("USE_CLOUD_TASKS", "false").lower() in ("true", "1")
    CLOUD_TASKS_LOCATION: str = os.getenv("CLOUD_TASKS_LOCATION", os.getenv("GCP_REGION", "us-central1"))
    CLOUD_TASKS_QUEUE: str = os.getenv("CLOUD_TASKS_QUEUE", "oncogemma-stage-queue")
    WORKER_SERVICE_URL: str = os.getenv("WORKER_SERVICE_URL", "http://localhost:8000")
    CLOUD_TASKS_SERVICE_ACCOUNT: str = os.getenv("CLOUD_TASKS_SERVICE_ACCOUNT", "")
    
    # Auth
    MOCK_AUTH_ENABLED: bool = True
    DEFAULT_MOCK_ROLE: str = "pathologist"
    DEFAULT_MOCK_USER_ID: str = "user_pathologist_001"
    
    # Config directory
    CONFIGS_DIR: str = os.path.join(os.path.dirname(__file__), "../../../configs")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
