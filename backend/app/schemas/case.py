from datetime import datetime, timezone
from uuid import UUID
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

class CaseCreate(BaseModel):
    pass

class CaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_by: str
    status: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime, _info) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

class SlideUploadUrlRequest(BaseModel):
    filename: str
    size_bytes: int
    content_type: str = "application/octet-stream"

class SlideUploadUrlResponse(BaseModel):
    upload_url: str
    gcs_uri: str

class SlideFinalizeRequest(BaseModel):
    gcs_uri: str
    client_sha256: str | None = None

    @field_validator("gcs_uri")
    @classmethod
    def validate_gcs_uri(cls, v: str) -> str:
        if not v or not v.strip().startswith("gs://") or len(v.strip()) <= 5:
            raise ValueError("gcs_uri must be a valid gs:// URI")
        return v.strip()

class SlideMppUpdateRequest(BaseModel):
    mpp_x: float
    mpp_y: float | None = None

class CaseDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_by: str
    status: str
    created_at: datetime
    slides: list[dict] = []
    stages: list[dict] = []
    tile_url_template: str | None = None
    cdn_base_url: str | None = None

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime, _info) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

class ApproveStageRequest(BaseModel):
    override_justification: str | None = None
