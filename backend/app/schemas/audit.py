from datetime import datetime
from pydantic import BaseModel, ConfigDict

class AuditEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    case_id: str | None
    actor: str
    event_type: str
    stage: str | None
    payload: dict | None
    created_at: datetime

class PaginatedAuditEvents(BaseModel):
    events: list[AuditEventResponse]
    total: int
    page: int
    page_size: int
