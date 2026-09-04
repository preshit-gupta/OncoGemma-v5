import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Text, DateTime, ForeignKey, UniqueConstraint, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB

from app.core.db import Base
from app.models.base import GUID

JSONType = JSON().with_variant(JSONB, "postgresql")

class StageExecution(Base):
    __tablename__ = "stage_executions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    
    stage: Mapped[str] = mapped_column(String, nullable=False) # ingest|preprocess|qc|triage|mitosis|grading|report
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued") # queued|running|awaiting_review|confirmed|rejected|done|failed
    
    input_ref: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    output_ref: Mapped[str | None] = mapped_column(Text, nullable=True) # GCS URI of result json blob
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    model_versions: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    reviewed_by: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    review_edits: Mapped[dict | None] = mapped_column(JSONType, nullable=True) # RFC-6902 JSON diff

    case = relationship("Case", back_populates="stage_executions")
    hotspots = relationship("Hotspot", cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (
        UniqueConstraint("case_id", "stage", "attempt", name="uq_stage_execution_attempt"),
        Index("ix_stage_poll", "status", "stage"),
    )
