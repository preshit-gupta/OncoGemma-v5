import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.base import GUID

class Slide(Base):
    __tablename__ = "slides"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    
    gcs_uri_original: Mapped[str] = mapped_column(String, nullable=False)
    gcs_uri_pyramid: Mapped[str | None] = mapped_column(String, nullable=True)
    gcs_uri_pyramid_norm: Mapped[str | None] = mapped_column(String, nullable=True)
    
    format: Mapped[str | None] = mapped_column(String, nullable=True)
    scanner: Mapped[str | None] = mapped_column(String, nullable=True)
    
    mpp_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    mpp_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_mag: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    checksum_sha256: Mapped[str | None] = mapped_column(String, nullable=True)
    label_stripped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    case = relationship("Case", back_populates="slides")
