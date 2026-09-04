"""
SQLAlchemy ORM model for Virtual HPF Sites (v4.3 Mitosis Detection & Virtual HPFs).
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB

from app.core.db import Base
from app.models.base import GUID

JSONType = JSON().with_variant(JSONB, "postgresql")


class HpfSite(Base):
    __tablename__ = "hpf_sites"

    case_id = Column(GUID, ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True, index=True)
    seq = Column(Integer, primary_key=True) # 1 to 10
    center_um = Column(JSONType, nullable=False) # [x, y] in base micrometers
    radius_um = Column(Float, nullable=False, default=262.0)
    mitotic_count = Column(Integer, nullable=False, default=0)
    source = Column(String, nullable=False, default="model") # model | pathologist
    image_patch_uri = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
