"""
SQLAlchemy ORM model for Mitotic Detections (v4.3 Mitosis Detection & Verification).
"""
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB

from app.core.db import Base
from app.models.base import GUID

JSONType = JSON().with_variant(JSONB, "postgresql")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(String, primary_key=True) # e.g. m_0001
    case_id = Column(GUID, ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True, index=True)
    hotspot_id = Column(String, nullable=True)
    centroid_um = Column(JSONType, nullable=False) # [x, y] in base micrometers
    det_conf = Column(Float, nullable=True) # YOLO detector confidence
    ver_conf = Column(Float, nullable=True) # HoVer-Net verifier confidence
    label = Column(String, nullable=False, default="unreviewed") # mitosis | not_mitosis | unreviewed
    label_source = Column(String, nullable=False, default="model") # model | pathologist | medgemma
    medgemma_verdict = Column(String, nullable=True) # CONFIRMED | REJECTED_APOPTOSIS | REJECTED_LYMPHOCYTE | REJECTED_RESTING_NUCLEUS | EQUIVOCAL
    medgemma_rationale = Column(Text, nullable=True) # Multimodal referee rationale
    medgemma_confidence = Column(String, nullable=True) # low | medium | high
    crop_uri = Column(Text, nullable=True) # Normalized 128x128 crop URI
    crop_orig_uri = Column(Text, nullable=True) # Original 128x128 crop URI
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
