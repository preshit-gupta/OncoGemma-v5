import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.base import GUID

# Use JSONB on PostgreSQL, JSON on SQLite
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")

class Report(Base):
    __tablename__ = "reports"

    case_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    specimen_type: Mapped[str] = mapped_column(String, nullable=False, default="core_biopsy")
    procedure: Mapped[str] = mapped_column(String, nullable=False, default="Core Needle Biopsy")
    laterality: Mapped[str] = mapped_column(String, nullable=False, default="right")
    tumor_site: Mapped[str] = mapped_column(String, nullable=False, default="upper_outer_quadrant")
    histologic_type: Mapped[str] = mapped_column(String, nullable=False, default="IDC-NST")
    tumor_size_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    lvi_status: Mapped[str] = mapped_column(String, nullable=False, default="absent")
    dcis_present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    margins: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON_TYPE,
        nullable=True,
        default=None
    )
    lymph_nodes: Mapped[Dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=lambda: {"examined_count": 0, "positive_count": 0, "extranodal_extension": False, "largest_metastasis_mm": 0.0}
    )
    biomarkers: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON_TYPE,
        nullable=True,
        default=None
    )
    staging: Mapped[Dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=lambda: {"ajcc_version": "8th/9th Edition", "pt_stage": "pTX", "pn_stage": "pNX", "pm_stage": "cM0", "stage_group": "Unknown"}
    )
    narrative: Mapped[Dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=lambda: {"diagnosis_line": "", "microscopic_findings": "", "clinical_correlation": ""}
    )
    visual_evidence: Mapped[Dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=lambda: {"heatmap_url": "", "mitotic_crop_url": "", "grading_patch_url": ""}
    )
    
    status: Mapped[str] = mapped_column(String, nullable=False, default="draft") # draft, in_review, signed, amended
    pdf_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pdf_sha256: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    signed_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    npi: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    attestation_statement: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    signed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    integrity_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    narrative_edited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    amendments: Mapped[List[Dict[str, Any]]] = mapped_column(JSON_TYPE, nullable=False, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    case = relationship("Case", back_populates="reports")
