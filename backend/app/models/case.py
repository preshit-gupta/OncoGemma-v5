import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.base import GUID

class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="open") # open, needs_rescan, done
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    slides = relationship("Slide", back_populates="case", cascade="all, delete-orphan", passive_deletes=True)
    stage_executions = relationship("StageExecution", back_populates="case", cascade="all, delete-orphan", passive_deletes=True)
    grading = relationship("Grading", back_populates="case", uselist=False, cascade="all, delete-orphan", passive_deletes=True)
    reports = relationship("Report", back_populates="case", cascade="all, delete-orphan", passive_deletes=True, order_by="desc(Report.version)")
    hotspots = relationship("Hotspot", cascade="all, delete-orphan", passive_deletes=True)

    @property
    def report(self):
        return self.reports[0] if self.reports else None
    detections = relationship("Detection", cascade="all, delete-orphan", passive_deletes=True)
    hpf_sites = relationship("HpfSite", cascade="all, delete-orphan", passive_deletes=True)
