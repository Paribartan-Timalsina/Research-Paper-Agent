from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Insight(Base):
    __tablename__ = "insights"

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid4)
    paper_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), unique=True
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    contributions: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    methodology: Mapped[str | None] = mapped_column(Text, nullable=True)
    limitations: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    future_work: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    paper = relationship("Paper", back_populates="insight")
