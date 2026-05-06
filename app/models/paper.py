from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(512))
    filename: Mapped[str] = mapped_column(String(512))
    raw_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tasks = relationship("AgentTask", back_populates="paper", cascade="all, delete-orphan")
    insight = relationship("Insight", back_populates="paper", uselist=False, cascade="all, delete-orphan")
    conversations = relationship(
        "Conversation", back_populates="paper", cascade="all, delete-orphan"
    )
    figures = relationship(
        "PaperFigure", back_populates="paper", cascade="all, delete-orphan",
        order_by="PaperFigure.page",
    )
