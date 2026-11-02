"""Database tables."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Table, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _new_id() -> str:
    return str(uuid.uuid4())


template_tags = Table(
    "resume_template_tags",
    Base.metadata,
    Column(
        "template_id",
        ForeignKey("resume_templates.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("tag_id", ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(100), unique=True)

    templates: Mapped[list["ResumeTemplate"]] = relationship(
        secondary=template_tags, back_populates="tags"
    )


class ResumeTemplate(Base):
    """A reference LaTeX resume whose style generated resumes follow."""

    __tablename__ = "resume_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    latex_src: Mapped[str] = mapped_column(Text)
    preview_image: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    tags: Mapped[list[Tag]] = relationship(
        secondary=template_tags, back_populates="templates", lazy="selectin"
    )
