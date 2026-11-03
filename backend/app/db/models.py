"""Database tables."""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    LargeBinary,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JsonColumn = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


def _new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


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


class RunKind(enum.StrEnum):
    SKILLS_FROM_RESUME = "skills_from_resume"
    SKILLS_FROM_GITHUB = "skills_from_github"
    JOB_MARKET = "job_market"
    LEARNING_PATH = "learning_path"
    RESUME = "resume"
    RESUME_RENDER = "resume_render"
    RESUME_REVIEW = "resume_review"
    INTERVIEW = "interview"


class RunStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_finished(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED)


class CareerSession(Base):
    """An anonymous user's workspace, identified only by its unguessable id."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    target_role: Mapped[str] = mapped_column(String(200))
    location: Mapped[str] = mapped_column(String(200))
    github_username: Mapped[str | None] = mapped_column(String(39))
    skills: Mapped[list] = mapped_column(JsonColumn, default=list)

    runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class AgentRun(Base):
    """One run of an agent for a session; interviews use the LangGraph thread id as id."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[RunKind] = mapped_column(Enum(RunKind, native_enum=False, length=40))
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, length=20), default=RunStatus.QUEUED
    )
    input: Mapped[dict] = mapped_column(JsonColumn, default=dict)
    result: Mapped[dict | None] = mapped_column(JsonColumn)
    error: Mapped[str | None] = mapped_column(Text)
    # Progress events, in order: [{"step": "...", "detail": {...}, "at": "..."}].
    events: Mapped[list] = mapped_column(JsonColumn, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    session: Mapped[CareerSession] = relationship(back_populates="runs")
    resume_file: Mapped["ResumeFile | None"] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )


class ResumeFile(Base):
    """The LaTeX source and compiled PDF produced by a resume run."""

    __tablename__ = "resume_files"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), primary_key=True
    )
    latex: Mapped[str] = mapped_column(Text)
    pdf: Mapped[bytes | None] = mapped_column(LargeBinary)
