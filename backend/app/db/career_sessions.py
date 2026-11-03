"""Queries for anonymous career sessions and their agent runs."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import AgentRun, CareerSession, RunKind, RunStatus, utcnow

UNFINISHED = (RunStatus.QUEUED, RunStatus.RUNNING)


def create_session(db: Session, target_role: str, location: str) -> CareerSession:
    career_session = CareerSession(target_role=target_role, location=location, skills=[])
    db.add(career_session)
    db.flush()
    return career_session


def get_session(db: Session, session_id: str) -> CareerSession | None:
    """The session, marked as just seen (so it is not expired), or None."""
    career_session = db.get(CareerSession, session_id)
    if career_session is not None:
        career_session.last_seen_at = utcnow()
    return career_session


def create_run(
    db: Session, career_session: CareerSession, kind: RunKind, run_input: dict, **fields
) -> AgentRun:
    run = AgentRun(session=career_session, kind=kind, input=run_input, **fields)
    db.add(run)
    db.flush()
    return run


def get_run(db: Session, run_id: str) -> AgentRun | None:
    return db.get(AgentRun, run_id)


def list_runs(db: Session, session_id: str, kind: RunKind | None = None) -> list[AgentRun]:
    query = select(AgentRun).where(AgentRun.session_id == session_id)
    if kind is not None:
        query = query.where(AgentRun.kind == kind)
    return list(db.scalars(query.order_by(AgentRun.created_at.desc(), AgentRun.id)))


def latest_succeeded_run(db: Session, session_id: str, kind: RunKind) -> AgentRun | None:
    query = (
        select(AgentRun)
        .where(AgentRun.session_id == session_id, AgentRun.kind == kind)
        .where(AgentRun.status == RunStatus.SUCCEEDED)
        .order_by(AgentRun.finished_at.desc().nulls_last(), AgentRun.created_at.desc())
        .limit(1)
    )
    return db.scalars(query).first()


def unfinished_run(db: Session, session_id: str, kind: RunKind) -> AgentRun | None:
    query = select(AgentRun).where(
        AgentRun.session_id == session_id, AgentRun.kind == kind, AgentRun.status.in_(UNFINISHED)
    )
    return db.scalars(query.limit(1)).first()


def fail_unfinished_runs(db: Session, reason: str) -> int:
    """Mark queued/running runs failed (used at startup: their worker threads are gone).

    Interviews are skipped: they are not worker jobs, and their state lives in the checkpointer.
    """
    runs = db.scalars(
        select(AgentRun).where(AgentRun.status.in_(UNFINISHED), AgentRun.kind != RunKind.INTERVIEW)
    ).all()
    for run in runs:
        run.status, run.error, run.finished_at = RunStatus.FAILED, reason, utcnow()
    return len(runs)


@dataclass(frozen=True)
class ExpiredSessions:
    session_ids: list[str]
    interview_thread_ids: list[str]


EXPIRY_BATCH_SIZE = 500  # keeps each IN (...) list well under database parameter limits


def interview_thread_ids(db: Session, session_ids: list[str]) -> list[str]:
    query = select(AgentRun.id).where(
        AgentRun.session_id.in_(session_ids), AgentRun.kind == RunKind.INTERVIEW
    )
    return list(db.scalars(query))


def delete_expired_sessions(
    db: Session, ttl: timedelta, now: datetime | None = None
) -> ExpiredSessions:
    """Delete sessions unseen for `ttl`, in batches; returns what was deleted."""
    cutoff = (now or utcnow()) - ttl
    deleted = ExpiredSessions(session_ids=[], interview_thread_ids=[])
    while batch := _delete_expired_batch(db, cutoff):
        deleted.session_ids.extend(batch.session_ids)
        deleted.interview_thread_ids.extend(batch.interview_thread_ids)
    return deleted


def _delete_expired_batch(db: Session, cutoff: datetime) -> ExpiredSessions | None:
    # One list of ids for both steps, so checkpoints are only deleted for deleted sessions.
    expired_ids = list(
        db.scalars(
            select(CareerSession.id)
            .where(CareerSession.last_seen_at < cutoff)
            .limit(EXPIRY_BATCH_SIZE)
        )
    )
    if not expired_ids:
        return None
    thread_ids = interview_thread_ids(db, expired_ids)
    db.execute(delete(AgentRun).where(AgentRun.session_id.in_(expired_ids)))
    db.execute(delete(CareerSession).where(CareerSession.id.in_(expired_ids)))
    return ExpiredSessions(session_ids=expired_ids, interview_thread_ids=thread_ids)
