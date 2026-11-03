"""Executes agent runs in background threads and records their progress in the database.

Each run row moves queued -> running -> succeeded/failed. Handlers report progress
through RunContext.progress(); every event is committed right away so a browser
streaming /runs/{id}/events sees it, and can reconnect without losing any.
"""

import logging
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field

from app.db.models import AgentRun, CareerSession, RunKind, RunStatus, utcnow
from app.db.session import session_scope

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 500


@dataclass(frozen=True)
class SessionSnapshot:
    """The session fields a handler may read (a copy, safe to use off the request thread)."""

    target_role: str
    location: str
    github_username: str | None
    skills: list[str] = field(default_factory=list)


@dataclass
class RunContext:
    run_id: str
    session_id: str
    input: dict
    session: SessionSnapshot

    def progress(self, step: str, detail: dict | None = None) -> None:
        with session_scope() as db:
            run = db.get(AgentRun, self.run_id)
            event = {"step": step, "detail": detail or {}, "at": utcnow().isoformat()}
            run.events = [*run.events, event]

    def update_session(self, **fields) -> None:
        """Write results that belong to the session itself (e.g. extracted skills)."""
        with session_scope() as db:
            career_session = db.get(CareerSession, self.session_id)
            for name, value in fields.items():
                setattr(career_session, name, value)


Handler = Callable[[RunContext], dict]


class UserFacingError(Exception):
    """A failure whose message is safe and useful to show the user as-is."""


def _snapshot(career_session: CareerSession) -> SessionSnapshot:
    return SessionSnapshot(
        target_role=career_session.target_role,
        location=career_session.location,
        github_username=career_session.github_username,
        skills=list(career_session.skills or []),
    )


def _error_message(error: Exception) -> str:
    if isinstance(error, UserFacingError):
        return str(error)[:MAX_ERROR_LENGTH]
    return f"{type(error).__name__}: the agent failed. Please try again."


class RunExecutor:
    def __init__(self, handlers: dict[RunKind, Handler], executor: Executor):
        self.handlers = handlers
        self.executor = executor

    @classmethod
    def with_threads(cls, handlers: dict[RunKind, Handler], workers: int) -> "RunExecutor":
        return cls(handlers, ThreadPoolExecutor(max_workers=workers, thread_name_prefix="run"))

    def submit(self, run_id: str) -> None:
        self.executor.submit(self.execute, run_id)

    def execute(self, run_id: str) -> None:
        """Run one queued run to completion; never raises."""
        try:
            context, handler = self._start(run_id)
        except Exception:  # a broken row must not kill the worker thread
            logger.exception("Could not start run %s", run_id)
            return
        try:
            result = handler(context)
        except Exception as error:  # every agent failure becomes a failed run
            logger.exception("Run %s failed", run_id)
            self._finish(run_id, RunStatus.FAILED, error=_error_message(error))
        else:
            self._finish(run_id, RunStatus.SUCCEEDED, result=result)

    def _start(self, run_id: str) -> tuple[RunContext, Handler]:
        with session_scope() as db:
            run = db.get(AgentRun, run_id)
            run.status, run.started_at = RunStatus.RUNNING, utcnow()
            context = RunContext(run.id, run.session_id, dict(run.input), _snapshot(run.session))
            return context, self.handlers[run.kind]

    def _finish(self, run_id: str, status: RunStatus, **fields) -> None:
        with session_scope() as db:
            run = db.get(AgentRun, run_id)
            run.status, run.finished_at = status, utcnow()
            for name, value in fields.items():
                setattr(run, name, value)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
