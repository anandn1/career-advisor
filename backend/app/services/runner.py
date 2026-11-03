"""Executes agent runs in background threads and records their progress in the database.

Each run row moves queued -> running -> succeeded/failed. Handlers report progress
through RunContext.progress(); every event is committed right away so a browser
streaming /runs/{id}/events sees it, and can reconnect without losing any.
"""

import json
import logging
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field

from app.db.career_sessions import unique_ignoring_case
from app.db.models import AgentRun, CareerSession, RunKind, RunStatus, utcnow
from app.db.session import session_scope

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 500
SAVE_FAILED_MESSAGE = "The run finished but its result could not be saved. Please try again."


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
            if run is None:
                raise RunGoneError("The session was deleted while this run was in progress.")
            event = {"step": step, "detail": detail or {}, "at": utcnow().isoformat()}
            run.events = [*run.events, event]

    def update_session(self, **fields) -> None:
        """Write results that belong to the session itself (e.g. its GitHub username)."""
        with session_scope() as db:
            career_session = self._live_session(db)
            for name, value in fields.items():
                setattr(career_session, name, value)

    def add_session_skills(self, found: list[str]) -> list[str]:
        """Merge `found` into the session's skills as stored now (not the start-of-run snapshot),
        so edits made while the run was working are kept. Returns the merged list."""
        with session_scope() as db:
            career_session = self._live_session(db)
            career_session.skills = unique_ignoring_case([*(career_session.skills or []), *found])
            return list(career_session.skills)

    def _live_session(self, db) -> CareerSession:
        career_session = db.get(CareerSession, self.session_id)
        if career_session is None:
            raise RunGoneError("The session was deleted while this run was in progress.")
        return career_session


Handler = Callable[[RunContext], dict]


class UserFacingError(Exception):
    """A failure whose message is safe and useful to show the user as-is."""


class RunGoneError(UserFacingError):
    """The run's row (or its whole session) was deleted while it was running."""


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
        except Exception as error:  # a broken row must not kill the worker thread
            logger.exception("Could not start run %s", run_id)
            self._finish_safely(run_id, RunStatus.FAILED, error=_error_message(error))
            return
        try:
            result = handler(context)
            json.dumps(result)  # fail here, not at commit, if the result cannot be stored
        except Exception as error:  # every agent failure becomes a failed run
            logger.exception("Run %s failed", run_id)
            self._finish_safely(run_id, RunStatus.FAILED, error=_error_message(error))
        else:
            self._finish_safely(run_id, RunStatus.SUCCEEDED, result=result)

    def _start(self, run_id: str) -> tuple[RunContext, Handler]:
        with session_scope() as db:
            run = db.get(AgentRun, run_id)
            run.status, run.started_at = RunStatus.RUNNING, utcnow()
            context = RunContext(run.id, run.session_id, dict(run.input), _snapshot(run.session))
            return context, self.handlers[run.kind]

    def _finish(self, run_id: str, status: RunStatus, **fields) -> None:
        with session_scope() as db:
            run = db.get(AgentRun, run_id)
            if run is None:  # deleted along with its session; nothing left to update
                return
            run.status, run.finished_at = status, utcnow()
            for name, value in fields.items():
                setattr(run, name, value)

    def _finish_safely(self, run_id: str, status: RunStatus, **fields) -> None:
        """Finish the run; if that fails, still mark it failed so its kind is not blocked."""
        try:
            self._finish(run_id, status, **fields)
        except Exception:  # e.g. the database rejected the update
            logger.exception("Could not save the outcome of run %s", run_id)
            try:
                self._finish(run_id, RunStatus.FAILED, error=SAVE_FAILED_MESSAGE)
            except Exception:  # database unreachable: startup will mark it failed
                logger.exception("Could not mark run %s failed", run_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
