"""Creating, reading, editing and deleting anonymous sessions, plus the summary report."""

from fastapi import APIRouter, status

from app.api.deps import CurrentSession, Db
from app.api.schemas import SessionCreate, SessionOut, SessionUpdate
from app.db import career_sessions
from app.db.models import RunKind

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SessionOut)
def create_session(body: SessionCreate, db: Db):
    return career_sessions.create_session(db, body.target_role, body.location)


@router.get("/{session_id}", response_model=SessionOut)
def read_session(career_session: CurrentSession):
    return career_session


@router.patch("/{session_id}", response_model=SessionOut)
def update_session(body: SessionUpdate, career_session: CurrentSession):
    changes = body.model_dump(exclude_none=True)
    if "skills" in changes:
        changes["skills"] = list(dict.fromkeys(skill for skill in changes["skills"] if skill))
    for name, value in changes.items():
        setattr(career_session, name, value)
    return career_session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(career_session: CurrentSession, db: Db):
    db.delete(career_session)


def _latest_result(db, session_id: str, kind: RunKind) -> dict | None:
    run = career_sessions.latest_succeeded_run(db, session_id, kind)
    return None if run is None else {"run_id": run.id, **(run.result or {})}


@router.get("/{session_id}/report")
def read_report(career_session: CurrentSession, db: Db) -> dict:
    """Everything the session has produced so far: the latest result of each agent."""
    interviews = career_sessions.list_runs(db, career_session.id, RunKind.INTERVIEW)
    return {
        "session": SessionOut.model_validate(career_session).model_dump(mode="json"),
        "job_market": _latest_result(db, career_session.id, RunKind.JOB_MARKET),
        "learning_path": _latest_result(db, career_session.id, RunKind.LEARNING_PATH),
        "resume": _latest_resume(db, career_session.id),
        "resume_review": _latest_result(db, career_session.id, RunKind.RESUME_REVIEW),
        "interviews": [
            {"id": run.id, "status": run.status, "summary": (run.result or {}).get("summary")}
            for run in interviews
        ],
    }


def _latest_resume(db, session_id: str) -> dict | None:
    """The newer of the latest generated or re-rendered resume."""
    candidates = [
        run
        for kind in (RunKind.RESUME, RunKind.RESUME_RENDER)
        if (run := career_sessions.latest_succeeded_run(db, session_id, kind))
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda run: run.finished_at)
    return {"run_id": newest.id, **(newest.result or {})}
