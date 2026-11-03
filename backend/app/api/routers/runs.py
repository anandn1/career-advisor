"""Starting agent runs, polling them, streaming their progress and downloading resume files."""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.exc import IntegrityError

from app.agents.resume_strategist import ResumeProfile
from app.api.deps import CurrentSession, Db, Runner
from app.api.schemas import (
    MAX_REQUEST_LENGTH,
    GithubSkillsRequest,
    ResumeRenderRequest,
    ResumeRequest,
    RunOut,
)
from app.core.config import get_settings
from app.db import career_sessions
from app.db.models import AgentRun, CareerSession, RunKind
from app.db.session import session_scope
from app.db.templates import get_template
from app.services import documents, github

router = APIRouter(prefix="/sessions/{session_id}/runs", tags=["runs"])

EVENT_POLL_SECONDS = 0.5
MAX_STREAM_SECONDS = 10 * 60
RESUME_KINDS = (RunKind.RESUME, RunKind.RESUME_RENDER)


def _start_run(db, runner, career_session: CareerSession, kind: RunKind, run_input: dict):
    """Queue a run; at most one unfinished run of each kind per session."""
    conflict = HTTPException(status.HTTP_409_CONFLICT, f"A {kind} run is already in progress")
    if career_sessions.unfinished_run(db, career_session.id, kind):
        raise conflict
    try:
        run = career_sessions.create_run(db, career_session, kind, run_input)
        db.commit()  # the worker thread reads the row from its own transaction
    except IntegrityError as error:  # a concurrent request won the race (unique index)
        db.rollback()
        raise conflict from error
    runner.submit(run.id)
    return run


async def _read_upload(upload: UploadFile) -> str:
    """Text of an uploaded resume; the file itself is never stored."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in documents.SUPPORTED_SUFFIXES:
        allowed = ", ".join(documents.SUPPORTED_SUFFIXES)
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Upload one of: {allowed}")
    limit = get_settings().max_upload_bytes
    content = await upload.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"Files are limited to {limit} bytes"
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(content)
        try:
            text = await asyncio.to_thread(documents.read_document_text, path)
        except documents.DocumentError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    text = text.replace("\x00", "")  # NUL characters cannot be stored in Postgres JSON
    if not text.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No text found in the file")
    return text


@router.post("/skills/resume", status_code=status.HTTP_202_ACCEPTED, response_model=RunOut)
async def extract_skills_from_resume(
    career_session: CurrentSession, db: Db, runner: Runner, file: Annotated[UploadFile, File()]
):
    text = await _read_upload(file)
    return _start_run(db, runner, career_session, RunKind.SKILLS_FROM_RESUME, {"resume_text": text})


@router.post("/skills/github", status_code=status.HTTP_202_ACCEPTED, response_model=RunOut)
def extract_skills_from_github(
    body: GithubSkillsRequest, career_session: CurrentSession, db: Db, runner: Runner
):
    try:
        username = github.parse_username(body.github_username)
    except github.InvalidUsernameError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    run_input = {"github_username": username}
    return _start_run(db, runner, career_session, RunKind.SKILLS_FROM_GITHUB, run_input)


@router.post("/job-market", status_code=status.HTTP_202_ACCEPTED, response_model=RunOut)
def analyze_job_market(career_session: CurrentSession, db: Db, runner: Runner):
    return _start_run(db, runner, career_session, RunKind.JOB_MARKET, {})


@router.post("/learning-path", status_code=status.HTTP_202_ACCEPTED, response_model=RunOut)
def build_learning_path(career_session: CurrentSession, db: Db, runner: Runner):
    analysis_run = career_sessions.latest_succeeded_run(db, career_session.id, RunKind.JOB_MARKET)
    gaps = ((analysis_run.result or {}).get("analysis", {}) if analysis_run else {}).get(
        "skill_gaps", []
    )
    if not gaps:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Run a job market analysis that finds skill gaps first"
        )
    run_input = {"skill_gaps": gaps, "job_market_run_id": analysis_run.id}
    return _start_run(db, runner, career_session, RunKind.LEARNING_PATH, run_input)


def _check_template(db, template_id: str | None) -> None:
    if template_id is not None and get_template(db, template_id) is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown template_id")


@router.post("/resume", status_code=status.HTTP_202_ACCEPTED, response_model=RunOut)
def generate_resume(body: ResumeRequest, career_session: CurrentSession, db: Db, runner: Runner):
    _check_template(db, body.template_id)
    profile = ResumeProfile(
        target_role=career_session.target_role,
        contact=body.contact,
        github_username=career_session.github_username if body.include_github else None,
        education=body.education,
        work_experience=body.work_experience,
        skills=list(dict.fromkeys([*career_session.skills, *body.extra_skills])),
    )
    run_input = {"profile": profile.model_dump(), "template_id": body.template_id}
    return _start_run(db, runner, career_session, RunKind.RESUME, run_input)


@router.post("/resume-render", status_code=status.HTTP_202_ACCEPTED, response_model=RunOut)
def rerender_resume(
    body: ResumeRenderRequest, career_session: CurrentSession, db: Db, runner: Runner
):
    _check_template(db, body.template_id)
    run_input = {"draft": body.draft.model_dump(), "template_id": body.template_id}
    return _start_run(db, runner, career_session, RunKind.RESUME_RENDER, run_input)


@router.post("/resume-review", status_code=status.HTTP_202_ACCEPTED, response_model=RunOut)
async def review_resume(
    career_session: CurrentSession,
    db: Db,
    runner: Runner,
    file: Annotated[UploadFile, File()],
    request: Annotated[str, Form(max_length=MAX_REQUEST_LENGTH)] = "",
):
    text = await _read_upload(file)
    run_input = {"resume_text": text, "request": request}
    return _start_run(db, runner, career_session, RunKind.RESUME_REVIEW, run_input)


@router.get("", response_model=list[RunOut])
def list_runs(career_session: CurrentSession, db: Db, kind: RunKind | None = None):
    return career_sessions.list_runs(db, career_session.id, kind)


def _get_owned_run(db, career_session: CareerSession, run_id: str) -> AgentRun:
    run = career_sessions.get_run(db, run_id)
    if run is None or run.session_id != career_session.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run


@router.get("/{run_id}", response_model=RunOut)
def read_run(run_id: str, career_session: CurrentSession, db: Db):
    return _get_owned_run(db, career_session, run_id)


def _run_snapshot(session_id: str, run_id: str) -> dict | None:
    with session_scope() as db:
        run = career_sessions.get_run(db, run_id)
        if run is None or run.session_id != session_id:
            return None
        return {
            "events": list(run.events),
            "finished": run.status.is_finished,
            "run": RunOut.model_validate(run).model_dump(mode="json"),
        }


def _resume_index(last_event_id: str | None) -> int:
    """Index of the first event to send after a reconnect (Last-Event-ID + 1)."""
    if last_event_id and last_event_id.isascii() and last_event_id.isdigit():
        return int(last_event_id) + 1
    return 0


def _require_streamable_run(session_id: str, run_id: str) -> dict:
    """Runs before streaming starts, so an unknown run is a plain 404 response."""
    snapshot = _run_snapshot(session_id, run_id)
    if snapshot is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return snapshot


@router.get("/{run_id}/events", response_class=EventSourceResponse)
async def stream_run_events(
    session_id: str,
    run_id: str,
    request: Request,
    snapshot: Annotated[dict, Depends(_require_streamable_run)],
    last_event_id: Annotated[str | None, Header()] = None,
):
    """Progress events as they happen (each with its index as id), then a final 'done' event.

    Each poll uses its own short transaction, so a long stream holds no database connection.
    """
    sent = _resume_index(last_event_id)
    deadline = time.monotonic() + MAX_STREAM_SECONDS
    while True:
        for index in range(sent, len(snapshot["events"])):
            yield ServerSentEvent(data=snapshot["events"][index], event="progress", id=str(index))
        sent = len(snapshot["events"])
        if snapshot["finished"]:
            yield ServerSentEvent(data=snapshot["run"], event="done")
            return
        if time.monotonic() >= deadline or await request.is_disconnected():
            return  # browsers reconnect with Last-Event-ID, so nothing is lost
        await asyncio.sleep(EVENT_POLL_SECONDS)
        snapshot = await asyncio.to_thread(_run_snapshot, session_id, run_id)
        if snapshot is None:  # deleted while streaming
            return


def _resume_file(db, career_session: CareerSession, run_id: str):
    run = _get_owned_run(db, career_session, run_id)
    if run.kind not in RESUME_KINDS or run.resume_file is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This run has no resume file")
    return run.resume_file


@router.get("/{run_id}/resume.tex")
def download_resume_latex(run_id: str, career_session: CurrentSession, db: Db):
    resume_file = _resume_file(db, career_session, run_id)
    headers = {"Content-Disposition": 'attachment; filename="resume.tex"'}
    return Response(resume_file.latex, media_type="application/x-tex", headers=headers)


@router.get("/{run_id}/resume.pdf")
def download_resume_pdf(
    run_id: str, career_session: CurrentSession, db: Db, download: bool = False
):
    resume_file = _resume_file(db, career_session, run_id)
    if resume_file.pdf is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No PDF was produced for this resume")
    disposition = "attachment" if download else "inline"
    headers = {"Content-Disposition": f'{disposition}; filename="resume.pdf"'}
    return Response(resume_file.pdf, media_type="application/pdf", headers=headers)
