"""Starting agent runs, polling them, streaming their progress and downloading resume files."""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.exc import IntegrityError
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.agents.resume_strategist import ResumeProfile
from app.api.deps import CurrentSession, Db, Runner
from app.api.schemas import (
    MAX_REQUEST_LENGTH,
    UPLOAD_KINDS,
    GithubSkillsRun,
    JobMarketRun,
    LearningPathRun,
    ResumeRenderRun,
    ResumeRun,
    RunOut,
    RunRequest,
    RunUpload,
)
from app.core.config import get_settings
from app.db import career_sessions
from app.db.career_sessions import unique_ignoring_case
from app.db.models import AgentRun, CareerSession, RunKind, RunStatus, utcnow
from app.db.session import session_scope
from app.db.templates import get_template
from app.services import documents, github

router = APIRouter(prefix="/sessions/{session_id}/runs", tags=["runs"])

logger = logging.getLogger(__name__)
NOT_SCHEDULED = "The server could not start this run right now. Please try again."
EVENT_POLL_SECONDS = 0.5
MAX_STREAM_SECONDS = 10 * 60
MAX_EVENT_ID_DIGITS = 10
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
    try:
        runner.submit(run.id)
    except Exception as error:  # e.g. the pool is shutting down: don't leave a queued orphan
        logger.exception("Could not schedule run %s", run.id)
        run.status, run.finished_at, run.error = RunStatus.FAILED, utcnow(), NOT_SCHEDULED
        db.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_SCHEDULED) from error
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


def _github_input(db, career_session: CareerSession, body: GithubSkillsRun) -> dict:
    try:
        username = github.parse_username(body.github_username)
    except github.InvalidUsernameError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    return {"github_username": username}


def _job_market_input(db, career_session: CareerSession, body: JobMarketRun) -> dict:
    return {}


def _learning_path_input(db, career_session: CareerSession, body: LearningPathRun) -> dict:
    analysis_run, gaps = career_sessions.latest_skill_gaps(db, career_session.id)
    if not gaps:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Run a job market analysis that finds skill gaps first"
        )
    return {"skill_gaps": gaps, "job_market_run_id": analysis_run.id}


def _check_template(db, template_id: str | None) -> None:
    if template_id is not None and get_template(db, template_id) is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown template_id")


def _resume_input(db, career_session: CareerSession, body: ResumeRun) -> dict:
    _check_template(db, body.template_id)
    profile = ResumeProfile(
        target_role=career_session.target_role,
        contact=body.contact,
        github_username=career_session.github_username if body.include_github else None,
        education=body.education,
        work_experience=body.work_experience,
        skills=unique_ignoring_case([*career_session.skills, *body.extra_skills]),
    )
    return {"profile": profile.model_dump(), "template_id": body.template_id}


def _resume_render_input(db, career_session: CareerSession, body: ResumeRenderRun) -> dict:
    _check_template(db, body.template_id)
    return {"draft": body.draft.model_dump(), "template_id": body.template_id}


_INPUT_FOR_KIND = {
    RunKind.SKILLS_FROM_GITHUB: _github_input,
    RunKind.JOB_MARKET: _job_market_input,
    RunKind.LEARNING_PATH: _learning_path_input,
    RunKind.RESUME: _resume_input,
    RunKind.RESUME_RENDER: _resume_render_input,
}
_RUN_REQUEST = TypeAdapter(RunRequest)


def _json_body(raw: bytes):
    try:
        return _RUN_REQUEST.validate_json(raw)
    except ValidationError as error:  # reported like FastAPI's own body errors
        errors = error.errors(include_url=False)
        raise RequestValidationError([{**e, "loc": ("body", *e["loc"])} for e in errors]) from None


async def _upload_input(form) -> tuple[RunKind, dict]:
    """(kind, run input) from the multipart form of a run that reads a resume file."""
    kind, upload, request = form.get("kind"), form.get("file"), form.get("request") or ""
    if kind not in UPLOAD_KINDS:
        allowed = ", ".join(UPLOAD_KINDS)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"A form upload needs a kind of: {allowed}"
        )
    if not isinstance(upload, StarletteUploadFile):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Attach the resume as 'file'")
    if not isinstance(request, str) or len(request) > MAX_REQUEST_LENGTH:
        limit = f"'request' is limited to {MAX_REQUEST_LENGTH} characters"
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, limit)
    run_input = {"resume_text": await _read_upload(upload)}
    if kind == RunKind.RESUME_REVIEW:
        run_input["request"] = request
    return RunKind(kind), run_input


JSON, FORM = "application/json", "multipart/form-data"
# The bodies POST /runs reads itself, by media type: (schema name, type). main.py adds the
# schemas to the API description, since FastAPI cannot list a body it does not parse.
RUN_BODIES = {JSON: ("RunRequest", RunRequest), FORM: ("RunUpload", RunUpload)}
SCHEMA_REF = "#/components/schemas/{model}"
START_RUN_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            media: {"schema": {"$ref": SCHEMA_REF.format(model=name)}}
            for media, (name, _) in RUN_BODIES.items()
        },
    }
}


@router.post(
    "", status_code=status.HTTP_202_ACCEPTED, response_model=RunOut, openapi_extra=START_RUN_BODY
)
async def start_run(request: Request, career_session: CurrentSession, db: Db, runner: Runner):
    """Start a run of any kind: JSON picked by "kind", or a form for a resume file."""
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type == FORM:
        kind, run_input = await _upload_input(await request.form())
    elif media_type == JSON:
        body = _json_body(await request.body())
        kind, run_input = body.kind, _INPUT_FOR_KIND[body.kind](db, career_session, body)
    else:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Send JSON, or a multipart form to upload a resume",
        )
    return _start_run(db, runner, career_session, kind, run_input)


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
    if (
        last_event_id
        and len(last_event_id) <= MAX_EVENT_ID_DIGITS
        and last_event_id.isascii()
        and last_event_id.isdigit()
    ):
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


@router.get("/{run_id}/resume")
def download_resume(
    run_id: str,
    career_session: CurrentSession,
    db: Db,
    format: Literal["pdf", "tex"] = "pdf",  # noqa: A002 (the query parameter's name)
    download: bool = False,
):
    """The resume a run wrote: the PDF (shown in the page unless `download`) or its LaTeX."""
    resume_file = _resume_file(db, career_session, run_id)
    if format == "tex":
        headers = {"Content-Disposition": 'attachment; filename="resume.tex"'}
        return Response(resume_file.latex, media_type="application/x-tex", headers=headers)
    if resume_file.pdf is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No PDF was produced for this resume")
    disposition = "attachment" if download else "inline"
    headers = {"Content-Disposition": f'{disposition}; filename="resume.pdf"'}
    return Response(resume_file.pdf, media_type="application/pdf", headers=headers)
