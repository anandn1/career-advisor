"""FastAPI application: routes, CORS, background run workers and session expiry.

Run from backend/:  uv run uvicorn app.main:create_app --factory --reload
(A factory, so importing this module reads no settings and needs no credentials.)
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import TypeAdapter

from app.api.routers import interviews, runs, sessions, templates
from app.api.schemas import RunRequest, RunUpload
from app.core.config import get_settings
from app.db import career_sessions
from app.db.session import session_scope
from app.services.checkpoints import delete_interview_checkpoints
from app.services.run_handlers import HANDLERS
from app.services.runner import RunExecutor

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
SCHEMA_REF = "#/components/schemas/{model}"
CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60
RESTART_REASON = "The server restarted while this run was in progress. Please run it again."


def _delete_checkpoints(thread_ids: list[str]) -> None:
    delete_interview_checkpoints(thread_ids)


def expire_old_sessions() -> int:
    """Delete sessions unused for SESSION_TTL_DAYS, with their interview checkpoints."""
    ttl = timedelta(days=get_settings().session_ttl_days)
    with session_scope() as db:
        expired = career_sessions.delete_expired_sessions(db, ttl)
    try:
        _delete_checkpoints(expired.interview_thread_ids)
    except Exception:  # checkpoints are best effort; the session rows are already gone
        logger.exception("Could not delete interview checkpoints")
    if expired.session_ids:
        logger.info("Deleted %d expired sessions", len(expired.session_ids))
    return len(expired.session_ids)


async def _expire_periodically() -> None:
    while True:
        try:
            await asyncio.to_thread(expire_old_sessions)
        except Exception:  # keep the loop alive; retry next interval
            logger.exception("Session cleanup failed")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with session_scope() as db:
        stale = career_sessions.fail_unfinished_runs(db, RESTART_REASON)
    if stale:
        logger.info("Marked %d interrupted runs as failed", stale)
    app.state.runner = RunExecutor.with_threads(HANDLERS, get_settings().run_workers)
    cleanup = asyncio.create_task(_expire_periodically())
    try:
        yield
    finally:
        cleanup.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup
        app.state.runner.shutdown()


def _describe_run_bodies(app: FastAPI) -> None:
    """Add the run request bodies to the API description: POST /runs reads them itself,
    because it takes JSON or a multipart form, so FastAPI cannot list them."""
    build = app.openapi

    def openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = build()
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, body in {"RunRequest": RunRequest, "RunUpload": RunUpload}.items():
            body_schema = TypeAdapter(body).json_schema(ref_template=SCHEMA_REF)
            for model, model_schema in body_schema.pop("$defs", {}).items():
                components.setdefault(model, model_schema)
            components[name] = body_schema
        return schema

    app.openapi = openapi


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic AI Career Advisor API", version="2.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (
        sessions.router,
        runs.router,
        interviews.router,
        interviews.topics_router,
        templates.router,
    ):
        app.include_router(router, prefix=API_PREFIX)
    _describe_run_bodies(app)

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok"}

    return app
