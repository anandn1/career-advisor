"""Shared FastAPI dependencies."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db import career_sessions
from app.db.models import CareerSession
from app.db.session import session_scope
from app.services.runner import RunExecutor


def get_db() -> Iterator[Session]:
    with session_scope() as db:
        yield db


# scope="function": commit before the response is sent, so a 2xx means the data is saved.
Db = Annotated[Session, Depends(get_db, scope="function")]


def get_career_session(session_id: str, db: Db) -> CareerSession:
    career_session = career_sessions.get_session(db, session_id)
    if career_session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or expired")
    return career_session


CurrentSession = Annotated[CareerSession, Depends(get_career_session)]


def get_runner(request: Request) -> RunExecutor:
    return request.app.state.runner


Runner = Annotated[RunExecutor, Depends(get_runner)]
