"""Mock interviews: start, answer turn by turn (resumed from the checkpoint), read the report."""

import uuid

from fastapi import APIRouter, HTTPException, status

from app.agents import interview_coach
from app.api.deps import CurrentSession, Db
from app.api.schemas import AnswerRequest, InterviewCreate, InterviewOut, TurnOut
from app.db import career_sessions
from app.db.models import AgentRun, CareerSession, RunKind, RunStatus, utcnow

router = APIRouter(prefix="/sessions/{session_id}/interviews", tags=["interviews"])


def _default_topics(db, career_session: CareerSession) -> list[str]:
    analysis_run = career_sessions.latest_succeeded_run(db, career_session.id, RunKind.JOB_MARKET)
    analysis = (analysis_run.result or {}).get("analysis", {}) if analysis_run else {}
    return interview_coach.topics_for_skill_gaps(analysis.get("skill_gaps", []))


def _out(run: AgentRun, turn: interview_coach.InterviewTurn | None = None) -> InterviewOut:
    return InterviewOut(
        id=run.id,
        status=run.status,
        company_focus=run.input["company_focus"],
        topics=run.input["topics"],
        turn=TurnOut(**turn.__dict__) if turn else None,
        report=(run.result or {}).get("report", []),
    )


def _record_turn(run: AgentRun, turn: interview_coach.InterviewTurn) -> None:
    if turn.finished:
        report = interview_coach.evaluation_report(run.id)
        run.status, run.finished_at = RunStatus.SUCCEEDED, utcnow()
        run.result = {"summary": turn.summary, "report": report}


@router.post("", status_code=status.HTTP_201_CREATED, response_model=InterviewOut)
def start_interview(body: InterviewCreate, career_session: CurrentSession, db: Db):
    topics = body.topics or _default_topics(db, career_session)
    run_input = {"company_focus": body.company_focus, "topics": topics}
    run = career_sessions.create_run(
        db,
        career_session,
        RunKind.INTERVIEW,
        run_input,
        id=str(uuid.uuid4()),
        status=RunStatus.RUNNING,
        started_at=utcnow(),
    )
    turn = interview_coach.start_interview(run.id, topics, body.company_focus)
    _record_turn(run, turn)
    return _out(run, turn)


def _owned_interview(db, career_session: CareerSession, interview_id: str) -> AgentRun:
    run = career_sessions.get_run(db, interview_id)
    if run is None or run.session_id != career_session.id or run.kind != RunKind.INTERVIEW:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Interview not found")
    return run


@router.post("/{interview_id}/answers", response_model=InterviewOut)
def answer(interview_id: str, body: AnswerRequest, career_session: CurrentSession, db: Db):
    run = _owned_interview(db, career_session, interview_id)
    try:
        turn = interview_coach.submit_answer(run.id, body.answer)
    except interview_coach.InterviewStateError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
    _record_turn(run, turn)
    return _out(run, turn)


@router.get("/{interview_id}", response_model=InterviewOut)
def read_interview(interview_id: str, career_session: CurrentSession, db: Db):
    return _out(_owned_interview(db, career_session, interview_id))
