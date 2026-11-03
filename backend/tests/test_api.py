"""HTTP API tests on a temporary SQLite file, with agent runs executed inline (no threads)."""

import io
import json
from concurrent.futures import Executor, Future
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app import main
from app.agents import interview_coach
from app.core.config import BACKEND_DIR
from app.db import career_sessions
from app.db.models import (
    AgentRun,
    CareerSession,
    ResumeFile,
    ResumeTemplate,
    RunKind,
)
from app.db.models import RunStatus as Status
from app.db.session import session_scope
from app.services.runner import RunExecutor, UserFacingError

API = "/api/v1"


class InlineExecutor(Executor):
    """Runs submitted work immediately, so a run is finished when the POST returns."""

    def __init__(self):
        self.submitted = []

    def submit(self, fn, /, *args, **kwargs):
        self.submitted.append(args)
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


class ParkedExecutor(InlineExecutor):
    """Accepts work but never runs it, leaving runs queued."""

    def submit(self, fn, /, *args, **kwargs):
        self.submitted.append(args)
        return Future()


@pytest.fixture
def engine(sqlite_engine):
    return sqlite_engine


def ok_handler(result=None, events=()):
    def handle(context):
        for step in events:
            context.progress(step)
        return result if result is not None else {"input": context.input}

    return handle


@pytest.fixture
def handlers():
    """Fake agent handlers, replaceable per test: handlers[RunKind.X] = ..."""
    return {kind: ok_handler() for kind in RunKind if kind != RunKind.INTERVIEW}


@pytest.fixture
def executor():
    return InlineExecutor()


@pytest.fixture(autouse=True)
def no_background_cleanup(monkeypatch):
    """The real 6-hourly expiry loop would run against the test DB concurrently with tests."""

    async def idle():
        return None

    monkeypatch.setattr(main, "_expire_periodically", idle)


@pytest.fixture
def client(engine, handlers, executor, monkeypatch):
    monkeypatch.setattr(
        main.RunExecutor, "with_threads", classmethod(lambda cls, _h, _w: cls(handlers, executor))
    )
    with TestClient(main.create_app()) as test_client:
        yield test_client


@pytest.fixture
def session_id(client):
    response = client.post(
        f"{API}/sessions", json={"target_role": "Backend Dev", "location": "Pune"}
    )
    return response.json()["id"]


def run_url(session_id, suffix=""):
    return f"{API}/sessions/{session_id}/runs{suffix}"


def upload(name="cv.txt", content=b"Python and Docker", media="text/plain"):
    return {"file": (name, io.BytesIO(content), media)}


def stored_run(run_id):
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        db.expunge(run)
        return run


def add_run(session_id, kind, status=Status.SUCCEEDED, result=None, **fields):
    with session_scope() as db:
        career_session = db.get(CareerSession, session_id)
        run = career_sessions.create_run(
            db, career_session, kind, {}, status=status, result=result, **fields
        )
        if status.is_finished:
            run.finished_at = fields.get("finished_at") or main.career_sessions.utcnow()
        return run.id


# --- Health and sessions ---


def test_health_is_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_created_session_has_an_unguessable_id(client):
    body = client.post(f"{API}/sessions", json={"target_role": "Dev", "location": "Pune"}).json()
    assert len(body["id"]) == 36 and body["skills"] == []


def test_session_fields_are_trimmed(client):
    body = client.post(
        f"{API}/sessions", json={"target_role": "  Dev ", "location": " Pune "}
    ).json()
    assert (body["target_role"], body["location"]) == ("Dev", "Pune")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"target_role": "", "location": "Pune"},
        {"target_role": "Dev"},
        {"target_role": "x" * 201, "location": "Pune"},
    ],
)
def test_invalid_session_payloads_are_rejected(client, payload):
    assert client.post(f"{API}/sessions", json=payload).status_code == 422


def test_session_can_be_read_back(client, session_id):
    assert client.get(f"{API}/sessions/{session_id}").json()["target_role"] == "Backend Dev"


def test_unknown_session_is_not_found(client):
    assert client.get(f"{API}/sessions/not-a-session").status_code == 404


def test_reading_a_session_refreshes_its_last_seen_time(client, session_id):
    with session_scope() as db:
        db.get(CareerSession, session_id).last_seen_at = main.career_sessions.utcnow() - timedelta(
            days=9
        )
    client.get(f"{API}/sessions/{session_id}")
    with session_scope() as db:
        age = main.career_sessions.utcnow() - db.get(
            CareerSession, session_id
        ).last_seen_at.replace(tzinfo=main.career_sessions.utcnow().tzinfo)
    assert age < timedelta(minutes=1)


def test_session_skills_can_be_edited_and_are_deduplicated(client, session_id):
    body = client.patch(
        f"{API}/sessions/{session_id}", json={"skills": ["Go", " Go ", "", "SQL"]}
    ).json()
    assert body["skills"] == ["Go", "SQL"]


def test_patch_leaves_unspecified_fields_alone(client, session_id):
    body = client.patch(f"{API}/sessions/{session_id}", json={"location": "Remote"}).json()
    assert (body["target_role"], body["location"]) == ("Backend Dev", "Remote")


def test_deleting_a_session_removes_it_and_its_runs(client, session_id):
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    assert client.delete(f"{API}/sessions/{session_id}").status_code == 204
    assert client.get(f"{API}/sessions/{session_id}").status_code == 404
    with session_scope() as db:
        assert db.get(AgentRun, run_id) is None


# --- Starting runs ---


def test_job_market_run_is_accepted_and_executed(client, session_id):
    response = client.post(run_url(session_id, "/job-market"))
    assert (
        response.status_code == 202 and stored_run(response.json()["id"]).status == Status.SUCCEEDED
    )


def test_run_result_is_returned_by_the_run_endpoint(client, session_id, handlers):
    handlers[RunKind.JOB_MARKET] = ok_handler({"analysis": {"skill_gaps": ["docker"]}})
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    assert client.get(run_url(session_id, f"/{run_id}")).json()["result"] == {
        "analysis": {"skill_gaps": ["docker"]}
    }


def test_a_second_unfinished_run_of_the_same_kind_conflicts(engine, handlers, monkeypatch):
    parked = ParkedExecutor()
    monkeypatch.setattr(
        main.RunExecutor, "with_threads", classmethod(lambda cls, _h, _w: cls(handlers, parked))
    )
    with TestClient(main.create_app()) as client:
        sid = client.post(
            f"{API}/sessions", json={"target_role": "Dev", "location": "Pune"}
        ).json()["id"]
        assert client.post(run_url(sid, "/job-market")).status_code == 202
        assert client.post(run_url(sid, "/job-market")).status_code == 409


def test_finished_runs_do_not_block_new_ones(client, session_id):
    client.post(run_url(session_id, "/job-market"))
    assert client.post(run_url(session_id, "/job-market")).status_code == 202


def test_different_kinds_can_run_together(engine, handlers, monkeypatch):
    parked = ParkedExecutor()
    monkeypatch.setattr(
        main.RunExecutor, "with_threads", classmethod(lambda cls, _h, _w: cls(handlers, parked))
    )
    with TestClient(main.create_app()) as client:
        sid = client.post(
            f"{API}/sessions", json={"target_role": "Dev", "location": "Pune"}
        ).json()["id"]
        client.post(run_url(sid, "/job-market"))
        assert (
            client.post(
                run_url(sid, "/skills/github"), json={"github_username": "octocat"}
            ).status_code
            == 202
        )


def test_runs_cannot_be_started_for_an_unknown_session(client):
    assert client.post(run_url("missing", "/job-market")).status_code == 404


def test_github_skills_run_stores_the_parsed_username(client, session_id):
    run_id = client.post(
        run_url(session_id, "/skills/github"),
        json={"github_username": "https://github.com/octocat/"},
    ).json()["id"]
    assert stored_run(run_id).input == {"github_username": "octocat"}


@pytest.mark.parametrize("username", ["../etc", "https://gitlab.com/octocat", "a b"])
def test_invalid_github_usernames_are_rejected(client, session_id, username):
    response = client.post(
        run_url(session_id, "/skills/github"), json={"github_username": username}
    )
    assert response.status_code == 422


def test_learning_path_needs_a_job_market_analysis_first(client, session_id):
    assert client.post(run_url(session_id, "/learning-path")).status_code == 409


def test_learning_path_needs_skill_gaps(client, session_id):
    add_run(session_id, RunKind.JOB_MARKET, result={"analysis": {"skill_gaps": []}})
    assert client.post(run_url(session_id, "/learning-path")).status_code == 409


def test_learning_path_uses_the_latest_analysis_gaps(client, session_id):
    add_run(
        session_id,
        RunKind.JOB_MARKET,
        result={"analysis": {"skill_gaps": ["docker", "k8s"]}},
    )
    run_id = client.post(run_url(session_id, "/learning-path")).json()["id"]
    assert stored_run(run_id).input["skill_gaps"] == ["docker", "k8s"]


def test_failed_analyses_are_not_used_for_the_learning_path(client, session_id):
    add_run(
        session_id,
        RunKind.JOB_MARKET,
        status=Status.FAILED,
        result={"analysis": {"skill_gaps": ["docker"]}},
    )
    assert client.post(run_url(session_id, "/learning-path")).status_code == 409


# --- Uploads ---


def test_resume_upload_text_is_passed_to_the_run(client, session_id):
    run_id = client.post(run_url(session_id, "/skills/resume"), files=upload()).json()["id"]
    assert stored_run(run_id).input == {"resume_text": "Python and Docker"}


@pytest.mark.parametrize("name", ["cv.doc", "cv.exe", "cv"])
def test_unsupported_upload_types_are_rejected(client, session_id, name):
    assert client.post(run_url(session_id, "/skills/resume"), files=upload(name)).status_code == 415


def test_uploads_over_the_size_limit_are_rejected(client, session_id, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10")
    main.get_settings.cache_clear()
    response = client.post(run_url(session_id, "/skills/resume"), files=upload(content=b"x" * 11))
    assert response.status_code == 413


def test_upload_exactly_at_the_size_limit_is_accepted(client, session_id, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10")
    main.get_settings.cache_clear()
    response = client.post(run_url(session_id, "/skills/resume"), files=upload(content=b"x" * 10))
    assert response.status_code == 202


def test_empty_upload_text_is_rejected(client, session_id):
    response = client.post(run_url(session_id, "/skills/resume"), files=upload(content=b"   \n"))
    assert response.status_code == 422


def test_corrupt_upload_is_rejected(client, session_id):
    files = upload("cv.pdf", b"not a pdf", "application/pdf")
    assert client.post(run_url(session_id, "/skills/resume"), files=files).status_code == 422


def test_uploaded_files_are_not_kept(client, session_id, monkeypatch):
    from pathlib import Path

    written = []
    real_write = Path.write_bytes

    def recording_write(path, data):
        written.append(path)
        return real_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", recording_write)
    client.post(run_url(session_id, "/skills/resume"), files=upload())
    assert written and not any(path.exists() for path in written)


def test_resume_review_passes_the_request_text(client, session_id):
    response = client.post(
        run_url(session_id, "/resume-review"), files=upload(), data={"request": "Focus on impact"}
    )
    assert stored_run(response.json()["id"]).input == {
        "resume_text": "Python and Docker",
        "request": "Focus on impact",
    }


def test_overlong_review_requests_are_rejected(client, session_id):
    response = client.post(
        run_url(session_id, "/resume-review"), files=upload(), data={"request": "x" * 1001}
    )
    assert response.status_code == 422


# --- Resume runs and files ---


def test_resume_profile_combines_session_skills_and_extras(client, session_id):
    client.patch(f"{API}/sessions/{session_id}", json={"skills": ["Go", "SQL"]})
    run_id = client.post(
        run_url(session_id, "/resume"), json={"extra_skills": ["SQL", "Rust"]}
    ).json()["id"]
    profile = stored_run(run_id).input["profile"]
    assert (profile["target_role"], profile["skills"]) == (
        "Backend Dev",
        ["Go", "SQL", "Rust"],
    )


def test_resume_uses_the_session_github_user_unless_excluded(client, session_id):
    with session_scope() as db:
        db.get(CareerSession, session_id).github_username = "octocat"
    with_github = client.post(run_url(session_id, "/resume"), json={}).json()["id"]
    without = client.post(run_url(session_id, "/resume"), json={"include_github": False}).json()[
        "id"
    ]
    assert stored_run(with_github).input["profile"]["github_username"] == "octocat"
    assert stored_run(without).input["profile"]["github_username"] is None


def test_unknown_template_is_rejected(client, session_id):
    response = client.post(run_url(session_id, "/resume"), json={"template_id": "missing"})
    assert response.status_code == 422


def test_known_template_is_accepted(client, session_id):
    with session_scope() as db:
        template = ResumeTemplate(name="t1", latex_src="\\documentclass{article}")
        db.add(template)
        db.flush()
        template_id = template.id
    response = client.post(run_url(session_id, "/resume"), json={"template_id": template_id})
    assert response.status_code == 202


def _resume_with_files(session_id, latex="\\documentclass{a}", pdf=b"%PDF-1"):
    run_id = add_run(session_id, RunKind.RESUME, result={"has_pdf": pdf is not None})
    with session_scope() as db:
        db.add(ResumeFile(run_id=run_id, latex=latex, pdf=pdf))
    return run_id


def test_resume_latex_can_be_downloaded(client, session_id):
    run_id = _resume_with_files(session_id)
    response = client.get(run_url(session_id, f"/{run_id}/resume.tex"))
    assert (
        response.text == "\\documentclass{a}"
        and "attachment" in response.headers["content-disposition"]
    )


def test_resume_pdf_is_shown_inline_by_default(client, session_id):
    run_id = _resume_with_files(session_id)
    response = client.get(run_url(session_id, f"/{run_id}/resume.pdf"))
    assert response.content == b"%PDF-1" and response.headers["content-disposition"].startswith(
        "inline"
    )


def test_resume_pdf_can_be_downloaded(client, session_id):
    run_id = _resume_with_files(session_id)
    response = client.get(run_url(session_id, f"/{run_id}/resume.pdf"), params={"download": True})
    assert response.headers["content-disposition"].startswith("attachment")


def test_missing_pdf_is_not_found(client, session_id):
    run_id = _resume_with_files(session_id, pdf=None)
    assert client.get(run_url(session_id, f"/{run_id}/resume.pdf")).status_code == 404


def test_non_resume_runs_have_no_files(client, session_id):
    run_id = add_run(session_id, RunKind.JOB_MARKET)
    assert client.get(run_url(session_id, f"/{run_id}/resume.tex")).status_code == 404


def test_another_sessions_run_is_not_visible(client, session_id):
    other = client.post(f"{API}/sessions", json={"target_role": "Dev", "location": "X"}).json()[
        "id"
    ]
    run_id = _resume_with_files(other)
    assert client.get(run_url(session_id, f"/{run_id}")).status_code == 404
    assert client.get(run_url(session_id, f"/{run_id}/resume.tex")).status_code == 404


def test_rerender_accepts_an_edited_draft(client, session_id):
    draft = {"contact": {"name": "Ada"}, "summary": "Edited summary"}
    run_id = client.post(run_url(session_id, "/resume-render"), json={"draft": draft}).json()["id"]
    assert stored_run(run_id).input["draft"]["summary"] == "Edited summary"


def test_rerender_rejects_a_malformed_draft(client, session_id):
    response = client.post(run_url(session_id, "/resume-render"), json={"draft": {"summary": 3}})
    assert response.status_code == 422


# --- Listing and streaming ---


def test_runs_can_be_listed_and_filtered_by_kind(client, session_id):
    client.post(run_url(session_id, "/job-market"))
    client.post(run_url(session_id, "/skills/github"), json={"github_username": "octocat"})
    kinds = [
        r["kind"] for r in client.get(run_url(session_id), params={"kind": "job_market"}).json()
    ]
    assert kinds == ["job_market"] and len(client.get(run_url(session_id)).json()) == 2


def parse_sse(text):
    events, current = [], {}
    for line in text.splitlines():
        if not line:
            if current:
                events.append(current)
            current = {}
        elif ":" in line and not line.startswith(":"):
            key, value = line.split(":", 1)
            current[key] = value.strip()
    if current:
        events.append(current)
    return events


def test_event_stream_replays_progress_then_done(client, session_id, handlers):
    handlers[RunKind.JOB_MARKET] = ok_handler({"ok": True}, events=("search", "scrape", "analyze"))
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    events = parse_sse(client.get(run_url(session_id, f"/{run_id}/events")).text)
    assert [e["event"] for e in events] == ["progress", "progress", "progress", "done"]
    assert [json.loads(e["data"])["step"] for e in events[:3]] == [
        "search",
        "scrape",
        "analyze",
    ]


def test_event_stream_ids_allow_resuming(client, session_id, handlers):
    handlers[RunKind.JOB_MARKET] = ok_handler({}, events=("a", "b", "c"))
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    response = client.get(run_url(session_id, f"/{run_id}/events"), headers={"Last-Event-ID": "0"})
    steps = [
        json.loads(e["data"])["step"] for e in parse_sse(response.text) if e["event"] == "progress"
    ]
    assert steps == ["b", "c"]


def test_done_event_carries_the_final_run(client, session_id, handlers):
    handlers[RunKind.JOB_MARKET] = ok_handler({"answer": 42})
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    done = parse_sse(client.get(run_url(session_id, f"/{run_id}/events")).text)[-1]
    assert json.loads(done["data"])["result"] == {"answer": 42}


def test_event_stream_for_an_unknown_run_is_not_found(client, session_id):
    assert client.get(run_url(session_id, "/missing/events")).status_code == 404


def test_event_stream_for_another_sessions_run_is_not_found(client, session_id):
    other = client.post(f"{API}/sessions", json={"target_role": "Dev", "location": "X"}).json()[
        "id"
    ]
    run_id = add_run(other, RunKind.JOB_MARKET)
    assert client.get(run_url(session_id, f"/{run_id}/events")).status_code == 404


# --- Run engine ---


def test_failed_handler_marks_the_run_failed_without_leaking_details(client, session_id, handlers):
    def boom(context):
        raise RuntimeError("secret internal path /etc/x")

    handlers[RunKind.JOB_MARKET] = boom
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    run = stored_run(run_id)
    assert run.status == Status.FAILED and "secret" not in run.error


def test_user_facing_errors_are_shown_as_is(client, session_id, handlers):
    def no_path(context):
        raise UserFacingError("No learning path could be built.")

    handlers[RunKind.JOB_MARKET] = no_path
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    assert stored_run(run_id).error == "No learning path could be built."


def test_runs_record_start_and_finish_times(client, session_id):
    run = stored_run(client.post(run_url(session_id, "/job-market")).json()["id"])
    assert run.started_at is not None and run.finished_at >= run.started_at


def test_handlers_can_update_the_session(client, session_id, handlers):
    def extract(context):
        context.update_session(skills=["Go"])
        return {}

    handlers[RunKind.JOB_MARKET] = extract
    client.post(run_url(session_id, "/job-market"))
    assert client.get(f"{API}/sessions/{session_id}").json()["skills"] == ["Go"]


def test_handlers_see_a_snapshot_of_the_session(client, session_id, handlers):
    seen = {}
    handlers[RunKind.JOB_MARKET] = lambda context: (
        seen.update(role=context.session.target_role) or {}
    )
    client.post(run_url(session_id, "/job-market"))
    assert seen == {"role": "Backend Dev"}


def test_executing_an_unknown_run_does_not_raise(engine):
    RunExecutor({}, InlineExecutor()).execute("no-such-run")


# --- Interviews ---


class InterviewLLM:
    def invoke(self, prompt):
        text = "GRADE: good\nSKILL: x\nFEEDBACK: Fine." if "Evaluate" in prompt else "Summary."
        return type("Reply", (), {"content": text})()


@pytest.fixture
def coach(monkeypatch):
    graph = interview_coach.build_interview_graph().compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(interview_coach, "get_interview_graph", lambda: graph)
    monkeypatch.setattr(interview_coach, "get_llm", lambda: InterviewLLM())
    monkeypatch.setattr(interview_coach.random, "choice", lambda items: items[0])
    batches = iter(
        [
            [
                Document(
                    page_content=f"Q{i}?",
                    metadata={
                        "question_id": f"q{i}",
                        "topic": "system design",
                        "difficulty": "easy",
                    },
                )
            ]
            for i in range(3)
        ]
    )
    monkeypatch.setattr(interview_coach, "_retrieve_questions", lambda s, d: next(batches, []))


def interview_url(session_id, suffix=""):
    return f"{API}/sessions/{session_id}/interviews{suffix}"


def test_starting_an_interview_returns_the_first_question(client, session_id, coach):
    response = client.post(interview_url(session_id), json={"company_focus": "google"})
    body = response.json()
    assert (
        response.status_code == 201
        and body["turn"]["question"] == "Q0?"
        and body["status"] == "running"
    )


def test_interview_topics_default_to_the_latest_skill_gaps(client, session_id, coach):
    add_run(session_id, RunKind.JOB_MARKET, result={"analysis": {"skill_gaps": ["docker"]}})
    body = client.post(interview_url(session_id), json={}).json()
    assert body["topics"] == ["Cloud & DevOps", "System Design"]


def test_interview_topics_without_analysis_use_the_defaults(client, session_id, coach):
    assert (
        client.post(interview_url(session_id), json={}).json()["topics"]
        == interview_coach.DEFAULT_TOPICS
    )


@pytest.mark.parametrize("payload", [{"company_focus": "initech"}, {"topics": []}])
def test_invalid_interview_requests_are_rejected(client, session_id, coach, payload):
    assert client.post(interview_url(session_id), json=payload).status_code == 422


def test_answers_advance_the_interview(client, session_id, coach):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]
    body = client.post(
        interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "LRU"}
    ).json()
    assert body["turn"]["question"] == "Q1?" and body["turn"]["feedback"].startswith("Grade: GOOD")


def test_finishing_stores_the_summary_and_report(client, session_id, coach):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]
    client.post(interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "LRU"})
    client.post(interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "stop"})
    body = client.get(interview_url(session_id, f"/{interview_id}")).json()
    assert body["status"] == "succeeded" and [e["question_id"] for e in body["report"]] == ["q0"]


def test_answering_a_finished_interview_conflicts(client, session_id, coach):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]
    client.post(interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "stop"})
    response = client.post(
        interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "more"}
    )
    assert response.status_code == 409


def test_overlong_answers_are_rejected(client, session_id, coach):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]
    response = client.post(
        interview_url(session_id, f"/{interview_id}/answers"),
        json={"answer": "x" * 5001},
    )
    assert response.status_code == 422


def test_interviews_of_other_sessions_are_not_found(client, session_id, coach):
    other = client.post(f"{API}/sessions", json={"target_role": "Dev", "location": "X"}).json()[
        "id"
    ]
    interview_id = client.post(interview_url(other), json={}).json()["id"]
    assert client.get(interview_url(session_id, f"/{interview_id}")).status_code == 404


def test_non_interview_runs_are_not_interviews(client, session_id):
    run_id = add_run(session_id, RunKind.JOB_MARKET)
    assert client.get(interview_url(session_id, f"/{run_id}")).status_code == 404


# --- Report and templates ---


def test_report_collects_the_latest_results(client, session_id):
    add_run(
        session_id,
        RunKind.JOB_MARKET,
        result={"analysis": {"skill_gaps": ["old"]}},
        finished_at=main.career_sessions.utcnow() - timedelta(hours=1),
    )
    add_run(session_id, RunKind.JOB_MARKET, result={"analysis": {"skill_gaps": ["new"]}})
    add_run(session_id, RunKind.JOB_MARKET, status=Status.FAILED, result={"analysis": {}})
    report = client.get(f"{API}/sessions/{session_id}/report").json()
    assert (
        report["job_market"]["analysis"]["skill_gaps"] == ["new"]
        and report["learning_path"] is None
    )


def test_report_prefers_the_newest_resume_of_either_kind(client, session_id):
    add_run(
        session_id,
        RunKind.RESUME,
        result={"draft": "first"},
        finished_at=main.career_sessions.utcnow() - timedelta(hours=1),
    )
    newest = add_run(session_id, RunKind.RESUME_RENDER, result={"draft": "edited"})
    assert client.get(f"{API}/sessions/{session_id}/report").json()["resume"]["run_id"] == newest


def test_templates_are_listed_with_sorted_tags(client):
    with session_scope() as db:
        from app.db.templates import get_or_create_tag

        db.add(
            ResumeTemplate(
                name="b",
                latex_src="x",
                tags=[get_or_create_tag(db, "Tech"), get_or_create_tag(db, "Startup")],
            )
        )
        db.add(ResumeTemplate(name="a", latex_src="y"))
    body = client.get(f"{API}/templates").json()
    assert [(t["name"], t["tags"]) for t in body] == [
        ("a", []),
        ("b", ["Startup", "Tech"]),
    ]


def test_cors_allows_the_configured_frontend(client):
    response = client.options(
        f"{API}/sessions",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_other_origins(client):
    response = client.options(
        f"{API}/sessions",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


# --- Startup and expiry ---


def test_startup_marks_interrupted_runs_failed(engine, handlers, monkeypatch):
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        stuck = career_sessions.create_run(
            db, career_session, RunKind.JOB_MARKET, {}, status=Status.RUNNING
        )
        live = career_sessions.create_run(
            db, career_session, RunKind.INTERVIEW, {}, status=Status.RUNNING
        )
        stuck_id, live_id = stuck.id, live.id
    monkeypatch.setattr(
        main.RunExecutor,
        "with_threads",
        classmethod(lambda cls, _h, _w: cls(handlers, InlineExecutor())),
    )
    with TestClient(main.create_app()):
        pass
    assert (
        stored_run(stuck_id).status == Status.FAILED
        and stored_run(live_id).status == Status.RUNNING
    )


def _session_seen(days_ago, with_interview=False):
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        career_session.last_seen_at = main.career_sessions.utcnow() - timedelta(days=days_ago)
        if with_interview:
            career_sessions.create_run(db, career_session, RunKind.INTERVIEW, {}, id="thread-1")
        return career_session.id


def test_expiry_deletes_only_old_sessions(engine, monkeypatch):
    monkeypatch.setattr(main, "_delete_checkpoints", lambda ids: None)
    old, fresh = _session_seen(31), _session_seen(29)
    main.expire_old_sessions()
    with session_scope() as db:
        remaining = set(db.scalars(select(CareerSession.id)))
    assert remaining == {fresh} and old not in remaining


def test_expiry_deletes_interview_checkpoints(engine, monkeypatch):
    deleted = []
    monkeypatch.setattr(main, "_delete_checkpoints", deleted.extend)
    _session_seen(40, with_interview=True)
    main.expire_old_sessions()
    assert deleted == ["thread-1"]


def test_checkpoint_cleanup_failure_still_expires_sessions(engine, monkeypatch):
    def broken(ids):
        raise ConnectionError("postgres down")

    monkeypatch.setattr(main, "_delete_checkpoints", broken)
    _session_seen(40, with_interview=True)
    main.expire_old_sessions()
    with session_scope() as db:
        assert db.scalars(select(CareerSession)).first() is None


def test_session_ttl_is_configurable(engine, monkeypatch):
    monkeypatch.setattr(main, "_delete_checkpoints", lambda ids: None)
    monkeypatch.setenv("SESSION_TTL_DAYS", "7")
    main.get_settings.cache_clear()
    _session_seen(8)
    main.expire_old_sessions()
    with session_scope() as db:
        assert db.scalars(select(CareerSession)).first() is None


# --- Round-1 audit regressions ---


@pytest.mark.parametrize("field", ["target_role", "location"])
def test_blank_fields_are_rejected_on_create(client, field):
    payload = {"target_role": "Dev", "location": "Pune", field: "   "}
    assert client.post(f"{API}/sessions", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["target_role", "location"])
def test_blank_fields_are_rejected_on_update(client, session_id, field):
    assert client.patch(f"{API}/sessions/{session_id}", json={field: "  "}).status_code == 422


def test_role_of_exactly_200_characters_is_accepted(client):
    payload = {"target_role": "x" * 200, "location": "Pune"}
    assert client.post(f"{API}/sessions", json=payload).status_code == 201


def test_patch_trims_strings(client, session_id):
    body = client.patch(f"{API}/sessions/{session_id}", json={"location": "  Remote  "}).json()
    assert body["location"] == "Remote"


def test_unserialisable_result_fails_the_run_without_blocking_its_kind(
    client, session_id, handlers
):
    handlers[RunKind.JOB_MARKET] = lambda context: {"x": object()}
    first = client.post(run_url(session_id, "/job-market")).json()["id"]
    handlers[RunKind.JOB_MARKET] = lambda context: {}
    assert stored_run(first).status == Status.FAILED
    assert client.post(run_url(session_id, "/job-market")).status_code == 202


def test_session_deleted_mid_run_stops_the_handler_quietly(engine):
    from app.services.runner import RunGoneError

    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        run_id = career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {}).id
        session_id = career_session.id
    outcome = {}

    def delete_then_report(context):
        with session_scope() as db:
            db.delete(db.get(CareerSession, session_id))
        try:
            context.progress("too late")
        except RunGoneError:
            outcome["stopped"] = True
            raise
        outcome["continued"] = True
        return {}

    RunExecutor({RunKind.JOB_MARKET: delete_then_report}, InlineExecutor()).execute(run_id)
    assert outcome == {"stopped": True} and stored_run_or_none(run_id) is None


def stored_run_or_none(run_id):
    with session_scope() as db:
        return db.get(AgentRun, run_id)


def test_failure_to_save_the_outcome_still_marks_the_run_failed(engine, monkeypatch):
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        run_id = career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {}).id
    executor = RunExecutor({RunKind.JOB_MARKET: lambda context: {"ok": True}}, InlineExecutor())
    real_finish = executor._finish
    calls = []

    def flaky_finish(run_id, status, **fields):
        calls.append(status)
        if len(calls) == 1:
            raise RuntimeError("database hiccup")
        real_finish(run_id, status, **fields)

    monkeypatch.setattr(executor, "_finish", flaky_finish)
    executor.execute(run_id)
    assert stored_run(run_id).status == Status.FAILED and calls == [Status.SUCCEEDED, Status.FAILED]


def test_long_user_facing_errors_are_truncated(client, session_id, handlers):
    def wordy(context):
        raise UserFacingError("x" * 600)

    handlers[RunKind.JOB_MARKET] = wordy
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    assert len(stored_run(run_id).error) == 500


def test_run_is_marked_running_while_its_handler_works(client, session_id, handlers):
    seen = {}

    def check(context):
        seen["status"] = stored_run(context.run_id).status
        return {}

    handlers[RunKind.JOB_MARKET] = check
    client.post(run_url(session_id, "/job-market"))
    assert seen == {"status": Status.RUNNING}


def test_concurrent_duplicate_is_blocked_by_the_database(engine):
    from sqlalchemy.exc import IntegrityError

    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {})
        session_id = career_session.id
    with pytest.raises(IntegrityError), session_scope() as db:
        career_sessions.create_run(db, db.get(CareerSession, session_id), RunKind.JOB_MARKET, {})


def test_several_interviews_may_run_at_once(engine):
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        for _ in range(2):
            career_sessions.create_run(
                db, career_session, RunKind.INTERVIEW, {}, status=Status.RUNNING
            )
        session_id = career_session.id
    with session_scope() as db:
        assert len(career_sessions.list_runs(db, session_id, RunKind.INTERVIEW)) == 2


def test_runs_are_listed_newest_first(client, session_id):
    older = add_run(
        session_id,
        RunKind.JOB_MARKET,
        created_at=main.career_sessions.utcnow() - timedelta(hours=1),
    )
    newer = add_run(session_id, RunKind.RESUME)
    assert [r["id"] for r in client.get(run_url(session_id)).json()] == [newer, older]


def test_learning_path_uses_the_newest_analysis(client, session_id):
    add_run(
        session_id,
        RunKind.JOB_MARKET,
        result={"analysis": {"skill_gaps": ["old"]}},
        finished_at=main.career_sessions.utcnow() - timedelta(hours=1),
    )
    newest = add_run(session_id, RunKind.JOB_MARKET, result={"analysis": {"skill_gaps": ["new"]}})
    run_id = client.post(run_url(session_id, "/learning-path")).json()["id"]
    assert stored_run(run_id).input == {"skill_gaps": ["new"], "job_market_run_id": newest}


def test_resume_profile_keeps_every_extra_skill(client, session_id):
    client.patch(f"{API}/sessions/{session_id}", json={"skills": ["Go"]})
    run_id = client.post(
        run_url(session_id, "/resume"), json={"extra_skills": ["Rust", "SQL"]}
    ).json()["id"]
    assert stored_run(run_id).input["profile"]["skills"] == ["Go", "Rust", "SQL"]


def test_files_attached_to_non_resume_runs_are_not_served(client, session_id):
    run_id = add_run(session_id, RunKind.JOB_MARKET)
    with session_scope() as db:
        db.add(ResumeFile(run_id=run_id, latex="x", pdf=b"y"))
    assert client.get(run_url(session_id, f"/{run_id}/resume.tex")).status_code == 404


def test_uppercase_upload_suffix_is_accepted(client, session_id):
    assert (
        client.post(run_url(session_id, "/skills/resume"), files=upload("CV.TXT")).status_code
        == 202
    )


def test_empty_and_corrupt_uploads_give_different_reasons(client, session_id):
    empty = client.post(run_url(session_id, "/skills/resume"), files=upload(content=b" ")).json()
    corrupt = client.post(
        run_url(session_id, "/skills/resume"), files=upload("cv.pdf", b"junk", "application/pdf")
    ).json()
    assert (
        empty["detail"] == "No text found in the file" and "Could not read PDF" in corrupt["detail"]
    )


def test_deleting_a_session_also_deletes_its_resume_files(client, session_id):
    run_id = _resume_with_files(session_id)
    client.delete(f"{API}/sessions/{session_id}")
    with session_scope() as db:
        assert db.get(ResumeFile, run_id) is None


def test_non_numeric_last_event_id_replays_everything(client, session_id, handlers):
    handlers[RunKind.JOB_MARKET] = ok_handler({}, events=("a", "b"))
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    response = client.get(
        run_url(session_id, f"/{run_id}/events"), headers={"Last-Event-ID": "abc"}
    )
    assert len([e for e in parse_sse(response.text) if e["event"] == "progress"]) == 2


@pytest.fixture
def parked_client(engine, handlers, monkeypatch):
    parked = ParkedExecutor()
    monkeypatch.setattr(
        main.RunExecutor, "with_threads", classmethod(lambda cls, _h, _w: cls(handlers, parked))
    )
    monkeypatch.setattr(main_runs, "EVENT_POLL_SECONDS", 0.01)
    with TestClient(main.create_app()) as test_client:
        yield test_client


from app.api.routers import runs as main_runs  # noqa: E402


def test_interview_summary_is_stored_on_the_run(client, session_id, coach):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]
    client.post(interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "stop"})
    assert (
        stored_run(interview_id).result["summary"]
        == "Interview complete!\n\nNo questions were answered in this session."
    )


def test_interview_start_failure_is_a_503_and_recorded(client, session_id, monkeypatch):
    def broken(*args, **kwargs):
        raise ConnectionError("vector store down")

    monkeypatch.setattr(interview_coach, "start_interview", broken)
    response = client.post(interview_url(session_id), json={})
    runs = client.get(run_url(session_id), params={"kind": "interview"}).json()
    assert response.status_code == 503 and [r["status"] for r in runs] == ["failed"]


def test_interview_answer_failure_is_a_retryable_503(client, session_id, coach, monkeypatch):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]

    def broken(*args, **kwargs):
        raise TimeoutError("LLM timed out")

    monkeypatch.setattr(interview_coach, "submit_answer", broken)
    response = client.post(
        interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "x"}
    )
    assert response.status_code == 503 and stored_run(interview_id).status == Status.RUNNING


def test_report_lists_interviews_and_reviews(client, session_id):
    interview_id = add_run(session_id, RunKind.INTERVIEW, result={"summary": "S"})
    add_run(session_id, RunKind.RESUME_REVIEW, result={"review": {"summary": "R"}})
    report = client.get(f"{API}/sessions/{session_id}/report").json()
    assert report["interviews"] == [{"id": interview_id, "status": "succeeded", "summary": "S"}]
    assert report["resume_review"]["review"] == {"summary": "R"}
    assert report["session"]["id"] == session_id


def test_report_prefers_a_newer_generated_resume_over_an_older_render(client, session_id):
    add_run(
        session_id,
        RunKind.RESUME_RENDER,
        result={},
        finished_at=main.career_sessions.utcnow() - timedelta(hours=1),
    )
    newest = add_run(session_id, RunKind.RESUME, result={})
    assert client.get(f"{API}/sessions/{session_id}/report").json()["resume"]["run_id"] == newest


def test_startup_fails_queued_runs_too(engine, handlers, monkeypatch):
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        queued_id = career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {}).id
    monkeypatch.setattr(
        main.RunExecutor,
        "with_threads",
        classmethod(lambda cls, _h, _w: cls(handlers, InlineExecutor())),
    )
    with TestClient(main.create_app()):
        pass
    assert stored_run(queued_id).status == Status.FAILED


def test_expiry_keeps_a_session_exactly_at_the_cutoff(engine):
    now = main.career_sessions.utcnow()
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        career_session.last_seen_at = now - timedelta(days=30)
    with session_scope() as db:
        expired = career_sessions.delete_expired_sessions(db, timedelta(days=30), now=now)
    assert expired.session_ids == []


def test_expiry_returns_only_interview_threads_for_checkpoint_cleanup(engine):
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        career_session.last_seen_at = main.career_sessions.utcnow() - timedelta(days=40)
        career_sessions.create_run(db, career_session, RunKind.INTERVIEW, {}, id="thread-9")
        career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {})
    with session_scope() as db:
        expired = career_sessions.delete_expired_sessions(db, timedelta(days=30))
    assert expired.interview_thread_ids == ["thread-9"] and len(expired.session_ids) == 1


def test_expiry_reports_the_number_of_sessions_deleted(engine, monkeypatch):
    monkeypatch.setattr(main, "_delete_checkpoints", lambda ids: None)
    _session_seen(40), _session_seen(50), _session_seen(1)
    assert main.expire_old_sessions() == 2


def test_real_thread_pool_runs_and_shuts_down(engine):
    import time

    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        run_id = career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {}).id
    executor = RunExecutor.with_threads({RunKind.JOB_MARKET: lambda context: {"ok": 1}}, workers=1)
    executor.submit(run_id)
    for _ in range(100):
        if stored_run(run_id).status == Status.SUCCEEDED:
            break
        time.sleep(0.02)
    executor.shutdown()
    assert stored_run(run_id).result == {"ok": 1}


# --- Round-2 audit regressions ---


def drive_stream(parked_client, sid, run_id, on_poll):
    """Stream a run's events; on_poll(n) runs before the n-th database poll (n starts at 1)."""
    real_snapshot = main_runs._run_snapshot
    polls = []

    def snapshot(session_id, run_id_):
        polls.append(1)
        on_poll(len(polls))
        return real_snapshot(session_id, run_id_)

    import unittest.mock

    with unittest.mock.patch.object(main_runs, "_run_snapshot", snapshot):
        return parse_sse(parked_client.get(run_url(sid, f"/{run_id}/events")).text)


def new_session(test_client):
    return test_client.post(f"{API}/sessions", json={"target_role": "Dev", "location": "P"}).json()[
        "id"
    ]


def test_stream_delivers_events_that_happen_while_it_is_open(parked_client):
    sid = new_session(parked_client)
    run_id = parked_client.post(run_url(sid, "/job-market")).json()["id"]

    def advance(poll):
        with session_scope() as db:
            run = db.get(AgentRun, run_id)
            if poll == 2:  # the first poll (before streaming) saw a queued run with no events
                run.events = [{"step": "search", "detail": {}, "at": "t"}]
            if poll == 3:
                run.status = Status.SUCCEEDED

    events = drive_stream(parked_client, sid, run_id, advance)
    assert [(e["event"], json.loads(e["data"]).get("step")) for e in events] == [
        ("progress", "search"),
        ("done", None),
    ]


def test_stream_ends_without_done_when_the_session_is_deleted(parked_client):
    sid = new_session(parked_client)
    run_id = parked_client.post(run_url(sid, "/job-market")).json()["id"]

    def delete_on_second_poll(poll):
        if poll == 2:
            with session_scope() as db:
                db.delete(db.get(CareerSession, sid))

    assert drive_stream(parked_client, sid, run_id, delete_on_second_poll) == []


def test_stream_of_an_unfinished_run_stops_at_the_time_limit(parked_client, monkeypatch):
    monkeypatch.setattr(main_runs, "MAX_STREAM_SECONDS", 0)
    sid = new_session(parked_client)
    run_id = parked_client.post(run_url(sid, "/job-market")).json()["id"]
    events = parse_sse(parked_client.get(run_url(sid, f"/{run_id}/events")).text)
    assert all(e.get("event") != "done" for e in events)


@pytest.mark.parametrize(
    ("header", "index"), [("4", 5), ("0", 1), ("²", 0), ("abc", 0), ("", 0), (None, 0)]
)
def test_resume_index_accepts_only_ascii_digits(header, index):
    assert main_runs._resume_index(header) == index


def test_requests_commit_before_the_response_is_sent(engine, handlers, monkeypatch):
    from sqlalchemy.orm import Session as OrmSession

    order = []
    real_commit = OrmSession.commit

    def recording_commit(self):
        order.append("commit")
        return real_commit(self)

    monkeypatch.setattr(OrmSession, "commit", recording_commit)
    monkeypatch.setattr(
        main.RunExecutor,
        "with_threads",
        classmethod(lambda cls, _h, _w: cls(handlers, InlineExecutor())),
    )
    app = main.create_app()

    async def recording_app(scope, receive, send):
        async def recording_send(message):
            if message["type"] == "http.response.body":
                order.append("body-sent")
            await send(message)

        await app(scope, receive, recording_send)

    with TestClient(recording_app) as test_client:
        order.clear()
        test_client.post(f"{API}/sessions", json={"target_role": "Dev", "location": "Pune"})
    assert order.index("commit") < order.index("body-sent")


def test_concurrent_duplicate_request_gets_409_not_500(parked_client, monkeypatch):
    sid = new_session(parked_client)
    parked_client.post(run_url(sid, "/job-market"))
    monkeypatch.setattr(career_sessions, "unfinished_run", lambda db, session_id, kind: None)
    assert parked_client.post(run_url(sid, "/job-market")).status_code == 409


def test_rerendered_resume_files_can_be_downloaded(client, session_id):
    run_id = add_run(session_id, RunKind.RESUME_RENDER, result={})
    with session_scope() as db:
        db.add(ResumeFile(run_id=run_id, latex="L", pdf=b"P"))
    assert client.get(run_url(session_id, f"/{run_id}/resume.tex")).text == "L"
    assert client.get(run_url(session_id, f"/{run_id}/resume.pdf")).content == b"P"


def test_rerender_with_an_unknown_template_is_rejected(client, session_id):
    body = {"draft": {"contact": {}, "summary": "s"}, "template_id": "missing"}
    assert client.post(run_url(session_id, "/resume-render"), json=body).status_code == 422


def test_rerender_requires_a_contact(client, session_id):
    response = client.post(run_url(session_id, "/resume-render"), json={"draft": {"summary": "s"}})
    assert response.status_code == 422


def test_generic_error_message_is_exact(client, session_id, handlers):
    def boom(context):
        raise RuntimeError("internal detail")

    handlers[RunKind.JOB_MARKET] = boom
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    assert stored_run(run_id).error == "RuntimeError: the agent failed. Please try again."


def test_unserialisable_result_is_reported_as_an_agent_failure(client, session_id, handlers):
    from app.services import runner

    handlers[RunKind.JOB_MARKET] = lambda context: {"x": object()}
    run_id = client.post(run_url(session_id, "/job-market")).json()["id"]
    assert stored_run(run_id).error not in (None, runner.SAVE_FAILED_MESSAGE)


def test_finishing_a_missing_run_is_a_no_op(engine):
    RunExecutor({}, InlineExecutor())._finish("missing", Status.FAILED, error="x")


def test_patch_with_null_leaves_the_field_unchanged(client, session_id):
    body = client.patch(f"{API}/sessions/{session_id}", json={"location": None}).json()
    assert body["location"] == "Pune"


def test_startup_records_why_runs_failed(engine, handlers, monkeypatch):
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        run_id = career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {}).id
    monkeypatch.setattr(
        main.RunExecutor,
        "with_threads",
        classmethod(lambda cls, _h, _w: cls(handlers, InlineExecutor())),
    )
    with TestClient(main.create_app()):
        pass
    assert stored_run(run_id).error == main.RESTART_REASON


def test_shutdown_stops_the_run_executor(engine, handlers, monkeypatch):
    class RecordingExecutor(InlineExecutor):
        shut = []

        def shutdown(self, wait=True, *, cancel_futures=False):
            self.shut.append(cancel_futures)

    recording = RecordingExecutor()
    monkeypatch.setattr(
        main.RunExecutor, "with_threads", classmethod(lambda cls, _h, _w: cls(handlers, recording))
    )
    with TestClient(main.create_app()):
        pass
    assert recording.shut == [True]


def test_finished_interviews_record_their_finish_time(client, session_id, coach):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]
    client.post(interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "stop"})
    assert stored_run(interview_id).finished_at is not None


@pytest.mark.parametrize("topics", [[""], ["   "], ["x" * 201]])
def test_interview_topics_must_be_short_non_blank_text(client, session_id, coach, topics):
    assert client.post(interview_url(session_id), json={"topics": topics}).status_code == 422


def test_review_request_of_exactly_the_limit_is_accepted(client, session_id):
    response = client.post(
        run_url(session_id, "/resume-review"), files=upload(), data={"request": "x" * 1000}
    )
    assert response.status_code == 202


def test_answer_of_exactly_the_limit_is_accepted(client, session_id, coach):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]
    response = client.post(
        interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "x" * 5000}
    )
    assert response.status_code == 200


@pytest.mark.parametrize(("count", "code"), [(200, 200), (201, 422)])
def test_skill_list_size_limit(client, session_id, count, code):
    skills = [f"s{i}" for i in range(count)]
    assert client.patch(f"{API}/sessions/{session_id}", json={"skills": skills}).status_code == code


def test_nul_characters_are_removed_from_uploads(client, session_id):
    run_id = client.post(
        run_url(session_id, "/skills/resume"), files=upload(content=b"Py\x00thon")
    ).json()["id"]
    assert stored_run(run_id).input == {"resume_text": "Python"}


def test_latest_run_ignores_runs_without_a_finish_time(engine):
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        timed = career_sessions.create_run(
            db,
            career_session,
            RunKind.JOB_MARKET,
            {},
            status=Status.SUCCEEDED,
            finished_at=main.career_sessions.utcnow(),
        )
        career_sessions.create_run(
            db, career_session, RunKind.JOB_MARKET, {}, status=Status.SUCCEEDED
        )
        timed_id, session_id = timed.id, career_session.id
    with session_scope() as db:
        assert (
            career_sessions.latest_succeeded_run(db, session_id, RunKind.JOB_MARKET).id == timed_id
        )


def test_importing_the_app_module_needs_no_settings(tmp_path):
    """Import app.main in a fresh process with no credentials and .env loading disabled."""
    import os
    import subprocess
    import sys

    code = (
        "import app.core.config as config\n"
        "config.Settings.model_config['env_file'] = None\n"
        "import app.main\n"
    )
    # Only PATH: no real credentials from this shell reach the child process.
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "PYTHONPATH": str(BACKEND_DIR),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-500:]


# --- Round-3 audit regressions ---


class RecordingCheckpointer:
    def __init__(self):
        self.deleted = []

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)


@pytest.fixture
def checkpointer(monkeypatch):
    recording = RecordingCheckpointer()
    monkeypatch.setattr("app.core.checkpointer.get_checkpointer", lambda: recording)
    return recording


def test_deleting_a_session_deletes_its_interview_checkpoints(client, session_id, checkpointer):
    interview_id = add_run(session_id, RunKind.INTERVIEW, status=Status.RUNNING)
    add_run(session_id, RunKind.JOB_MARKET)
    client.delete(f"{API}/sessions/{session_id}")
    assert checkpointer.deleted == [interview_id]


def test_checkpoint_failure_does_not_undo_a_session_delete(client, session_id, monkeypatch):
    def broken():
        raise ConnectionError("postgres down")

    monkeypatch.setattr("app.core.checkpointer.get_checkpointer", broken)
    add_run(session_id, RunKind.INTERVIEW, status=Status.RUNNING)
    assert client.delete(f"{API}/sessions/{session_id}").status_code == 204
    assert client.get(f"{API}/sessions/{session_id}").status_code == 404


def test_expiry_really_deletes_checkpoints(engine, checkpointer):
    _session_seen(40, with_interview=True)
    main.expire_old_sessions()
    assert checkpointer.deleted == ["thread-1"]


def test_report_failure_on_the_final_answer_still_closes_the_interview(
    client, session_id, coach, monkeypatch
):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]

    def broken(thread_id):
        raise ConnectionError("checkpointer down")

    monkeypatch.setattr(interview_coach, "evaluation_report", broken)
    response = client.post(
        interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "stop"}
    )
    run = stored_run(interview_id)
    assert response.status_code == 200 and run.status == Status.SUCCEEDED
    assert run.result == {"summary": response.json()["turn"]["summary"], "report": []}


def test_scheduling_failure_does_not_leave_a_queued_run(engine, handlers, monkeypatch):
    class Refusing(InlineExecutor):
        def submit(self, fn, /, *args, **kwargs):
            raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(
        main.RunExecutor, "with_threads", classmethod(lambda cls, _h, _w: cls(handlers, Refusing()))
    )
    with TestClient(main.create_app()) as test_client:
        sid = new_session(test_client)
        first = test_client.post(run_url(sid, "/job-market"))
        statuses = [r["status"] for r in test_client.get(run_url(sid)).json()]
        second = test_client.post(run_url(sid, "/job-market"))
    assert (first.status_code, statuses, second.status_code) == (503, ["failed"], 503)


def test_expiry_handles_more_sessions_than_one_batch(engine, monkeypatch):
    monkeypatch.setattr(main, "_delete_checkpoints", lambda ids: None)
    monkeypatch.setattr(career_sessions, "EXPIRY_BATCH_SIZE", 3)
    for _ in range(7):
        _session_seen(40)
    _session_seen(1)
    assert main.expire_old_sessions() == 7


def test_expiry_of_forty_thousand_sessions_does_not_hit_parameter_limits(engine):
    from sqlalchemy import insert

    old = main.career_sessions.utcnow() - timedelta(days=40)
    rows = [
        {
            "id": f"s{i:05d}",
            "target_role": "Dev",
            "location": "P",
            "skills": [],
            "created_at": old,
            "last_seen_at": old,
        }
        for i in range(40_000)
    ]
    with session_scope() as db:
        db.execute(insert(CareerSession), rows)
    with session_scope() as db:
        expired = career_sessions.delete_expired_sessions(db, timedelta(days=30))
    assert len(expired.session_ids) == 40_000


def test_cleanup_loop_keeps_going_after_an_error(monkeypatch):
    import asyncio

    calls, sleeps = [], []

    def flaky_expiry():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("database hiccup")

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(main, "expire_old_sessions", flaky_expiry)
    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(REAL_EXPIRE_LOOP())
    assert len(calls) == 2 and sleeps == [main.CLEANUP_INTERVAL_SECONDS] * 2


REAL_EXPIRE_LOOP = main._expire_periodically  # captured before the autouse fixture replaces it


@pytest.mark.parametrize(
    ("payload", "code"),
    [({"skills": ["x" * 100]}, 200), ({"skills": ["x" * 101]}, 422)],
)
def test_skill_names_are_limited_to_100_characters(client, session_id, payload, code):
    assert client.patch(f"{API}/sessions/{session_id}", json=payload).status_code == code


@pytest.mark.parametrize(("count", "code"), [(10, 201), (11, 422)])
def test_at_most_ten_interview_topics(client, session_id, coach, count, code):
    topics = [f"topic {i}" for i in range(count)]
    assert client.post(interview_url(session_id), json={"topics": topics}).status_code == code


@pytest.mark.parametrize(("count", "code"), [(100, 202), (101, 422)])
def test_extra_skill_list_limit(client, session_id, count, code):
    response = client.post(
        run_url(session_id, "/resume"), json={"extra_skills": [f"s{i}" for i in range(count)]}
    )
    assert response.status_code == code


def test_blank_extra_skills_are_dropped_and_trimmed(client, session_id):
    run_id = client.post(
        run_url(session_id, "/resume"), json={"extra_skills": ["", "  ", " Rust "]}
    ).json()["id"]
    assert stored_run(run_id).input["profile"]["skills"] == ["Rust"]


def test_failed_interview_start_records_the_reason_and_time(client, session_id, monkeypatch):
    from app.api.routers import interviews as interviews_router

    def broken(*args, **kwargs):
        raise ConnectionError("vector store down")

    monkeypatch.setattr(interview_coach, "start_interview", broken)
    client.post(interview_url(session_id), json={})
    run_id = client.get(run_url(session_id), params={"kind": "interview"}).json()[0]["id"]
    run = stored_run(run_id)
    assert run.error == interviews_router.UNAVAILABLE and run.finished_at is not None


def test_stream_stops_when_the_browser_disconnects(engine):
    import asyncio

    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        run_id = career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {}).id
        session_id = career_session.id

    class DisconnectedRequest:
        async def is_disconnected(self):
            return True

    async def collect():
        snapshot = main_runs._run_snapshot(session_id, run_id)
        stream = main_runs.stream_run_events(session_id, run_id, DisconnectedRequest(), snapshot)
        return [event async for event in stream]

    assert asyncio.run(collect()) == []


# --- Round-4 audit regressions ---


def test_failure_while_starting_a_run_marks_it_failed(client, session_id, monkeypatch):
    from app.services import runner

    def broken(career_session):
        raise RuntimeError("bad row")

    monkeypatch.setattr(runner, "_snapshot", broken)
    first = client.post(run_url(session_id, "/job-market")).json()["id"]
    assert stored_run(first).status == Status.FAILED
    assert client.post(run_url(session_id, "/job-market")).status_code == 202


def test_run_without_a_handler_is_marked_failed(engine):
    with session_scope() as db:
        career_session = career_sessions.create_session(db, "Dev", "Pune")
        run_id = career_sessions.create_run(db, career_session, RunKind.JOB_MARKET, {}).id
    RunExecutor({}, InlineExecutor()).execute(run_id)
    assert stored_run(run_id).status == Status.FAILED


def test_handlers_receive_the_sessions_current_fields(client, session_id, handlers):
    seen = {}

    def capture(context):
        snapshot = context.session
        seen.update(
            role=snapshot.target_role,
            location=snapshot.location,
            github=snapshot.github_username,
            skills=snapshot.skills,
        )
        return {}

    with session_scope() as db:
        db.get(CareerSession, session_id).github_username = "octocat"
    client.patch(
        f"{API}/sessions/{session_id}", json={"skills": ["Go", "SQL"], "location": "Remote"}
    )
    handlers[RunKind.JOB_MARKET] = capture
    client.post(run_url(session_id, "/job-market"))
    assert seen == {
        "role": "Backend Dev",
        "location": "Remote",
        "github": "octocat",
        "skills": ["Go", "SQL"],
    }


def test_interview_that_finishes_immediately_is_closed(client, session_id, coach, monkeypatch):
    monkeypatch.setattr(interview_coach, "_retrieve_questions", lambda state, difficulty_map: [])
    body = client.post(interview_url(session_id), json={}).json()
    assert body["turn"]["finished"] and stored_run(body["id"]).status == Status.SUCCEEDED


def test_failed_interview_cannot_be_answered(client, session_id, coach):
    interview_id = client.post(interview_url(session_id), json={}).json()["id"]
    with session_scope() as db:
        db.get(AgentRun, interview_id).status = Status.FAILED
    response = client.post(
        interview_url(session_id, f"/{interview_id}/answers"), json={"answer": "stop"}
    )
    assert response.status_code == 409 and stored_run(interview_id).status == Status.FAILED


def test_interview_topics_ignore_failed_analyses(client, session_id, coach):
    add_run(
        session_id,
        RunKind.JOB_MARKET,
        status=Status.FAILED,
        result={"analysis": {"skill_gaps": ["docker"]}},
    )
    assert (
        client.post(interview_url(session_id), json={}).json()["topics"]
        == interview_coach.DEFAULT_TOPICS
    )


@pytest.mark.parametrize(
    ("header", "index"), [("9" * 5000, 0), ("9" * 11, 0), ("9" * 10, 10_000_000_000)]
)
def test_resume_index_caps_the_header_length(header, index):
    assert main_runs._resume_index(header) == index


def test_report_handles_a_resume_run_without_a_finish_time(client, session_id):
    with session_scope() as db:
        career_session = db.get(CareerSession, session_id)
        run = career_sessions.create_run(
            db, career_session, RunKind.RESUME, {}, status=Status.SUCCEEDED, result={}
        )
        run_id = run.id
    assert client.get(f"{API}/sessions/{session_id}/report").json()["resume"]["run_id"] == run_id


def test_session_skills_are_deduplicated_ignoring_case(client, session_id):
    body = client.patch(
        f"{API}/sessions/{session_id}", json={"skills": ["Go", "go", "GO", "SQL"]}
    ).json()
    assert body["skills"] == ["Go", "SQL"]


def test_resume_skills_are_deduplicated_ignoring_case(client, session_id):
    client.patch(f"{API}/sessions/{session_id}", json={"skills": ["Python"]})
    run_id = client.post(
        run_url(session_id, "/resume"), json={"extra_skills": ["python", "Rust"]}
    ).json()["id"]
    assert stored_run(run_id).input["profile"]["skills"] == ["Python", "Rust"]


def test_shutdown_cancels_the_cleanup_loop(engine, handlers, monkeypatch):
    import asyncio

    cancelled = []

    async def forever():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    monkeypatch.setattr(main, "_expire_periodically", forever)
    monkeypatch.setattr(
        main.RunExecutor,
        "with_threads",
        classmethod(lambda cls, _h, _w: cls(handlers, InlineExecutor())),
    )
    with TestClient(main.create_app()):
        pass
    assert cancelled == [True]
