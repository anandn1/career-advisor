"""The agent-run handlers, with the agents themselves faked."""

import pytest

from app.agents.resume_reviewer import ResumeReview
from app.agents.resume_strategist import ContactInfo, ResumeDraft, ResumeResult
from app.db import career_sessions
from app.db.models import CareerSession, ResumeFile, ResumeTemplate, RunKind
from app.db.session import session_scope
from app.services import latex, run_handlers
from app.services.runner import RunContext, SessionSnapshot, UserFacingError

VALID = "\\documentclass{article}\\begin{document}x\\end{document}"


@pytest.fixture
def context(sqlite_engine):
    """A real session + run row, and the context a handler would receive."""

    def make(kind=RunKind.JOB_MARKET, run_input=None, skills=("Go",)):
        with session_scope() as db:
            career_session = career_sessions.create_session(db, "Backend Dev", "Pune")
            career_session.skills = list(skills)
            run = career_sessions.create_run(db, career_session, kind, run_input or {})
            ids = (run.id, career_session.id)
        snapshot = SessionSnapshot("Backend Dev", "Pune", None, list(skills))
        return RunContext(ids[0], ids[1], run_input or {}, snapshot)

    return make


def stored_session(session_id):
    with session_scope() as db:
        career_session = db.get(CareerSession, session_id)
        db.expunge(career_session)
        return career_session


def events(context):
    with session_scope() as db:
        return [e["step"] for e in career_sessions.get_run(db, context.run_id).events]


def test_resume_skills_are_saved_on_the_session(context, monkeypatch):
    monkeypatch.setattr(run_handlers.skills, "extract_skills_from_text", lambda text: ["Python"])
    ctx = context(RunKind.SKILLS_FROM_RESUME, {"resume_text": "I write Python"})
    assert run_handlers.skills_from_resume(ctx) == {"skills": ["Python"]}
    assert stored_session(ctx.session_id).skills == ["Python"]


def test_github_skills_also_save_the_username(context, monkeypatch):
    monkeypatch.setattr(run_handlers.skills, "extract_skills_from_github", lambda name: ["Rust"])
    ctx = context(RunKind.SKILLS_FROM_GITHUB, {"github_username": "octocat"})
    run_handlers.skills_from_github(ctx)
    saved = stored_session(ctx.session_id)
    assert (saved.skills, saved.github_username) == (["Rust"], "octocat")


def test_job_market_reports_each_graph_step(context, monkeypatch):
    def fake_analysis(role, location, skills, on_step):
        for node in ("search_cached_postings", "scrape_fresh_postings", "analyze_skill_gaps"):
            on_step(node)
        return {"job_data": [{"title": "Dev"}], "skill_gap_data": {"skill_gaps": ["docker"]}}

    monkeypatch.setattr(run_handlers.job_market_analyst, "run_job_market_analysis", fake_analysis)
    ctx = context()
    result = run_handlers.job_market(ctx)
    assert result == {"jobs": [{"title": "Dev"}], "analysis": {"skill_gaps": ["docker"]}}
    assert events(ctx) == ["search_cached_postings", "scrape_fresh_postings", "analyze_skill_gaps"]


def test_job_market_uses_the_session_role_location_and_skills(context, monkeypatch):
    seen = {}

    def fake_analysis(role, location, skills, on_step):
        seen.update(role=role, location=location, skills=skills)
        return {}

    monkeypatch.setattr(run_handlers.job_market_analyst, "run_job_market_analysis", fake_analysis)
    run_handlers.job_market(context(skills=("Go", "SQL")))
    assert seen == {"role": "Backend Dev", "location": "Pune", "skills": ["Go", "SQL"]}


def test_learning_path_without_a_result_is_a_user_facing_error(context, monkeypatch):
    monkeypatch.setattr(
        run_handlers.curriculum_architect,
        "build_learning_path",
        lambda gaps, inventory, on_step: {},
    )
    with pytest.raises(UserFacingError):
        run_handlers.learning_path(context(RunKind.LEARNING_PATH, {"skill_gaps": ["docker"]}))


def test_learning_path_skips_soft_skill_gaps(context, monkeypatch):
    seen = {}

    def fake_build(gaps, inventory, on_step):
        seen["gaps"] = gaps
        return {}

    monkeypatch.setattr(run_handlers.curriculum_architect, "build_learning_path", fake_build)
    with pytest.raises(UserFacingError):
        run_handlers.learning_path(
            context(RunKind.LEARNING_PATH, {"skill_gaps": ["docker", "communication"]})
        )
    assert seen["gaps"] == "docker"


def draft():
    return ResumeDraft(contact=ContactInfo(name="Ada"), summary="Engineer")


def resume_context(context, template_id=None):
    profile = {"target_role": "Backend Dev", "skills": ["Go"]}
    return context(RunKind.RESUME, {"profile": profile, "template_id": template_id})


@pytest.fixture
def fake_generation(monkeypatch):
    seen = {}

    def generate(profile, template):
        seen.update(profile=profile, template=template)
        return ResumeResult(draft=draft(), latex=VALID)

    monkeypatch.setattr(run_handlers, "generate_resume", generate)
    return seen


def stored_file(run_id):
    with session_scope() as db:
        resume_file = db.get(ResumeFile, run_id)
        db.expunge(resume_file)
        return resume_file


def test_resume_without_tex_stores_only_latex(context, fake_generation, monkeypatch):
    monkeypatch.setattr(latex, "is_available", lambda: False)
    ctx = resume_context(context)
    result = run_handlers.resume(ctx)
    saved = stored_file(ctx.run_id)
    assert (saved.latex, saved.pdf, result["has_pdf"]) == (VALID, None, False)
    assert result["pdf_error"]


def test_resume_with_tex_stores_the_pdf(context, fake_generation, monkeypatch):
    monkeypatch.setattr(latex, "is_available", lambda: True)
    monkeypatch.setattr(latex, "compile_pdf", lambda source: b"%PDF")
    ctx = resume_context(context)
    result = run_handlers.resume(ctx)
    assert stored_file(ctx.run_id).pdf == b"%PDF" and result == {
        "draft": draft().model_dump(),
        "template_id": None,
        "has_pdf": True,
        "pdf_error": None,
    }


def test_resume_compile_failure_still_stores_the_latex(context, fake_generation, monkeypatch):
    def fail(source):
        raise latex.LatexCompileError("compilation errors", "! Undefined control sequence")

    monkeypatch.setattr(latex, "is_available", lambda: True)
    monkeypatch.setattr(latex, "compile_pdf", fail)
    ctx = resume_context(context)
    result = run_handlers.resume(ctx)
    assert stored_file(ctx.run_id).latex == VALID and not result["has_pdf"]


def test_resume_uses_the_selected_template(context, fake_generation, monkeypatch):
    monkeypatch.setattr(latex, "is_available", lambda: False)
    with session_scope() as db:
        template = ResumeTemplate(name="fancy", latex_src="FANCY TEMPLATE")
        db.add(template)
        db.flush()
        template_id = template.id
    run_handlers.resume(resume_context(context, template_id))
    assert fake_generation["template"] == "FANCY TEMPLATE"


def test_resume_without_a_template_uses_the_default(context, fake_generation, monkeypatch):
    monkeypatch.setattr(latex, "is_available", lambda: False)
    run_handlers.resume(resume_context(context))
    assert fake_generation["template"] == run_handlers.default_template()


def test_deleted_template_is_a_user_facing_error(context, fake_generation):
    with pytest.raises(UserFacingError):
        run_handlers.resume(resume_context(context, "deleted-template"))


def test_rerender_typesets_the_edited_draft(context, monkeypatch):
    seen = {}

    def render(edited, template):
        seen["summary"] = edited.summary
        return VALID

    monkeypatch.setattr(run_handlers, "render_latex", render)
    monkeypatch.setattr(latex, "is_available", lambda: False)
    edited = draft().model_copy(update={"summary": "Edited"})
    ctx = context(RunKind.RESUME_RENDER, {"draft": edited.model_dump(), "template_id": None})
    run_handlers.resume_render(ctx)
    assert seen == {"summary": "Edited"} and stored_file(ctx.run_id).latex == VALID


def test_review_passes_the_role_and_request(context, monkeypatch):
    seen = {}

    def review(text, role, request):
        seen.update(text=text, role=role, request=request)
        return ResumeReview(summary="s", strengths=["a"], weaknesses=["b"], suggestions=["c"])

    monkeypatch.setattr(run_handlers, "review_resume", review)
    ctx = context(RunKind.RESUME_REVIEW, {"resume_text": "CV", "request": "Be harsh"})
    assert run_handlers.resume_review(ctx)["review"]["suggestions"] == ["c"]
    assert seen == {"text": "CV", "role": "Backend Dev", "request": "Be harsh"}


def test_every_background_kind_has_a_handler():
    assert set(run_handlers.HANDLERS) == set(RunKind) - {RunKind.INTERVIEW}


def test_progress_events_are_appended_in_order(context):
    ctx = context()
    ctx.progress("one")
    ctx.progress("two", {"n": 2})
    with session_scope() as db:
        saved = career_sessions.get_run(db, ctx.run_id).events
    assert [(e["step"], e["detail"]) for e in saved] == [("one", {}), ("two", {"n": 2})]


def test_review_with_an_empty_request_uses_the_default(monkeypatch):
    from app.agents import resume_reviewer

    seen = {}

    class LLM:
        def with_structured_output(self, schema):
            from langchain_core.runnables import RunnableLambda

            def respond(prompt):
                seen["prompt"] = prompt.to_string()
                return ResumeReview(summary="s", strengths=[], weaknesses=[], suggestions=[])

            return RunnableLambda(respond)

    monkeypatch.setattr(resume_reviewer, "get_llm", lambda: LLM())
    resume_reviewer.review_resume("CV text", "Dev", "   ")
    assert resume_reviewer.DEFAULT_REQUEST in seen["prompt"] and "CV text" in seen["prompt"]


def test_learning_path_success_returns_a_json_ready_path(context, monkeypatch):
    import json

    from app.agents.curriculum_architect import CourseRecommendation, PersonalizedLearningPath

    path = PersonalizedLearningPath(
        user_summary="s",
        skill_gaps=["docker"],
        recommendations=[
            CourseRecommendation(
                skill_name="docker", course_title="D", course_url="u", description="d"
            )
        ],
    )
    monkeypatch.setattr(
        run_handlers.curriculum_architect,
        "build_learning_path",
        lambda gaps, inventory, on_step: {"draft_path": path},
    )
    result = run_handlers.learning_path(context(RunKind.LEARNING_PATH, {"skill_gaps": ["docker"]}))
    assert json.loads(json.dumps(result))["path"]["skill_gaps"] == ["docker"]


def test_compile_failure_message_is_reported(context, fake_generation, monkeypatch):
    def fail(source):
        raise latex.LatexCompileError("compilation errors")

    monkeypatch.setattr(latex, "is_available", lambda: True)
    monkeypatch.setattr(latex, "compile_pdf", fail)
    assert (
        run_handlers.resume(resume_context(context))["pdf_error"]
        == "pdflatex failed: compilation errors"
    )


def test_progress_after_the_run_is_deleted_stops_the_handler(context):
    from app.services.runner import RunGoneError

    ctx = context()
    with session_scope() as db:
        db.delete(db.get(CareerSession, ctx.session_id))
    with pytest.raises(RunGoneError):
        ctx.progress("anything")


def test_github_skills_report_progress(context, monkeypatch):
    monkeypatch.setattr(run_handlers.skills, "extract_skills_from_github", lambda name: [])
    ctx = context(RunKind.SKILLS_FROM_GITHUB, {"github_username": "octocat"})
    run_handlers.skills_from_github(ctx)
    assert events(ctx) == ["reading_github"]


def test_resume_reports_pdf_compilation(context, fake_generation, monkeypatch):
    monkeypatch.setattr(latex, "is_available", lambda: True)
    monkeypatch.setattr(latex, "compile_pdf", lambda source: b"%PDF")
    ctx = resume_context(context)
    run_handlers.resume(ctx)
    assert events(ctx) == ["drafting_resume", "compiling_pdf"]


def test_learning_path_inventory_has_the_session_skills(context, monkeypatch):
    import json

    seen = {}

    def fake_build(gaps, inventory, on_step):
        seen["inventory"] = json.loads(inventory)
        return {}

    monkeypatch.setattr(run_handlers.curriculum_architect, "build_learning_path", fake_build)
    with pytest.raises(UserFacingError):
        run_handlers.learning_path(
            context(RunKind.LEARNING_PATH, {"skill_gaps": ["x"]}, skills=("Go", "SQL"))
        )
    assert seen["inventory"] == {"skills": ["Go", "SQL"], "target_role": "Backend Dev"}


def test_rerender_returns_the_template_and_pdf_status(context, monkeypatch):
    monkeypatch.setattr(run_handlers, "render_latex", lambda draft, template: VALID)
    monkeypatch.setattr(latex, "is_available", lambda: False)
    ctx = context(RunKind.RESUME_RENDER, {"draft": draft().model_dump(), "template_id": None})
    result = run_handlers.resume_render(ctx)
    assert (result["template_id"], result["has_pdf"], bool(result["pdf_error"])) == (
        None,
        False,
        True,
    )


def test_reviewer_prompt_names_the_target_role(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from app.agents import resume_reviewer

    seen = {}

    class LLM:
        def with_structured_output(self, schema):
            def respond(prompt):
                seen["prompt"] = prompt.to_string()
                return ResumeReview(summary="s", strengths=[], weaknesses=[], suggestions=[])

            return RunnableLambda(respond)

    monkeypatch.setattr(resume_reviewer, "get_llm", lambda: LLM())
    resume_reviewer.review_resume("CV", "Site Reliability Engineer")
    assert "Site Reliability Engineer" in seen["prompt"]


def test_run_graph_reports_each_step_and_returns_the_final_state():
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    from app.core.graphs import run_graph

    class State(TypedDict):
        n: int

    graph = StateGraph(State)
    graph.add_node("add_one", lambda state: {"n": state["n"] + 1})
    graph.add_node("times_ten", lambda state: {"n": state["n"] * 10})
    graph.add_edge(START, "add_one")
    graph.add_edge("add_one", "times_ten")
    graph.add_edge("times_ten", END)
    steps = []
    assert run_graph(graph.compile(), {"n": 1}, on_step=steps.append) == {"n": 20}
    assert steps == ["add_one", "times_ten"]


def test_run_graph_works_without_a_step_callback():
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    from app.core.graphs import run_graph

    class State(TypedDict):
        n: int

    graph = StateGraph(State)
    graph.add_node("double", lambda state: {"n": state["n"] * 2})
    graph.add_edge(START, "double")
    graph.add_edge("double", END)
    assert run_graph(graph.compile(), {"n": 4}) == {"n": 8}
