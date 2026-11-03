"""What each kind of agent run does. Every handler takes a RunContext, returns a JSON result."""

from app.agents import curriculum_architect, job_market_analyst, skills
from app.agents.resume_reviewer import review_resume
from app.agents.resume_strategist import (
    ResumeDraft,
    ResumeProfile,
    default_template,
    generate_resume,
    render_latex,
)
from app.db.models import ResumeFile, RunKind
from app.db.session import session_scope
from app.db.templates import get_template
from app.services import latex
from app.services.runner import Handler, RunContext, UserFacingError


def _step_reporter(context: RunContext):
    return lambda node: context.progress(node)


def skills_from_resume(context: RunContext) -> dict:
    context.progress("extracting_skills")
    found = skills.extract_skills_from_text(context.input["resume_text"])
    context.update_session(skills=found)
    return {"skills": found}


def skills_from_github(context: RunContext) -> dict:
    username = context.input["github_username"]
    context.progress("reading_github", {"username": username})
    found = skills.extract_skills_from_github(username)
    context.update_session(skills=found, github_username=username)
    return {"skills": found}


def job_market(context: RunContext) -> dict:
    snapshot = context.session
    state = job_market_analyst.run_job_market_analysis(
        snapshot.target_role,
        snapshot.location,
        snapshot.skills,
        on_step=_step_reporter(context),
    )
    return {
        "jobs": state.get("job_data", []),
        "analysis": state.get("skill_gap_data", {}),
    }


def learning_path(context: RunContext) -> dict:
    gaps_text, inventory = curriculum_architect.learning_path_inputs(
        context.input["skill_gaps"], context.session.skills, context.session.target_role
    )
    state = curriculum_architect.build_learning_path(
        gaps_text, inventory, on_step=_step_reporter(context)
    )
    path = state.get("draft_path")
    if path is None:
        raise UserFacingError("No learning path could be built for these skill gaps.")
    return {"path": path.model_dump()}


def _template_latex(template_id: str | None) -> str:
    if template_id is None:
        return default_template()
    with session_scope() as db:
        template = get_template(db, template_id)
        if template is None:
            raise UserFacingError("The selected resume template no longer exists.")
        return template.latex_src


def _store_resume_file(context: RunContext, latex_source: str) -> dict:
    """Save the .tex (and PDF when TeX is installed); returns what was produced."""
    pdf, pdf_error = None, None
    if latex.is_available():
        context.progress("compiling_pdf")
        try:
            pdf = latex.compile_pdf(latex_source)
        except latex.LatexCompileError as error:
            pdf_error = str(error)
    else:
        pdf_error = "PDF compilation is not available on this server."
    with session_scope() as db:
        db.add(ResumeFile(run_id=context.run_id, latex=latex_source, pdf=pdf))
    return {"has_pdf": pdf is not None, "pdf_error": pdf_error}


def resume(context: RunContext) -> dict:
    profile = ResumeProfile.model_validate(context.input["profile"])
    template_id = context.input.get("template_id")
    context.progress("drafting_resume")
    result = generate_resume(profile, _template_latex(template_id))
    files = _store_resume_file(context, result.latex)
    return {"draft": result.draft.model_dump(), "template_id": template_id, **files}


def resume_render(context: RunContext) -> dict:
    draft = ResumeDraft.model_validate(context.input["draft"])
    template_id = context.input.get("template_id")
    context.progress("typesetting_resume")
    latex_source = render_latex(draft, _template_latex(template_id))
    files = _store_resume_file(context, latex_source)
    return {"draft": draft.model_dump(), "template_id": template_id, **files}


def resume_review(context: RunContext) -> dict:
    context.progress("reviewing_resume")
    review = review_resume(
        context.input["resume_text"],
        context.session.target_role,
        context.input.get("request", ""),
    )
    return {"review": review.model_dump()}


HANDLERS: dict[RunKind, Handler] = {
    RunKind.SKILLS_FROM_RESUME: skills_from_resume,
    RunKind.SKILLS_FROM_GITHUB: skills_from_github,
    RunKind.JOB_MARKET: job_market,
    RunKind.LEARNING_PATH: learning_path,
    RunKind.RESUME: resume,
    RunKind.RESUME_RENDER: resume_render,
    RunKind.RESUME_REVIEW: resume_review,
}
