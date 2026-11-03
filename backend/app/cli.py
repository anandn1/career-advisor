"""Terminal front-end for the career advisor agents (development use).

Run from backend/:  uv run python -m app.cli
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.agents import curriculum_architect, interview_coach, job_market_analyst, skills
from app.agents.resume_strategist import ContactInfo, ResumeProfile, generate_resume
from app.core.config import get_settings
from app.services import github, latex

TOP_GAPS_SHOWN = 10
COMMANDS = {
    "interview": ("interview", "mock", "practice"),
    "jobs": ("job", "market", "search"),
    "learn": ("learn", "course", "curriculum", "path"),
    "resume": ("resume", "cv"),
    "skills": ("skill",),
    "report": ("report", "summary"),
}
EXIT_WORDS = {"exit", "quit", "q", "bye"}
HELP = """Commands: jobs | learn | resume | interview | skills | report | exit"""


@dataclass
class CliSession:
    target_role: str
    location: str
    github_username: str | None = None
    user_skills: list[str] = field(default_factory=list)
    skill_gaps: list[str] = field(default_factory=list)
    job_count: int = 0
    learning_path: curriculum_architect.PersonalizedLearningPath | None = None
    resume_path: Path | None = None
    interview_summaries: list[str] = field(default_factory=list)


def ask(prompt: str, default: str = "") -> str:
    answer = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return answer or default


def collect_background(session: CliSession) -> None:
    choice = ask("Provide background via (1) resume file, (2) GitHub, (3) skip", "3")
    if choice == "1":
        path = Path(ask("Path to resume (PDF/DOCX/TXT)").strip("\"'"))
        if path.exists():
            session.user_skills = skills.extract_skills_from_resume(path)
        else:
            print("File not found; skipping.")
    elif choice == "2":
        try:
            username = github.parse_username(ask("GitHub username or profile URL"))
        except github.InvalidUsernameError as error:
            print(f"{error}; skipping.")
            return
        session.github_username = username
        session.user_skills = skills.extract_skills_from_github(username)
    print(f"Skills: {', '.join(session.user_skills) or 'none found'}")


def run_job_analysis(session: CliSession) -> None:
    result = job_market_analyst.run_job_market_analysis(
        session.target_role, session.location, session.user_skills
    )
    analysis = result["skill_gap_data"]
    session.skill_gaps = analysis["skill_gaps"]
    session.job_count = len(result["job_data"])
    print(f"Analyzed {session.job_count} postings. Top gaps:")
    for gap in session.skill_gaps[:TOP_GAPS_SHOWN]:
        print(f"  - {gap}")


def ensure_skill_gaps(session: CliSession) -> None:
    if not session.skill_gaps:
        print("Finding your skill gaps first...")
        run_job_analysis(session)


def run_learning_path(session: CliSession) -> None:
    ensure_skill_gaps(session)
    session.learning_path = curriculum_architect.plan_learning_path(
        session.skill_gaps, session.user_skills, session.target_role
    )
    if session.learning_path is None:
        print("Could not build a learning path.")
        return
    print(session.learning_path.user_summary)
    for rec in session.learning_path.recommendations:
        print(f"  - [{rec.skill_name}] {rec.course_title}: {rec.course_url}")


def run_resume(session: CliSession) -> None:
    contact = ContactInfo(
        name=ask("Full name"),
        email=ask("Email"),
        phone=ask("Phone"),
        location=session.location,
    )
    profile = ResumeProfile(
        target_role=session.target_role,
        contact=contact,
        github_username=session.github_username,
        skills=session.user_skills,
    )
    result = generate_resume(profile)
    output_dir = get_settings().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    session.resume_path = output_dir / "resume.tex"
    session.resume_path.write_text(result.latex, encoding="utf-8")
    if latex.is_available():
        pdf_path = session.resume_path.with_suffix(".pdf")
        pdf_path.write_bytes(latex.compile_pdf(result.latex))
        session.resume_path = pdf_path
    print(f"Resume saved to {session.resume_path}")


def run_interview(session: CliSession) -> None:
    ensure_skill_gaps(session)
    topics = interview_coach.topics_for_skill_gaps(session.skill_gaps)
    company = ask(f"Company focus ({', '.join(interview_coach.COMPANIES)}, all)", "generic")
    thread_id = f"cli-{uuid.uuid4()}"
    print(f"Topics: {', '.join(topics)}. Type 'stop' to finish.")
    turn = interview_coach.start_interview(thread_id, topics, company)
    while not turn.finished:
        print(f"\nINTERVIEWER: {turn.question}")
        turn = interview_coach.submit_answer(thread_id, input("YOU: "))
        if turn.feedback:
            print(f"\n{turn.feedback}")
    print(f"\n{turn.summary}")
    session.interview_summaries.append(turn.summary or "")


def show_skills(session: CliSession) -> None:
    print(", ".join(session.user_skills) or "No skills extracted yet.")


def save_report(session: CliSession) -> Path:
    lines = [
        "CAREER ADVISORY REPORT",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Target role: {session.target_role} ({session.location})",
        f"Skills ({len(session.user_skills)}): {', '.join(session.user_skills)}",
        f"Postings analyzed: {session.job_count}",
        f"Skill gaps: {', '.join(session.skill_gaps)}",
    ]
    if session.learning_path:
        lines.append("Recommended courses:")
        lines += [f"  - {r.course_title} ({r.skill_name}): {r.course_url}"
                  for r in session.learning_path.recommendations]  # fmt: skip
    if session.resume_path:
        lines.append(f"Resume: {session.resume_path}")
    lines += [f"Interview {i}: {s}" for i, s in enumerate(session.interview_summaries, 1)]

    output_dir = get_settings().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved to {path}")
    return path


ACTIONS = {
    "interview": run_interview,
    "jobs": run_job_analysis,
    "learn": run_learning_path,
    "resume": run_resume,
    "skills": show_skills,
    "report": save_report,
}


def command_for(text: str) -> str | None:
    lowered = text.lower()
    return next(
        (name for name, words in COMMANDS.items() if any(w in lowered for w in words)),
        None,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    session = CliSession(
        target_role=ask("Target job role", "Software Engineer"),
        location=ask("Preferred location (city or 'Remote')", "Remote"),
    )
    collect_background(session)
    print(HELP)
    while (text := input("\nYOU: ").strip().lower()) not in EXIT_WORDS:
        command = command_for(text)
        if command is None:
            print(HELP)
            continue
        try:
            ACTIONS[command](session)
        except Exception as error:  # keep the session alive after an agent failure
            logging.exception("%s failed: %s", command, error)
    save_report(session)


if __name__ == "__main__":
    main()
