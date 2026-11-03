"""Resume strategist: turns a user's profile (plus GitHub repos) into a tailored LaTeX resume.

Two LLM steps: first a structured resume draft tailored to the target role,
then a full LaTeX document that follows a reference template's style.
"""

import json
import logging
import re
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.resources import get_resume_llm
from app.services import github

logger = logging.getLogger(__name__)

REPOS_FOR_PROJECTS = 6
MAX_BULLET_LENGTH = 220
DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "1.tex"
LATEX_SPECIAL_CHARACTERS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)(?:\n```|$)", re.DOTALL)
# '%' starts a comment unless escaped as '\%'.
_LATEX_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
REQUIRED_LATEX_MARKERS = (r"\documentclass", r"\begin{document}", r"\end{document}")


class ContactInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    github: str = ""


class Education(BaseModel):
    institution: str
    degree: str = ""
    field: str = ""
    status: str = Field("", description="e.g. 'Currently pursuing' or 'Graduated May 2025'")
    location: str = ""
    gpa: str = ""


class WorkExperience(BaseModel):
    company: str
    position: str = ""
    location: str = ""
    duration: str = ""
    description_bullets: list[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description_bullets: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)


class SkillCategory(BaseModel):
    category_name: str = Field(description="e.g. 'Programming Languages', 'Frameworks', 'Tools'")
    skills: list[str]


class ResumeProfile(BaseModel):
    """What the user tells us about themselves."""

    target_role: str
    contact: ContactInfo = Field(default_factory=ContactInfo)
    github_username: str | None = None
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class ResumeDraft(BaseModel):
    """Structured resume content written by the LLM."""

    contact: ContactInfo
    summary: str = Field(description="2-4 sentence professional summary tailored to the role")
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    skills: list[SkillCategory] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)


class ResumeResult(BaseModel):
    draft: ResumeDraft
    latex: str


class InvalidLatexError(ValueError):
    pass


DRAFT_PROMPT = ChatPromptTemplate.from_template(
    """You are a professional resume writer. Using ONLY the data below, write a resume
tailored to the target role "{target_role}".
- Write a 2-4 sentence summary aimed at that role.
- Group skills into categories and put the most role-relevant ones first.
- Rewrite work and project bullets to be concise, action-oriented and quantified
  where the data allows. Never invent employers, dates, degrees or numbers.

DATA:
{profile}"""
)

LATEX_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert LaTeX resume typesetter. Produce a complete, compile-ready .tex
document for the resume data below, following the reference template's style.

REFERENCE TEMPLATE:
{template}

RESUME DATA (JSON; special characters are already LaTeX-escaped):
{resume}

Rules:
1. Copy the template's \\documentclass line and entire preamble (packages, \\newcommand
   definitions, formatting settings) exactly.
2. Use the template's own custom commands and section layout for every entry.
3. Replace only the content with the resume data. Leave out sections with no data.
4. Use only packages that the template already loads.
5. Output only raw LaTeX starting with \\documentclass: no markdown fences, no commentary."""
)


def escape_latex(value):
    """Recursively escape LaTeX special characters in strings inside dicts/lists."""
    if isinstance(value, str):
        return "".join(LATEX_SPECIAL_CHARACTERS.get(char, char) for char in value)
    if isinstance(value, dict):
        return {key: escape_latex(item) for key, item in value.items()}
    if isinstance(value, list):
        return [escape_latex(item) for item in value]
    return value


def strip_code_fences(text: str) -> str:
    """The LaTeX inside the first ``` fence, else the text from \\documentclass on."""
    fenced = _FENCED_BLOCK.search(text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find(r"\documentclass")
    return (text[start:] if start >= 0 else text).strip()


def _without_comments(latex: str) -> str:
    return _LATEX_COMMENT.sub("", latex)


def validate_latex(latex: str) -> str:
    """Require the document markers outside comments; returns the LaTeX unchanged."""
    live = _without_comments(latex)
    missing = [marker for marker in REQUIRED_LATEX_MARKERS if marker not in live]
    if missing:
        raise InvalidLatexError(f"Generated LaTeX is missing {', '.join(missing)}")
    return latex


def _shorten(text: str) -> str:
    if len(text) <= MAX_BULLET_LENGTH:
        return text
    return text[: MAX_BULLET_LENGTH - 3] + "..."


def projects_from_repos(repos: list[dict]) -> list[Project]:
    return [
        Project(
            name=repo["name"],
            description_bullets=[
                _shorten(repo["description"]) or f"Developed the {repo['name']} repository.",
                f"Source: {repo['html_url']}",
            ],
            tech_stack=[repo["language"]] if repo["language"] else [],
        )
        for repo in repos
    ]


def _profile_with_github(profile: ResumeProfile) -> dict:
    """The profile as a dict, enriched with GitHub projects and contact defaults."""
    data = profile.model_dump()
    if not profile.github_username:
        return data
    github_data = github.fetch_profile_and_repos(profile.github_username, REPOS_FOR_PROJECTS)
    gh_profile = github_data["profile"]
    contact = data["contact"]
    contact["name"] = contact["name"] or gh_profile.get("name") or ""
    contact["location"] = contact["location"] or gh_profile.get("location") or ""
    contact["github"] = contact["github"] or gh_profile.get("html_url", "")
    repo_languages = [repo["language"] for repo in github_data["repos"] if repo["language"]]
    data["skills"] = list(dict.fromkeys([*profile.skills, *repo_languages]))
    data["projects"] = [p.model_dump() for p in projects_from_repos(github_data["repos"])]
    return data


def draft_resume(profile: ResumeProfile) -> ResumeDraft:
    chain = DRAFT_PROMPT | get_resume_llm().with_structured_output(ResumeDraft)
    return chain.invoke(
        {
            "target_role": profile.target_role,
            "profile": json.dumps(_profile_with_github(profile), indent=2),
        }
    )


def render_latex(draft: ResumeDraft, template_latex: str) -> str:
    chain = LATEX_PROMPT | get_resume_llm() | StrOutputParser()
    raw = chain.invoke(
        {
            "template": template_latex,
            "resume": json.dumps(escape_latex(draft.model_dump()), indent=2),
        }
    )
    return validate_latex(strip_code_fences(raw))


def default_template() -> str:
    return DEFAULT_TEMPLATE.read_text(encoding="utf-8")


def generate_resume(profile: ResumeProfile, template_latex: str | None = None) -> ResumeResult:
    """Draft a tailored resume and typeset it as LaTeX (default template if none given)."""
    draft = draft_resume(profile)
    latex = render_latex(draft, template_latex or default_template())
    return ResumeResult(draft=draft, latex=latex)
