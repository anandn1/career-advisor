"""Resume strategist: turns a user's profile (plus GitHub repos) into a tailored LaTeX resume.

Two LLM steps: first a structured resume draft tailored to the target role,
then a full LaTeX document that follows a reference template's style.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

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
    # The default font encoding prints these as "¡", "¿", "—" and "”".
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    '"': "''",
    # A blank line ends a paragraph, which LaTeX rejects inside a command argument.
    "\n": " ",
    "\r": " ",
    "\t": " ",
}
# Unicode that models often write and pdflatex cannot print, as LaTeX that it can.
UNICODE_REPLACEMENTS = {
    "\u00a0": "~",  # no-break space
    "\u202f": "~",  # narrow no-break space
    "\u2009": r"\,",  # thin space
    "\u2002": " ",  # en space
    "\u2003": " ",  # em space
    "\u2007": " ",  # figure space
    "\u2008": " ",  # punctuation space
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "--",  # figure dash
    "\u2013": "--",  # en dash
    "\u2014": "---",  # em dash
    "\u2212": "-",  # minus sign
    "\u2032": "'",  # prime
    "\u2033": "''",  # double prime
    "\u2264": r"$\leq$",
    "\u2265": r"$\geq$",
    "\u2248": r"$\approx$",
    "\u2260": r"$\neq$",
    "\u21d2": r"$\Rightarrow$",
    "\u2219": r"\textbullet{}",  # bullet operator
    "\u25aa": r"\textbullet{}",  # small black square
    "\u25cf": r"\textbullet{}",  # black circle
    "\u25e6": r"\textbullet{}",  # white bullet
}
# Every other character is kept only if pdflatex prints it under both the default and the
# T1 font encoding (each was compiled in the backend image); the rest, such as emoji, are
# dropped, because one unprintable character fails the whole PDF.
PRINTABLE_RANGES = (
    (0x0020, 0x007E),
    (0x00A1, 0x00AA),
    (0x00AC, 0x00BA),
    (0x00BC, 0x00CF),
    (0x00D1, 0x00DD),
    (0x00DF, 0x00EF),
    (0x00F1, 0x00FD),
    (0x00FF, 0x0103),
    (0x0106, 0x010F),
    (0x0112, 0x0117),
    (0x011A, 0x0125),
    (0x0128, 0x012D),
    (0x0130, 0x0137),
    (0x0139, 0x013E),
    (0x0141, 0x0148),
    (0x014C, 0x0165),
    (0x0168, 0x0171),
    (0x0174, 0x017E),
    (0x0192, 0x0192),
    (0x2015, 0x2016),
    (0x2018, 0x2019),
    (0x201C, 0x201D),
    (0x2020, 0x2022),
    (0x2026, 0x2026),
    (0x2030, 0x2031),
    (0x20AC, 0x20AC),
    (0x2122, 0x2122),
    (0x2190, 0x2193),
)
_FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)(?:\n```|$)", re.DOTALL)
# '%' starts a comment unless escaped as '\%' ('\\%' is a line break, then a comment).
_LATEX_COMMENT = re.compile(r"(?<!\\)((?:\\\\)*)%.*$", re.MULTILINE)
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

    @field_validator("github_username")
    @classmethod
    def _normalize_github_username(cls, value: str | None) -> str | None:
        """Accept a username or a github.com profile URL; empty means none."""
        if value is None or not value.strip():
            return None
        return github.parse_username(value)


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
3. Replace only the content with the resume data. Leave out every section, entry and
   contact item (with its icon and separator) that has no data; never leave a placeholder.
4. Use only packages that the template already loads.
5. Output only raw LaTeX starting with \\documentclass: no markdown fences, no commentary."""
)


def _is_printable(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in PRINTABLE_RANGES)


def _escape_character(char: str) -> str:
    if char in LATEX_SPECIAL_CHARACTERS:
        return LATEX_SPECIAL_CHARACTERS[char]
    if char in UNICODE_REPLACEMENTS:
        return UNICODE_REPLACEMENTS[char]
    return char if _is_printable(char) else ""


def escape_latex(value):
    """Recursively make strings inside dicts/lists safe for pdflatex.

    LaTeX special characters are escaped, accents typed as two characters are joined
    ("e" + U+0301 becomes "é"), and characters pdflatex cannot print are converted or dropped.
    """
    if isinstance(value, str):
        text = unicodedata.normalize("NFC", value)
        escaped = [(char, _escape_character(char)) for char in text]
        dropped = {char for char, out in escaped if not out and char.isprintable()}
        if dropped:
            left_out = "".join(sorted(dropped))
            logger.warning("Left out characters pdflatex cannot print: %s", left_out)
        return "".join(out for _, out in escaped)
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
    return _LATEX_COMMENT.sub(r"\1", latex)


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
            # Non-ASCII text as itself: a "\u00e9" escape copied into LaTeX is an accent command.
            "resume": json.dumps(escape_latex(draft.model_dump()), indent=2, ensure_ascii=False),
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
