"""Extracts a user's technical skills from a resume file or a GitHub profile."""

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.resources import get_llm
from app.services import github
from app.services.documents import read_document_text
from app.services.job_scraper import extract_skills as match_known_skills

logger = logging.getLogger(__name__)

RESUME_SYSTEM_PROMPT = """You are an expert tech recruiter and skill extractor.
Read the resume and extract all relevant technical skills: programming languages,
frameworks, databases, tools and technical concepts.
Return only a single comma-separated list, with no explanation, categories or preamble."""

GITHUB_SYSTEM_PROMPT = """You are an expert tech skill extractor.
Create one deduplicated list of technical skills from two sources:
1. Repo languages: languages detected in the user's code.
2. Profile README: the user's self-described profile.
Return only a single comma-separated list, with no explanation, categories or preamble."""


def _parse_skill_list(text: str) -> list[str]:
    skills = (skill.strip().strip("'\"") for skill in text.split(","))
    return sorted({skill for skill in skills if skill}, key=str.lower)


def _ask_llm_for_skills(system_prompt: str, user_prompt: str) -> list[str] | None:
    """Ask the LLM for a comma-separated skill list; None if the call fails."""
    try:
        reply = get_llm().invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except Exception as error:  # network/auth/rate-limit errors from the provider
        logger.warning("Skill extraction LLM call failed: %s", error)
        return None
    return _parse_skill_list(str(reply.content))


def extract_skills_from_text(resume_text: str) -> list[str]:
    llm_skills = _ask_llm_for_skills(
        RESUME_SYSTEM_PROMPT, f"Here is the resume text:\n\n{resume_text}"
    )
    return llm_skills if llm_skills is not None else match_known_skills(resume_text)


def extract_skills_from_resume(path: Path) -> list[str]:
    return extract_skills_from_text(read_document_text(path))


def extract_skills_from_github(username: str) -> list[str]:
    languages = github.fetch_repo_languages(github.fetch_repos(username))
    readme = github.fetch_profile_readme(username) or "No README content provided."
    user_prompt = f'Repo Languages: "{", ".join(languages)}"\nProfile README: "{readme}"'
    llm_skills = _ask_llm_for_skills(GITHUB_SYSTEM_PROMPT, user_prompt)
    return llm_skills if llm_skills is not None else languages
