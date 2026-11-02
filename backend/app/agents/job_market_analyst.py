"""Job market analyst: finds postings for a role and computes the user's skill gaps.

Pipeline: search the cached postings in Chroma, scrape fresh postings from
LinkedIn (and cache them), then compare required skills with the user's.
"""

import logging
import re
from functools import cache
from typing import Any, TypedDict

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import get_settings
from app.core.resources import get_chroma, get_reranker
from app.services import job_scraper

logger = logging.getLogger(__name__)

CANDIDATE_POSTINGS = 15
RERANKED_POSTINGS = 5
POSTINGS_TO_SCRAPE = 5
MIN_SKILL_LENGTH = 3
DEFAULT_TITLE = "Untitled role"

_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
_TITLE_LINE = re.compile(r"(?:job title|title|position):\s*([^\n\r]+)", re.IGNORECASE)


class JobMarketState(TypedDict, total=False):
    role: str
    location: str
    user_skills: list[str]
    job_data: list[dict]
    skill_gap_data: dict


def _market_store():
    return get_chroma(get_settings().market_chroma_dir)


def _retriever() -> BaseRetriever:
    base = _market_store().as_retriever(search_kwargs={"k": CANDIDATE_POSTINGS})
    reranker = get_reranker(top_n=RERANKED_POSTINGS)
    if reranker is None:
        return base
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)


def _posting_to_document(posting: dict[str, Any]) -> Document:
    content = "\n".join(
        [
            f"Job Title: {posting.get('title', 'N/A')}",
            f"Company: {posting.get('company', 'N/A')}",
            f"Location: {posting.get('location', 'N/A')}",
            f"Seniority Level: {posting.get('seniority level', 'N/A')}",
            f"Employment Type: {posting.get('employment type', 'N/A')}",
            f"Job Function: {posting.get('job function', 'N/A')}",
            f"Industries: {posting.get('industries', 'N/A')}",
            f"Skills Required: {', '.join(posting.get('skills required', []))}",
        ]
    )
    metadata = {
        "source": posting.get("link", "N/A"),
        "company": posting.get("company", "N/A"),
        "location": posting.get("location", "N/A"),
        "seniority": posting.get("seniority level", "N/A"),
        "title": posting.get("title", "N/A"),
    }
    return Document(page_content=content, metadata=metadata)


def _locations_match(wanted: str, found: str) -> bool:
    return wanted in found or found in wanted


def _skills_from_content(content: str) -> list[str]:
    if "skills required:" not in content:
        return []
    skills_line = content.split("skills required:")[-1]
    skills = (skill.strip().strip(",.") for skill in skills_line.split(","))
    return [skill for skill in skills if len(skill) >= MIN_SKILL_LENGTH]


def _title_from(document: Document) -> str:
    title = document.metadata.get("title", "N/A")
    if title.lower() != "n/a":
        return title
    match = _TITLE_LINE.search(document.page_content)
    return match.group(1).strip().title() if match else DEFAULT_TITLE


def _document_to_job(document: Document) -> dict:
    return {
        "title": _title_from(document),
        "company": document.metadata.get("company", "N/A"),
        "location": document.metadata.get("location", "N/A"),
        "skills_required": _skills_from_content(document.page_content.lower()),
    }


def _dedupe_jobs(jobs: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for job in jobs:
        key = (job["company"], tuple(sorted(s.lower() for s in job["skills_required"])))
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def search_cached_postings(state: JobMarketState) -> JobMarketState:
    """Find previously scraped postings for the role in the requested location."""
    wanted_location = state["location"].lower()
    documents = _retriever().invoke(f"{state['role']} jobs in {state['location']}")
    jobs = [
        _document_to_job(doc)
        for doc in documents
        if _locations_match(wanted_location, doc.metadata.get("location", "").lower())
    ]
    return {"job_data": _dedupe_jobs(jobs)}


def _scrape_posting(summary: dict) -> tuple[dict, Document] | None:
    try:
        details = job_scraper.fetch_job_description(summary["link"])
    except job_scraper.ScraperError as error:
        logger.warning("Skipping posting: %s", error)
        return None
    posting = {**summary, **details}
    job = {
        "title": summary.get("title", DEFAULT_TITLE),
        "company": summary.get("company", "N/A"),
        "location": summary.get("location", "N/A"),
        "skills_required": details.get("skills required", []),
    }
    return job, _posting_to_document(posting)


def scrape_fresh_postings(state: JobMarketState) -> JobMarketState:
    """Scrape live postings, cache them in Chroma, and add them to the results."""
    summaries = job_scraper.fetch_jobs(state["role"], state["location"])
    scraped = [
        result
        for summary in summaries[:POSTINGS_TO_SCRAPE]
        if summary.get("link") and (result := _scrape_posting(summary))
    ]
    if scraped:
        chunks = _text_splitter.split_documents([document for _, document in scraped])
        _market_store().add_documents(chunks)
    return {"job_data": state.get("job_data", []) + [job for job, _ in scraped]}


def _normalize_skill(skill: str) -> str:
    return re.sub(r"[\s-]+", " ", skill.lower().strip(".,;-"))


def analyze_skill_gaps(state: JobMarketState) -> JobMarketState:
    """Compare the skills each posting asks for with the user's skills."""
    jobs = state.get("job_data", [])
    user_skills = {_normalize_skill(s) for s in state.get("user_skills", [])}

    gaps_per_job = []
    for job in jobs:
        required = {_normalize_skill(s) for s in job.get("skills_required", []) if s}
        gaps = sorted(required - user_skills)
        gaps_per_job.append(
            {
                "title": job.get("title", "N/A"),
                "company": job.get("company", "N/A"),
                "gaps": gaps,
                "gap_count": len(gaps),
            }
        )

    all_required = sorted(
        {_normalize_skill(s) for job in jobs for s in job.get("skills_required", []) if s}
    )
    total_gaps = sum(job["gap_count"] for job in gaps_per_job)
    analysis = {
        "role": state["role"],
        "location": state["location"],
        "user_current_skills": state.get("user_skills", []),
        "all_required_skills": all_required,
        "skill_gaps": [skill for skill in all_required if skill not in user_skills],
        "gaps_per_job": gaps_per_job,
        "average_gaps_per_job": round(total_gaps / len(jobs), 1) if jobs else 0,
    }
    return {"skill_gap_data": analysis}


@cache
def get_job_market_graph() -> CompiledStateGraph:
    graph = StateGraph(JobMarketState)
    graph.add_node("search_cached_postings", search_cached_postings)
    graph.add_node("scrape_fresh_postings", scrape_fresh_postings)
    graph.add_node("analyze_skill_gaps", analyze_skill_gaps)
    graph.add_edge(START, "search_cached_postings")
    graph.add_edge("search_cached_postings", "scrape_fresh_postings")
    graph.add_edge("scrape_fresh_postings", "analyze_skill_gaps")
    graph.add_edge("analyze_skill_gaps", END)
    return graph.compile()


def initial_state(role: str, location: str, user_skills: list[str]) -> JobMarketState:
    return {
        "role": role,
        "location": location,
        "user_skills": user_skills,
        "job_data": [],
    }


def run_job_market_analysis(role: str, location: str, user_skills: list[str]) -> JobMarketState:
    """Run the whole pipeline and return the final state (job_data, skill_gap_data)."""
    return get_job_market_graph().invoke(initial_state(role, location, user_skills))
