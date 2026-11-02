"""Curriculum architect: builds a personalized learning path for a user's skill gaps.

Graph outline:
  planner -> for each skill gap:
                 search the course catalog (Chroma) -> web search + store if empty
                 -> parallel checks (prerequisites, interests, workload)
                 -> aggregate scores -> recommend top courses (or alternatives)
          -> architect assembles the path via a tool call -> critic reviews
          -> revise until approved or MAX_REVISIONS is reached.
"""

import json
import logging
import operator
import uuid
from functools import cache
from pathlib import Path
from pprint import pformat
from typing import Annotated, Any, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.resources import get_chroma, get_llm, get_web_search

logger = logging.getLogger(__name__)

CATALOG_CANDIDATES = 10
COURSES_PER_SKILL = 3
MAX_ARCHITECT_ATTEMPTS = 2
MAX_REVISIONS = 2
NO_PREREQUISITE = 0
NO_INTEREST_OVERLAP_SCORE = 0.5
INTEREST_SCORE_BOOST = 1.5
MAX_SCORE = 10
MAX_DIFFICULTY_BY_LEVEL = {"Beginner": 2, "Intermediate": 4, "Advanced": 5}
DEFAULT_MAX_DIFFICULTY = 4
RECURSION_LIMIT = 300


class Course(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    prereqs: list[int] = Field(default_factory=list, description="Course IDs (use 0 if none)")
    difficulty: int = Field(..., ge=1, le=5, description="1=easy, 5=hard")
    topics: list[str] = Field(default_factory=list)
    url: str
    description: str


class NewCourseList(BaseModel):
    courses: list[Course]


class UserProfile(BaseModel):
    interests: list[str]
    skill_level: str = Field(..., description="'Beginner', 'Intermediate' or 'Advanced'")
    past_courses: list[int] = Field(..., description="Course IDs from projects (use 0 if none)")


class PlannerOutput(BaseModel):
    user_profile: UserProfile
    skills_to_find: list[str] = Field(
        ..., description="EVERY skill gap mentioned in the analysis (typically 5-20)."
    )


class CourseRecommendation(BaseModel):
    skill_name: str
    course_title: str
    course_url: str
    description: str


class PersonalizedLearningPath(BaseModel):
    user_summary: str
    skill_gaps: list[str]
    recommendations: list[CourseRecommendation]


class Critique(BaseModel):
    is_approved: bool
    revisions_needed: str


SEED_COURSES_FILE = Path(__file__).parent / "data" / "seed_courses.json"


# --- Course catalog (Chroma) ---


def _course_to_document(course: Course) -> Document:
    content = (
        f"Course: {course.title}. Topics: {', '.join(course.topics)}. "
        f"Description: {course.description}"
    )
    metadata = course.model_dump()
    metadata["prereqs"] = ",".join(map(str, course.prereqs))
    metadata["topics"] = ",".join(course.topics)
    return Document(page_content=content, metadata=metadata)


def _document_to_course(document: Document) -> Course:
    metadata = dict(document.metadata)
    metadata["prereqs"] = [int(p) for p in str(metadata.get("prereqs", "")).split(",") if p]
    metadata["topics"] = [t.strip() for t in str(metadata.get("topics", "")).split(",") if t]
    return Course(**metadata)


def _store_courses(store: Chroma, courses: list[Course]) -> None:
    store.add_documents([_course_to_document(c) for c in courses], ids=[c.id for c in courses])


def _load_seed_courses() -> list[Course]:
    return [Course(**course) for course in json.loads(SEED_COURSES_FILE.read_text())]


@cache
def get_course_catalog() -> Chroma:
    """The course catalog store, seeded from SEED_COURSES_FILE on first use."""
    store = get_chroma(get_settings().courses_chroma_dir)
    seed_courses = _load_seed_courses()
    existing_ids = set(store.get(ids=[c.id for c in seed_courses])["ids"])
    missing = [c for c in seed_courses if c.id not in existing_ids]
    if missing:
        logger.info("Seeding course catalog with %d courses", len(missing))
        _store_courses(store, missing)
    return store


# --- Graph state ---


def list_overwrite_reducer(existing: list | None, new_chunk: list | None) -> list:
    """Append like operator.add, but an explicit empty list resets the value."""
    if new_chunk == []:
        return []
    return (existing or []) + (new_chunk or [])


class CurriculumGraphState(TypedDict, total=False):
    skill_gap_analysis: str
    resume_inventory: str

    user_profile: UserProfile
    skills_to_find: list[str]

    current_skill: str | None
    course_catalog: list[Course]
    raw_search_results: list[dict]

    evaluation_logs: Annotated[list[dict[str, Any]], list_overwrite_reducer]
    all_found_courses: Annotated[list[dict[str, Any]], operator.add]
    evaluated_courses: list[dict[str, Any]]

    messages: Annotated[list[BaseMessage], operator.add]
    draft_path: PersonalizedLearningPath | None
    structured_critique: Critique | None
    revision_count: Annotated[int, operator.add]


# --- Prompts ---

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a senior career coach analyzing skill gaps and user background.

Your job:
1. Create a structured UserProfile from the resume inventory.
2. Extract ALL critical skills from the skill gap analysis (typically 5-20 skills).

Rules:
- If 10+ skills are listed as gaps, extract ALL of them.
- Prioritize skills that appear multiple times or are marked as critical.
- Include both technical skills and tools.
- Infer skill_level from the resume (Beginner/Intermediate/Advanced).
- Estimate past_courses from mentioned projects (use 0 if unclear).""",
        ),
        (
            "human",
            "Resume inventory:\n{resume}\n\nSkill gap analysis:\n{gaps}\n\n"
            "Extract ALL skill gaps mentioned.",
        ),
    ]
)

PROCESS_SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a data extraction expert. Convert web search results into Course objects.
- Create unique IDs (e.g. 'aws-udemy-architect').
- Estimate difficulty realistically (1-5).
- Extract prerequisites (use [0] if none) and ALL relevant topics.
- Every course MUST have a valid URL and description.
- Ignore results that are not courses.""",
        ),
        ("human", 'Process these search results for "{skill}":\n\n{search_results}'),
    ]
)

ARCHITECT_SYSTEM_PROMPT = """You are the Curriculum Architect. Assemble the FINAL personalized
learning path from the user profile and the vetted course list.

Call the submit_learning_path tool with arguments in EXACTLY this structure:
{
  "path": {
    "user_summary": "A 2-3 sentence summary of the learner.",
    "skill_gaps": ["skill1", "skill2"],
    "recommendations": [
      {"skill_name": "...", "course_title": "...", "course_url": "https://...",
       "description": "Short helpful description"}
    ]
  }
}

Rules:
- Always wrap everything in "path"; never rename fields or add new ones.
- Build skill_gaps from the unique skills in the vetted course list.
- Include EVERY vetted course in recommendations; do not filter."""

CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a senior curriculum quality reviewer.

Criteria: a clear 2-3 sentence user_summary; skill_gaps that match the recommendations;
at least one course per skill gap; coherent and actionable overall.

Be lenient. Approve when each gap has 1+ recommendations, the summary is clear and the
structure is valid. Reject only when >50% of gaps have no recommendation, the summary is
missing or nonsensical, the structure is broken, or recommendations do not align with
the gaps. Be specific but concise about needed improvements.""",
        ),
        ("human", "Review this learning path:\n\n{draft_path_json}"),
    ]
)


@tool("submit_learning_path")
def submit_learning_path(path: dict) -> str:
    """Submit the finalized learning path: {"path": {user_summary, skill_gaps, recommendations}}."""
    return "Learning path submitted successfully."


# --- Planning and iteration nodes ---


def entry_node(state: CurriculumGraphState) -> dict:
    return {
        "messages": [],
        "revision_count": 0,
        "all_found_courses": [],
        "evaluation_logs": [],
    }


def planner_node(state: CurriculumGraphState) -> dict:
    chain = PLANNER_PROMPT | get_llm().with_structured_output(PlannerOutput)
    output = chain.invoke(
        {"resume": state["resume_inventory"], "gaps": state["skill_gap_analysis"]}
    )
    logger.info("Planner found %d skills to cover", len(output.skills_to_find))
    return {
        "user_profile": output.user_profile,
        "skills_to_find": output.skills_to_find,
    }


def iteration_controller_node(state: CurriculumGraphState) -> dict:
    remaining = state.get("skills_to_find", [])
    if not remaining:
        return {"current_skill": None}
    current, *rest = remaining
    logger.info("Finding courses for %s (%d left)", current, len(rest))
    return {
        "current_skill": current,
        "skills_to_find": rest,
        "evaluation_logs": [],
        "course_catalog": [],
    }


def _course_mentions_any(course: Course, keywords: list[str]) -> bool:
    haystacks = [
        course.title.lower(),
        course.description.lower(),
        *(t.lower() for t in course.topics),
    ]
    return any(keyword in text for keyword in keywords for text in haystacks)


def fetch_course_catalog(state: CurriculumGraphState) -> dict:
    """Similar courses from the catalog, keeping those that mention the skill's keywords."""
    skill = state["current_skill"]
    documents = get_course_catalog().similarity_search(skill, k=CATALOG_CANDIDATES)
    courses = [_document_to_course(doc) for doc in documents]
    keywords = skill.lower().split()
    matching = [course for course in courses if _course_mentions_any(course, keywords)]
    return {"course_catalog": matching or courses}


def web_search_for_courses(state: CurriculumGraphState) -> dict:
    search = get_web_search()
    if search is None:
        return {"raw_search_results": []}
    query = f"best online courses for {state['current_skill']} udemy coursera edx"
    try:
        response = search.invoke({"query": query})
    except Exception as error:  # third-party API failures should not abort the path
        logger.warning("Course web search failed: %s", error)
        return {"raw_search_results": []}
    results = response.get("results", []) if isinstance(response, dict) else response
    return {"raw_search_results": results}


def process_and_store_courses(state: CurriculumGraphState) -> dict:
    """Turn search results into Course objects and add them to the catalog."""
    results = state.get("raw_search_results") or []
    if not results:
        return {"course_catalog": []}
    chain = PROCESS_SEARCH_PROMPT | get_llm().with_structured_output(NewCourseList)
    try:
        courses = chain.invoke(
            {"skill": state["current_skill"], "search_results": pformat(results)}
        ).courses
    except Exception as error:  # malformed structured output from the LLM
        logger.warning("Could not extract courses from search results: %s", error)
        return {"course_catalog": []}
    if courses:
        _store_courses(get_course_catalog(), courses)
    return {"course_catalog": courses}


# --- Parallel evaluation nodes ---


def start_evaluation(state: CurriculumGraphState) -> dict:
    return {}


def check_prerequisites(state: CurriculumGraphState) -> dict:
    completed = set(state["user_profile"].past_courses)
    logs = []
    for course in state["course_catalog"]:
        required = {p for p in course.prereqs if p != NO_PREREQUISITE}
        logs.append({"type": "prereq", "course_id": course.id, "pass": required <= completed})
    return {"evaluation_logs": logs}


def _interest_score(interests: set[str], topics: set[str]) -> float:
    overlap = interests & topics
    if not overlap:
        return NO_INTEREST_OVERLAP_SCORE
    return min(len(overlap) / len(interests) * INTEREST_SCORE_BOOST, 1.0)


def match_interests(state: CurriculumGraphState) -> dict:
    interests = {i.lower() for i in state["user_profile"].interests}
    logs = [
        {
            "type": "interest",
            "course_id": course.id,
            "score": round(_interest_score(interests, {t.lower() for t in course.topics}), 2),
        }
        for course in state["course_catalog"]
    ]
    return {"evaluation_logs": logs}


def estimate_workload(state: CurriculumGraphState) -> dict:
    level = state["user_profile"].skill_level
    max_difficulty = MAX_DIFFICULTY_BY_LEVEL.get(level, DEFAULT_MAX_DIFFICULTY)
    logs = [
        {
            "type": "workload",
            "course_id": c.id,
            "load_ok": c.difficulty <= max_difficulty,
        }
        for c in state["course_catalog"]
    ]
    return {"evaluation_logs": logs}


_LOG_FIELDS = {"prereq": ("prereq_pass", "pass"), "interest": ("interest_score", "score"),
               "workload": ("workload_ok", "load_ok")}  # fmt: skip


def aggregate_scores(state: CurriculumGraphState) -> dict:
    """Combine the three checks: interest score out of 10, zero if prereqs/workload fail."""
    courses = {course.id: course for course in state["course_catalog"]}
    checks: dict[str, dict] = {}
    for log in state["evaluation_logs"]:
        field, source = _LOG_FIELDS[log["type"]]
        checks.setdefault(log["course_id"], {})[field] = log[source]

    evaluated = []
    for course_id, details in checks.items():
        if course_id not in courses:
            continue
        eligible = details.get("prereq_pass", False) and details.get("workload_ok", False)
        score = details.get("interest_score", 0) * MAX_SCORE if eligible else 0
        evaluated.append(
            {"course": courses[course_id], "score": round(score, 2), "details": details}
        )
    return {"evaluated_courses": evaluated}


def _recommendations(state: CurriculumGraphState, label: str, candidates: list[dict]) -> dict:
    skill = state["current_skill"]
    recommendations = [
        {
            "skill_name": skill,
            "course_title": entry["course"].title,
            "course_url": entry["course"].url,
            "description": f"{label} for {skill} (Score: {entry['score']}/{MAX_SCORE}). "
            f"{entry['course'].description}",
        }
        for entry in candidates[:COURSES_PER_SKILL]
    ]
    return {"all_found_courses": recommendations}


def _ranked(state: CurriculumGraphState) -> list[dict]:
    return sorted(state["evaluated_courses"], key=lambda entry: entry["score"], reverse=True)


def recommend_enrollment(state: CurriculumGraphState) -> dict:
    eligible = [entry for entry in _ranked(state) if entry["score"] > 0]
    return _recommendations(state, "Best match", eligible)


def suggest_alternatives(state: CurriculumGraphState) -> dict:
    return _recommendations(state, "Alternative", _ranked(state))


# --- Architect / critic loop ---


def _first_prompt(state: CurriculumGraphState) -> list[BaseMessage]:
    courses = state["all_found_courses"]
    skills = {course.get("skill_name", "Unknown") for course in courses}
    content = (
        f"User profile:\n{state['user_profile'].model_dump_json(indent=2)}\n\n"
        f"Vetted courses ({len(courses)} total):\n{pformat(courses)}\n\n"
        f"Write a 2-3 sentence user_summary, list the {len(skills)} unique skills as "
        f"skill_gaps, include ALL {len(courses)} courses, then call submit_learning_path."
    )
    return [
        SystemMessage(content=ARCHITECT_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]


def architect_node(state: CurriculumGraphState) -> dict:
    """Ask the LLM to submit the path via the tool, nudging it once if it does not."""
    is_first_pass = state.get("revision_count", 0) == 0
    messages = _first_prompt(state) if is_first_pass else list(state["messages"])
    model = get_llm().bind_tools([submit_learning_path])
    response = None
    for _ in range(MAX_ARCHITECT_ATTEMPTS):
        response = model.invoke(messages)
        if getattr(response, "tool_calls", None):
            break
        messages += [
            response,
            HumanMessage(content="You must call the submit_learning_path tool. Try again."),
        ]
    new_messages = messages + [response] if is_first_pass else [response]
    return {"messages": new_messages}


def _path_from_tool_args(args) -> PersonalizedLearningPath:
    if isinstance(args, str):
        args = json.loads(args)
    path = args.get("path", args)
    if isinstance(path, str):
        path = json.loads(path)
    return PersonalizedLearningPath(**path)


def _rejected_submission(reason: str, tool_call_id: str) -> dict:
    return {
        "structured_critique": Critique(is_approved=False, revisions_needed=reason),
        "messages": [
            ToolMessage(content=f"Submission failed: {reason}", tool_call_id=tool_call_id)
        ],
        "revision_count": 1,
    }


def handle_submission_node(state: CurriculumGraphState) -> dict:
    tool_calls = getattr(state["messages"][-1], "tool_calls", None)
    if not tool_calls:
        reason = "Call submit_learning_path with user_summary, skill_gaps and ALL recommendations."
        return {
            "structured_critique": Critique(is_approved=False, revisions_needed=reason),
            "revision_count": 1,
        }
    call = tool_calls[0]
    try:
        draft = _path_from_tool_args(call["args"])
    except (ValueError, TypeError, KeyError) as error:
        return _rejected_submission(f"Could not parse your submission: {error}", call["id"])
    logger.info("Draft learning path has %d recommendations", len(draft.recommendations))
    return {
        "draft_path": draft,
        "messages": [ToolMessage(content="Learning path submitted.", tool_call_id=call["id"])],
    }


def critic_node(state: CurriculumGraphState) -> dict:
    chain = CRITIC_PROMPT | get_llm().with_structured_output(Critique)
    critique = chain.invoke({"draft_path_json": state["draft_path"].model_dump_json(indent=2)})
    return {"structured_critique": critique, "revision_count": 1}


def prepare_revision_node(state: CurriculumGraphState) -> dict:
    critique = state.get("structured_critique")
    feedback = critique.revisions_needed if critique else "Include ALL recommendations."
    content = (
        f"Reviewer feedback:\n{feedback}\n\nAddress it and resubmit with submit_learning_path."
    )
    return {"messages": [HumanMessage(content=content)]}


# --- Routing ---


def route_after_iteration(state: CurriculumGraphState) -> str:
    return "architect" if state["current_skill"] is None else "fetch_course_catalog"


def route_after_catalog(state: CurriculumGraphState) -> str:
    return "start_evaluation" if state["course_catalog"] else "web_search_for_courses"


def route_after_scoring(state: CurriculumGraphState) -> str:
    has_eligible = any(entry["score"] > 0 for entry in state["evaluated_courses"])
    return "recommend_enrollment" if has_eligible else "suggest_alternatives"


def route_after_submission(state: CurriculumGraphState) -> str:
    if state.get("draft_path"):
        return "critic"
    return END if state["revision_count"] >= MAX_REVISIONS else "prepare_revision"


def route_after_critique(state: CurriculumGraphState) -> str:
    if state["structured_critique"].is_approved or state["revision_count"] >= MAX_REVISIONS:
        return END
    return "prepare_revision"


@cache
def get_curriculum_graph() -> CompiledStateGraph:
    graph = StateGraph(CurriculumGraphState)
    nodes = [
        entry_node, planner_node, iteration_controller_node, fetch_course_catalog,
        web_search_for_courses, process_and_store_courses, start_evaluation,
        check_prerequisites, match_interests, estimate_workload, aggregate_scores,
        recommend_enrollment, suggest_alternatives, architect_node, handle_submission_node,
        critic_node, prepare_revision_node,
    ]  # fmt: skip
    for node in nodes:
        graph.add_node(node.__name__.removesuffix("_node"), node)

    parallel_checks = ["check_prerequisites", "match_interests", "estimate_workload"]
    graph.set_entry_point("entry")
    graph.add_edge("entry", "planner")
    graph.add_edge("planner", "iteration_controller")
    graph.add_conditional_edges(
        "iteration_controller",
        route_after_iteration,
        ["fetch_course_catalog", "architect"],
    )
    graph.add_conditional_edges(
        "fetch_course_catalog",
        route_after_catalog,
        ["start_evaluation", "web_search_for_courses"],
    )
    graph.add_edge("web_search_for_courses", "process_and_store_courses")
    for check in parallel_checks:
        graph.add_edge("process_and_store_courses", check)
        graph.add_edge("start_evaluation", check)
        graph.add_edge(check, "aggregate_scores")
    graph.add_conditional_edges(
        "aggregate_scores",
        route_after_scoring,
        ["recommend_enrollment", "suggest_alternatives"],
    )
    graph.add_edge("recommend_enrollment", "iteration_controller")
    graph.add_edge("suggest_alternatives", "iteration_controller")
    graph.add_edge("architect", "handle_submission")
    graph.add_conditional_edges(
        "handle_submission", route_after_submission, ["critic", "prepare_revision", END]
    )
    graph.add_conditional_edges("critic", route_after_critique, ["prepare_revision", END])
    graph.add_edge("prepare_revision", "architect")
    return graph.compile()


def build_learning_path(skill_gap_analysis: str, resume_inventory: str) -> CurriculumGraphState:
    """Run the full graph; the result's `draft_path` holds the learning path."""
    return get_curriculum_graph().invoke(
        {
            "skill_gap_analysis": skill_gap_analysis,
            "resume_inventory": resume_inventory,
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )


SOFT_SKILLS = {
    "communication", "teamwork", "collaboration", "problem solving", "scrum", "agile",
    "user experience", "user interface", "leadership", "presentation", "time management",
}  # fmt: skip


def plan_learning_path(
    skill_gaps: list[str], user_skills: list[str], target_role: str
) -> PersonalizedLearningPath | None:
    """Learning path for the technical gaps (soft skills are skipped when possible)."""
    technical = [gap for gap in skill_gaps if gap.lower() not in SOFT_SKILLS] or skill_gaps
    inventory = json.dumps({"skills": user_skills, "target_role": target_role}, indent=2)
    return build_learning_path("\n".join(technical), inventory).get("draft_path")
