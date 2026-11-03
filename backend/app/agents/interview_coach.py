"""Interview coach: an adaptive mock interview that pauses for each answer.

The graph asks a question retrieved from the interview-question store, then
interrupts. Each answer resumes it from the Postgres checkpoint (keyed by
thread ID), so a session can span many HTTP requests. Difficulty adapts per
topic, and weak answers may get an LLM-written follow-up.
"""

import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import Literal

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from app.core.checkpointer import get_checkpointer
from app.core.config import get_settings
from app.core.resources import get_chroma, get_llm, get_reranker

logger = logging.getLogger(__name__)

MAX_QUESTIONS_PER_SESSION = 15
CANDIDATE_QUESTIONS = 15
RERANKED_QUESTIONS = 5
RANDOM_PICK_FROM_TOP = 3
TOPIC_HISTORY_LENGTH = 5
INTERVIEWER = "Interviewer"
COMPANIES = ("amazon", "apple", "google", "meta", "microsoft", "netflix", "nvidia", "generic")
ALL_DIFFICULTIES = ["easy", "medium", "hard"]
STOP_WORDS = {"stop", "done", "end", "finish", "quit"}
GOOD_GRADES = {"excellent", "good"}
GRADES = ("excellent", "good", "moderate", "bad")
DEFAULT_GRADE = "moderate"
EVALUATION_ERROR = "error"
MAX_FOLLOW_UPS_PER_QUESTION = 2
NO_QUESTION_IDS = {"NONE", "ERROR", "DONE"}
DEFAULT_TOPICS = ["Data Structures & Algorithms", "System Design"]

# Skill-gap keywords and the interview topic each maps to.
TOPIC_FOR_KEYWORD = {
    "angular": "Frontend Frameworks",
    "react": "Frontend Frameworks",
    "vue": "Frontend Frameworks",
    "aws": "Cloud & DevOps",
    "gcp": "Cloud & DevOps",
    "google cloud": "Cloud & DevOps",
    "azure": "Cloud & DevOps",
    "docker": "Cloud & DevOps",
    "kubernetes": "Cloud & DevOps",
    "python": "Programming Languages",
    "java": "Programming Languages",
    "go": "Programming Languages",
    "javascript": "Programming Languages",
    "typescript": "Programming Languages",
    "sql": "Databases",
    "postgresql": "Databases",
    "mongodb": "Databases",
    "system design": "System Design",
    "api": "System Design",
    "microservices": "System Design",
    "css": "Frontend Basics",
    "html": "Frontend Basics",
    "bootstrap": "Frontend Basics",
}
SKILL_GAPS_TO_MAP = 15

InterviewFocus = Literal["all", "proficient", "gaps"]


class InterviewAgentState(MessagesState):
    interview_focus: InterviewFocus
    company_focus: str
    topic_list: list[str]
    evaluation_report: list[dict]
    covered_questions: list[str]
    current_question_id: str
    current_question_topic: str
    current_question_difficulty: str
    start_time: datetime
    question_count: int
    topic_performance: dict[str, list[dict]]
    current_difficulty_per_topic: dict[str, list[str]]
    follow_up_count: int


EVALUATION_PROMPT = PromptTemplate.from_template(
    """You are a senior technical interviewer. Evaluate this answer:

Question: {question}
Candidate Answer: {answer}

Provide:
GRADE: [bad/moderate/good/excellent]
SKILL: [primary technical skill tested]
FEEDBACK: [2-3 sentences]"""
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize this interview performance concisely:\n\n{report}"
)

ACTION_CHOICE_PROMPT = PromptTemplate.from_template(
    """You are an interview agent's routing logic.

The user was asked: {question}
The user answered: {answer}
The evaluation was: {grade}

Should you:
1. Ask a probing follow-up (the answer was incomplete, vague or 'moderate'), or
2. Move on to a new question (the answer was strong, or bad and unsalvageable)?

Ask a follow-up only when it is really necessary.
Respond with ONLY the word "FOLLOW_UP" or "NEW_QUESTION"."""
)

FOLLOW_UP_PROMPT = PromptTemplate.from_template("""You are a technical interviewer.

Original Question: {question}
Candidate's Answer: {answer}
Your internal evaluation: {grade}

Probe deeper with a SINGLE, concise follow-up question based on their answer.
Do not say "Good answer" or "Okay"; just ask the question.
Example: if the answer was about sharding, ask "How would you handle a hot shard?"

Your follow-up question:""")


_KEYWORDS_LONGEST_FIRST = sorted(TOPIC_FOR_KEYWORD, key=len, reverse=True)


def _has_word(text: str, keyword: str) -> bool:
    """Whole-word match, so 'go' is not found in 'mongodb' or 'algorithms'."""
    pattern = rf"(?<![a-z0-9+#]){re.escape(keyword)}(?![a-z0-9+#])"
    return re.search(pattern, text.lower()) is not None


def topics_for_skill_gaps(skill_gaps: list[str]) -> list[str]:
    """Map skill gaps to interview topics; System Design is always included."""
    topics = set()
    for gap in skill_gaps[:SKILL_GAPS_TO_MAP]:
        topic = next(
            (TOPIC_FOR_KEYWORD[kw] for kw in _KEYWORDS_LONGEST_FIRST if _has_word(gap, kw)), None
        )
        if topic:
            topics.add(topic)
    if not topics:
        return list(DEFAULT_TOPICS)
    return sorted(topics | {"System Design"})


# --- Difficulty adaptation ---


def _harder(difficulties: list[str]) -> list[str]:
    return difficulties if set(difficulties) == {"hard"} else ["hard", "medium"]


def _easier(difficulties: list[str]) -> list[str]:
    return difficulties if set(difficulties) == {"easy"} else ["easy", "medium"]


def adapt_difficulty(topic_performance: dict, current: dict, topics: list[str]) -> dict:
    """Adapt from the topic's last two answers.

    Both bad on hard questions -> easier; both excellent on easy ones -> harder.
    """
    updated = dict(current)
    for topic in (t.lower() for t in topics):
        recent = topic_performance.get(topic, [])[-2:]
        if len(recent) < 2:
            continue
        difficulties = current.get(topic, ALL_DIFFICULTIES)
        if all(e["difficulty"] == "hard" and e["evaluation"] == "bad" for e in recent):
            updated[topic] = _easier(difficulties)
        elif all(e["difficulty"] == "easy" and e["evaluation"] == "excellent" for e in recent):
            updated[topic] = _harder(difficulties)
        if updated.get(topic, difficulties) != difficulties:
            logger.info("Difficulty for %s: %s -> %s", topic, difficulties, updated[topic])
    return updated


# --- Question retrieval ---


def _question_store():
    return get_chroma(get_settings().interview_chroma_dir)


def _retriever(metadata_filter: dict) -> BaseRetriever:
    base = _question_store().as_retriever(
        search_kwargs={"k": CANDIDATE_QUESTIONS, "filter": metadata_filter}
    )
    reranker = get_reranker(top_n=RERANKED_QUESTIONS)
    if reranker is None:
        return base
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)


def _companies_for(company_focus: str) -> list[str]:
    return list(COMPANIES) if company_focus == "all" else [company_focus]


def _topic_condition(topics: list[str], difficulty_map: dict | None) -> dict:
    """Topic filter; with a difficulty map, each topic keeps its own allowed difficulties."""
    names = [t.lower() for t in topics]
    if difficulty_map is None:
        return {"topic": {"$in": names}}
    per_topic = [
        {
            "$and": [
                {"topic": name},
                {"difficulty": {"$in": difficulty_map.get(name, ALL_DIFFICULTIES)}},
            ]
        }
        for name in names
    ]
    return per_topic[0] if len(per_topic) == 1 else {"$or": per_topic}


def _question_filter(state: InterviewAgentState, difficulty_map: dict | None) -> dict:
    conditions = [
        _topic_condition(state["topic_list"], difficulty_map),
        {"company": {"$in": _companies_for(state["company_focus"])}},
    ]
    if state.get("covered_questions"):
        conditions.append({"question_id": {"$nin": state["covered_questions"]}})
    return {"$and": conditions}


def _retrieve_questions(state: InterviewAgentState, difficulty_map: dict):
    """Search with each topic's difficulties, then without difficulty limits."""
    query = f"{state['company_focus']} interview questions on {', '.join(state['topic_list'])}"
    for allowed in (difficulty_map, None):
        try:
            results = _retriever(_question_filter(state, allowed)).invoke(query)
        except Exception as error:  # Chroma/Cohere failures: try the relaxed filter
            logger.warning("Question retrieval failed: %s", error)
            continue
        if results:
            return results
    return []


# --- Graph nodes ---


def start_interview_node(state: InterviewAgentState) -> dict:
    if not state.get("topic_list"):
        raise ValueError("topic_list is required")
    config_text = (
        f"Interview configuration\nFocus: {state['interview_focus']}\n"
        f"Company: {state['company_focus']}\nTopics: {', '.join(state['topic_list'])}"
    )
    return {
        "messages": [SystemMessage(content=config_text)],
        "start_time": datetime.now(),
        "question_count": 0,
        "current_question_id": "",
        "current_question_topic": "",
        "current_question_difficulty": "",
        "covered_questions": [],
        "evaluation_report": [],
        "topic_performance": {},
        "current_difficulty_per_topic": {},
        "follow_up_count": 0,
    }


def generate_question_node(state: InterviewAgentState) -> dict:
    difficulty_map = adapt_difficulty(
        state.get("topic_performance", {}),
        state.get("current_difficulty_per_topic", {}),
        state["topic_list"],
    )
    results = _retrieve_questions(state, difficulty_map)
    if not results:
        return {
            "messages": [SystemMessage(content="No more questions available.")],
            "current_question_id": "NONE",
        }
    chosen = random.choice(results[:RANDOM_PICK_FROM_TOP])
    return {
        "messages": [AIMessage(content=chosen.page_content.strip(), name=INTERVIEWER)],
        "current_question_id": chosen.metadata["question_id"],
        "current_question_topic": chosen.metadata.get("topic", "general").lower(),
        "current_question_difficulty": chosen.metadata.get("difficulty", "medium"),
        "question_count": state.get("question_count", 0) + 1,
        "current_difficulty_per_topic": difficulty_map,
        "follow_up_count": 0,
    }


def ask_follow_up_node(state: InterviewAgentState) -> dict:
    """LLM-written follow-up on the same question; does not count as a new question."""
    last = state["evaluation_report"][-1]
    try:
        prompt = FOLLOW_UP_PROMPT.format(
            question=last["question"], answer=last["answer"], grade=last["evaluation"]
        )
        follow_up = str(get_llm().invoke(prompt).content).strip()
    except Exception as error:  # provider errors: fall back to moving on
        logger.warning("Follow-up generation failed: %s", error)
        follow_up = "Let's move on."
    return {
        "messages": [AIMessage(content=follow_up, name=INTERVIEWER)],
        "follow_up_count": state.get("follow_up_count", 0) + 1,
    }


def wait_for_answer_node(state: InterviewAgentState) -> dict:
    """Pause until resumed with Command(resume={"messages": [HumanMessage(...)]})."""
    return interrupt("Waiting for the candidate's answer")


def _last_question_and_answer(messages: list[BaseMessage]) -> tuple[str, str] | None:
    answer = question = None
    for message in reversed(messages):
        if answer is None and isinstance(message, HumanMessage):
            answer = message.content
        elif answer is not None and isinstance(message, AIMessage) and message.name == INTERVIEWER:
            question = message.content
            break
    # An empty answer is still an answer: it gets graded rather than ignored.
    return (question, answer) if question is not None and answer is not None else None


def _parse_field(name: str, text: str, default: str, flags: int = 0) -> str:
    """Value after 'NAME:', tolerating Markdown emphasis such as '**NAME:** value'."""
    match = re.search(rf"{name}[*_\s]*:[*_\s]*(.+)", text, re.IGNORECASE | flags)
    return match.group(1).strip().strip("*_ ") if match else default


def _parse_grade(text: str) -> str:
    value = _parse_field("GRADE", text, "").lower()
    return next((g for g in GRADES if re.search(rf"\b{g}\b", value)), DEFAULT_GRADE)


def _grade_answer(question: str, answer: str) -> tuple[str, str, str]:
    """Return (grade, skill, feedback) from the LLM evaluation."""
    try:
        text = str(
            get_llm().invoke(EVALUATION_PROMPT.format(question=question, answer=answer)).content
        )
    except Exception as error:  # provider errors: record and carry on
        logger.warning("Answer evaluation failed: %s", error)
        return EVALUATION_ERROR, "unknown", str(error)
    return (
        _parse_grade(text),
        _parse_field("SKILL", text, "general").lower(),
        _parse_field("FEEDBACK", text, "No feedback.", re.DOTALL),
    )


def evaluate_answer_node(state: InterviewAgentState) -> dict:
    """Grade the answer; a follow-up answer updates the previous question's entry."""
    pair = _last_question_and_answer(state["messages"])
    if pair is None or _user_asked_to_stop(state["messages"]):
        return {}
    question, answer = pair
    grade, skill, feedback = _grade_answer(question, answer)
    question_id = state["current_question_id"]
    topic = state.get("current_question_topic", "general").lower()
    now = datetime.now().isoformat()

    report = [dict(entry) for entry in state.get("evaluation_report", [])]
    performance = {t: list(entries) for t, entries in state.get("topic_performance", {}).items()}
    topic_entries = performance.setdefault(topic, [])
    is_follow_up = bool(report) and report[-1]["question_id"] == question_id

    if is_follow_up:
        last = report[-1]
        last.update(evaluation=grade, feedback=feedback, timestamp=now)
        last["answer"] = f"{last.get('answer', '')}\n[FOLLOW-UP]: {answer}"
        if topic_entries and topic_entries[-1]["question_id"] == question_id:
            topic_entries[-1] = {
                **topic_entries[-1],
                "evaluation": grade,
                "timestamp": now,
            }
    else:
        difficulty = state.get("current_question_difficulty", "medium")
        report.append(
            {
                "question_id": question_id, "question": question, "answer": answer,
                "evaluation": grade, "feedback": feedback, "skill": skill,
                "timestamp": now, "topic": topic, "difficulty": difficulty,
            }
        )  # fmt: skip
        topic_entries.append(
            {
                "question_id": question_id,
                "evaluation": grade,
                "difficulty": difficulty,
                "timestamp": now,
            }
        )
        performance[topic] = topic_entries[-TOPIC_HISTORY_LENGTH:]

    return {
        "messages": [SystemMessage(content=f"Grade: {grade.upper()}\nFeedback: {feedback}")],
        "evaluation_report": report,
        "topic_performance": performance,
    }


def update_coverage_node(state: InterviewAgentState) -> dict:
    current = state.get("current_question_id")
    covered = state.get("covered_questions", [])
    if not current or current in NO_QUESTION_IDS or current in covered:
        return {}
    return {"covered_questions": [*covered, current]}


def _user_asked_to_stop(messages: list[BaseMessage]) -> bool:
    last_answer = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    return last_answer is not None and str(last_answer.content).lower().strip() in STOP_WORDS


def decide_action_router(
    state: InterviewAgentState,
) -> Literal["ask_follow_up", "update_coverage"]:
    if _user_asked_to_stop(state["messages"]) or not state.get("evaluation_report"):
        return "update_coverage"
    last = state["evaluation_report"][-1]
    if last["question_id"] != state.get("current_question_id"):
        return "update_coverage"
    out_of_follow_ups = state.get("follow_up_count", 0) >= MAX_FOLLOW_UPS_PER_QUESTION
    if last["evaluation"] in GOOD_GRADES | {EVALUATION_ERROR} or out_of_follow_ups:
        return "update_coverage"
    try:
        prompt = ACTION_CHOICE_PROMPT.format(
            question=last["question"], answer=last["answer"], grade=last["evaluation"]
        )
        choice = str(get_llm().invoke(prompt).content).upper()
    except Exception as error:  # provider errors: default to a new question
        logger.warning("Follow-up decision failed: %s", error)
        return "update_coverage"
    return "ask_follow_up" if "FOLLOW_UP" in choice else "update_coverage"


def route_after_question(
    state: InterviewAgentState,
) -> Literal["wait_for_answer", "synthesize_and_update"]:
    out_of_questions = state.get("current_question_id") in {"NONE", "ERROR"}
    return "synthesize_and_update" if out_of_questions else "wait_for_answer"


def should_continue_router(
    state: InterviewAgentState,
) -> Literal["generate_question", "synthesize_and_update"]:
    finished = (
        _user_asked_to_stop(state["messages"])
        or state.get("current_question_id") in {"NONE", "ERROR"}
        or state.get("question_count", 0) >= MAX_QUESTIONS_PER_SESSION
    )
    return "synthesize_and_update" if finished else "generate_question"


def synthesize_and_update_node(state: InterviewAgentState) -> dict:
    report = state.get("evaluation_report", [])
    if not report:
        summary = "No questions were answered in this session."
    else:
        lines = "\n".join(f"- {e['skill']}: {e['evaluation']} | {e['feedback']}" for e in report)
        try:
            summary = str(get_llm().invoke(SUMMARY_PROMPT.format(report=lines)).content)
        except Exception as error:  # provider errors: plain fallback summary
            logger.warning("Interview summary failed: %s", error)
            summary = f"Completed {len(report)} questions."
    return {
        "messages": [SystemMessage(content=f"Interview complete!\n\n{summary}")],
        "current_question_id": "DONE",
    }


def build_interview_graph() -> StateGraph:
    graph = StateGraph(InterviewAgentState)
    graph.add_node("start_interview", start_interview_node)
    graph.add_node("generate_question", generate_question_node)
    graph.add_node("wait_for_answer", wait_for_answer_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("ask_follow_up", ask_follow_up_node)
    graph.add_node("update_coverage", update_coverage_node)
    graph.add_node("synthesize_and_update", synthesize_and_update_node)

    graph.add_edge(START, "start_interview")
    graph.add_edge("start_interview", "generate_question")
    graph.add_conditional_edges("generate_question", route_after_question)
    graph.add_edge("wait_for_answer", "evaluate_answer")
    graph.add_conditional_edges("evaluate_answer", decide_action_router)
    graph.add_edge("ask_follow_up", "wait_for_answer")
    graph.add_conditional_edges("update_coverage", should_continue_router)
    graph.add_edge("synthesize_and_update", END)
    return graph


@cache
def get_interview_graph() -> CompiledStateGraph:
    return build_interview_graph().compile(checkpointer=get_checkpointer())


# --- Session API ---


@dataclass(frozen=True)
class InterviewTurn:
    """What the candidate sees after each step of the interview."""

    question: str | None
    feedback: str | None
    finished: bool
    summary: str | None
    question_count: int


def _turn_from(result: dict) -> InterviewTurn:
    messages = result["messages"]
    finished = "__interrupt__" not in result
    last_answer_index = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
    )
    since_answer = messages[last_answer_index + 1 :]
    feedback = next(
        (
            m.content
            for m in since_answer
            if isinstance(m, SystemMessage) and m.content.startswith("Grade:")
        ),
        None,
    )
    question = None
    if not finished:
        question = next(
            (
                m.content
                for m in reversed(messages)
                if isinstance(m, AIMessage) and m.name == INTERVIEWER
            ),
            None,
        )
    return InterviewTurn(
        question=question,
        feedback=feedback,
        finished=finished,
        summary=messages[-1].content if finished else None,
        question_count=result.get("question_count", 0),
    )


class InterviewStateError(RuntimeError):
    """The thread is not in a state that allows this call."""


def _thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def start_interview(
    thread_id: str,
    topics: list[str],
    company_focus: str,
    focus: InterviewFocus = "gaps",
) -> InterviewTurn:
    """Begin a session on a new `thread_id`; returns the first question."""
    if not topics:
        raise ValueError("topic_list is required")
    if get_interview_graph().get_state(_thread(thread_id)).values:
        raise InterviewStateError(f"Interview {thread_id!r} already exists")
    initial = {
        "interview_focus": focus,
        "company_focus": company_focus,
        "topic_list": topics,
        "messages": [],
    }
    return _turn_from(get_interview_graph().invoke(initial, config=_thread(thread_id)))


def submit_answer(thread_id: str, answer: str) -> InterviewTurn:
    """Resume the paused session with the candidate's answer ('stop' ends it)."""
    if not get_interview_graph().get_state(_thread(thread_id)).next:
        raise InterviewStateError(f"Interview {thread_id!r} is not waiting for an answer")
    resume = Command(resume={"messages": [HumanMessage(content=answer)]})
    return _turn_from(get_interview_graph().invoke(resume, config=_thread(thread_id)))


def evaluation_report(thread_id: str) -> list[dict]:
    snapshot = get_interview_graph().get_state(_thread(thread_id))
    return snapshot.values.get("evaluation_report", [])
