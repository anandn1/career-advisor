import os
from typing import TypedDict, List, Literal, Annotated, Optional, Dict
import operator
import random
from datetime import datetime
import re
import uuid 
import time # Added for session ID

from langgraph.types import Command, interrupt
from langchain_core.messages import ( 
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage
)

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver


from langchain_cohere import CohereRerank
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)

# --- Project Imports ---
import sys

try:
    from settings import llm, embedding_function, DB_URL, COHERE_API_KEY
except ImportError:
    print("[FATAL ERROR] settings.py not found. Make sure it's in the correct path.")
    sys.exit(1)

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INTERVIEW_CHROMA_DIR = os.path.join(SCRIPT_DIR, "ingestion", "chroma_db_interview_questions")
MAX_QUESTIONS_PER_SESSION = 15
MESSAGE_HISTORY_LIMIT = 10

# --- 1. State Definition ---
class InterviewAgentState(MessagesState):
    """Extended MessagesState for interview-specific fields."""
    user_id: str
    interview_focus: Literal["all", "proficient", "gaps"]
    company_focus: str
    topic_list: List[str]
    
    # Store the full report and coverage list
    evaluation_report: List[dict]
    covered_questions: List[str]
    
    current_question_id: str
    current_question_topic: str  
    current_question_difficulty: str  
    start_time: datetime
    question_count: int
    topic_performance: Dict[str, List[dict]]  
    current_difficulty_per_topic: Dict[str, List[str]]


print("\n" + "="*70)
print("INITIALIZING INTERVIEW COACH DEPENDENCIES")
print("="*70)

# Vector DB
try:
    interview_db = Chroma(
        persist_directory=INTERVIEW_CHROMA_DIR,
        embedding_function=embedding_function
    )
    count = interview_db._collection.count()
    print(f"Interview DB: {count} questions loaded")
except Exception as e:
    print(f"ChromaDB Error: {e}")
    raise RuntimeError(f"Failed to load interview DB: {e}")

# Reranker
cohere_reranker = None
if COHERE_API_KEY:
    try:
        cohere_reranker = CohereRerank(
            cohere_api_key=COHERE_API_KEY, 
            model="rerank-english-v3.0"
        )
        print("CohereRerank initialized")
    except Exception as e:
        print(f"Reranker Warning: {e}")
else:
    print("No COHERE_API_KEY set. Reranker disabled.")

print("="*70)

# --- 3. LLM Prompts ---
EVALUATION_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are a senior technical interviewer. Evaluate this answer:

Question: {question}
Candidate Answer: {answer}

Provide:
GRADE: [bad/moderate/good/excellent]
SKILL: [primary technical skill tested]
FEEDBACK: [2-3 sentences]"""
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["report"],
    template="""Summarize this interview performance:

{report}

Provide a concise summary."""
)

ACTION_CHOICE_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "grade"],
    template="""You are an interview agent's routing logic.

The user was asked: {question}
The user answered: {answer}
The evaluation was: {grade}

Based on this evaluation, should you:
1. Ask a probing follow-up question (because the answer was incomplete, vague, or 'moderate').
2. Move on to a completely new question (because the answer was 'excellent', 'good', or 'bad' and unsalvageable).

Ask follow up question strictly when its necessary, avoid unnecessary follow up questions.

Important : Respond with ONLY the word "FOLLOW_UP" or "NEW_QUESTION"."""
)

FOLLOW_UP_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "grade"],
    template="""You are a technical interviewer.

Original Question: {question}
Candidate's Answer: {answer}
Your internal evaluation: {grade}

Your goal is to probe deeper. Ask a SINGLE, concise follow-up question based on their answer.
Do NOT say "Good answer" or "Okay". Just ask the follow-up question.

Example: If the answer was about 'sharding', a good follow-up is 'How would you handle a hot shard?'

Your follow-up question:"""
)

# --- Helper function for difficulty transitions ---
def get_next_difficulty_state(current_diffs: List[str], upgrade: bool = False, downgrade: bool = False) -> List[str]:
    """Return next difficulty state based on transition."""
    
    current_set = set(current_diffs)

    if upgrade:
        if current_set != {"hard"}:
            return ["hard", "medium"]
        else:
            return current_diffs
        
    elif downgrade:
        if current_set != {"easy"}:
            return ["easy", "medium"]
        else:
            return current_diffs
        
    return current_diffs

# --- 4. Graph Node Functions ---

def start_interview_node(state: InterviewAgentState) -> dict:
    """Initialize interview session."""
    print("\n" + "="*70)
    print("INTERVIEW SESSION STARTING")
    print("="*70)
    
    if not state.get("topic_list"):
        raise ValueError("topic_list is required")
    
    current_difficulty_per_topic = {}
    print(f"Initial difficulty settings: {current_difficulty_per_topic}")
    
    return {
        "messages": [
            SystemMessage(
                content=(
                    f"Interview Configuration\n"
                    f"Focus: {state['interview_focus']}\n"
                    f"Company: {state['company_focus']}\n"
                    f"Topics: {', '.join(state['topic_list'])}"
                )
            )
        ],
        "start_time": datetime.now(),
        "question_count": 0,
        "current_question_id": "",
        "current_question_topic": "",
        "current_question_difficulty": "",
        "covered_questions": [],
        "evaluation_report": [],
        "topic_performance": {},
        "current_difficulty_per_topic": current_difficulty_per_topic
    }


def generate_question_node(state: InterviewAgentState) -> dict:
    """RAG node to fetch the next question, using raw topics and $and filter."""
    print(f"\n--- Node: generate_question (Q#{state.get('question_count', 0) + 1}) ---")
    
    topics = state["topic_list"]
    company = state["company_focus"]
    covered_ids = state.get("covered_questions", [])
    
    topic_performance = state.get("topic_performance", {})
    current_difficulty_per_topic = state.get("current_difficulty_per_topic", {})
    difficulty_updates = {}
    messages_to_print = []
    
    for topic_original_case in topics:
        topic = topic_original_case.lower()
        
        recent_evals = topic_performance.get(topic, [])
        curr_diffs = current_difficulty_per_topic.get(topic, ["easy", "medium", "hard"])
        
        hard_evals = [e for e in recent_evals if e["difficulty"] == "hard"]
        if len(hard_evals) >= 2:
            last_two_hard = hard_evals[-2:]
            if all(e["evaluation"] == "bad" for e in last_two_hard):
                new_diffs = get_next_difficulty_state(curr_diffs, downgrade=True)
                if new_diffs != curr_diffs:
                    messages_to_print.append(f"\n{'='*70}\n>>> Difficulty decreased for {topic_original_case}: {curr_diffs} → {new_diffs}\n{'='*70}")
                    difficulty_updates[topic] = new_diffs

        easy_evals = [e for e in recent_evals if e["difficulty"] == "easy"]
        if len(easy_evals) >= 2 and topic not in difficulty_updates:
            last_two_easy = easy_evals[-2:]
            if all(e["evaluation"] == "excellent" for e in last_two_easy):
                new_diffs = get_next_difficulty_state(curr_diffs, upgrade=True)
                if new_diffs != curr_diffs:
                    messages_to_print.append(f"\n{'='*70}\n>>> Difficulty increased for {topic_original_case}: {curr_diffs} → {new_diffs}\n{'='*70}\n{'='*70}")
                    difficulty_updates[topic] = new_diffs

    updated_difficulty_map = {**current_difficulty_per_topic, **difficulty_updates}
    for msg in messages_to_print:
        print(msg)
    
    all_companies_list = ["amazon", "microsoft", "nvidia", "netflix", "meta", "google", "generic"]
    
    if company == "all":
        companies_to_search = all_companies_list
    elif company == "generic":
        companies_to_search = ["generic"]
    else:
        companies_to_search = [company]
    
    print(f"Searching for companies: {companies_to_search}")

    allowed_difficulties = set()
    for topic_original_case in topics:
        topic = topic_original_case.lower()
        diffs = updated_difficulty_map.get(topic, ["easy", "medium", "hard"])
        allowed_difficulties.update(diffs)
    
    allowed_difficulties = list(allowed_difficulties)
    
    filter_conditions = [
        {"topic": {"$in": [t.lower() for t in topics]}},
        {"company": {"$in": companies_to_search}},
        {"difficulty": {"$in": allowed_difficulties}},
    ]
    
    if covered_ids:
        filter_conditions.append({"question_id": {"$nin": covered_ids}})
    
    metadata_filter = {"$and": filter_conditions}
    print(f"Filter (Attempt 1): {metadata_filter}")
    
    results = []
    query_text = f"{company} interview questions on {', '.join(topics)}"
    
    try:
        base_retriever = interview_db.as_retriever(
            search_kwargs={"k": 15, "filter": metadata_filter}
        )
        
        if cohere_reranker:
            print("Using Cohere reranker...")
            compressor = ContextualCompressionRetriever(
                base_compressor=cohere_reranker, 
                base_retriever=base_retriever
            )
            results = compressor.invoke(query_text)
        else:
            print("Using vector search...")
            results = base_retriever.invoke(query_text)
    except Exception as e:
        print(f"Retrieval (Attempt 1) failed: {e}")
    
    if not results:
        print("No questions found with specific filter. Relaxing filter (ignoring difficulty)...")
        
        relaxed_filter_conditions = [
            {"topic": {"$in": [t.lower() for t in topics]}},
            {"company": {"$in": companies_to_search}},
        ]
        if covered_ids:
            relaxed_filter_conditions.append({"question_id": {"$nin": covered_ids}})
        
        relaxed_metadata_filter = {"$and": relaxed_filter_conditions}
        print(f"Filter (Attempt 2): {relaxed_metadata_filter}")
        
        try:
            base_retriever = interview_db.as_retriever(
                search_kwargs={"k": 15, "filter": relaxed_metadata_filter}
            )
            
            if cohere_reranker:
                print("Using Cohere reranker (Fallback)...")
                compressor = ContextualCompressionRetriever(
                    base_compressor=cohere_reranker, 
                    base_retriever=base_retriever
                )
                results = compressor.invoke(query_text)
            else:
                print("Using vector search (Fallback)...")
                results = base_retriever.invoke(query_text)
        except Exception as e:
            print(f"Retrieval (Attempt 2) failed: {e}")
            return {
                "messages": [SystemMessage(content=f"Error: {e}")],
                "current_question_id": "ERROR" 
            }

    if not results:
        print("No questions found even with relaxed filter!")
        return {
            "messages": [SystemMessage(content="No more questions available.")],
            "current_question_id": "NONE"
        }
    
    N = 3
    selectable_results = results[:N]
    next_doc = random.choice(selectable_results)
    
    next_question = next_doc.page_content.strip()
    next_id = next_doc.metadata["question_id"]
    next_topic = next_doc.metadata.get("topic", "general").lower()
    next_difficulty = next_doc.metadata.get("difficulty", "medium")
    
    print(f"Selected: {next_id}")
    print(f"Topic: {next_topic} | Difficulty: {next_difficulty}")
    
    messages = state.get("messages", [])
    if len(messages) > MESSAGE_HISTORY_LIMIT:
        messages = messages[-MESSAGE_HISTORY_LIMIT:]
    
    return {
        "messages": messages + [AIMessage(content=next_question, name="Interviewer")],
        "current_question_id": next_id,
        "current_question_topic": next_topic,
        "current_question_difficulty": next_difficulty,
        "question_count": state.get("question_count", 0) + 1,
        "current_difficulty_per_topic": updated_difficulty_map
    }

def ask_follow_up_node(state: InterviewAgentState) -> dict:
    """Generates an LLM-based follow-up question."""
    print("\n--- Node: ask_follow_up ---")
    
    last_eval = state.get("evaluation_report", [])[-1]
    question = last_eval["question"]
    answer = last_eval["answer"]
    grade = last_eval["evaluation"]
    
    try:
        prompt = FOLLOW_UP_PROMPT.format(question=question, answer=answer, grade=grade)
        response = llm.invoke(prompt)
        follow_up_question = response.content.strip()
    except Exception as e:
        print(f"Follow-up generation failed: {e}")
        follow_up_question = "Let's move on. "
    
    print(f"Asking follow-up: {follow_up_question}")
    
    messages = state["messages"]
        
    return {
        "messages": messages + [AIMessage(content=follow_up_question, name="Interviewer")]
    }


def wait_for_answer_node(state: InterviewAgentState) -> dict:
    """Pauses the graph to wait for the user's answer."""
    print("--- Node: wait_for_answer (PAUSING) ---")
    resumed_value = interrupt(value="Waiting for user's answer")
    print("--- Node: wait_for_answer (RESUMED) ---")
    return resumed_value


def evaluate_answer_node(state: InterviewAgentState) -> dict:
    """Evaluates the user's answer and updates/appends to the report."""
    print("\n--- Node: evaluate_answer ---")
    
    messages = state["messages"]
    
    question_msg = answer_msg = None
    for i, msg in enumerate(reversed(messages)):
        if isinstance(msg, HumanMessage) and msg.name != "Interviewer":
            answer_msg = msg
            for j in range(len(messages) - i - 2, -1, -1):
                if isinstance(messages[j], AIMessage) and messages[j].name == "Interviewer":
                    question_msg = messages[j]
                    break
            break
    
    if not question_msg or not answer_msg:
        print("Q/A pair not found (Normal for 'stop' command)")
        return {}
    
    current_q_id = state["current_question_id"]
    topic = state.get("current_question_topic", "general").lower()
    difficulty = state.get("current_question_difficulty", "medium")
    
    try:
        prompt = EVALUATION_PROMPT.format(
            question=question_msg.content,
            answer=answer_msg.content
        )
        response = llm.invoke(prompt)
        
        grade = re.search(r"GRADE:\s*(.+)", response.content, re.IGNORECASE)
        skill = re.search(r"SKILL:\s*(.+)", response.content, re.IGNORECASE)
        feedback = re.search(r"FEEDBACK:\s*(.+)", response.content, re.IGNORECASE | re.DOTALL)
        
        grade = grade.group(1).strip().lower() if grade else "moderate"
        skill = skill.group(1).strip().lower() if skill else "general"
        feedback = feedback.group(1).strip() if feedback else "No feedback."
        
        print(f"Grade: {grade.upper()} | Skill: {skill} | Topic: {topic} | Difficulty: {difficulty}")
        print("\n" + "="*40 + "\nFEEDBACK*" + "="*40)
        print(f"Grade: {grade.upper()}\nFeedback: {feedback}")
        print("="*40)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        grade, skill, feedback = "error", "unknown", str(e)

    current_report = state.get("evaluation_report", []).copy()
    topic_performance = state.get("topic_performance", {}).copy()
    if topic not in topic_performance:
        topic_performance[topic] = []

    is_follow_up = bool(
        current_report and 
        current_report[-1]["question_id"] == current_q_id
    )

    if is_follow_up:
        print("This is a follow-up. UPDATING previous grade.")
        
        last_eval_report = current_report[-1]
        last_eval_report["evaluation"] = grade
        last_eval_report["feedback"] = feedback
        last_eval_report["answer"] = f"{last_eval_report.get('answer', '')}\n[FOLLOW-UP]: {answer_msg.content}"
        last_eval_report["timestamp"] = datetime.now().isoformat()
        
        if topic_performance[topic]:
            last_perf_entry = topic_performance[topic][-1]
            if last_perf_entry["question_id"] == current_q_id:
                last_perf_entry["evaluation"] = grade
                last_perf_entry["timestamp"] = datetime.now().isoformat()
        
        return {
            "messages": [SystemMessage(content=f"Grade (Updated): {grade.upper()}\nFeedback: {feedback}")],
            "evaluation_report": current_report,
            "topic_performance": topic_performance
        }
        
    else:
        print("This is a new question. APPENDING to report.")
        
        new_eval_entry = {
            "question_id": current_q_id,
            "question": question_msg.content,
            "answer": answer_msg.content,
            "evaluation": grade,
            "feedback": feedback,
            "skill": skill,
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "difficulty": difficulty
        }
        
        topic_performance[topic].append({
            "question_id": current_q_id,
            "evaluation": grade,
            "difficulty": difficulty,
            "timestamp": datetime.now().isoformat()
        })
        topic_performance[topic] = topic_performance[topic][-5:]

        return {
            "messages": [SystemMessage(content=f"Grade: {grade.upper()}\nFeedback: {feedback}")],
            "evaluation_report": current_report + [new_eval_entry],
            "topic_performance": topic_performance
        }
    
def update_coverage_node(state: InterviewAgentState) -> dict:
    """Update covered questions."""
    print("--- Node: update_coverage ---")
    
    current_id = state.get("current_question_id")
    if not current_id or current_id in ["NONE", "ERROR", "DONE"]:
        return {}

    covered = state.get("covered_questions", []).copy()
    
    if current_id not in covered:
        print(f"Covered: {current_id}")
        covered.append(current_id)
        return {"covered_questions": covered}
    
    return {}

def decide_action_router(state: InterviewAgentState) -> Literal["ask_follow_up", "get_new_question"]:
    """Decides whether to ask a follow-up or move to a new question."""
    print("--- Router: decide_action ---")
    
    try:
        last_eval = state.get("evaluation_report", [])[-1]
        grade = last_eval["evaluation"].lower()
        
        if grade in ["excellent", "good"]:
            print("Answer was good. Moving to new question.")
            return "get_new_question"
        
        question = last_eval["question"]
        answer = last_eval["answer"]
        
        prompt = ACTION_CHOICE_PROMPT.format(question=question, answer=answer, grade=grade)
        response = llm.invoke(prompt)
        choice = response.content.upper().strip()
        
        if "FOLLOW_UP" in choice:
            print("LLM decided to ask a follow-up.")
            return "ask_follow_up"
        else:
            print("LLM decided to move on.")
            return "get_new_question"
            
    except Exception as e:
        print(f"Action router failed: {e}. Defaulting to new question.")
        return "get_new_question"


def should_continue_router(state: InterviewAgentState) -> Literal["generate_question", "synthesize_and_update"]:
    """Route to next node."""
    print("--- Router: should_continue ---")
    
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and msg.name != "Interviewer":
            content = msg.content.lower().strip()
            
            stop_triggers = ["stop", "done", "end", "finish", "quit"]
            if content in stop_triggers:
                print("User stop detected")
                return "synthesize_and_update"
            
            break
    
    if state.get("current_question_id") in ["NONE", "ERROR"]:
        print("No more questions available")
        return "synthesize_and_update"
    
    if state.get("question_count", 0) >= MAX_QUESTIONS_PER_SESSION:
        print(f"Max questions ({MAX_QUESTIONS_PER_SESSION}) reached")
        return "synthesize_and_update"
    
    print("Continuing...")
    return "generate_question"

def synthesize_and_update_node(state: InterviewAgentState) -> dict:
    """Generate final summary."""
    print("\n" + "="*70)
    print("INTERVIEW SESSION COMPLETE")
    print("="*70)
    
    report = state.get("evaluation_report", [])
    if not report:
        summary = "No questions were answered in this session."
    else:
        try:
            report_str = "\n".join([
                f"- {e['skill']}: {e['evaluation']} | {e['feedback']}"
                for e in report
            ])
            prompt = SUMMARY_PROMPT.format(report=report_str)
            summary = llm.invoke(prompt).content
        except Exception as e:
            print(f"Summary error: {e}")
            summary = f"Completed {len(report)} questions."
    
    print("\nFinal Summary:\n")
    print(summary)
    
    return {
        "messages": [SystemMessage(content=f"Interview Complete!\n\n{summary}")],
        "current_question_id": "DONE"
    }

# --- 5. Build Graph ---
print("\nBuilding LangGraph Graph...")

workflow = StateGraph(InterviewAgentState)

workflow.add_node("start_interview", start_interview_node)
workflow.add_node("generate_question", generate_question_node)
workflow.add_node("wait_for_answer", wait_for_answer_node) 
workflow.add_node("evaluate_answer", evaluate_answer_node)
workflow.add_node("ask_follow_up", ask_follow_up_node)
workflow.add_node("update_coverage", update_coverage_node)
workflow.add_node("synthesize_and_update", synthesize_and_update_node)

workflow.add_edge(START, "start_interview")
workflow.add_edge("start_interview", "generate_question")
workflow.add_edge("generate_question", "wait_for_answer")
workflow.add_edge("wait_for_answer", "evaluate_answer")

workflow.add_conditional_edges(
    "evaluate_answer",
    decide_action_router,
    {
        "ask_follow_up": "ask_follow_up",
        "get_new_question": "update_coverage"
    }
)

workflow.add_edge("ask_follow_up", "wait_for_answer")

workflow.add_conditional_edges(
    "update_coverage",
    should_continue_router,
    {
        "generate_question": "generate_question",
        "synthesize_and_update": "synthesize_and_update"
    }
)

workflow.add_edge("synthesize_and_update", END)

# --- 6. Interactive Functions (to be called by supervisor.py) ---

def print_agent_output(messages: List[BaseMessage]):
    """Finds and prints ONLY the latest AI Interviewer question."""
    question = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.name == "Interviewer":
            question = msg.content
            break
            
    if question:
        print("\n" + "="*40 + "\nINTERVIEWER\n" + "="*40)
        print(question)

def prompt_for_choice(prompt: str, choices: List[str]) -> str:
    """Helper function to get validated user input from a list."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    
    while True:
        try:
            user_input = input(f"Enter number (1-{len(choices)}): ")
            idx = int(user_input) - 1
            if 0 <= idx < len(choices):
                selected = choices[idx]
                print(f"Selected: {selected}")
                return selected
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def run_interactive_interview(checkpointer, user_skills: list):
    """
    Execute a live, interactive interview session.
    This function is called by the supervisor's main router loop.
    """
    
    # Compile the agent *with the shared checkpointer*
    interview_agent = workflow.compile(checkpointer=checkpointer)
    print("[Interview Coach] Agent compiled.")
    
    session_id = f"interview_session_{int(time.time())}"
    config = {"configurable": {"thread_id": session_id}}
    print(f"\nStarting new interview session: {session_id}")
    
    # --- 1. Get Interview Config ---
    
    # --- MODIFIED: Use user_skills to define topic lists ---
    # This is a simple heuristic; you can make this logic more complex
    all_topics = list(set([s.lower() for s in user_skills if s.lower() in ["data structures & algorithms", "system design", "python", "machine learning"]]))
    if not all_topics:
        all_topics = ["data structures & algorithms", "system design"] # Default
    
    print(f"\nBased on your resume, we can focus on these topics: {', '.join(all_topics)}")
    
    focus_choices = ["all_my_topics", "generic", "specific_company"]
    focus_choice = prompt_for_choice("Choose your interview focus:", focus_choices)
    
    topic_list = all_topics
    company_choices = ["amazon", "microsoft", "nvidia", "netflix", "meta", "google", "generic", "all"]
    company_focus = "generic"

    if focus_choice == "generic":
        company_focus = "generic"
    elif focus_choice == "specific_company":
        company_focus = prompt_for_choice("Choose your company focus:", company_choices)

    # Note: 'interview_focus' isn't used as heavily now, but we set it.
    interview_focus = "all" 
            
    print(f"Topics set: {', '.join(topic_list)}")
    print(f"Company set: {company_focus}")
    
    initial_state = {
        "user_id": "interactive_user",
        "interview_focus": interview_focus,
        "company_focus": company_focus,
        "topic_list": topic_list,
        "messages": [],
    }
    
    # --- 2. Start Interview (First Invoke) ---
    print("\nStarting interview... (Type 'stop' or 'exit' at any time to end)")
    result = interview_agent.invoke(initial_state, config=config)
    
    print_agent_output(result["messages"])
    
    # --- 3. Interaction Loop ---
    while "__interrupt__" in result:
        try:
            user_input = input("\nYour Answer: ")
            
            if user_input.lower().strip() in ["stop", "exit", "quit", "finish"]:
                print("Ending interview session...")
                break # Exit the loop
            
            resume_payload = {"messages": [HumanMessage(content=user_input)]}
            result = interview_agent.invoke(Command(resume=resume_payload), config=config)
            print_agent_output(result["messages"])
            
        except KeyboardInterrupt:
            print("\n\nInterview interrupted. Ending session.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break
    
    # --- 4. Print Final Summary ---
    if "__interrupt__" not in result:
        print("\n" + "="*50 + "\nINTERVIEW COMPLETE\n" + "="*50)
        print(result['messages'][-1].content)
    
    print("\nReturning to Career Supervisor...")

# This file is now a module. It should not run itself.
# The 'if __name__ == "__main__":' block is removed.
# The supervisor's main file will handle the checkpointer and
# call 'run_interactive_interview'.