# interview_agent_v3_interactive.py
import os
from typing import TypedDict, List, Literal, Annotated, Optional, Dict
import operator
import random
from datetime import datetime
import re
import uuid # <-- IMPORTED
from langgraph.types import Command, interrupt
from langchain_core.messages import ( # <-- CLEANED UP IMPORTS
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage
)

# --- LangGraph v1.0 Imports ---
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver

# --- LangChain Imports ---
from langchain_cohere import CohereRerank
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)

# --- Project Imports ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from settings import llm, embedding_function, DB_URL, COHERE_API_KEY

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INTERVIEW_CHROMA_DIR = os.path.join(SCRIPT_DIR, "ingestion", "chroma_db_interview_questions")
MAX_QUESTIONS_PER_SESSION = 15
MESSAGE_HISTORY_LIMIT = 10
USE_IN_MEMORY_CHECKPOINTER = False

# --- 1. State Definition ---
class InterviewAgentState(MessagesState):
    """Extended MessagesState for interview-specific fields."""
    user_id: str
    interview_focus: Literal["all", "proficient", "gaps"]
    company_focus: str
    topic_list: List[str]
    evaluation_report: Annotated[List[dict], operator.add]
    covered_questions: Annotated[List[str], operator.add]
    current_question_id: str
    current_question_topic: str  # NEW: Track current question's topic
    current_question_difficulty: str  # NEW: Track current question's difficulty
    start_time: datetime
    question_count: int
    topic_performance: Dict[str, List[dict]]  # NEW: Track performance per topic
    current_difficulty_per_topic: Dict[str, List[str]]
# --- 2. Direct Dependency Initialization (CORRECTED v1.0 API) ---
print("\n" + "="*70)
print("INITIALIZING LANGGRAPH v1.0 DEPENDENCIES")
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



# --- Helper function for difficulty transitions ---
def get_next_difficulty_state(current_diffs: List[str], upgrade: bool = False, downgrade: bool = False) -> List[str]:
    """Return next difficulty state based on transition."""
    
    current_set = set(current_diffs)

    if upgrade:
        # --- Upgrade Logic ---
        # Called when user aces 2 easy questions.
        # The new state should be ["hard"].
        if current_set != {"hard"}:
            return ["hard", "medium"]
        else:
            return current_diffs # Already at ["hard"], no change
        
    elif downgrade:
        # --- Downgrade Logic ---
        # Called when user fails 2 hard questions.
        # The new state should be ["easy"].
        if current_set != {"easy"}:
            return ["easy", "medium"]
        else:
            return current_diffs # Already at ["easy"], no change
        
    return current_diffs
# ------------------------------------------------------------


# --- 4. Node Functions ---
# REPLACE the entire start_interview_node function with:



def start_interview_node(state: InterviewAgentState) -> dict:
    """Initialize interview session."""
    print("\n" + "="*70)
    print("INTERVIEW SESSION STARTING")
    print("="*70)
    
    if not state.get("topic_list"):
        raise ValueError("topic_list is required")
    
    # --- MODIFIED: Removed initial difficulty logic based on 'focus' ---
    # The generate_question_node will default to ["easy", "medium", "hard"]
    # if a topic is not found in the map.
    
    # Initialize per-topic difficulty settings (as empty)
    current_difficulty_per_topic = {}
    
    # Show initial difficulty settings
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
    focus = state["interview_focus"]
    company = state["company_focus"]
    covered_ids = state.get("covered_questions", [])
    
    # --- Adaptive Difficulty Logic ---
    topic_performance = state.get("topic_performance", {})
    current_difficulty_per_topic = state.get("current_difficulty_per_topic", {})
    difficulty_updates = {}
    messages_to_print = []
    
    # --- FIX: Loop using original case, but lookup using lowercase ---
    for topic_original_case in topics:
        topic = topic_original_case.lower() # Use lowercase key for lookups
        
        recent_evals = topic_performance.get(topic, [])
        curr_diffs = current_difficulty_per_topic.get(topic, ["easy", "medium", "hard"])
        
        # --- Downgrade Logic: Check last 2 HARD questions ---
        hard_evals = [e for e in recent_evals if e["difficulty"] == "hard"]
        if len(hard_evals) >= 2:
            last_two_hard = hard_evals[-2:]
            
            if all(e["evaluation"] == "bad" for e in last_two_hard):
                new_diffs = get_next_difficulty_state(curr_diffs, downgrade=True)
                if new_diffs != curr_diffs:
                    # Print with original case for readability
                    messages_to_print.append(f"\n{'='*70}\n>>> Difficulty decreased for {topic_original_case}: {curr_diffs} → {new_diffs}\n{'='*70}")
                    difficulty_updates[topic] = new_diffs # Update map with lowercase key

        # --- Upgrade Logic: Check last 2 EASY questions ---
        easy_evals = [e for e in recent_evals if e["difficulty"] == "easy"]
        
        if len(easy_evals) >= 2 and topic not in difficulty_updates:
            last_two_easy = easy_evals[-2:]
            
            if all(e["evaluation"] == "excellent" for e in last_two_easy):
                new_diffs = get_next_difficulty_state(curr_diffs, upgrade=True)
                if new_diffs != curr_diffs:
                    # Print with original case for readability
                    messages_to_print.append(f"\n{'='*70}\n>>> Difficulty increased for {topic_original_case}: {curr_diffs} → {new_diffs}\n{'='*70}\n{'='*70}")
                    difficulty_updates[topic] = new_diffs # Update map with lowercase key
    # --- END OF FIX ---

    # Apply difficulty updates
    updated_difficulty_map = {**current_difficulty_per_topic, **difficulty_updates}
    
    # Print difficulty change notifications
    for msg in messages_to_print:
        print(msg)
    
# Build filter using current difficulty settings
    all_companies_list = ["amazon", "microsoft", "nvidia", "netflix", "meta", "google", "generic"]
    
    if company == "all":
        companies_to_search = all_companies_list
    elif company == "generic":
        companies_to_search = ["generic"]
    else:
        # Search for the specific company + generic
        companies_to_search = [company]
    
    print(f"Searching for companies: {companies_to_search}")

    # Aggregate allowed difficulties across all topics
    allowed_difficulties = set()
    for topic_original_case in topics:
        # --- FIX: Must use .lower() here too ---
        topic = topic_original_case.lower()
        diffs = updated_difficulty_map.get(topic, ["easy", "medium", "hard"])
        allowed_difficulties.update(diffs)
    
    allowed_difficulties = list(allowed_difficulties)
    
    # --- Construct metadata filter (Attempt 1: Specific) ---
    filter_conditions = [
        {"topic": {"$in": [t.lower() for t in topics]}},
        {"company": {"$in": companies_to_search}},
        {"difficulty": {"$in": allowed_difficulties}},
    ]
    
    if covered_ids:
        filter_conditions.append({"question_id": {"$nin": covered_ids}})
    
    metadata_filter = {"$and": filter_conditions}
    
    print(f"Filter (Attempt 1): {metadata_filter}")
    
    # --- Retrieval (Attempt 1) ---
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
        # We can continue and try the relaxed filter
    
    
    # --- Fallback Retrieval (Attempt 2: Relaxed) ---
    if not results:
        print("No questions found with specific filter. Relaxing filter (ignoring difficulty)...")
        
        # Build relaxed filter: only topic, company, and covered_ids
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

    # --- Final Check ---
    if not results:
        print("No questions found even with relaxed filter!")
        return {
            "messages": [SystemMessage(content="No more questions available.")],
            "current_question_id": "NONE"
        }
    
    # MODIFIED: Select randomly from top 3 for variety 
    N = 3
    selectable_results = results[:N]
    next_doc = random.choice(selectable_results)
    
    
    next_question = next_doc.page_content.strip()
    next_id = next_doc.metadata["question_id"]
    next_topic = next_doc.metadata.get("topic", "general").lower() # Use lowercase
    next_difficulty = next_doc.metadata.get("difficulty", "medium")
    
    print(f"Selected: {next_id}")
    print(f"Topic: {next_topic} | Difficulty: {next_difficulty}")
    
    # Trim message history
    messages = state.get("messages", [])
    if len(messages) > MESSAGE_HISTORY_LIMIT:
        messages = messages[-MESSAGE_HISTORY_LIMIT:]
    
    return {
        "messages": messages + [AIMessage(content=next_question, name="Interviewer")],
        "current_question_id": next_id,
        "current_question_topic": next_topic,
        "current_question_difficulty": next_difficulty,
        "question_count": state.get("question_count", 0) + 1,
        "current_difficulty_per_topic": updated_difficulty_map # Pass the updated map
    }
def wait_for_answer_node(state: InterviewAgentState) -> dict:
    """
    Pause the graph to wait for the user's answer.
    When resumed, it will return the user's message
    to be added to the state.
    """
    print("--- Node: wait_for_answer (PAUSING) ---")
    
    # The interrupt will pause execution here.
    # When resumed, the 'Command(resume=...)' payload
    # will be returned by this interrupt() call.
    resumed_value = interrupt(value="Waiting for user's answer")
    
    print("--- Node: wait_for_answer (RESUMED) ---")

    return resumed_value



def evaluate_answer_node(state: InterviewAgentState) -> dict:
    """Evaluate the user's answer."""
    print("\n--- Node: evaluate_answer ---")
    
    messages = state["messages"]
    
    # Find question-answer pair
    question_msg = answer_msg = None
    for i, msg in enumerate(reversed(messages)):
        if isinstance(msg, HumanMessage) and msg.name != "Interviewer":
            answer_msg = msg
            # Find the preceding AI Interviewer message
            for j in range(len(messages) - i - 2, -1, -1):
                if isinstance(messages[j], AIMessage) and messages[j].name == "Interviewer":
                    question_msg = messages[j]
                    break
            break
    
    if not question_msg or not answer_msg:
        print("Q/A pair not found (This is normal if user just typed 'stop')")
        return {"evaluation_report": []}
    
    # Get topic and difficulty from state
    topic = state.get("current_question_topic", "general").lower()
    difficulty = state.get("current_question_difficulty", "medium")
    
    # LLM Evaluation
    try:
        prompt = EVALUATION_PROMPT.format(
            question=question_msg.content,
            answer=answer_msg.content
        )
        response = llm.invoke(prompt)
        
        # Parse with regex
        grade = re.search(r"GRADE:\s*(.+)", response.content, re.IGNORECASE)
        skill = re.search(r"SKILL:\s*(.+)", response.content, re.IGNORECASE)
        feedback = re.search(r"FEEDBACK:\s*(.+)", response.content, re.IGNORECASE | re.DOTALL)
        
        grade = grade.group(1).strip().lower() if grade else "moderate"
        skill = skill.group(1).strip().lower() if skill else "general"
        feedback = feedback.group(1).strip() if feedback else "No feedback."
        
        print(f"Grade: {grade.upper()} | Skill: {skill} | Topic: {topic} | Difficulty: {difficulty}")
        
        # --- MINIMAL CHANGE IS HERE ---
        # Print the feedback immediately to the console
        print("\n" + "="*40 + "\nFEEDBACK*" + "="*40)
        print(f"Grade: {grade.upper()}\nFeedback: {feedback}")
        print("="*40)
        # --- END OF CHANGE ---

    except Exception as e:
        print(f"Evaluation failed: {e}")
        grade, skill, feedback = "error", "unknown", str(e)

    # Update topic performance tracking
    topic_performance = state.get("topic_performance", {}).copy()
    if topic not in topic_performance:
        topic_performance[topic] = []
    
    topic_performance[topic].append({
        "question_id": state["current_question_id"],
        "evaluation": grade,
        "difficulty": difficulty,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 5 evaluations per topic
    topic_performance[topic] = topic_performance[topic][-5:]
    
    return {
        "messages": [SystemMessage(content=f"Grade: {grade.upper()}\nFeedback: {feedback}")],
        "evaluation_report": [{
            "question_id": state["current_question_id"],
            "question": question_msg.content,
            "answer": answer_msg.content,
            "evaluation": grade,
            "feedback": feedback,
            "skill": skill,
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "difficulty": difficulty
        }],
        "topic_performance": topic_performance
    }
def update_coverage_node(state: InterviewAgentState) -> dict:
    """Update covered questions."""
    print("--- Node: update_coverage ---")
    
    current_id = state.get("current_question_id")
    if current_id and current_id not in ["NONE", "ERROR", "DONE"]:
        covered = state.get("covered_questions", [])
        if current_id not in covered:
            print(f"Covered: {current_id}")
            return {"covered_questions": [current_id]}
    
    return {}

def should_continue_router(state: InterviewAgentState) -> Literal["generate_question", "synthesize_and_update"]:
    """Route to next node."""
    print("--- Router: should_continue ---")
    
    # Check user stop intent - only in the most recent human message
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and msg.name != "Interviewer":
            content = msg.content.lower().strip() # Get the clean, lowercase content
            
            # --- FIX: Check if message *IS* a stop word, not if it *contains* one ---
            stop_triggers = ["stop", "done", "end", "finish", "quit"]
            if content in stop_triggers:
                print("User stop detected")
                return "synthesize_and_update"
            # --- END FIX ---
            
            break  # Only check the most recent human message
    
    # Check termination
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

# Add nodes
workflow.add_node("start_interview", start_interview_node)
workflow.add_node("generate_question", generate_question_node)
workflow.add_node("wait_for_answer", wait_for_answer_node) 
workflow.add_node("evaluate_answer", evaluate_answer_node)
workflow.add_node("update_coverage", update_coverage_node)
workflow.add_node("synthesize_and_update", synthesize_and_update_node)

# --- Define flow (UPDATED) ---
workflow.add_edge(START, "start_interview")
workflow.add_edge("start_interview", "generate_question")

# The graph now loops:
workflow.add_edge("generate_question", "wait_for_answer")
workflow.add_edge("wait_for_answer", "evaluate_answer")
workflow.add_edge("evaluate_answer", "update_coverage")

# Router
workflow.add_conditional_edges(
    "update_coverage",
    should_continue_router,
    {
        "generate_question": "generate_question", # Loop back
        "synthesize_and_update": "synthesize_and_update"
    }
)

workflow.add_edge("synthesize_and_update", END)

# --- 6. Interactive Main Function (REPLACED) ---

# REPLACE your print_agent_output function with this:

def print_agent_output(messages: List[BaseMessage]):
    """Finds and prints ONLY the latest AI Interviewer question."""
    
    question = None
    
    # Search from the end for the newest question
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.name == "Interviewer":
            question = msg.content
            break
            
    # Print what we found
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


def run_interactive_interview():
    """Execute a live, interactive interview session."""
    
    session_id = f"interview_{str(uuid.uuid4())}"
    config = {"configurable": {"thread_id": session_id}}
    print(f"Starting new interview session: {session_id}")
    
    # --- 1. Get Interview Config ---
    
    # --- MODIFIED: Define topic lists ---
    # (Adjust these lists to match your actual topics)
    proficient_topics = ["Data Structures & Algorithms", "System Design"]
    all_topics = ["Data Structures & Algorithms", "System Design", "Python", "Machine Learning"]
    gap_topics = ["Python", "Machine Learning"]
    # --- END MODIFIED ---

    # --- MODIFIED: Prompt for focus ---
    focus_choices = ["proficient", "all", "gaps"] # From your state definition
    interview_focus = prompt_for_choice("Choose your interview focus:", focus_choices)
    
    topic_list = []
    if interview_focus == "proficient":
        topic_list = proficient_topics
    elif interview_focus == "all":
        topic_list = all_topics
    elif interview_focus == "gaps":
        topic_list = gap_topics
            
    print(f"Topics set: {', '.join(topic_list)}")
    # --- END MODIFIED ---

    # --- MODIFIED: Prompt for company ---
    company_choices = ["amazon", "microsoft", "nvidia",  "netflix", "meta", "google", "generic", "all"]
    company_focus = prompt_for_choice("Choose your company focus:", company_choices)
    # --- END MODIFIED ---
    
    initial_state = {
        "user_id": "interactive_user",
        "interview_focus": interview_focus, # Set from user input
        "company_focus": company_focus,     # Set from user input
        "topic_list": topic_list,           # Set from user input
        "messages": [],
    }
    
    # --- 2. Start Interview (First Invoke) ---
    print("\nStarting interview... (Type 'stop' at any time to end)")
    # This runs the graph until the first interrupt()
    result = interview_agent.invoke(initial_state, config=config)
    
    # Print the first question
    print_agent_output(result["messages"])
    
    # --- 3. Interaction Loop ---
    # (The rest of this function remains unchanged)
    while "__interrupt__" in result:
        try:
            # Get user input from the command line
            user_input = input("\nYour Answer: ")
            
            # Create the payload to resume the graph
            resume_payload = {"messages": [HumanMessage(content=user_input)]}
            
            # Resume the graph
            result = interview_agent.invoke(Command(resume=resume_payload), config=config)
            
            # Print new output (feedback + new question)
            print_agent_output(result["messages"])
            
        except KeyboardInterrupt:
            print("\n\nInterview interrupted by user. Ending session.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break
    
    # --- 4. Print Final Summary ---
    if "__interrupt__" not in result:
        print("\n" + "="*50 + "\nINTERVIEW COMPLETE\n" + "="*50)
        # The final message is the summary
        print(result['messages'][-1].content)


with PostgresSaver.from_conn_string(DB_URL) as interview_checkpointer:
    interview_agent = workflow.compile(checkpointer=interview_checkpointer)
    print("Agent compiled and ready!")
    if __name__ == "__main__":
        run_interactive_interview()