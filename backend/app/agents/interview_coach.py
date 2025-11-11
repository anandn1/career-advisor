# interview_agent_v3_interactive.py
import os
from typing import TypedDict, List, Literal, Annotated, Optional
import operator
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
INTERVIEW_CHROMA_DIR = "./chroma_db_interview_questions"
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
    start_time: datetime
    question_count: int

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

# --- 4. Node Functions ---
def start_interview_node(state: InterviewAgentState) -> dict:
    """Initialize interview session."""
    print("\n" + "="*70)
    print("INTERVIEW SESSION STARTING")
    print("="*70)
    
    if not state.get("topic_list"):
        raise ValueError("topic_list is required")
    
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
        "covered_questions": [],
        "evaluation_report": []
    }

def generate_question_node(state: InterviewAgentState) -> dict:
    """RAG node to fetch the next question, using raw topics and $and filter."""
    print(f"\n--- Node: generate_question (Q#{state.get('question_count', 0) + 1}) ---")
    
    topics = state["topic_list"]
    focus = state["interview_focus"]
    company = state["company_focus"]
    covered_ids = state.get("covered_questions", [])
    
    # Build filter
    companies_to_search = ["generic"]
    if company != "generic":
        companies_to_search.append(company)

    # --- CHROMA FILTER FIX ---
    filter_conditions = [
        {"topic": {"$in": [t.lower() for t in topics]}}, # Query 'topic' and lowercase it
        {"company": {"$in": companies_to_search}},
    ]
    
    if covered_ids:
        filter_conditions.append({"question_id": {"$nin": covered_ids}})
    
    if focus == "gaps":
        filter_conditions.append({"difficulty": {"$in": ["easy", "medium"]}})
    elif focus == "proficient":
        filter_conditions.append({"difficulty": {"$in": ["medium", "hard"]}})
    
    metadata_filter = {"$and": filter_conditions}
    # --- END FIX ---
    
    print(f"Filter: {metadata_filter}")
    
    # Retrieval
    try:
        base_retriever = interview_db.as_retriever(
            search_kwargs={"k": 15, "filter": metadata_filter}
        )
        query_text = f"{company} interview questions on {', '.join(topics)}"
        
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
        print(f"Retrieval failed: {e}")
        # Set flag for router
        return {
            "messages": [SystemMessage(content=f"Error: {e}")],
            "current_question_id": "ERROR" 
        }
    
    if not results:
        print("No questions found!")
        # Set flag for router
        return {
            "messages": [SystemMessage(content="No more questions available.")],
            "current_question_id": "NONE"
        }
    
    # Get top result
    next_doc = results[0]
    next_question = next_doc.page_content.strip()
    next_id = next_doc.metadata["question_id"]
    
    print(f"Selected: {next_id}")
    print(f"Topic: {next_doc.metadata.get('topic', 'N/A')} | Difficulty: {next_doc.metadata['difficulty']}")
    
    # Trim message history
    messages = state.get("messages", [])
    if len(messages) > MESSAGE_HISTORY_LIMIT:
        messages = messages[-MESSAGE_HISTORY_LIMIT:]
    
    # Return a dict, NOT a Command.
    return {
        "messages": messages + [AIMessage(content=next_question, name="Interviewer")],
        "current_question_id": next_id,
        "question_count": state.get("question_count", 0) + 1
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
    
    # 'resumed_value' is the {"messages": [HumanMessage(...)]}
    # We return it so LangGraph merges it into the state
    # before moving to the next node (evaluate_answer).
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
        
        print(f"Grade: {grade.upper()} | Skill: {skill}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        grade, skill, feedback = "error", "unknown", str(e)
    
    return {
        "messages": [SystemMessage(content=f"Grade: {grade.upper()}\nFeedback: {feedback}")],
        "evaluation_report": [{
            "question_id": state["current_question_id"],
            "question": question_msg.content,
            "answer": answer_msg.content,
            "evaluation": grade,
            "feedback": feedback,
            "skill": skill,
            "timestamp": datetime.now().isoformat()
        }]
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
    
    # Check user stop intent
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and msg.name != "Interviewer":
            if any(trigger in msg.content.lower() for trigger in ["stop", "done", "end", "finish", "quit"]):
                print("User stop detected")
                return "synthesize_and_update"
            break # Only check the most recent human message
    
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

def print_agent_output(messages: List[BaseMessage]):
    """Finds and prints the latest Feedback (System) and Question (AI)."""
    feedback = None
    question = None
    
    # Search from the end of the list for the *newest*
    # feedback and question
    for msg in reversed(messages):
        if question is None and isinstance(msg, AIMessage) and msg.name == "Interviewer":
            question = msg.content
        elif feedback is None and isinstance(msg, SystemMessage):
            # Don't print the initial config message as feedback
            if "Interview Configuration" not in msg.content:
                feedback = msg.content
        
        # Stop if we've found both of the most recent
        if question and feedback:
            break
            
    # Print what we found
    if feedback:
        print("\n" + "="*40 + "\nFEEDBACK*" + "="*40)
        print(feedback)
    if question:
        print("\n" + "="*40 + "\nINTERVIEWER\n" + "="*40)
        print(question)

def run_interactive_interview():
    """Execute a live, interactive interview session."""
    
    session_id = f"interview_{str(uuid.uuid4())}"
    config = {"configurable": {"thread_id": session_id}}
    print(f"Starting new interview session: {session_id}")
    
    # --- 1. Get Interview Config ---
    # In a real app, you'd get this from the user.
    # We'll hard-code it for this example.
    topics = ["Data Structures & Algorithms", "System Design"]
    print(f"Topics: {', '.join(topics)}")
    
    initial_state = {
        "user_id": "interactive_user",
        "interview_focus": "all",
        "company_focus": "generic",
        "topic_list": topics,
        "messages": [],
    }
    
    # --- 2. Start Interview (First Invoke) ---
    print("Starting interview... (Type 'stop' at any time to end)")
    # This runs the graph until the first interrupt()
    result = interview_agent.invoke(initial_state, config=config)
    
    # Print the first question
    print_agent_output(result["messages"])
    
    # --- 3. Interaction Loop ---
    # The loop continues as long as the graph is interrupted
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

# Compile
with PostgresSaver.from_conn_string(DB_URL) as interview_checkpointer:
    interview_agent = workflow.compile(checkpointer=interview_checkpointer)
    print("Agent compiled and ready!")
    if __name__ == "__main__":
        run_interactive_interview()