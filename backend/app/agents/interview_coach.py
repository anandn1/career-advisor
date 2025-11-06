import os
import requests
from typing import TypedDict, List, Literal, Annotated
import operator

# --- Core LangChain/LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

# --- Imports from our project ---
# We'll need the shared LLM, embedder, DB_URI, and checkpointer
from settings import (
    llm,
    embedding_function,
    CHROMA_PERSIST_DIR,
    DB_URI,
    COHERE_API_KEY
)
from langchain_chroma import Chroma

# --- 1. Define the Agent's State ---
# This is the "memory" for the interview flow

class InterviewAgentState(TypedDict):
    """The state for our interview agent."""
    
    user_id: str
    interview_focus: Literal["all", "proficient", "gaps"]
    
    # This list will be loaded from our user profile in Postgres
    skill_list: List[str] 
    
    # This list will be built up during the interview
    evaluation_report: Annotated[list, operator.add]
    
    # This list tracks what we've already asked
    covered_questions: Annotated[list, operator.add]
    
    # This is the main "messages" list for the chat
    messages: Annotated[list, operator.add]

# --- 2. Initialize Dependencies ---
# (We'll need to define a *new* Chroma DB for questions)
INTERVIEW_CHROMA_DIR = "./chroma_db_interview_questions"
interview_db = Chroma(
    persist_directory=INTERVIEW_CHROMA_DIR,
    embedding_function=embedding_function
)
print(f"Interview Question DB loaded from {INTERVIEW_CHROMA_DIR}")

# This checkpointer will store the state of the interview
interview_checkpointer = PostgresSaver.from_conn_string(DB_URI, table_name="interview_checkpoints")
print("Interview Agent checkpointer (Postgres) initialized.")

# --- 3. Define the Agent's Nodes (The Functions) ---

def start_interview_node(state: InterviewAgentState):
    """
    Entry node: Loads the user's profile and focus.
    (This is a stub, we'd load this from a User Profile DB)
    """
    print("--- Node: start_interview ---")
    
    # --- STUBBED DATA (for now) ---
    # In production, we'd query Postgres for this
    user_id = "user_123" 
    focus = state.get("interview_focus", "all") # Get focus from initial message
    
    if focus == "gaps":
        skill_list = ["Python", "SQL", "LangGraph"] # STUB
    elif focus == "proficient":
        skill_list = ["React", "CSS"] # STUB
    else:
        skill_list = ["Python", "SQL", "LangGraph", "React", "CSS"] # STUB
    # --- END STUB ---

    return {
        "user_id": user_id,
        "skill_list": skill_list,
        "messages": [SystemMessage(content="Interview started. Let's begin.")]
    }

def generate_question_node(state: InterviewAgentState):
    """
    The RAG Node: Queries Chroma for the next question.
    """
    print("--- Node: generate_question ---")
    
    # This node's logic will be the most complex:
    # 1. Get state.get("skill_list")
    # 2. Get state.get("covered_questions")
    # 3. Build a Chroma query with metadata filters for skills, difficulty,
    #    and to exclude covered_questions.
    
    # --- STUBBED (for now) ---
    next_question = "Tell me about a time you used Python to solve a difficult problem."
    # --- END STUB ---
    
    return {"messages": [HumanMessage(content=next_question, name="Interviewer")]}

def evaluate_answer_node(state: InterviewAgentState):
    """
    The Grading Node: Uses an LLM to evaluate the user's last answer.
    """
    print("--- Node: evaluate_answer ---")
    
    # The last message is the user's answer
    user_answer = state["messages"][-1].content
    # The second-to-last message is the question
    question = state["messages"][-2].content
    
    # --- LLM Call (Stubbed for now) ---
    # We'd create a prompt like:
    # "Question: {question}\nAnswer: {user_answer}\n\nGrade this answer
    #  on a scale of [bad, moderate, good, excellent]..."
    
    evaluation = "good" # STUB
    feedback = "This is a solid answer." # STUB
    skill_tag = "Python" # STUB
    # --- END STUB ---
    
    # Add to our running report
    new_report_entry = {
        "question": question,
        "answer": user_answer,
        "evaluation": evaluation,
        "feedback": feedback,
        "skill": skill_tag
    }
    
    return {
        "evaluation_report": [new_report_entry], # 'operator.add' appends this
        "messages": [SystemMessage(content=f"Evaluation: {evaluation}. Feedback: {feedback}")]
    }

def synthesize_and_update_node(state: InterviewAgentState):
    """
    The Final Node: Summarizes the report and updates the user's
    global profile in the main Postgres DB.
    """
    print("--- Node: synthesize_and_update ---")
    
    report = state["evaluation_report"]
    
    # --- LLM Call (Stubbed for now) ---
    # We'd call an LLM to summarize the 'report' list.
    # "The user showed 'good' skill in Python..."
    summary = "User did well. Skills 'Python' and 'SQL' are now 'proficient'."
    # --- END STUB ---
    
    # --- DB Update (Stubbed for now) ---
    # Here, we would write to our main Postgres user_profile table
    # E.g., user_profile.add_skill("Python"), user_profile.remove_gap("Python")
    print("Updating user's global profile in Postgres... (STUBBED)")
    # --- END STUB ---
    
    return {"messages": [SystemMessage(content=f"Interview complete!\nSummary: {summary}")]}

# --- 4. Define the Graph's Edges (The Flow) ---

def should_continue_node(state: InterviewAgentState):
    """
    This is our "Conditional Edge" - the main loop.
    """
    print("--- Node: should_continue (Router) ---")
    
    # If the user's last message was "stop", we end.
    last_message = state["messages"][-1].content.lower()
    if "stop" in last_message or "i'm done" in last_message:
        return "end_interview"
    else:
        # Otherwise, loop back to generate another question
        return "continue_interview"

# --- 5. Build the Graph ---

print("Building Interview Agent Graph...")

# Initialize the graph
workflow = StateGraph(InterviewAgentState)

# Add the nodes
workflow.add_node("start_interview", start_interview_node)
workflow.add_node("generate_question", generate_question_node)
workflow.add_node("evaluate_answer", evaluate_answer_node)
workflow.add_node("update_session_state", lambda state: {"covered_questions": [state["messages"][-2].content]}) # Simple updater
workflow.add_node("synthesize_and_update", synthesize_and_update_node)

# Add the "wait for user" interrupt
workflow.add_node("await_user_answer", lambda state: state)

# Add the edges
workflow.set_entry_point("start_interview")
workflow.add_edge("start_interview", "generate_question")

# This is the "wait" step
workflow.add_edge("generate_question", "await_user_answer")
workflow.add_node("await_user_answer", END) # This pauses the graph

# This is the main evaluation loop
workflow.add_edge("evaluate_answer", "update_session_state")

# This is the conditional edge
workflow.add_conditional_edges(
    "update_session_state",
    should_continue_node,
    {
        "continue_interview": "generate_question", # The loop
        "end_interview": "synthesize_and_update"   # The exit
    }
)
workflow.add_edge("synthesize_and_update", END) # Final end

# Compile the agent, connecting the checkpointer
interview_agent_executor = workflow.compile(checkpointer=interview_checkpointer)

print("--- Interview Agent is Ready ---")

# --- 6. Example Test Script ---
if __name__ == "__main__":
    
    print("Running checkpointer setup for test...")
    interview_checkpointer.setup()
    print("Checkpointer setup complete.")

    print("--- Interview Agent Test Script ---")
    
    test_session_id = "interview_thread_1"
    
    # --- Start the interview ---
    print("\n--- STARTING INTERVIEW ---")
    # We pass in the initial focus
    config = {"configurable": {"thread_id": test_session_id}}
    start_input = {"interview_focus": "gaps"}
    
    response = interview_agent_executor.invoke(start_input, config)
    print(f"[AGENT]: {response['messages'][-1].content}") # Prints the first question
    
    # --- Answer the first question ---
    print("\n--- PROVIDING ANSWER 1 ---")
    user_answer_1 = "I once used Python's pandas library to deduplicate a massive dataset."
    print(f"[USER]: {user_answer_1}")
    
    response = interview_agent_executor.invoke(
        {"messages": [HumanMessage(content=user_answer_1)]}, 
        config
    )
    print(f"[AGENT]: {response['messages'][-2].content}") # Prints the evaluation
    print(f"[AGENT]: {response['messages'][-1].content}") # Prints the next question
    
    # --- Answer the second question ---
    print("\n--- PROVIDING ANSWER 2 & STOPPING ---")
    user_answer_2 = "I don't know that one. I'm done for today, stop."
    print(f"[USER]: {user_answer_2}")
    
    response = interview_agent_executor.invoke(
        {"messages": [HumanMessage(content=user_answer_2)]}, 
        config
    )
    print(f"[AGENT]: {response['messages'][-2].content}") # Prints the final evaluation
    print(f"[AGENT]: {response['messages'][-1].content}") # Prints the summary