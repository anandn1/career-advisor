"""
Resume Analyzer Agent (Non-Interactive Tool)

This agent is called as a tool by the supervisor.
Its job is to:
1.  Read a resume file from a path using the ResumeParser.
2.  Analyze it based on a user's request.
3.  Return a single JSON blob with the analysis.
"""

import os
import sys
import json
from typing import TypedDict, List
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

# --- Project Imports ---
try:
    # --- MODIFIED: Importing 'llm' as requested ---
    from settings import llm
    from utils import ResumeParser
except ImportError:
    print("[FATAL ERROR] settings.py or utils.py not found.")
    sys.exit(1)

# --- 1. Pydantic Output Schema ---

class ResumeAnalysis(BaseModel):
    """The structured analysis of a user's resume."""
    summary: str = Field(description="A 2-3 sentence summary of the resume's overall quality and fit for the user's request.")
    strengths: List[str] = Field(description="A list of 3-5 key strengths identified in the resume.")
    weaknesses: List[str] = Field(description="A list of 3-5 key weaknesses or areas for improvement.")
    actionable_suggestions: List[str] = Field(description="A list of 3-5 specific, actionable suggestions for the user (e.g., 'Quantify project 1 with metrics').")

# --- 2. Graph State ---

class ResumeAnalyzerState(TypedDict):
    """The state for the resume analysis graph."""
    request: str
    resume_path: str
    resume_text: str
    analysis_report: dict
    final_json: str

# --- 3. Graph Node Functions ---

def extract_text_node(state: ResumeAnalyzerState) -> dict:
    """Reads the full text from the resume file."""
    print("\n[Resume Analyzer] Node: extract_text_node")
    try:
        path = state['resume_path']
        # Use the imported ResumeParser from utils.py
        resume_text = ResumeParser.extract_text(path)
        if not resume_text:
            raise ValueError("No text could be extracted from the file.")
        print(f"--- Extracted {len(resume_text)} characters from resume.")
        return {"resume_text": resume_text}
    except Exception as e:
        print(f"--- Error extracting text: {e}")
        return {"resume_text": f"Error: Could not read resume. {e}"}

# Prompt for the LLM
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional career coach and resume reviewer.
Your task is to provide a critical, constructive, and actionable analysis of a user's resume.
The user has provided their full resume text and a specific request.

You MUST respond with a valid `ResumeAnalysis` JSON object.
"""),
    ("human", """
**User's Request:**
{request}

**Full Resume Text:**
---
{resume_text}
---

Please provide your analysis based on their request.
""")
])

def generate_analysis_node(state: ResumeAnalyzerState) -> dict:
    """Uses an LLM to analyze the resume text against the request."""
    print("\n[Resume Analyzer] Node: generate_analysis_node")
    
    if "Error:" in state['resume_text']:
        # Handle the error from the previous node
        return {"analysis_report": {
            "summary": "Analysis failed.",
            "strengths": [],
            "weaknesses": ["Could not read or parse the resume file."],
            "actionable_suggestions": ["Please ensure the file path is correct and the file is not corrupted."]
        }}

    try:
        # --- MODIFIED: Using 'llm' as requested ---
        chain = analysis_prompt | llm.with_structured_output(ResumeAnalysis)
        
        # Invoke the chain
        analysis_obj = chain.invoke({
            "request": state['request'],
            "resume_text": state['resume_text']
        })
        
        print("--- Analysis generated successfully.")
        return {"analysis_report": analysis_obj.model_dump()}

    except Exception as e:
        print(f"--- Error during LLM analysis: {e}")
        return {"analysis_report": {
            "summary": "Analysis failed due to an LLM error.",
            "strengths": [],
            "weaknesses": [f"An internal error occurred: {e}"],
            "actionable_suggestions": ["Please try again."]
        }}

def format_output_node(state: ResumeAnalyzerState) -> dict:
    """Converts the final analysis dict to a JSON string."""
    print("\n[Resume Analyzer] Node: format_output_node")
    final_json = json.dumps(state['analysis_report'], indent=2)
    return {"final_json": final_json}

# --- 4. Graph Definition ---

def create_workflow():
    """Builds the simple, non-interactive graph."""
    workflow = StateGraph(ResumeAnalyzerState)
    
    workflow.add_node("extract_text_node", extract_text_node)
    workflow.add_node("generate_analysis_node", generate_analysis_node)
    workflow.add_node("format_output_node", format_output_node)
    
    workflow.set_entry_point("extract_text_node")
    workflow.add_edge("extract_text_node", "generate_analysis_node")
    workflow.add_edge("generate_analysis_node", "format_output_node")
    workflow.add_edge("format_output_node", END)
    
    return workflow.compile()

# Compile the app once when the file is imported
app = create_workflow()
print("\n[Resume Analyzer] Graph compiled and ready.")

# --- 5. Main Function for Supervisor ---

def run_resume_analyzer_graph(request: str, resume_path: str,checkpointer) -> str:
    """
    The main entry point for the supervisor to call this agent.
    
    Args:
        request (str): The natural language request (e.g., "critique my resume").
        resume_path (str): The full file path to the user's resume.
                            
    Returns:
        str: A JSON string of the ResumeAnalysis.
    """
    
    print(f"\n[Resume Analyzer] Received request: '{request}'")
    print(f"[Resume Analyzer] Received resume path: '{resume_path}'")

    initial_state = {
        "request": request,
        "resume_path": resume_path,
    }
    
    try:
        # This is a non-interactive, blocking call
        result = app.invoke(initial_state)
        
        final_json = result.get("final_json", json.dumps({"error": "No report generated."}))
        print("[Resume Analyzer] ✅ Analysis complete. Returning JSON.")
        return final_json
        
    except Exception as e:
        print(f"\n❌ ERROR: Resume Analyzer workflow failed!")
        print(f"Error details: {e}")
        return json.dumps({"error": str(e), "message": "The resume analyzer agent failed."})