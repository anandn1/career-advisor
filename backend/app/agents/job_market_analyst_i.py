"""
Job Market Analyst Agent
(Refactored as an importable module)

This agent is triggered by the supervisor. It receives a user request
(e.g., "find AI jobs in London") and a list of user skills.
It runs its internal graph to find jobs, scrape details,
and perform a skill-gap analysis, returning the result as a JSON string.
"""

import os
import requests
import json
import re
import time
from typing import Dict, Any, Literal
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated

# --- 1. Settings Import ---
print("Loading agent settings from settings...")
try:
    from settings import (
        llm,
        embedding_function,
        CHROMA_PERSIST_DIR,
        DB_URL,
        COHERE_API_KEY
    )
except ImportError:
    print("[ERROR] Could not import from settings.py. Make sure it exists.")
    # Provide sensible defaults or raise an error
    llm = None # This will likely fail, but shows the dependency
    embedding_function = None
    CHROMA_PERSIST_DIR = "./chroma_db"
    DB_URL = "postgresql://user:pass@localhost:5432/db"
    COHERE_API_KEY = ""


# --- 2. Configuration ---
SCRAPER_API_BASE_URL = "http://localhost:8001"
print(f"Job Market Analyst configured to use scraper at: {SCRAPER_API_BASE_URL}")

# --- 3. State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    role: str
    location: str
    jobs_found: bool
    scraped: bool
    analysis_done: bool
    job_data: list
    skill_gap_data: dict
    next_step: str
    user_skills: list  # This will be populated by the supervisor

# --- 4. Vector Store Management ---
# (Functions: get_vectorstore, _format_job_as_document)
def get_vectorstore() -> Chroma:
    """Initializes or loads the persistent Chroma vector store."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Creating new/empty vector store at {CHROMA_PERSIST_DIR}...")
        placeholder_doc = [
            Document(
                page_content="The job market database is ready.",
                metadata={"source": "system_init"}
            )
        ]
        vectorstore = Chroma.from_documents(
            documents=placeholder_doc,
            embedding=embedding_function,
            persist_directory=CHROMA_PERSIST_DIR
        )
    else:
        print(f"Loading existing vector store from {CHROMA_PERSIST_DIR}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_function
        )
    return vectorstore

def _format_job_as_document(job_details: Dict[str, Any]) -> Document:
    """Formats job JSON into a Document for embedding."""
    content = f"""
    Job Title: {job_details.get('title', 'N/A')}
    Company: {job_details.get('company', 'N/A')}
    Location: {job_details.get('location', 'N/A')}
    Seniority Level: {job_details.get('seniority level', 'N/A')}
    Employment Type: {job_details.get('employment type', 'N/A')}
    Job Function: {job_details.get('job function', 'N/A')}
    Industries: {job_details.get('industries', 'N/A')}
    Skills Required: {', '.join(job_details.get('skills required', []))}
    """
    
    metadata = {
        "source": job_details.get('link', 'N/A'),
        "company": job_details.get('company', 'N/A'),
        "location": job_details.get('location', 'N/A'),
        "seniority": job_details.get('seniority level', 'N/A'),
        "title": job_details.get('title', 'N/A'),
    }
    
    return Document(page_content=content.strip(), metadata=metadata)

# --- 5. Initialize Vector Store & Retriever ---
print("Initializing Job Market Analyst Vector Store...")
db = get_vectorstore()
print("Vector store initialized and ready.")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
)

base_retriever = db.as_retriever(search_kwargs={"k": 15})
reranker = CohereRerank(
    cohere_api_key=COHERE_API_KEY, 
    model="rerank-english-v3.0", 
    top_n=5
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, 
    base_retriever=base_retriever
)

print("Job Market Analyst Vector Store is ready.")

# --- 6. Workflow Node Functions ---
# (parse_query, search_db, scrape_web, analyze_skills, format_output, route_next_step)
def parse_query(state: AgentState) -> AgentState:
    """Extract role and location from user message using LLM."""
    print("\n[NODE] Parsing Query...")
    # Handle both dict and HumanMessage objects
    last_message = state["messages"][-1]
    if hasattr(last_message, 'content'):
        user_message = last_message.content
    else:
        user_message = last_message.get("content", "Find Full Stack Developer jobs in Kolkata")
    
    # LLM prompt for extraction
    extraction_prompt = f"""
    Extract the job role (e.g., "Full Stack Developer") and location (e.g., "Kolkata") from this query: "{user_message}"
    Respond ONLY with JSON: {{"role": "extracted_role", "location": "extracted_location"}}
    If unclear, default role to "Full Stack Developer" and location to "Kolkata".
    """
    
    try:
        response = llm.invoke(extraction_prompt)
        parsed = json.loads(response.content.strip())
        role = parsed.get("role", "Full Stack Developer").strip()
        location = parsed.get("location", "Kolkata").strip()
    except:
        # Fallback to simple split
        role = "Full Stack Developer"
        location = "Hyderabad"
    
    print(f"Parsed: role='{role}', location='{location}'")
    
    return {
        **state,
        "role": role,
        "location": location,
        "next_step": "search_db"
    }

def search_db(state: AgentState) -> AgentState:
    """Search existing database for jobs."""
    print(f"\n[NODE] Searching DB: {state['role']} in {state['location']}")
    
    query = f"{state['role']} jobs in {state['location']}"
    retrieved_docs = compression_retriever.invoke(query)
    
    raw_jobs = []
    target_location = state['location'].lower()
    
    if retrieved_docs:
        for doc in retrieved_docs:
            content = doc.page_content.lower()
            doc_location = doc.metadata.get('location', '').lower()
            
            # Filter by location - only include jobs from the requested location
            if target_location not in doc_location and doc_location not in target_location:
                continue
            
            skills = []
            if "skills required:" in content:
                skills_line = content.split("skills required:")[-1].strip()
                skills = [s.strip().strip(',.') for s in skills_line.split(',') 
                         if s.strip() and len(s.strip()) > 2]
            
            # Enhanced title extraction with regex fallback from content
            title = doc.metadata.get('title', 'N/A')
            if title.lower() == 'n/a':
                title_match = re.search(r'(?:job title|title|position):\s*([^\n\r]+)', content, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip().title()
                else:
                    title_match = re.search(r'([a-z\s]+developer|[a-z\s]+engineer|[a-z\s]+lead)\s*(?:at|for|-)?\s*[a-z\s]+', content, re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip().title()
            
            job_data = {
                "title": title if title != 'N/A' else 'Full Stack Developer Role',
                "company": doc.metadata.get('company', 'N/A'),
                "location": doc.metadata.get('location', 'N/A'),
                "skills_required": skills
            }
            raw_jobs.append(job_data)
    
    # Dedupe jobs by company + skills signature
    seen = set()
    jobs = []
    for job in raw_jobs:
        key = (job['company'], tuple(sorted([s.lower() for s in job['skills_required']])))
        if key not in seen:
            seen.add(key)
            jobs.append(job)
    
    # Always scrape first to get fresh data for the specific location
    next_step = "scrape_web"
    
    return {
        **state,
        "jobs_found": len(jobs) > 0,
        "job_data": jobs,
        "next_step": next_step
    }

def scrape_web(state: AgentState) -> AgentState:
    """Scrape live jobs from web."""
    print(f"\n[NODE] Scraping Web: {state['role']} in {state['location']}")
    
    all_new_chunks = []
    new_jobs = []
    
    try:
        jobs_url = f"{SCRAPER_API_BASE_URL}/api/jobs"
        params = {"role": state['role'], "location": state['location'], "page": 1}
        response = requests.get(jobs_url, params=params, timeout=20)
        response.raise_for_status()
        jobs_list = response.json()
        
        if jobs_list:
            for job_summary in jobs_list[:5]: # Limit to 5 new scrapes per run
                job_link = job_summary.get('link')
                if not job_link:
                    continue
                
                try:
                    desc_url = f"{SCRAPER_API_BASE_URL}/api/jobs/description"
                    desc_params = {"url": job_link}
                    desc_response = requests.get(desc_url, params=desc_params, timeout=20)
                    
                    if desc_response.status_code == 200:
                        job_details = desc_response.json()
                        full_job_data = {**job_summary, **job_details}
                        
                        # Add to vector store
                        doc = _format_job_as_document(full_job_data)
                        chunks = text_splitter.split_documents([doc])
                        all_new_chunks.extend(chunks)
                        
                        # Add to jobs list
                        skills = job_details.get('skills required', [])
                        new_jobs.append({
                            "title": job_summary.get('title', full_job_data.get('title', 'Full Stack Developer Role')),
                            "company": job_summary.get('company', 'N/A'),
                            "location": job_summary.get('location', 'N/A'),
                            "skills_required": skills
                        })
                except Exception as e:
                    print(f"Error fetching job details: {e}")
                    continue

            if all_new_chunks:
                print(f"[NODE] Adding {len(all_new_chunks)} chunks to DB...")
                db.add_documents(all_new_chunks)
                
    except requests.RequestException as e:
        print(f"Scraping error: {e}")
    
    combined_jobs = state.get('job_data', []) + new_jobs
    
    return {
        **state,
        "scraped": True,
        "job_data": combined_jobs,
        "next_step": "analyze_skills"
    }

def analyze_skills(state: AgentState) -> AgentState:
    """Perform skill gap analysis using user's actual skills."""
    print(f"\n[NODE] Analyzing Skills: {state['role']} in {state['location']}")
    
    # Normalization function for fuzzy matching
    def normalize_skill(skill: str) -> str:
        if not skill:
            return ""
        return re.sub(r'[\s-]+', ' ', skill.lower().strip('.,;-'))
    
    # Normalize user skills once
    user_skills_normalized = {normalize_skill(s) for s in state['user_skills']}
    
    # Collect overall unique required skills (normalized)
    all_job_skills = set()
    for job in state['job_data']:
        skills = job.get('skills_required', [])
        all_job_skills.update([normalize_skill(s) for s in skills if s])
    
    all_job_skills = sorted(list(all_job_skills))
    
    # Overall unique gaps
    overall_gaps = [s for s in all_job_skills if s not in user_skills_normalized]
    
    # Per-job gaps
    gaps_per_job = []
    total_gap_count = 0
    if not state['job_data']:
        print("[WARNING] No job data found to analyze.")
        avg_gaps = 0
    else:
        for job in state['job_data']:
            job_skills = [normalize_skill(s) for s in job.get('skills_required', []) if s]
            job_unique_skills = set(job_skills)
            job_gaps = [s for s in job_unique_skills if s not in user_skills_normalized]
            gap_count = len(job_gaps)
            total_gap_count += gap_count
            
            gaps_per_job.append({
                "title": job.get("title", "N/A"),
                "company": job.get("company", "N/A"),
                "gaps": sorted(list(job_gaps)),
                "gap_count": gap_count
            })
        avg_gaps = total_gap_count / len(state['job_data']) if state['job_data'] else 0
    
    analysis = {
        "role": state['role'],
        "location": state['location'],
        "user_current_skills": state['user_skills'],
        "all_required_skills": all_job_skills,
        "skill_gaps": overall_gaps,
        "gaps_per_job": gaps_per_job,
        "average_gaps_per_job": round(avg_gaps, 1),
        "recommendation": f"Focus on learning the top 5 gaps to bridge 80% of requirements across {len(gaps_per_job)} opportunities."
    }
    
    return {
        **state,
        "analysis_done": True,
        "skill_gap_data": analysis,
        "next_step": "format_output"
    }

def format_output(state: AgentState) -> AgentState:
    """Format final JSON output."""
    print("\n[NODE] Formatting Output...")
    
    output = {
        "job_postings": state['job_data'][:10], # Limit output
        "skill_gap_analysis": state['skill_gap_data'],
        "summary": f"Found {len(state['job_data'])} {state['role']} jobs in {state['location']}. "
                   f"Identified average {state['skill_gap_data']['average_gaps_per_job']} skill gaps per job."
    }
    
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": json.dumps(output, indent=2)}],
        "next_step": "end"
    }

def route_next_step(state: AgentState) -> Literal["search_db", "scrape_web", "analyze_skills", "format_output", "end"]:
    """Route to next node based on state."""
    next_step = state.get("next_step", "end")
    print(f"[ROUTER] Next step: {next_step}")
    return next_step

# --- 7. Build Workflow Graph ---
def create_workflow():
    """Creates the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("parse_query", parse_query)
    workflow.add_node("search_db", search_db)
    workflow.add_node("scrape_web", scrape_web)
    workflow.add_node("analyze_skills", analyze_skills)
    workflow.add_node("format_output", format_output)
    
    # Set entry point
    workflow.set_entry_point("parse_query")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "parse_query",
        route_next_step,
        {"search_db": "search_db", "end": END}
    )
    workflow.add_conditional_edges(
        "search_db",
        route_next_step,
        {"scrape_web": "scrape_web", "analyze_skills": "analyze_skills", "end": END}
    )
    workflow.add_conditional_edges(
        "scrape_web",
        route_next_step,
        {"analyze_skills": "analyze_skills", "end": END}
    )
    workflow.add_conditional_edges(
        "analyze_skills",
        route_next_step,
        {"format_output": "format_output", "end": END}
    )
    workflow.add_conditional_edges(
        "format_output",
        route_next_step,
        {"end": END}
    )
    
    return workflow





# --- 9. Main Function for Supervisor ---
def run_job_market_analyst_graph(user_request: str, user_skills: list,checkpointer) -> str:
    """
    The main entry point for the supervisor to call this agent.
    
    Args:
        user_request (str): The natural language request from the user
                            (e.g., "find AI jobs in London").
        user_skills (list): A list of strings representing the user's skills
                            (e.g., ["Python", "PyTorch", "AWS"]).
                            
    Returns:
        str: A JSON string containing the job listings and skill-gap analysis.
    """
    print(f"\n[Job Market Analyst] Received request: '{user_request}'")
    print(f"[Job Market Analyst] Received {len(user_skills)} user skills.")
    app = create_workflow().compile(checkpointer=checkpointer)
    try:
        # Each run should have a unique thread_id
        thread_id = f"job_analyst_{int(time.time())}"
        
        initial_state = {
            "messages": [{"role": "user", "content": user_request}],
            "role": "",
            "location": "",
            "jobs_found": False,
            "scraped": False,
            "analysis_done": False,
            "job_data": [],
            "skill_gap_data": {},
            "next_step": "",
            "user_skills": user_skills # Pass in the skills from the supervisor
        }
        
        # Invoke the compiled graph
        result = app.invoke(
            initial_state,
            {"configurable": {"thread_id": thread_id}}
        )
        
        # Get the final JSON output string
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            response = last_message.content
        else:
            response = last_message.get("content", '{"error": "No response"}')
        
        return response
    
    except Exception as e:
        print(f"[Job Market Analyst] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e), "message": "The job market analyst agent failed."})