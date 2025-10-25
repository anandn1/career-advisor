import os
import requests
from typing import Dict, Any
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. Settings Import ---
# Import the shared LLM, embedding function, and DB path
print("Loading agent settings from core.settings...")
from settings import (
    llm,
    embedding_function,
    CHROMA_PERSIST_DIR
)

# --- 2. Configuration ---
# The URL where your scraper FastAPI app is running
SCRAPER_API_BASE_URL = "http://localhost:8001" # As per our design
print(f"Agent configured to use scraper at: {SCRAPER_API_BASE_URL}")

# --- 3. Vector Store Management ---

def get_vectorstore() -> Chroma:
    """
    Initializes or loads the persistent Chroma vector store.
    
    This function handles the automatic creation of the database
    if it doesn't exist, as per your requirement.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Creating new/empty vector store at {CHROMA_PERSIST_DIR}...")
        # Seed with a placeholder document to initialize the collection
        placeholder_doc = [
            Document(
                page_content="The job market database is ready.",
                metadata={"source": "system_init"}
            )
        ]
        # Create the persistent store
        vectorstore = Chroma.from_documents(
            documents=placeholder_doc,
            embedding=embedding_function,
            persist_directory=CHROMA_PERSIST_DIR
        )
    else:
        # Load the existing persistent store from disk
        print(f"Loading existing vector store from {CHROMA_PERSIST_DIR}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_function
        )
    return vectorstore

# --- 4. Helper Function (Our "In-Memory JSON Loader") ---

def _format_job_as_document(job_details: Dict[str, Any]) -> Document:
    """
    Takes the JSON response (as a Python dict) from the scraper API
    and formats it into a single text string for the embedding model.
    This is the "formatting" step you asked about.
    """
    # This string is the 'page_content' that gets embedded
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
    
    # Metadata is attached for filtering (e.g., "find jobs in Remote")
    metadata = {
        "source": job_details.get('link', 'N/A'),
        "company": job_details.get('company', 'N/A'),
        "location": job_details.get('location', 'N/A'),
        "seniority": job_details.get('seniority level', 'N/A'),
    }
    
    return Document(page_content=content.strip(), metadata=metadata)

# --- 5. Initialize the Vector Store ONCE ---
# This db object is our shared "long-term memory"
db = get_vectorstore()
print("Vector store initialized and ready.")

# --- 6. Define Agent Tools ---

# --- Tool 1: The RAG Pipeline ---
@tool
def search_internal_knowledge(query: str) -> str:
    """
    Searches the existing vector database for job postings and market analysis.
    Use this tool FIRST to answer general questions about skills, trends,
    and to find jobs already in the database.
    """
    print(f"\n--- AGENT ACTION: Calling RAG Tool ---")
    print(f"Query: {query}")
    
    # Use the globally defined 'db' to search
    retrieved_docs = db.similarity_search(query, k=10)
    
    if not retrieved_docs:
        return "No information found in the database for that query."
        
    # Format the results into a single string for the LLM
    # This gives us full control over the context
    serialized_context = "\n\n---\n\n".join(
        (f"Source: {doc.metadata.get('source', 'N/A')}\n"
         f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    return serialized_context


# --- Tool 2: The Live Scraper Tool ---
print("Building live scraper tool ('scrape_live_job_market')...")
@tool
def scrape_live_job_market(role: str, location: str, num_pages: int = 1) -> str:
    """
    Fetches *new* job postings directly from the live web scraper API.
    Use this tool ONLY when the user asks for 'live', 'recent', or 'new' jobs,
    or if 'search_internal_knowledge' fails to find specific, up-to-date information.
    
    This tool automatically adds the new jobs to the internal knowledge base.
    """
    print(f"\n--- AGENT ACTION: Calling Live Scraper ---")
    print(f"Role: {role}, Location: {location}")
    
    all_new_documents = []
    
    try:
        # Call /api/jobs on your scraper service
        jobs_url = f"{SCRAPER_API_BASE_URL}/api/jobs"
        params = {"role": role, "location": location, "page": 1} # Simplified for agent
        response = requests.get(jobs_url, params=params, timeout=20)
        response.raise_for_status()
        jobs_list = response.json() # Get JSON response
        
        if not jobs_list:
            return f"Scraper found 0 new jobs for '{role}' in '{location}'."
        
        # Call /api/jobs/description for each job
        for job_summary in jobs_list:
            job_link = job_summary.get('link')
            if not job_link: continue
            
            desc_url = f"{SCRAPER_API_BASE_URL}/api/jobs/description"
            desc_params = {"url": job_link}
            desc_response = requests.get(desc_url, params=desc_params, timeout=20)
            
            if desc_response.status_code == 200:
                job_details = desc_response.json() # Get JSON response
                full_job_data = {**job_summary, **job_details}
                
                # Use our "in-memory JSON loader" to format it
                doc = _format_job_as_document(full_job_data)
                all_new_documents.append(doc)

    except requests.RequestException as e:
        print(f"ERROR in scrape_live_job_market: {e}")
        return f"Error: Failed to contact the job scraper API. {e}"

    # --- This is the "self-improving" step ---
    if all_new_documents:
        print(f"Ingesting {len(all_new_documents)} new documents into vector store...")
        db.add_documents(all_new_documents)
        print("Ingestion complete.")
        
        return (
            f"Successfully scraped and ingested {len(all_new_documents)} new jobs. "
            "You should now analyze this new data (using search_internal_knowledge) "
            "to answer the user's question."
        )
    
    return "No new jobs were successfully scraped and ingested."

# --- 7. Create the Agent ---

print("Building the agent 'brain'...")

all_tools = [search_internal_knowledge, scrape_live_job_market]

# This is the agent's "brain" prompt. It's crucial for routing.
AGENT_SYSTEM_PROMPT = """
You are an expert job market analyst assistant.
Your goal is to provide data-driven insights to a Team Leader.

You have two tools:
1. `search_internal_knowledge`: Searches your existing database of saved job postings.
2. `scrape_live_job_market`: Fetches *new, live* job postings from the web.

**Your Execution Rules:**
1.  **Always try `search_internal_knowledge` first.** This is your primary source of information.
2.  **Only use `scrape_live_job_market` if:**
    a) The user explicitly asks for "live", "recent", "new", or "up-to-the-minute" jobs.
    b) `search_internal_knowledge` returns no relevant results for their specific query.
3.  When you use `scrape_live_job_market`, the new data is automatically saved.
    After scraping, use the tool's summary to tell the user what you found.
"""


# Create the agent
agent = create_agent(llm, tools=all_tools,system_prompt=AGENT_SYSTEM_PROMPT)

print("--- Job Market Analyst Agent is Ready ---")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Are there any jobs for a 'Rust Engineer' in 'Berlin'?"}]}
)
print(result["messages"][-1].content)