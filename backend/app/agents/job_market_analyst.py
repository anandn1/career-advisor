import os
import requests
from typing import Dict, Any
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
# --- 1. Settings Import ---
# Import the shared LLM, embedding function, and DB path
print("Loading agent settings from core.settings...")
from settings import (
    llm,
    embedding_function,
    CHROMA_PERSIST_DIR,
    DB_URI,
    COHERE_API_KEY
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
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
)
#Base retriver
base_retriever = db.as_retriever(search_kwargs={"k": 25})
#Reranker
reranker = CohereRerank(
    cohere_api_key=COHERE_API_KEY, 
    model="rerank-english-v3.0", 
    top_n=3 
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, 
    base_retriever=base_retriever
)
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
    
    
    retrieved_docs = compression_retriever.invoke(query)
    
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
    """Fetches *new* job postings..."""
    print(f"\n--- AGENT ACTION: Calling Live Scraper ---")
    
    # This list will now hold CHUNKS
    all_new_chunks = []
    
    try:
        # ... (API call logic) ...
        jobs_url = f"{SCRAPER_API_BASE_URL}/api/jobs"
        params = {"role": role, "location": location, "page": 1}
        response = requests.get(jobs_url, params=params, timeout=20)
        response.raise_for_status()
        jobs_list = response.json()
        
        if not jobs_list:
            return f"Scraper found 0 new jobs for '{role}' in '{location}'."

        for job_summary in jobs_list:
            # ... (API call for details is the same) ...
            job_link = job_summary.get('link')
            if not job_link: continue
            desc_url = f"{SCRAPER_API_BASE_URL}/api/jobs/description"
            desc_params = {"url": job_link}
            desc_response = requests.get(desc_url, params=desc_params, timeout=20)
            
            if desc_response.status_code == 200:
                job_details = desc_response.json()
                full_job_data = {**job_summary, **job_details}
                
                # 1. We create the single, long document
                doc = _format_job_as_document(full_job_data)
                
                # --- 2. THIS IS THE NEW STEP ---
                # We split the long document into smaller chunks
                chunks = text_splitter.split_documents([doc])
                
                # 3. We add the chunks to our list
                all_new_chunks.extend(chunks)

    except requests.RequestException as e:
        return f"Error: Failed to contact the job scraper API. {e}"

    # --- UPDATED INGESTION ---
    if all_new_chunks:
        # Now we add the chunks, not the full docs
        print(f"Ingesting {len(all_new_chunks)} new document chunks into vector store...")
        db.add_documents(all_new_chunks)
        
        return f"Successfully scraped and ingested {len(all_new_chunks)} new job chunks."
        
    return "No new jobs were successfully scraped and ingested."
# Message trimming ,We define how many messages to keep. 10 = 5 user, 5 assistant.
MAX_MESSAGES_TO_KEEP = 10

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    This middleware intercepts the agent's state *before* calling the LLM
    and truncates the message history to prevent token overflow.
    """
    messages = state["messages"]
    
    # +1 to account for the System Prompt
    if len(messages) <= MAX_MESSAGES_TO_KEEP + 1:
        return None  # No changes needed, history is short

    print(f"--- Trimming History: {len(messages)} messages found ---")

    # Keep the first message (the System Prompt) and the last N messages
    first_msg = messages[0] # This is our AGENT_SYSTEM_PROMPT
    recent_messages = messages[-MAX_MESSAGES_TO_KEEP:]
    
    new_messages = [first_msg] + recent_messages
    
    print(f"--- Trimming History: Reduced to {len(new_messages)} messages ---")

    # This replaces the *entire* history with our new, trimmed list
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
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
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
                        llm, 
                         tools=all_tools,
                         system_prompt=AGENT_SYSTEM_PROMPT,
                         checkpointer=checkpointer,
                         middleware=[trim_messages],
                         )

    print("--- Job Market Analyst Agent is Ready ---")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the  Full Stack job postings available in Chennai ?"}]},
        {"configurable": {"thread_id": "1"}}
    )
    print(result1["messages"][-1].content)
result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "From what location have I asked you to find job posting? what type of jobs did I ask you to find? In the job posting I have asked you to find, what are the most in-demand skills ?"}]},
        {"configurable": {"thread_id": "1"}}
    )
print(result2["messages"][-1].content)