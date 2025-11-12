# import os
# import requests
# from typing import Dict, Any
# from langchain.tools import tool
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from langchain.agents import create_agent
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_classic.retrievers.contextual_compression import (
#     ContextualCompressionRetriever,
# )
# from langchain_cohere import CohereRerank
# from langgraph.checkpoint.postgres import PostgresSaver
# from langchain.messages import RemoveMessage
# from langgraph.graph.message import REMOVE_ALL_MESSAGES
# from langchain.agents import create_agent, AgentState
# from langchain.agents.middleware import before_model
# from langgraph.runtime import Runtime
# from langchain_core.runnables import RunnableConfig
# # --- 1. Settings Import ---
# # Import the shared LLM, embedding function, and DB path
# print("Loading agent settings from core.settings...")
# from settings import (
#     llm,
#     embedding_function,
#     CHROMA_PERSIST_DIR,
#     DB_URI,
#     COHERE_API_KEY
# )

# # --- 2. Configuration ---
# # The URL where your scraper FastAPI app is running
# SCRAPER_API_BASE_URL = "http://localhost:8001" # As per our design
# print(f"Agent configured to use scraper at: {SCRAPER_API_BASE_URL}")

# # --- 3. Vector Store Management ---

# def get_vectorstore() -> Chroma:
#     """
#     Initializes or loads the persistent Chroma vector store.
    
#     This function handles the automatic creation of the database
#     if it doesn't exist, as per your requirement.
#     """
#     if not os.path.exists(CHROMA_PERSIST_DIR):
#         print(f"Creating new/empty vector store at {CHROMA_PERSIST_DIR}...")
#         # Seed with a placeholder document to initialize the collection
#         placeholder_doc = [
#             Document(
#                 page_content="The job market database is ready.",
#                 metadata={"source": "system_init"}
#             )
#         ]
#         # Create the persistent store
#         vectorstore = Chroma.from_documents(
#             documents=placeholder_doc,
#             embedding=embedding_function,
#             persist_directory=CHROMA_PERSIST_DIR
#         )
#     else:
#         # Load the existing persistent store from disk
#         print(f"Loading existing vector store from {CHROMA_PERSIST_DIR}...")
#         vectorstore = Chroma(
#             persist_directory=CHROMA_PERSIST_DIR,
#             embedding_function=embedding_function
#         )
#     return vectorstore

# # --- 4. Helper Function (Our "In-Memory JSON Loader") ---

# def _format_job_as_document(job_details: Dict[str, Any]) -> Document:
#     """
#     Takes the JSON response (as a Python dict) from the scraper API
#     and formats it into a single text string for the embedding model.
#     This is the "formatting" step you asked about.
#     """
#     # This string is the 'page_content' that gets embedded
#     content = f"""
#     Job Title: {job_details.get('title', 'N/A')}
#     Company: {job_details.get('company', 'N/A')}
#     Location: {job_details.get('location', 'N/A')}
#     Seniority Level: {job_details.get('seniority level', 'N/A')}
#     Employment Type: {job_details.get('employment type', 'N/A')}
#     Job Function: {job_details.get('job function', 'N/A')}
#     Industries: {job_details.get('industries', 'N/A')}
#     Skills Required: {', '.join(job_details.get('skills required', []))}
#     """
    
#     # Metadata is attached for filtering (e.g., "find jobs in Remote")
#     metadata = {
#         "source": job_details.get('link', 'N/A'),
#         "company": job_details.get('company', 'N/A'),
#         "location": job_details.get('location', 'N/A'),
#         "seniority": job_details.get('seniority level', 'N/A'),
#     }
    
#     return Document(page_content=content.strip(), metadata=metadata)

# # --- 5. Initialize the Vector Store ONCE ---
# # This db object is our shared "long-term memory"
# db = get_vectorstore()
# print("Vector store initialized and ready.")
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=150,
#     length_function=len,
# )
# #Base retriver
# base_retriever = db.as_retriever(search_kwargs={"k": 25})
# #Reranker
# reranker = CohereRerank(
#     cohere_api_key=COHERE_API_KEY, 
#     model="rerank-english-v3.0", 
#     top_n=3 
# )
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=reranker, 
#     base_retriever=base_retriever
# )

# # --- NEW: Dummy User Skills (from Resume Parsing - Placeholder for Now) ---
# DUMMY_USER_SKILLS = [
#     "Python", "JavaScript", "React", "Node.js", "SQL", "Git",
#     "Agile Methodology", "Problem Solving", "Team Collaboration"
# ]
# print(f"Dummy user skills loaded: {', '.join(DUMMY_USER_SKILLS)}")

# # --- 6. Define Agent Tools ---

# # --- Tool 1: The RAG Pipeline ---
# @tool
# def search_internal_knowledge(query: str) -> str:
#     """
#     Searches the existing vector database for job postings and market analysis.
#     Use this tool FIRST to answer general questions about skills, trends,
#     and to find jobs already in the database.
#     """
#     print(f"\n--- AGENT ACTION: Calling RAG Tool ---")
#     print(f"Query: {query}")
    
    
#     retrieved_docs = compression_retriever.invoke(query)
    
#     if not retrieved_docs:
#         return "No information found in the database for that query."
        
#     # Format the results into a single string for the LLM
#     # This gives us full control over the context
#     serialized_context = "\n\n---\n\n".join(
#         (f"Source: {doc.metadata.get('source', 'N/A')}\n"
#          f"Content: {doc.page_content}")
#         for doc in retrieved_docs
#     )
    
#     return serialized_context


# # --- Tool 2: The Live Scraper Tool ---
# print("Building live scraper tool ('scrape_live_job_market')...")
# @tool
# def scrape_live_job_market(role: str, location: str, num_pages: int = 1) -> str:
#     """Fetches *new* job postings..."""
#     print(f"\n--- AGENT ACTION: Calling Live Scraper ---")
    
#     # This list will now hold CHUNKS
#     all_new_chunks = []
    
#     try:
#         # ... (API call logic) ...
#         jobs_url = f"{SCRAPER_API_BASE_URL}/api/jobs"
#         params = {"role": role, "location": location, "page": 1}
#         response = requests.get(jobs_url, params=params, timeout=20)
#         response.raise_for_status()
#         jobs_list = response.json()
        
#         if not jobs_list:
#             return f"Scraper found 0 new jobs for '{role}' in '{location}'."

#         for job_summary in jobs_list:
#             # ... (API call for details is the same) ...
#             job_link = job_summary.get('link')
#             if not job_link: continue
#             desc_url = f"{SCRAPER_API_BASE_URL}/api/jobs/description"
#             desc_params = {"url": job_link}
#             desc_response = requests.get(desc_url, params=desc_params, timeout=20)
            
#             if desc_response.status_code == 200:
#                 job_details = desc_response.json()
#                 full_job_data = {**job_summary, **job_details}
                
#                 # 1. We create the single, long document
#                 doc = _format_job_as_document(full_job_data)
                
#                 # --- 2. THIS IS THE NEW STEP ---
#                 # We split the long document into smaller chunks
#                 chunks = text_splitter.split_documents([doc])
                
#                 # 3. We add the chunks to our list
#                 all_new_chunks.extend(chunks)

#     except requests.RequestException as e:
#         return f"Error: Failed to contact the job scraper API. {e}"

#     # --- UPDATED INGESTION ---
#     if all_new_chunks:
#         # Now we add the chunks, not the full docs
#         print(f"Ingesting {len(all_new_chunks)} new document chunks into vector store...")
#         db.add_documents(all_new_chunks)
        
#         return f"Successfully scraped and ingested {len(all_new_chunks)} new job chunks."
        
#     return "No new jobs were successfully scraped and ingested."

# # --- NEW Tool 3: Skill Gap Analysis ---
# @tool
# def perform_skill_gap_analysis(role: str, location: str) -> str:
#     """
#     Performs a skill gap analysis by comparing dummy user skills (from resume placeholder)
#     against skills required in relevant job postings for the given role and location.
#     Retrieves jobs from the internal database, extracts required skills, and identifies gaps.
#     """
#     print(f"\n--- AGENT ACTION: Calling Skill Gap Analysis Tool ---")
#     print(f"Role: {role}, Location: {location}")
    
#     # Retrieve relevant job docs
#     query = f"job postings for {role} in {location}"
#     retrieved_docs = compression_retriever.invoke(query)
    
#     if not retrieved_docs:
#         return "No relevant job postings found in the database for skill gap analysis."
    
#     # Extract all unique job-required skills across retrieved docs
#     job_skills = set()
#     for doc in retrieved_docs:
#         # Simple parsing: look for "Skills Required:" line in content
#         content = doc.page_content.lower()
#         if "skills required:" in content:
#             # Extract the skills string after the label (basic split for demo)
#             skills_line = content.split("skills required:")[-1].strip()
#             # Split by commas and clean
#             extracted = [skill.strip().strip(',.') for skill in skills_line.split(',') if skill.strip()]
#             job_skills.update(extracted)
    
#     job_skills = list(job_skills)  # Convert back to list
    
#     # Dummy user skills (replace with actual resume parsing data later)
#     user_skills_lower = {skill.lower() for skill in DUMMY_USER_SKILLS}
    
#     # Identify gaps: job skills not in user skills (case-insensitive match)
#     skill_gaps = [
#         skill for skill in job_skills
#         if skill.lower() not in user_skills_lower and len(skill) > 2  # Filter short/noise
#     ]
    
#     # Prepare analysis summary
#     analysis = f"""
#     Skill Gap Analysis for {role} in {location}:
    
#     Your Current Skills (Dummy): {', '.join(DUMMY_USER_SKILLS)}
    
#     Common Required Skills in Jobs: {', '.join(job_skills[:10])}... (top 10 shown)
    
#     Identified Skill Gaps (Missing/Recommended to Learn): {', '.join(skill_gaps[:10]) if skill_gaps else 'None - You match well!'}... (top 10 shown)
    
#     Total Gaps Found: {len(skill_gaps)}
#     Recommendation: Focus on learning the top gaps to bridge 80% of requirements.
#     """
    
#     return analysis.strip()

# # Message trimming ,We define how many messages to keep. 10 = 5 user, 5 assistant.
# MAX_MESSAGES_TO_KEEP = 10

# @before_model
# def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
#     """
#     This middleware intercepts the agent's state *before* calling the LLM
#     and truncates the message history to prevent token overflow.
#     """
#     messages = state["messages"]
    
#     # +1 to account for the System Prompt
#     if len(messages) <= MAX_MESSAGES_TO_KEEP + 1:
#         return None  # No changes needed, history is short

#     print(f"--- Trimming History: {len(messages)} messages found ---")

#     # Keep the first message (the System Prompt) and the last N messages
#     first_msg = messages[0] # This is our AGENT_SYSTEM_PROMPT
#     recent_messages = messages[-MAX_MESSAGES_TO_KEEP:]
    
#     new_messages = [first_msg] + recent_messages
    
#     print(f"--- Trimming History: Reduced to {len(new_messages)} messages ---")

#     # This replaces the *entire* history with our new, trimmed list
#     return {
#         "messages": [
#             RemoveMessage(id=REMOVE_ALL_MESSAGES),
#             *new_messages
#         ]
#     }
# # --- 7. Create the Agent ---

# print("Building the agent 'brain'...")

# all_tools = [search_internal_knowledge, scrape_live_job_market, perform_skill_gap_analysis]

# # This is the agent's "brain" prompt. It's crucial for routing.
# AGENT_SYSTEM_PROMPT = """
# You are an expert job market analyst assistant.
# Your goal is to provide data-driven insights to a Team Leader, including skill gap analysis.

# You have three tools:
# 1. `search_internal_knowledge`: Searches your existing database of saved job postings.
# 2. `scrape_live_job_market`: Fetches *new, live* job postings from the web.
# 3. `perform_skill_gap_analysis`: Analyzes skill gaps by comparing your (dummy) user skills to job requirements for a role/location.

# **Your Execution Rules:**
# 1.  **Always try `search_internal_knowledge` first.** This is your primary source of information.
# 2.  **Only use `scrape_live_job_market` if:**
#     a) The user explicitly asks for "live", "recent", "new", or "up-to-the-minute" jobs.
#     b) `search_internal_knowledge` returns no relevant results for their specific query.
# 3.  When you use `scrape_live_job_market`, the new data is automatically saved.
#     After scraping, use the tool's summary to tell the user what you found.
#     **Immediately after scraping, automatically perform skill gap analysis using `perform_skill_gap_analysis` with the same role and location to provide comprehensive insights.**
# 4. **Use `perform_skill_gap_analysis` when the user asks about skill gaps, missing skills, or recommendations for a specific role/location, or automatically after scraping.**
#     It uses dummy resume data for now; in production, integrate with real resume parsing.
# """


# # Create the agent
# with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
#     checkpointer.setup()
#     agent = create_agent(
#                         llm, 
#                          tools=all_tools,
#                          system_prompt=AGENT_SYSTEM_PROMPT,
#                          checkpointer=checkpointer,
#                          middleware=[trim_messages],
#                          )

#     print("--- Job Market Analyst Agent is Ready (with Skill Gap Analysis) ---")
#     result1 = agent.invoke(
#         {"messages": [{"role": "user", "content": "What are the  Full Stack job postings available in Kolkata ?"}]},
#         {"configurable": {"thread_id": "1"}}
#     )
#     print(result1["messages"][-1].content)
#     result2 = agent.invoke(
#         {"messages": [{"role": "user", "content": "From what location have I asked you to find job posting? what type of jobs did I ask you to find? In the job posting I have asked you to find, what are the most in-demand skills ?"}]},
#         {"configurable": {"thread_id": "1"}}
#     )
#     print(result2["messages"][-1].content)
#     result3 = agent.invoke(
#         {"messages": [{"role": "user", "content": "Perform a skill gap analysis for Full Stack Developer roles in Kolkata."}]},
#         {"configurable": {"thread_id": "1"}}
#     )
#     print(result3["messages"][-1].content)
#//Note:::::::: too much token consumption using langchain3-5+ (brain of the agent +tools selection+ LLM parsing for almost everything 3 times) so we use langgraph around 2 tokens
#// using langgraph we can have the functions defined that act as tools and then we decide the workflow and finally invoke our agent helps in lesser token consumption
#########Note##########
# Traditional LangChain agents (like create_agent) are loops: LLM sees everything (prompt + history + tools), decides "next action," repeats until done. Flexible, but unpredictable (LLM hallucinations, token spikes).
# LangGraph: Graph = Nodes (your functions, e.g., "parse query") connected by Edges (e.g., "if jobs <3, scrape"). It's declarative: "Do A, then if X, do B else C." No LLM guessing the sequence.

import os
import requests
import json
import re  # Added for regex title extraction
from typing import Dict, Any, Literal
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated

# --- 1. Settings Import ---
print("Loading agent settings from core.settings...")
from settings import (
    llm,
    embedding_function,
    CHROMA_PERSIST_DIR,
    DB_URI,
    COHERE_API_KEY
)

# --- 2. Configuration ---
SCRAPER_API_BASE_URL = "http://localhost:8001"
print(f"Agent configured to use scraper at: {SCRAPER_API_BASE_URL}")

# --- 3. Resume Parser ---
import PyPDF2
from pathlib import Path

class ResumeParser:
    """Parse resume files and extract skills"""
    
    COMMON_SKILLS = {
        # Programming Languages
        'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
        'typescript', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
        
        # Web Technologies
        'html', 'css', 'react', 'react.js', 'vue', 'vue.js', 'angular', 'angular.js',
        'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'asp.net',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'cassandra', 'redis', 'elasticsearch',
        'dynamodb', 'firestore',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
        'ci/cd', 'terraform', 'ansible',
        
        # Data & AI
        'machine learning', 'deep learning', 'nlp', 'cv', 'pytorch', 'tensorflow',
        'scikit-learn', 'pandas', 'numpy', 'spark', 'hadoop',
        
        # APIs & Architecture
        'rest', 'restful', 'restful api', 'restful apis', 'graphql', 'soap', 'microservices',
        'grpc', 'websocket',
        
        # Tools & Practices
        'git', 'agile', 'agile methodology', 'agile methodologies', 'scrum', 'kanban',
        'jira', 'confluence', 'postman', 'swagger',
        
        # Soft Skills
        'communication', 'teamwork', 'team collaboration', 'problem-solving',
        'leadership', 'collaboration', 'project management', 'time management',
        
        # Testing
        'unit testing', 'integration testing', 'automated testing', 'jest', 'pytest',
        'selenium', 'junit',
        
        # Other
        'next.js', 'laravel', 'automated testing', 'user experience', 'ux'
    }
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"[ERROR] Error extracting PDF: {e}")
            return ""
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"[ERROR] Error extracting TXT: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Extract text based on file type"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return cls.extract_from_pdf(file_path)
        elif file_ext == '.txt':
            return cls.extract_from_txt(file_path)
        else:
            raise ValueError(f"[ERROR] Unsupported file type: {file_ext}. Use PDF or TXT")
    
    @classmethod
    def extract_skills(cls, resume_text: str) -> list:
        """Extract skills from resume text"""
        resume_lower = resume_text.lower()
        found_skills = set()
        
        # Find skills by matching common skills list
        for skill in cls.COMMON_SKILLS:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, resume_lower):
                found_skills.add(skill.title())  # Convert to title case
        
        return sorted(list(found_skills))
    
    @classmethod
    def parse_resume(cls, file_path: str) -> dict:
        """Main method to parse resume and extract skills"""
        print(f"\n[RESUME PARSER] Reading resume from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"[ERROR] Resume file not found: {file_path}")
            return {"skills": [], "error": "File not found"}
        
        # Extract text
        resume_text = cls.extract_text(file_path)
        
        if not resume_text:
            print("[ERROR] Could not extract text from resume")
            return {"skills": [], "error": "Could not extract text"}
        
        print(f"[RESUME PARSER] Extracted {len(resume_text)} characters from resume")
        
        # Extract skills
        skills = cls.extract_skills(resume_text)
        print(f"[RESUME PARSER] Found {len(skills)} skills: {', '.join(skills)}")
        
        return {
            "skills": skills,
            "error": None
        }

def get_user_skills() -> list:
    """Get user skills from resume"""
    print("\n" + "="*70)
    print("SKILL EXTRACTION FROM RESUME")
    print("="*70)
    
    resume_path = input("\nEnter the full path to your resume (PDF or TXT): ").strip()
    # Remove quotes if user included them
    resume_path = resume_path.strip('"\'')
    
    if resume_path:
        resume_data = ResumeParser.parse_resume(resume_path)
        if resume_data.get('error'):
            print(f"[ERROR] {resume_data['error']}")
            print("[RETRY] Please try again with a valid file path\n")
            return get_user_skills()
        elif resume_data['skills']:
            return resume_data['skills']
        else:
            print("[WARNING] No skills extracted from resume")
            return get_user_skills()
    else:
        print("[ERROR] No file path provided. Please try again.")
        return get_user_skills()

# --- 4. State Definition ---
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
    user_skills: list

# --- 5. Vector Store Management ---
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

# --- 6. Helper Function ---
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

# --- 7. Initialize Vector Store ---
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

# --- 8. Workflow Node Functions ---
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
        location = "Kolkata"
    
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
            for job_summary in jobs_list[:5]:
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
        "job_postings": state['job_data'][:10],
        "skill_gap_analysis": state['skill_gap_data'],
        "summary": f"Found {len(state['job_data'])} {state['role']} jobs in {state['location']}. "
                   f"Identified average {state['skill_gap_data']['average_gaps_per_job']} skill gaps per job."
    }
    
    json_output = json.dumps(output, indent=2)
    
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": json_output}],
        "next_step": "end"
    }

def route_next_step(state: AgentState) -> Literal["search_db", "scrape_web", "analyze_skills", "format_output", "end"]:
    """Route to next node based on state."""
    next_step = state.get("next_step", "end")
    print(f"[ROUTER] Next step: {next_step}")
    return next_step

# --- 9. Build Workflow Graph ---
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
        {
            "search_db": "search_db",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "search_db",
        route_next_step,
        {
            "scrape_web": "scrape_web",
            "analyze_skills": "analyze_skills",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "scrape_web",
        route_next_step,
        {
            "analyze_skills": "analyze_skills",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_skills",
        route_next_step,
        {
            "format_output": "format_output",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "format_output",
        route_next_step,
        {
            "end": END
        }
    )
    
    return workflow

# --- 10. Main Execution ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("JOB MARKET ANALYST AGENT - WITH RESUME SKILLS")
    print("="*70)
    
    # Get user skills
    user_skills = get_user_skills()
    
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        
        workflow = create_workflow()
        app = workflow.compile(checkpointer=checkpointer)
        
        # Use unique thread ID
        import time
        thread_id = f"test_{int(time.time())}"
        print(f"\nThread ID: {thread_id}\n")
        
        # Get job search parameters
        role = input("Enter desired job role (e.g., Full Stack Developer): ").strip()
        location = input("Enter desired location (e.g., Kolkata): ").strip()
        
        if not role or not location:
            print("[ERROR] Role and location are required")
        else:
            print(f"\n[TEST] Find {role} jobs in {location}")
            print("Expected: Job listings + Skill Gap Analysis in JSON")
            print("-" * 70)
            
            try:
                initial_state = {
                    "messages": [{"role": "user", "content": f"Find {role} jobs in {location}"}],
                    "role": "",
                    "location": "",
                    "jobs_found": False,
                    "scraped": False,
                    "analysis_done": False,
                    "job_data": [],
                    "skill_gap_data": {},
                    "next_step": "",
                    "user_skills": user_skills
                }
                
                result = app.invoke(
                    initial_state,
                    {"configurable": {"thread_id": thread_id}}
                )
                
                # Handle both dict and AIMessage objects
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    response = last_message.content
                else:
                    response = last_message.get("content", "No response")
                
                print("\n" + "="*70)
                print("COMPLETE JSON OUTPUT:")
                print("="*70)
                print(response)
                
                print("\n" + "="*70)
                print("="*70)
            
            except Exception as e:
                print(f"\n ERROR: {e}")
                import traceback
                traceback.print_exc()