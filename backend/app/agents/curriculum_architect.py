import os
import operator
import uuid
import json
from typing import List, Optional, TypedDict, Annotated, Dict, Any
from pprint import pprint, pformat

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
    AIMessage,
)
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Try multiple import paths for Tavily compatibility
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    try:
        from langchain_tavily import TavilySearchResults
    except ImportError:
        print("WARNING: TavilySearchResults not found. Web search will be disabled.")
        TavilySearchResults = None

print("Loading environment variables...")
load_dotenv()

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found in .env file.")
    exit()
else:
    print("SUCCESS: GROQ API key loaded.")

if not TAVILY_API_KEY:
    print("WARNING: TAVILY_API_KEY not found. Web search features will be limited.")
else:
    print("SUCCESS: Tavily API key loaded.")

# Initialize LLM
llm_reasoning = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
llm_structured = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)

### -----------------------------------------------------------------
### TOOLS & DATABASE SETUP
### -----------------------------------------------------------------

print("Initializing tools and database...")

# Web Search Tool
if TavilySearchResults is not None and TAVILY_API_KEY:
    try:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
        tavily_tool = TavilySearchResults(max_results=5)
        print("SUCCESS: Tavily search tool initialized.")
    except Exception as e:
        print(f"WARNING: Could not initialize Tavily tool: {e}")
        tavily_tool = None
else:
    tavily_tool = None
    print("WARNING: Tavily search tool not available.")

# Embeddings
model_name = "google/embeddinggemma-300m"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(f"SUCCESS: Embeddings model ({model_name}) loaded.")

# ChromaDB Vector Store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="course_catalog"
)
print("SUCCESS: ChromaDB vector store initialized.")

### -----------------------------------------------------------------
### PYDANTIC SCHEMAS
### -----------------------------------------------------------------

class Course(BaseModel):
    """Database schema for a course."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    prereqs: List[int] = Field(default_factory=list, description="List of course IDs (use 0 if none)")
    difficulty: int = Field(..., ge=1, le=5, description="1=easy, 5=hard")
    topics: List[str] = Field(default_factory=list, description="List of relevant topics")
    url: str = Field(..., description="The URL to the course")
    description: str = Field(..., description="A short description of the course")

class NewCourseList(BaseModel):
    """Schema for extracting courses from web search results."""
    courses: List[Course]

class UserProfile(BaseModel):
    """Structured user profile derived from resume."""
    interests: List[str] = Field(..., description="List of technical interests")
    skill_level: str = Field(..., description="Estimated skill level: 'Beginner', 'Intermediate', or 'Advanced'")
    past_courses: List[int] = Field(..., description="List of course IDs from projects/resume (use 0 if none)")

class PlannerOutput(BaseModel):
    """The structured output from the initial Planner node."""
    user_profile: UserProfile
    skills_to_find: List[str] = Field(
        ..., 
        description="ALL critical skill gaps to search for (typically 5-20 skills). Extract EVERY skill mentioned in the gap analysis."
    )

class CourseRecommendation(BaseModel):
    """A single, recommended online course."""
    skill_name: str
    course_title: str
    course_url: str
    description: str

class PersonalizedLearningPath(BaseModel):
    """The final, structured learning path for the user."""
    user_summary: str
    skill_gaps: List[str]
    recommendations: List[CourseRecommendation]

class Critique(BaseModel):
    """A structured critique of the generated learning path."""
    is_approved: bool
    revisions_needed: str

### -----------------------------------------------------------------
### SEED DATABASE
### -----------------------------------------------------------------

SEED_COURSES_DATA = [
    Course(id="101", title="Introduction to Python", prereqs=[], difficulty=1, topics=["Python", "Beginner"], url="https://www.codecademy.com/learn/learn-python-3", description="Python basics for beginners."),
    Course(id="205", title="Advanced FastAPI", prereqs=[101], difficulty=4, topics=["Python", "Web Development", "FastAPI", "API"], url="https://testdriven.io/courses/fastapi/", description="Deep dive into FastAPI framework."),
    Course(id="206", title="FastAPI for Beginners", prereqs=[101], difficulty=2, topics=["Python", "Web Development", "FastAPI", "API"], url="https://fastapi.tiangolo.com/tutorial/", description="Get started with FastAPI."),
    Course(id="303", title="Data Science with Pandas", prereqs=[101], difficulty=3, topics=["Python", "Machine Learning", "Pandas", "Data Analysis"], url="https://www.datacamp.com/courses/pandas-foundations", description="Pandas for data analysis."),
    Course(id="501", title="PostgreSQL: SQL & Database Management", prereqs=[], difficulty=3, topics=["Database", "PostgreSQL", "SQL"], url="https://www.postgresql.org/docs/current/tutorial.html", description="Learn SQL and Postgres."),
    Course(id="502", title="Advanced SQL for Data Scientists", prereqs=[501], difficulty=4, topics=["Database", "PostgreSQL", "SQL"], url="https://mode.com/sql-tutorial/", description="Complex SQL queries."),
    Course(id="700", title="LangChain for LLM App Development", prereqs=[101], difficulty=4, topics=["Python", "LangChain", "AI", "LLM"], url="https://python.langchain.com/docs/get_started/introduction", description="Build LLM applications."),
    Course(id="701", title="LangChain & Vector Databases", prereqs=[101, 700], difficulty=5, topics=["Python", "LangChain", "AI", "LLM", "Database"], url="https://python.langchain.com/docs/modules/data_connection/", description="LangChain with Chroma/Pinecone."),
    Course(id="800", title="Docker & Kubernetes", prereqs=[101], difficulty=4, topics=["Deployment", "Docker", "Kubernetes", "DevOps"], url="https://www.docker.com/get-started", description="Containerization and orchestration."),
    Course(id="801", title="AWS Cloud Practitioner", prereqs=[], difficulty=2, topics=["Cloud", "AWS", "DevOps"], url="https://aws.amazon.com/training/", description="AWS fundamentals."),
    Course(id="802", title="AWS Solutions Architect", prereqs=[801], difficulty=4, topics=["Cloud", "AWS", "Architecture"], url="https://aws.amazon.com/certification/certified-solutions-architect-associate/", description="Design scalable AWS systems."),
    Course(id="900", title="React.js Complete Guide", prereqs=[], difficulty=3, topics=["React", "JavaScript", "Frontend", "Web Development"], url="https://react.dev/learn", description="Master React fundamentals."),
    Course(id="901", title="Angular for Beginners", prereqs=[], difficulty=3, topics=["Angular", "TypeScript", "Frontend", "Web Development"], url="https://angular.io/tutorial", description="Get started with Angular."),
    Course(id="902", title="Vue.js Essentials", prereqs=[], difficulty=2, topics=["Vue", "JavaScript", "Frontend", "Web Development"], url="https://vuejs.org/guide/quick-start.html", description="Learn Vue.js basics."),
]

def prime_database():
    """Adds seed courses to ChromaDB if they don't exist."""
    print("\n" + "="*80)
    print("PRIMING DATABASE (Initializing Course Catalog)")
    
    ids_to_check = [c.id for c in SEED_COURSES_DATA]
    existing = vectorstore.get(ids=ids_to_check)
    existing_ids = set(existing['ids'])
    
    courses_to_add = [c for c in SEED_COURSES_DATA if c.id not in existing_ids]
    
    if not courses_to_add:
        print("All seed courses already exist in ChromaDB.")
        print("="*80)
        return

    print(f"Adding {len(courses_to_add)} seed courses to ChromaDB...")
    documents = []
    for course in courses_to_add:
        content = f"Course: {course.title}. Topics: {', '.join(course.topics)}. Description: {course.description}"
        metadata = course.model_dump()
        metadata['prereqs'] = ','.join(map(str, metadata['prereqs']))
        metadata['topics'] = ','.join(metadata['topics'])
        documents.append(Document(page_content=content, metadata=metadata))

    vectorstore.add_documents(documents, ids=[c.id for c in courses_to_add])
    print("SUCCESS: Database priming complete.")
    print("="*80)

prime_database()

### -----------------------------------------------------------------
### GRAPH STATE
### -----------------------------------------------------------------

def list_overwrite_reducer(
    existing_list: Optional[List[Any]], 
    new_chunk: Optional[List[Any]]
) -> List[Any]:
    if new_chunk == []:
        return []
    if existing_list is None:
        existing_list = []
    if new_chunk is None:
        return existing_list
    return existing_list + new_chunk

class CurriculumGraphState(TypedDict):
    """State for the curriculum generation graph."""
    # Inputs
    skill_gap_analysis: str
    resume_inventory: str
    
    # Planner Output
    user_profile: Optional[UserProfile]
    skills_to_find: List[str]
    
    # Iteration State
    current_skill: Optional[str]
    course_catalog: List[Course]
    raw_search_results: Optional[List[dict]]
    
    # Evaluation State
    evaluation_logs: Annotated[List[Dict[str, Any]], list_overwrite_reducer]
    all_found_courses: Annotated[List[Dict[str, Any]], operator.add]
    evaluated_courses: List[Dict[str, Any]]
    
    # Architect/Critic State
    messages: Annotated[list[BaseMessage], operator.add]
    draft_path: Optional[PersonalizedLearningPath]
    structured_critique: Optional[Critique]
    revision_count: Annotated[int, operator.add]

### -----------------------------------------------------------------
### GRAPH NODES
### -----------------------------------------------------------------

def entry_node(state: CurriculumGraphState) -> dict:
    print("\n" + "="*80)
    print("INITIALIZING WORKFLOW")
    print("="*80)
    return {
        "messages": [],
        "revision_count": 0,
        "all_found_courses": [],
        "evaluation_logs": []
    }

# Planner Node
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior career coach analyzing skill gaps and user background.

Your job:
1. Create a structured UserProfile from the resume inventory
2. Extract ALL critical skills from the skill gap analysis (typically 5-20 skills)
3. Don't limit yourself - extract EVERY skill mentioned as a gap

IMPORTANT RULES:
- If 10+ skills are listed in gaps, extract ALL of them
- Prioritize skills that appear multiple times or are marked as "critical"
- Include both technical skills and tools
- Infer skill_level from the resume (Beginner/Intermediate/Advanced)
- Estimate past_courses from mentioned projects (use 0 if unclear)

Output a 'PlannerOutput' object."""),
    ("human", """Analyze this information:

**Resume Inventory:**
{resume}

**Skill Gap Analysis:**
{gaps}

Extract ALL skill gaps mentioned (aim for complete coverage, not just top 3-5).""")
])
planner_chain = planner_prompt | llm_structured.with_structured_output(PlannerOutput)

def planner_node(state: CurriculumGraphState) -> dict:
    print("\nEXECUTING: Planner Agent")
    output = planner_chain.invoke({
        "resume": state['resume_inventory'],
        "gaps": state['skill_gap_analysis']
    })
    print(f"Planner identified {len(output.skills_to_find)} skills: {output.skills_to_find}")
    print(f"User profile: {output.user_profile.skill_level} level")
    print(f"User interests: {', '.join(output.user_profile.interests[:5])}")
    return {
        "user_profile": output.user_profile,
        "skills_to_find": output.skills_to_find
    }

# Iteration Controller
def iteration_controller_node(state: CurriculumGraphState) -> dict:
    print("\nEXECUTING: Iteration Controller")
    skills_list = state.get('skills_to_find', [])
    
    if not skills_list:
        print("Iteration complete. Moving to final assembly.")
        
        # Show collection summary
        all_courses = state.get('all_found_courses', [])
        print(f"\nCOLLECTION SUMMARY:")
        print(f"  Total courses collected: {len(all_courses)}")
        
        # Group by skill
        by_skill = {}
        for c in all_courses:
            skill = c.get('skill_name', 'unknown')
            by_skill[skill] = by_skill.get(skill, 0) + 1
        
        for skill, count in sorted(by_skill.items()):
            print(f"  - {skill}: {count} courses")
        
        return {"current_skill": None}
    
    current_skill = skills_list.pop(0)
    remaining = len(skills_list)
    total = len(state.get('skills_to_find', [])) + remaining + 1
    processed = total - remaining - 1
    
    print(f"Processing skill [{processed + 1}/{total}]: {current_skill}")
    
    return {
        "current_skill": current_skill,
        "skills_to_find": skills_list,
        "evaluation_logs": [],
        "course_catalog": []
    }

# Fetch Course Catalog (IMPROVED)
def fetch_course_catalog(state: CurriculumGraphState) -> dict:
    """Retrieves courses with relaxed, fuzzy matching."""
    print("EXECUTING: FetchCourseCatalog (ChromaDB)")
    skill = state['current_skill']
    
    # Increase k for more candidates
    docs = vectorstore.similarity_search(skill, k=10)
    
    found_courses = []
    for doc in docs:
        metadata = doc.metadata.copy()
        if isinstance(metadata.get('prereqs'), str):
            metadata['prereqs'] = [int(p) for p in metadata['prereqs'].split(',') if p]
        if isinstance(metadata.get('topics'), str):
            metadata['topics'] = [t.strip() for t in metadata['topics'].split(',') if t]
        found_courses.append(Course(**metadata))
    
    print(f"Found {len(found_courses)} courses in ChromaDB for '{skill}'")
    
    # RELAXED FUZZY MATCHING
    skill_lower = skill.lower()
    skill_keywords = skill_lower.split()
    
    filtered_courses = []
    for course in found_courses:
        course_topics_lower = [t.lower() for t in course.topics]
        title_lower = course.title.lower()
        desc_lower = course.description.lower()
        
        # Match if ANY keyword appears ANYWHERE
        match = False
        for keyword in skill_keywords:
            # Check topics
            if any(keyword in topic for topic in course_topics_lower):
                match = True
                break
            # Check title
            if keyword in title_lower:
                match = True
                break
            # Check description
            if keyword in desc_lower:
                match = True
                break
        
        if match:
            filtered_courses.append(course)
    
    print(f"{len(filtered_courses)} courses after fuzzy matching.")
    
    # Fallback: return all if no matches
    if not filtered_courses and found_courses:
        print(f"No fuzzy matches. Returning all {len(found_courses)} courses for evaluation.")
        return {"course_catalog": found_courses}
    
    return {"course_catalog": filtered_courses}

# Web Search for Courses
def web_search_for_courses(state: CurriculumGraphState) -> dict:
    """Searches the web for courses if none are found in ChromaDB."""
    print("EXECUTING: WebSearchForCourses (Tavily)")
    skill = state['current_skill']
    query = f"best online courses for {skill} 2024 platform udemy coursera edx"
    
    if tavily_tool is None:
        print("Tavily tool not available. Skipping web search.")
        return {"raw_search_results": []}
    
    try:
        results = tavily_tool.invoke({"query": query})
        print(f"Found {len(results)} search results.")
        return {"raw_search_results": results}
    except Exception as e:
        print(f"ERROR during web search: {e}")
        return {"raw_search_results": []}

# Process and Store Courses
process_search_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data extraction expert. Convert web search results into structured Course objects.

Requirements:
- Create unique IDs (e.g., 'aws-udemy-architect')
- Estimate difficulty realistically (1-5)
- Extract prerequisites (use [0] if none)
- Extract ALL relevant topics
- MUST have valid URL and description
- Ignore results that aren't courses

Respond ONLY with a NewCourseList object."""),
    ("human", """Process these search results for "{skill}":

{search_results}""")
])
process_search_chain = process_search_prompt | llm_structured.with_structured_output(NewCourseList)

def process_and_store_courses(state: CurriculumGraphState) -> dict:
    """Uses LLM to process search results and adds to ChromaDB."""
    print("EXECUTING: ProcessAndStoreCourses")
    skill = state['current_skill']
    search_results = state['raw_search_results']
    
    if not search_results:
        print("No search results to process.")
        return {"course_catalog": []}
    
    try:
        new_courses_obj = process_search_chain.invoke({
            "skill": skill,
            "search_results": pformat(search_results)
        })
        new_courses = new_courses_obj.courses
        print(f"LLM extracted {len(new_courses)} new courses from search.")
    except Exception as e:
        print(f"ERROR during LLM extraction: {e}")
        return {"course_catalog": []}

    if not new_courses:
        print("LLM extracted no courses.")
        return {"course_catalog": []}

    documents = []
    for course in new_courses:
        content = f"Course: {course.title}. Topics: {', '.join(course.topics)}. Description: {course.description}"
        metadata = course.model_dump()
        metadata['prereqs'] = ','.join(map(str, metadata['prereqs']))
        metadata['topics'] = ','.join(metadata['topics'])
        documents.append(Document(page_content=content, metadata=metadata))

    try:
        vectorstore.add_documents(documents, ids=[c.id for c in new_courses])
        print(f"SUCCESS: Added {len(new_courses)} courses to ChromaDB.")
    except Exception as e:
        print(f"ERROR adding to ChromaDB: {e}")

    return {"course_catalog": new_courses}

# Start Evaluation
def start_evaluation(state: CurriculumGraphState) -> dict:
    """Fan-out point for parallel evaluation."""
    print("EXECUTING: StartEvaluation (Triggering parallel checks)")
    return {}

# Check Prerequisites
def check_prerequisites(state: CurriculumGraphState) -> dict:
    """Parallel check: Prerequisites."""
    print("  Parallel check (1/3): Prerequisites")
    profile = state['user_profile']
    catalog = state['course_catalog']
    logs = []
    
    past_courses_set = set(profile.past_courses)
    
    for course in catalog:
        course_prereqs_set = set(int(p) for p in course.prereqs)
        met_prereqs = course_prereqs_set.issubset(past_courses_set)
        logs.append({"type": "prereq", "course_id": course.id, "pass": met_prereqs})
        
    return {"evaluation_logs": logs}

# Match Interests (IMPROVED - More Generous)
def match_interests(state: CurriculumGraphState) -> dict:
    """Parallel check: Interest matching with generous scoring."""
    print("  Parallel check (2/3): Interest matching")
    profile = state['user_profile']
    catalog = state['course_catalog']
    user_interests = set(i.lower() for i in profile.interests)
    logs = []
    
    for course in catalog:
        course_topics = set(t.lower() for t in course.topics)
        overlap = user_interests.intersection(course_topics)
        
        # MORE GENEROUS SCORING
        if not overlap:
            score = 0.5  # 50% base score for any course
        else:
            base_score = len(overlap) / len(user_interests) if user_interests else 0
            score = min(base_score * 1.5, 1.0)  # Boost and cap at 1.0
        
        logs.append({"type": "interest", "course_id": course.id, "score": round(score, 2)})
        
    return {"evaluation_logs": logs}

# Estimate Workload
def estimate_workload(state: CurriculumGraphState) -> dict:
    """Parallel check: Workload appropriateness."""
    print("  Parallel check (3/3): Workload estimation")
    profile = state['user_profile']
    catalog = state['course_catalog']
    logs = []
    
    level_map = {"Beginner": 2, "Intermediate": 4, "Advanced": 5}
    max_difficulty = level_map.get(profile.skill_level, 4)
    
    for course in catalog:
        load_ok = course.difficulty <= max_difficulty
        logs.append({"type": "workload", "course_id": course.id, "load_ok": load_ok})
        
    return {"evaluation_logs": logs}

# Aggregate Scores
def aggregate_scores(state: CurriculumGraphState) -> dict:
    print("EXECUTING: AggregateScores (Reducer)")
    logs = state['evaluation_logs']
    catalog_dict = {c.id: c for c in state['course_catalog']}
    aggregated = {}
    
    for log in logs:
        course_id = log['course_id']
        if course_id not in aggregated:
            aggregated[course_id] = {}
        
        if log['type'] == 'prereq':
            aggregated[course_id]['prereq_pass'] = log['pass']
        elif log['type'] == 'interest':
            aggregated[course_id]['interest_score'] = log['score']
        elif log['type'] == 'workload':
            aggregated[course_id]['workload_ok'] = log['load_ok']
    
    evaluated_courses = []
    for course_id, data in aggregated.items():
        if course_id not in catalog_dict:
            continue
            
        total_score = 0
        if data.get('prereq_pass', False) and data.get('workload_ok', False):
            total_score = data.get('interest_score', 0) * 10
            
        evaluated_courses.append({
            "course": catalog_dict[course_id],
            "score": round(total_score, 2),
            "details": data
        })
        
    print(f"Aggregation complete. Processed {len(evaluated_courses)} courses.")
    return {"evaluated_courses": evaluated_courses}

# Recommend Enrollment (IMPROVED - Top 3)
def recommend_enrollment(state: CurriculumGraphState) -> dict:
    print("EXECUTING: RecommendEnrollment")
    skill = state['current_skill']
    recommendations = []
    
    sorted_evals = sorted(state['evaluated_courses'], key=lambda x: x['score'], reverse=True)
    
    # Include TOP 3 courses that passed
    count = 0
    for course_eval in sorted_evals:
        if course_eval['score'] > 0 and count < 3:
            course = course_eval['course']
            recommendations.append({
                "skill_name": skill,
                "course_title": course.title,
                "course_url": course.url,
                "description": f"Best match for {skill} (Score: {course_eval['score']}/10). {course.description}"
            })
            count += 1
            print(f"  Recommended {count}: {course.title} (Score: {course_eval['score']}/10)")
    
    if not recommendations:
        print("  No courses met enrollment criteria. Using alternatives.")

    return {"all_found_courses": recommendations}

# Suggest Alternatives (IMPROVED - Top 3)
def suggest_alternatives(state: CurriculumGraphState) -> dict:
    print("EXECUTING: SuggestAlternatives")
    skill = state['current_skill']
    alternatives = []
    
    sorted_evals = sorted(state['evaluated_courses'], key=lambda x: x['score'], reverse=True)
    
    # Suggest TOP 3 regardless of score
    for i, course_eval in enumerate(sorted_evals[:3]):
        course = course_eval['course']
        recommendation = {
            "skill_name": skill,
            "course_title": course.title,
            "course_url": course.url,
            "description": f"Alternative for {skill} (Score: {course_eval['score']}/10). {course.description}"
        }
        alternatives.append(recommendation)
        print(f"  Alternative {i+1}: {course.title} (Score: {course_eval['score']}/10)")
    
    if not alternatives:
        print("  No alternatives found.")
    
    return {"all_found_courses": alternatives}

# Architect Node (IMPROVED - With Retry)
@tool("submit_learning_path")
def submit_learning_path(path: dict) -> str:
    """
    Submit the finalized learning path. 
    REQUIRED INPUT FORMAT:

    {
        "path": {
            "user_summary": "...",
            "skill_gaps": [...],
            "recommendations": [...]
        }
    }
    """

    if not isinstance(path, dict):
        raise ValueError("Expected 'path' to be a dictionary.")

    print("\nTool called: submit_learning_path (received keys: %s)" % list(path.keys()))
    return "Learning path submitted successfully."


tools = [submit_learning_path]
model_with_tools = llm_reasoning.bind_tools(tools)


# =====================================================================
# ENFORCED ARCHITECT PROMPT — CORRECT SCHEMA GUARANTEED
# =====================================================================
architect_system_prompt = """
You are the **Curriculum Architect**. Your job is to assemble the FINAL
Personalized Learning Path using the provided Planner output and curated course list.

IMPORTANT — TOOL CALL RULES
------------------------------------------------------------
When calling the tool **submit_learning_path**, you MUST pass arguments
EXACTLY in this structure:

{
  "path": {
    "user_summary": "A 2–3 sentence summary of the learner.",
    "skill_gaps": ["skill1", "skill2", ...],
    "recommendations": [
        {
           "skill_name": "skill",
           "course_title": "Course Title",
           "course_url": "https://...",
           "description": "Short helpful description"
        },
        ... include EVERY vetted course ...
    ]
  }
}

CRITICAL RULES
------------------------------------------------------------
✔ ALWAYS wrap your entire output in `"path"`  
✔ NEVER output `user_summary`, `skill_gaps`, or `recommendations` at the top level  
✔ NEVER omit the `"path"` wrapper  
✔ NEVER rename fields  
✔ NEVER invent new fields  
✔ ALWAYS include **ALL courses provided**  
✔ NO filtering unless explicitly instructed  
✔ Output JSON ONLY inside the tool call  

YOUR TASK
------------------------------------------------------------
1. Summarize the user's background and goals (2–3 sentences).
2. Build skill_gaps using unique skills from the vetted course list.
3. Include EVERY course in the recommendations array.
4. Then call: 
   submit_learning_path({...})
with the EXACT schema above.

Follow the format precisely.
"""


def architect_node(state: CurriculumGraphState) -> dict:
    print(f"\nEXECUTING: ARCHITECT (Revision: {state.get('revision_count', 0)})")
    
    courses = state['all_found_courses']
    print(f"Working with {len(courses)} total courses")
    
    # Group by skill
    by_skill = {}
    for course in courses:
        skill = course.get('skill_name', 'Unknown')
        if skill not in by_skill:
            by_skill[skill] = []
        by_skill[skill].append(course.get('course_title', 'Unknown'))
    
    print("Course distribution:")
    for skill, titles in sorted(by_skill.items()):
        print(f"  {skill}: {len(titles)} courses")
    
    # Build prompt
    if state['revision_count'] == 0:
        prompt_content = f"""
**User_Profile:**
{state['user_profile'].model_dump_json(indent=2)}

**Vetted_Courses ({len(courses)} total):**
{pformat(courses)}

INSTRUCTIONS:
1. Write a 2-3 sentence user_summary
2. Extract {len(by_skill)} unique skill names for skill_gaps
3. Include ALL {len(courses)} courses in recommendations
4. Call submit_learning_path tool with the complete structure
"""
        messages = [
            SystemMessage(content=architect_system_prompt),
            HumanMessage(content=prompt_content)
        ]
    else:
        messages = state['messages']

    # Retry logic
    max_retries = 2
    response = None
    for attempt in range(max_retries):
        try:
            response = model_with_tools.invoke(messages)
            
            # The response object from groq/chat may have a tool_calls attribute or a tool_calls list.
            tool_calls = getattr(response, 'tool_calls', None)
            if tool_calls:
                print(f"SUCCESS: Tool called on attempt {attempt + 1}")
                return {"messages": [response]}
            else:
                print(f"WARNING: No tool call on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    messages.append(response)
                    messages.append(HumanMessage(
                        content="You must call the submit_learning_path tool. Please try again."
                    ))
                else:
                    return {"messages": [response]}
        except Exception as e:
            print(f"ERROR on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
    
    return {"messages": [response]}

# Handle Submission
def handle_submission_node(state: CurriculumGraphState) -> dict:
    print("\nEXECUTING: HandleSubmission")
    last_message = state['messages'][-1]
    
    # If the architect didn't produce tool_calls, return error critique
    tool_calls = getattr(last_message, 'tool_calls', None)
    if not tool_calls:
        print("ERROR: Architect did not call tool!")
        critique = Critique(
            is_approved=False, 
            revisions_needed="""You must call the submit_learning_path tool with a PersonalizedLearningPath object containing:
- user_summary: 2-3 sentences about the user
- skill_gaps: list of skill names from recommendations
- recommendations: list of ALL CourseRecommendation objects provided
Include ALL available courses in your recommendations."""
        )
        tool_msg = ToolMessage(
            content="Submission failed - tool not called.", 
            tool_call_id="error-handling"
        )
        return {
            "structured_critique": critique, 
            "messages": [tool_msg],
            "revision_count": 1
        }
        
    tc = tool_calls[0]
    # tc could be a dict-like structure depending on runtime; normalize access
    try:
        tc_name = tc.get('name') if isinstance(tc, dict) else getattr(tc, 'name', None)
        tc_args = tc.get('args') if isinstance(tc, dict) else getattr(tc, 'args', None)
    except Exception:
        tc_name = None
        tc_args = None

    if tc_name == 'submit_learning_path':
        try:
            # Robust extraction: args may be a dict or a JSON string
            path_data = None
            if isinstance(tc_args, dict):
                path_data = tc_args.get('path') or tc_args
            elif isinstance(tc_args, str):
                path_parsed = json.loads(tc_args)
                path_data = path_parsed.get('path') or path_parsed

            if path_data is None:
                raise ValueError('No "path" key found in tool args')

            # If path_data is a JSON string, decode it
            if isinstance(path_data, str):
                path_data = json.loads(path_data)

            # Validate and create Pydantic model
            draft = PersonalizedLearningPath(**path_data)
            print(f"SUCCESS: Draft path saved with {len(draft.recommendations)} recommendations.")
            tool_msg = ToolMessage(content="Learning path submitted.", tool_call_id=(tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', 'tool')))
            return {"draft_path": draft, "messages": [tool_msg]}
        except Exception as e:
            print(f"ERROR parsing path: {e}")
            critique = Critique(
                is_approved=False,
                revisions_needed=f"Error parsing your submission: {str(e)}. Please ensure all fields are valid."
            )
            tool_msg = ToolMessage(content=f"Error: {str(e)}", tool_call_id="error")
            return {
                "structured_critique": critique,
                "messages": [tool_msg],
                "revision_count": 1
            }
    
    return {}

# Critic Node (IMPROVED - More Lenient)
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior curriculum quality reviewer ensuring learning paths are complete and useful.

EVALUATION CRITERIA:
1. user_summary: Clear, relevant, 2-3 sentences? (REQUIRED)
2. skill_gaps: Matches skills in recommendations? (REQUIRED)
3. recommendations: At least 1 course per skill gap? (REQUIRED)
4. Overall: Coherent and actionable?

APPROVAL GUIDELINES (BE LENIENT):
- APPROVE if: Each skill gap has 1+ recommendations, summary is clear, structure is valid
- APPROVE if: 5 gaps with 8+ total recommendations (multiple per gap is GOOD)
- APPROVE if: 1-2 courses per gap (this is ACCEPTABLE)

REJECT ONLY IF:
- Missing recommendations for >50% of gaps
- User summary is nonsensical or missing
- Obvious structural errors
- Recommendations don't align with skill_gaps at all

Be SPECIFIC but CONCISE about improvements needed."""),
    ("human", "Review this learning path:\n\n{draft_path_json}")
])
critic_chain = critic_prompt | llm_reasoning.with_structured_output(Critique)

def critic_node(state: CurriculumGraphState) -> dict:
    print("\nEXECUTING: CRITIC")
    critique_result = critic_chain.invoke({
        "draft_path_json": state["draft_path"].model_dump_json(indent=2)
    })
    
    approval_status = "APPROVED" if critique_result.is_approved else "NEEDS REVISION"
    print(f"Critique result: {approval_status}")
    
    if not critique_result.is_approved:
        print(f"Issues: {critique_result.revisions_needed[:100]}...")
    
    return {
        "structured_critique": critique_result,
        "revision_count": 1
    }

# Prepare Revision
def prepare_revision_node(state: CurriculumGraphState) -> dict:
    print(f"\nPREPARING REVISION (Attempt {state.get('revision_count', 0) + 1})")
    critique = state.get('structured_critique')
    
    if critique is None:
        revision_msg = HumanMessage(
            content="""
You must revise the learning path assembly.
Call the submit_learning_path tool with a complete PersonalizedLearningPath object including ALL recommendations.
"""
        )
    else:
        revision_msg = HumanMessage(
            content=f"""
**Feedback from Reviewer:**
{critique.revisions_needed}

Address these points and resubmit using the submit_learning_path tool.
"""
        )
    return {"messages": [revision_msg]}

### -----------------------------------------------------------------
### CONDITIONAL ROUTING
### -----------------------------------------------------------------

def should_continue_iteration(state: CurriculumGraphState) -> str:
    """Router for iteration controller."""
    if state['current_skill'] is None:
        return "architect"
    else:
        return "fetch_course_catalog"

def should_search_web(state: CurriculumGraphState) -> str:
    """Router after fetching from ChromaDB."""
    if not state['course_catalog']:
        return "search_web"
    else:
        return "evaluate_courses"

def should_route_recommendation(state: CurriculumGraphState) -> str:
    """Router for recommendation branching."""
    if not state['evaluated_courses']:
        return "suggest_alternatives"
        
    if not any(c["score"] > 0 for c in state['evaluated_courses']):
        return "suggest_alternatives"
    else:
        return "recommend_enrollment"

def should_call_submit_tool(state: CurriculumGraphState) -> str:
    """Router after architect runs."""
    if state.get('draft_path'):
        return "critic"
    else:
        return "handle_submission"
    
def should_continue_or_end_critique(state: CurriculumGraphState) -> str:
    """Router for critic's decision."""
    MAX_REVISIONS = 2
    
    if state["structured_critique"].is_approved:
        return "__end__"
    
    if state["revision_count"] >= MAX_REVISIONS:
        print(f"Reached max revisions ({MAX_REVISIONS}). Ending.")
        return "__end__"
    
    return "prepare_revision"

### -----------------------------------------------------------------
### BUILD GRAPH
### -----------------------------------------------------------------

print("\n" + "="*80)
print("BUILDING CURRICULUM ARCHITECT GRAPH")
print("="*80)

workflow = StateGraph(CurriculumGraphState)

# Add all nodes
workflow.add_node("entry", entry_node)
workflow.add_node("planner", planner_node)
workflow.add_node("iteration_controller", iteration_controller_node)
workflow.add_node("fetch_course_catalog", fetch_course_catalog)
workflow.add_node("web_search_for_courses", web_search_for_courses)
workflow.add_node("process_and_store_courses", process_and_store_courses)
workflow.add_node("start_evaluation", start_evaluation)
workflow.add_node("check_prerequisites", check_prerequisites)
workflow.add_node("match_interests", match_interests)
workflow.add_node("estimate_workload", estimate_workload)
workflow.add_node("aggregate_scores", aggregate_scores)
workflow.add_node("recommend_enrollment", recommend_enrollment)
workflow.add_node("suggest_alternatives", suggest_alternatives)
workflow.add_node("architect", architect_node)
workflow.add_node("handle_submission", handle_submission_node)
workflow.add_node("critic", critic_node)
workflow.add_node("prepare_revision", prepare_revision_node)

# Add edges
workflow.set_entry_point("entry")
workflow.add_edge("entry", "planner")
workflow.add_edge("planner", "iteration_controller")

# Iteration loop
workflow.add_conditional_edges(
    "iteration_controller",
    should_continue_iteration,
    {
        "fetch_course_catalog": "fetch_course_catalog",
        "architect": "architect"
    }
)

# RAG/Search branch
workflow.add_conditional_edges(
    "fetch_course_catalog",
    should_search_web,
    {
        "evaluate_courses": "start_evaluation",
        "search_web": "web_search_for_courses"
    }
)

# Web search branch
workflow.add_edge("web_search_for_courses", "process_and_store_courses")
workflow.add_edge("process_and_store_courses", "check_prerequisites")
workflow.add_edge("process_and_store_courses", "match_interests")
workflow.add_edge("process_and_store_courses", "estimate_workload")

# Evaluation branch
workflow.add_edge("start_evaluation", "check_prerequisites")
workflow.add_edge("start_evaluation", "match_interests")
workflow.add_edge("start_evaluation", "estimate_workload")

# Reducer edges
workflow.add_edge("check_prerequisites", "aggregate_scores")
workflow.add_edge("match_interests", "aggregate_scores")
workflow.add_edge("estimate_workload", "aggregate_scores")

# Conditional routing
workflow.add_conditional_edges(
    "aggregate_scores",
    should_route_recommendation,
    {
        "recommend_enrollment": "recommend_enrollment",
        "suggest_alternatives": "suggest_alternatives"
    }
)

# Loop back to controller
workflow.add_edge("recommend_enrollment", "iteration_controller")
workflow.add_edge("suggest_alternatives", "iteration_controller")

# Architect/Critic loop
workflow.add_conditional_edges(
    "architect",
    should_call_submit_tool,
    {
        "critic": "handle_submission",
        "handle_submission": "handle_submission"
    }
)

workflow.add_conditional_edges(
    "handle_submission",
    lambda state: "critic" if state.get('draft_path') else "prepare_revision",
    {
        "critic": "critic",
        "prepare_revision": "prepare_revision"
    }
)

workflow.add_conditional_edges(
    "critic",
    should_continue_or_end_critique,
    {
        "prepare_revision": "prepare_revision",
        "__end__": END
    }
)
workflow.add_edge("prepare_revision", "architect")

# Compile
app = workflow.compile()

print("SUCCESS: Graph compilation complete!")
try:
    app.get_graph().draw_png("curriculum_workflow_graph.png")
    print("SUCCESS: Graph diagram saved to 'curriculum_workflow_graph.png'")
except ImportError:
    print("INFO: Install 'pygraphviz' to visualize the graph: pip install pygraphviz")

### -----------------------------------------------------------------
### MAIN EXECUTION
### -----------------------------------------------------------------

if __name__ == "__main__":
    """
    Standalone execution of Curriculum Architect.
    For production use, this agent is orchestrated by supervisor_agent.py
    """
    print("\n" + "="*80)
    print("CURRICULUM ARCHITECT - Standalone Mode")
    print("="*80)
    print("\nThis agent is designed to be orchestrated by the supervisor agent.")
    print("For standalone testing, provide skill_gap_analysis and resume_inventory.\n")
    
    # Get user input
    print("Enter skill gap analysis (or press Enter for example):")
    skill_gap_input = input().strip()
    if not skill_gap_input:
        skill_gap_input = """
Target Role: Full Stack Developer
Location: Delhi

CRITICAL SKILL GAPS (Top Priority):
  1. angular
  2. aws
  3. bootstrap
  4. ci/cd
  5. docker
  6. kubernetes
  7. css
  8. html
  9. react
  10. vue.js

USER'S CURRENT SKILLS:
C, C++, JavaScript, Python, Node.js, Express.js, MongoDB, PostgreSQL

MARKET ANALYSIS:
- Jobs Found: 8
- Total Gaps Identified: 20
- Most In-Demand: angular, aws, css, docker, react
"""
    
    print("\nEnter resume inventory (or press Enter for example):")
    resume_input = input().strip()
    if not resume_input:
        resume_input = """
USER PROFILE:
- Target Role: Full Stack Developer
- Experience Level: Intermediate
- Location: Delhi

TECHNICAL SKILLS (33 total):
  C, C++, Chakra UI, Chroma, Docker, Express.js
  Git, GitHub, JavaScript, LangChain, LangGraph
  Machine Learning, MongoDB, Next.js, Node.js, NumPy
  OpenCV, Pandas, PostgreSQL, Python, React.js
  REST APIs, Scikit-learn, Socket.io, TypeScript

INTERESTS (inferred from skills):
  Frontend Development (React, Next.js)
  Backend Development (Node.js, APIs)
  Machine Learning & AI
"""
    
    inputs = {
        "skill_gap_analysis": skill_gap_input,
        "resume_inventory": resume_input,
    }
    
    config = {"recursion_limit": 100}
    
    try:
        final_state = app.invoke(inputs, config=config)
        
        print("\n" + "="*80)
        print("CURRICULUM ARCHITECT WORKFLOW COMPLETE")
        print("="*80)
        
        if final_state.get('draft_path'):
            print("\nFINAL PERSONALIZED LEARNING PATH:\n")
            path = final_state['draft_path']
            
            print("USER SUMMARY:")
            print(f"  {path.user_summary}\n")
            
            print("SKILL GAPS ADDRESSED:")
            for i, gap in enumerate(path.skill_gaps, 1):
                print(f"  {i}. {gap}")
            
            print(f"\nRECOMMENDED COURSES ({len(path.recommendations)} total):")
            
            # Group by skill
            by_skill = {}
            for rec in path.recommendations:
                skill = rec.skill_name
                if skill not in by_skill:
                    by_skill[skill] = []
                by_skill[skill].append(rec)
            
            # Print grouped
            for skill, courses in sorted(by_skill.items()):
                print(f"\n  {skill.upper()}:")
                for course in courses:
                    print(f"    - {course.course_title}")
                    print(f"      URL: {course.course_url}")
                    print(f"      {course.description}\n")
        else:
            print("\nWARNING: No draft_path was generated.")
            print("Final Critique:", final_state.get('structured_critique'))
        
    except Exception as e:
        print(f"\nERROR: Workflow failed!")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
