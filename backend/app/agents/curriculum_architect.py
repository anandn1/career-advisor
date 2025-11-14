### -----------------------------------------------------------------
### 1. IMPORTS & SETUP
### -----------------------------------------------------------------
import os
import operator
import uuid
from typing import List, Optional, TypedDict, Annotated, Dict, Any
from pprint import pprint, pformat

# Pydantic is used for structured output and state management
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
from langchain_tavily import TavilySearchResults

print("Loading environment variables...")
load_dotenv()

# --- Set your API Keys ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    print("❌ ERROR: API keys (GROQ_API_KEY, TAVILY_API_KEY) not found in .env file.")
    exit()
else:
    print("✅ GROQ API key loaded.")
    print("✅ Tavily API key loaded.")

# --- Initialize LLM ---
llm_reasoning = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
llm_structured = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)

### -----------------------------------------------------------------
### 2. TOOLS & DB SETUP
### -----------------------------------------------------------------

print("Initializing tools and database...")

# --- Initialize Web Search Tool ---
tavily_tool = TavilySearchResults(max_results=5)
print("✅ Tavily search tool initialized.")

# --- Initialize Embeddings ---
model_name="google/embeddinggemma-300m"
model_kwargs={'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(f"✅ Embeddings model ({model_name}) loaded.")

# --- Initialize ChromaDB (Vector Store) ---
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="course_catalog"
)
print("✅ ChromaDB vector store initialized.")


### -----------------------------------------------------------------
### 3. PYDANTIC SCHEMAS (Data Structures)
### -----------------------------------------------------------------

class Course(BaseModel):
    """Database schema for a course."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    prereqs: List[int] = Field(default_factory=list, description="List of course IDs (use 0 if none)")
    difficulty: int = Field(..., ge=1, le=5, description="1=easy, 5=hard")
    topics: List[str] = Field(default_factory=list, description="List of relevant topics, e.g., 'Python', 'FastAPI'")
    url: str = Field(..., description="The URL to the course")
    description: str = Field(..., description="A short description of the course")

class NewCourseList(BaseModel):
    """Schema for extracting courses from web search results."""
    courses: List[Course]

class UserProfile(BaseModel):
    """Structured user profile derived from resume."""
    interests: List[str] = Field(..., description="List of technical interests, e.g., 'Machine Learning', 'Web Development'")
    skill_level: str = Field(..., description="Estimated skill level: 'Beginner', 'Intermediate', or 'Advanced'")
    past_courses: List[int] = Field(..., description="List of course IDs from projects/resume (use 0 if none)")

class PlannerOutput(BaseModel):
    """The structured output from the initial Planner node."""
    user_profile: UserProfile
    skills_to_find: List[str] = Field(..., description="The list of 3-5 critical skill gaps to search for.")

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
### 3b. PRIME THE DATABASE
### -----------------------------------------------------------------

MOCK_COURSES_DATA = [
    Course(id="101", title="Introduction to Python", prereqs=[], difficulty=1, topics=["Python", "Beginner"], url="https://mock.com/101", description="Python basics."),
    Course(id="205", title="Advanced FastAPI", prereqs=[101], difficulty=4, topics=["Python", "Web Development", "FastAPI", "API"], url="https://mock.com/205", description="Deep dive into FastAPI."),
    Course(id="206", title="FastAPI for Beginners", prereqs=[101], difficulty=2, topics=["Python", "Web Development", "FastAPI", "API"], url="https://mock.com/206", description="Get started with FastAPI."),
    Course(id="303", title="Data Science with Pandas", prereqs=[101], difficulty=3, topics=["Python", "Machine Learning", "Pandas"], url="https://mock.com/303", description="Pandas for data analysis."),
    Course(id="501", title="PostgreSQL: SQL & Database Management", prereqs=[], difficulty=3, topics=["Database", "PostgreSQL", "SQL"], url="https://mock.com/501", description="Learn SQL and Postgres."),
    Course(id="502", title="Advanced SQL for Data Scientists", prereqs=[501], difficulty=4, topics=["Database", "PostgreSQL", "SQL"], url="https://mock.com/502", description="Complex SQL queries."),
    Course(id="700", title="LangChain for LLM App Development", prereqs=[101], difficulty=4, topics=["Python", "LangChain", "AI", "LLM"], url="https://mock.com/700", description="Build LLM apps."),
    Course(id="701", title="LangChain & Vector Databases", prereqs=[101, 700], difficulty=5, topics=["Python", "LangChain", "AI", "LLM", "Database"], url="https://mock.com/701", description="LangChain with Chroma/Pinecone."),
    Course(id="800", title="Docker & Kubernetes", prereqs=[101], difficulty=4, topics=["Deployment", "Docker", "Kubernetes"], url="https://mock.com/800", description="Containerization and orchestration.")
]

def prime_database():
    """Adds the mock courses to ChromaDB if they don't exist."""
    print("\n" + "="*80)
    print("===> 💾 PRIMING DATABASE")
    
    ids_to_check = [c.id for c in MOCK_COURSES_DATA]
    existing = vectorstore.get(ids=ids_to_check)
    existing_ids = set(existing['ids'])
    
    courses_to_add = [c for c in MOCK_COURSES_DATA if c.id not in existing_ids]
    
    if not courses_to_add:
        print("--- All mock courses already exist in ChromaDB.")
        print("="*80)
        return

    print(f"--- Adding {len(courses_to_add)} new mock courses to ChromaDB...")
    documents = []
    for course in courses_to_add:
        content = f"Course: {course.title}. Topics: {', '.join(course.topics)}. Description: {course.description}"
        metadata = course.model_dump()
        metadata['prereqs'] = ','.join(map(str, metadata['prereqs']))
        metadata['topics'] = ','.join(metadata['topics'])
        documents.append(Document(page_content=content, metadata=metadata))

    vectorstore.add_documents(documents, ids=[c.id for c in courses_to_add])
    print("--- ✅ Database priming complete.")
    print("="*80)

prime_database()


### -----------------------------------------------------------------
### 4. GRAPH STATE
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
    """Represents the state of our combined graph."""
    # --- Inputs ---
    skill_gap_analysis: str
    resume_inventory: str
    
    # --- Planner Output ---
    user_profile: Optional[UserProfile]
    skills_to_find: List[str]
    
    # --- Iteration State ---
    current_skill: Optional[str]
    course_catalog: List[Course]
    raw_search_results: Optional[List[dict]]
    
    # --- Evaluation State ---
    evaluation_logs: Annotated[List[Dict[str, Any]], list_overwrite_reducer]
    all_found_courses: Annotated[List[Dict[str, Any]], operator.add]
    evaluated_courses: List[Dict[str, Any]]
    
    # --- Architect/Critic State ---
    messages: Annotated[list[BaseMessage], operator.add]
    draft_path: Optional[PersonalizedLearningPath]
    structured_critique: Optional[Critique]
    revision_count: Annotated[int, operator.add]


### -----------------------------------------------------------------
### 5. GRAPH NODES
### -----------------------------------------------------------------

# --- Node 1: Entry Node ---
def entry_node(state: CurriculumGraphState) -> dict:
    print("\n" + "="*80)
    print("===> 🚀 INITIALIZING WORKFLOW")
    print("="*80)
    return {
        "messages": [],
        "revision_count": 0,
        "all_found_courses": [],
        "evaluation_logs": []
    }

# --- Node 2: Planner ---
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior career coach. Your job is to read a user's resume and a skill gap analysis.
Based *only* on this information, you must:
1.  Create a structured UserProfile. (Estimate 'past_courses' from resume projects, e.g., 'Basic ML model with Scikit-learn (used course 101)' means they have [101]).
2.  Extract the list of critical skills to search for.
You must output a 'PlannerOutput' object."""),
    ("human", """Here is the information:

**Resume Inventory:**
{resume}

**Skill Gap Analysis:**
{gaps}""")
])
planner_chain = planner_prompt | llm_structured.with_structured_output(PlannerOutput)

def planner_node(state: CurriculumGraphState) -> dict:
    print("\n===> 🧠 EXECUTING: Planner Agent")
    output = planner_chain.invoke({
        "resume": state['resume_inventory'],
        "gaps": state['skill_gap_analysis']
    })
    print(f"--- Planner identified skills: {output.skills_to_find}")
    print(f"--- Planner created profile for: {output.user_profile.skill_level} user")
    return {
        "user_profile": output.user_profile,
        "skills_to_find": output.skills_to_find
    }

# --- Node 3: Iteration Controller ---
def iteration_controller_node(state: CurriculumGraphState) -> dict:
    print("\n===> 🔄 EXECUTING: Iteration Controller")
    skills_list = state.get('skills_to_find', [])
    if not skills_list:
        print("--- Iteration complete. Routing to Architect.")
        return {"current_skill": None}
    current_skill = skills_list.pop(0)
    print(f"--- Next skill to process: {current_skill}")
    return {
        "current_skill": current_skill,
        "skills_to_find": skills_list,
        "evaluation_logs": [],
        "course_catalog": []
    }

# --- Node 4: FetchCourseCatalog ---
def fetch_course_catalog(state: CurriculumGraphState) -> dict:
    """Retrieves available courses for the current skill from ChromaDB."""
    print("===> 📚 EXECUTING: FetchCourseCatalog (from ChromaDB)")
    skill = state['current_skill']
    
    docs = vectorstore.similarity_search(skill, k=5)
    
    found_courses = []
    for doc in docs:
        metadata = doc.metadata.copy()
        if isinstance(metadata.get('prereqs'), str):
            metadata['prereqs'] = [int(p) for p in metadata['prereqs'].split(',') if p]
        if isinstance(metadata.get('topics'), str):
            metadata['topics'] = [t.strip() for t in metadata['topics'].split(',') if t]
        found_courses.append(Course(**metadata))
    
    print(f"--- Found {len(found_courses)} courses in ChromaDB for '{skill}'")
    
    skill_lower = skill.lower()
    filtered_courses = [
        course for course in found_courses
        if skill_lower in (topic.lower() for topic in course.topics)
    ]
    
    print(f"--- {len(filtered_courses)} courses after precise topic filtering.")
    
    return {"course_catalog": filtered_courses}

# --- Node 5: WebSearchForCourses ---
def web_search_for_courses(state: CurriculumGraphState) -> dict:
    """Searches the web for courses if none are found in ChromaDB."""
    print("===> 🌐 EXECUTING: WebSearchForCourses (using Tavily)")
    skill = state['current_skill']
    query = f"best online courses for {skill} including platform and description"
    
    try:
        results = tavily_tool.invoke({"query": query})
        print(f"--- Found {len(results)} search results.")
        return {"raw_search_results": results}
    except Exception as e:
        print(f"--- ❌ ERROR during web search: {e}")
        return {"raw_search_results": []}

# --- Node 6: ProcessAndStoreCourses ---
process_search_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data extraction expert. Your job is to convert raw web search results into a list of structured `Course` objects.
- Create a short, unique `id` for each course (e.g., 'fastapi-on-coursera').
- Realistically estimate `difficulty` (1-5).
- Extract or infer `prereqs`. Use `[0]` if none are obvious.
- Extract key `topics`.
- You MUST populate `url` and `description`.
- If a search result is not a course, ignore it.
Respond ONLY with a `NewCourseList` object."""),
    ("human", """Please process these search results for "{skill}":

{search_results}""")
])
process_search_chain = process_search_prompt | llm_structured.with_structured_output(NewCourseList)

def process_and_store_courses(state: CurriculumGraphState) -> dict:
    """Uses an LLM to process search results and adds new courses to ChromaDB."""
    print("===> 🛠️ EXECUTING: ProcessAndStoreCourses")
    skill = state['current_skill']
    search_results = state['raw_search_results']
    
    if not search_results:
        print("--- No search results to process. Returning empty catalog.")
        return {"course_catalog": []}
    
    try:
        new_courses_obj = process_search_chain.invoke({
            "skill": skill,
            "search_results": pformat(search_results)
        })
        new_courses = new_courses_obj.courses
        print(f"--- LLM extracted {len(new_courses)} new courses from search.")
    except Exception as e:
        print(f"--- ❌ ERROR during LLM extraction: {e}")
        return {"course_catalog": []}

    if not new_courses:
        print("--- LLM extracted no courses. Returning empty catalog.")
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
        print(f"--- ✅ Successfully added {len(new_courses)} new courses to ChromaDB.")
    except Exception as e:
        print(f"--- ❌ ERROR adding to ChromaDB: {e}")

    return {"course_catalog": new_courses}

# --- Node 7: StartEvaluation ---
def start_evaluation(state: CurriculumGraphState) -> dict:
    """Dummy node to act as a clear fan-out point for parallel evaluation."""
    print("===> 🏁 EXECUTING: StartEvaluation (Triggering parallel checks)")
    return {}

# --- Node 8: CheckPrerequisites ---
def check_prerequisites(state: CurriculumGraphState) -> dict:
    """Node 8 (Parallel): Checks prerequisites."""
    print("===>  parallel: (1/3) CheckPrerequisites")
    profile = state['user_profile']
    catalog = state['course_catalog']
    logs = []
    
    past_courses_set = set(profile.past_courses)
    
    for course in catalog:
        course_prereqs_set = set(int(p) for p in course.prereqs)
        met_prereqs = course_prereqs_set.issubset(past_courses_set)
        logs.append({"type": "prereq", "course_id": course.id, "pass": met_prereqs})
        
    return {"evaluation_logs": logs}

# --- Node 9: MatchInterests (IMPROVED) ---
def match_interests(state: CurriculumGraphState) -> dict:
    """Node 9 (Parallel): Matches interests with relaxed scoring."""
    print("===>  parallel: (2/3) MatchInterests")
    profile = state['user_profile']
    catalog = state['course_catalog']
    user_interests = set(i.lower() for i in profile.interests)
    logs = []
    
    for course in catalog:
        course_topics = set(t.lower() for t in course.topics)
        overlap = user_interests.intersection(course_topics)
        
        # Relaxed matching: give partial credit even if no direct match
        if not overlap:
            score = 0.3  # 30% score for non-matched topics
        else:
            score = len(overlap) / len(user_interests) if user_interests else 0
        
        logs.append({"type": "interest", "course_id": course.id, "score": round(score, 2)})
        
    return {"evaluation_logs": logs}

# --- Node 10: EstimateWorkload ---
def estimate_workload(state: CurriculumGraphState) -> dict:
    """Node 10 (Parallel): Estimates workload."""
    print("===>  parallel: (3/3) EstimateWorkload")
    profile = state['user_profile']
    catalog = state['course_catalog']
    logs = []
    
    level_map = {"Beginner": 2, "Intermediate": 4, "Advanced": 5}
    max_difficulty = level_map.get(profile.skill_level, 3)
    
    for course in catalog:
        load_ok = course.difficulty <= max_difficulty
        logs.append({"type": "workload", "course_id": course.id, "load_ok": load_ok})
        
    return {"evaluation_logs": logs}

# --- Node 11: AggregateScores ---
def aggregate_scores(state: CurriculumGraphState) -> dict:
    print("===> 📉 EXECUTING: AggregateScores (Reducer)")
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
        
    print(f"--- Aggregation complete. Processed {len(evaluated_courses)} courses.")
    return {"evaluated_courses": evaluated_courses}

# --- Node 12: RecommendEnrollment (IMPROVED) ---
def recommend_enrollment(state: CurriculumGraphState) -> dict:
    print("===> ✅ EXECUTING: RecommendEnrollment")
    skill = state['current_skill']
    recommendations = []
    
    sorted_evals = sorted(state['evaluated_courses'], key=lambda x: x['score'], reverse=True)
    
    # Include TOP 2 courses that passed (score > 0)
    count = 0
    for course_eval in sorted_evals:
        if course_eval['score'] > 0 and count < 2:
            course = course_eval['course']
            recommendations.append({
                "skill_name": skill,
                "course_title": course.title,
                "course_url": course.url,
                "description": f"Best match for {skill} (Score: {course_eval['score']}/10). {course.description}"
            })
            count += 1
            print(f"--- Identified recommendation {count}: {course.title}")
    
    if not recommendations:
        print("--- No courses met the criteria for enrollment. Will use alternatives.")

    return {"all_found_courses": recommendations}

# --- Node 13: SuggestAlternatives (IMPROVED) ---
def suggest_alternatives(state: CurriculumGraphState) -> dict:
    print("===> ⚠️  EXECUTING: SuggestAlternatives")
    skill = state['current_skill']
    alternatives = []
    
    sorted_evals = sorted(state['evaluated_courses'], key=lambda x: x['score'], reverse=True)
    
    # Suggest TOP 3 courses regardless of score (if any exist)
    for i, course_eval in enumerate(sorted_evals[:3]):
        course = course_eval['course']
        recommendation = {
            "skill_name": skill,
            "course_title": f"{course.title}",
            "course_url": course.url,
            "description": f"Suggested alternative for {skill} (Score: {course_eval['score']}/10). {course.description}"
        }
        alternatives.append(recommendation)
        print(f"--- Suggested alternative {i+1}: {course.title}")
    
    if not alternatives:
        print("--- No suitable alternatives found.")
    
    return {"all_found_courses": alternatives}


# --- Nodes 14-18: Architect/Critic Workflow ---

@tool
def submit_learning_path(path: PersonalizedLearningPath) -> str:
    """Submits the finalized, complete PersonalizedLearningPath for review."""
    print("\n---: `submit_learning_path` tool called ---")
    return "Learning path submitted successfully for review."

tools = [submit_learning_path]
model_with_tools = llm_reasoning.bind_tools(tools)

architect_system_prompt = """You are the 'Curriculum Architect'.
Your job is to assemble a complete PersonalizedLearningPath from the vetted courses.

CRITICAL REQUIREMENTS:
1. Read the `User_Profile` - understand their goals and interests.
2. Read the `Vetted_Courses` list - these are the only courses you can recommend.
3. Create a 2-3 sentence `user_summary` based on their profile.
4. Extract unique skill names from the recommendations to create `skill_gaps`.
5. Populate `recommendations` list with ALL available courses from Vetted_Courses.
6. If there are fewer courses than expected, still include all available ones.

You MUST use the `submit_learning_path` tool with a complete PersonalizedLearningPath.
Do NOT invent courses. Use ONLY what's in Vetted_Courses."""

def architect_node(state: CurriculumGraphState) -> dict:
    print(f"\n===> 📝 EXECUTING ARCHITECT (Revision: {state.get('revision_count', 0)})")
    
    # Debug: Show what we're working with
    print(f"--- DEBUG: Found {len(state['all_found_courses'])} courses total")
    for i, course in enumerate(state['all_found_courses']):
        print(f"    {i+1}. {course.get('course_title', 'Unknown')} ({course.get('skill_name', 'Unknown skill')})")
    
    if state['revision_count'] == 0:
        prompt_content = f"""
**Inputs:**
**User_Profile:**
{state['user_profile'].model_dump_json(indent=2)}

**Vetted_Courses ({len(state['all_found_courses'])} total):**
{pformat(state['all_found_courses'])}

Please assemble ALL of these into a comprehensive `PersonalizedLearningPath` and submit it.
IMPORTANT: Include ALL courses provided, not just some of them.
"""
        messages = [
            SystemMessage(content=architect_system_prompt),
            HumanMessage(content=prompt_content)
        ]
    else:
        messages = state['messages']

    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def handle_submission_node(state: CurriculumGraphState) -> dict:
    print(f"\n===> 💾 SAVING DRAFT")
    last_message = state['messages'][-1]
    
    if not last_message.tool_calls:
        print("--- [Graph]: ❌ ERROR - Architect did not call tool!")
        critique = Critique(
            is_approved=False, 
            revisions_needed="""You must call the `submit_learning_path` tool with a PersonalizedLearningPath object containing:
- user_summary: 2-3 sentences about the user
- skill_gaps: list of skill names from recommendations
- recommendations: list of ALL CourseRecommendation objects provided
Make sure you include ALL available courses in your recommendations."""
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
        
    tc = last_message.tool_calls[0]
    if tc['name'] == 'submit_learning_path':
        try:
            path_data = tc['args']['path']
            draft = PersonalizedLearningPath(**path_data)
            print(f"--- ✅ Draft path saved with {len(draft.recommendations)} recommendations.")
            tool_msg = ToolMessage(content="Learning path submitted.", tool_call_id=tc['id'])
            return {"draft_path": draft, "messages": [tool_msg]}
        except Exception as e:
            print(f"--- ❌ ERROR parsing path: {e}")
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

# --- Critic Node (IMPROVED) ---
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior curriculum quality reviewer.
Your job is to ensure the learning path is complete and useful.
Focus on:
1. Is the user_summary clear and matches their profile?
2. Are there AT LEAST 2-3 recommendations (one for each skill gap)?
3. Do the recommendations match the skill_gaps list?
4. If only 1 recommendation exists when there should be 3, mark as NOT APPROVED.
5. Overall coherence and presentation.

Be SPECIFIC about what needs to be improved.
"""),
    ("human", "Review this learning path:\n\n{draft_path_json}")
])
critic_chain = critic_prompt | llm_reasoning.with_structured_output(Critique)

def critic_node(state: CurriculumGraphState) -> dict:
    print(f"\n===> 🧐 EXECUTING CRITIC")
    critique_result = critic_chain.invoke({
        "draft_path_json": state["draft_path"].model_dump_json(indent=2)
    })
    
    print(f"--- Critique approved: {critique_result.is_approved}")
    return {
        "structured_critique": critique_result,
        "revision_count": 1
    }

# --- Revision Node (IMPROVED) ---
def prepare_revision_node(state: CurriculumGraphState) -> dict:
    print(f"\n===> 🔄 PREPARING REVISION (Attempt {state.get('revision_count', 0) + 1})")
    critique = state.get('structured_critique')
    
    if critique is None:
        revision_msg = HumanMessage(
            content="""
You must revise your assembly of the learning path.
Ensure you call the `submit_learning_path` tool with a complete PersonalizedLearningPath object that includes ALL recommendations.
"""
        )
    else:
        revision_msg = HumanMessage(
            content=f"""
**Feedback from Reviewer:**
{critique.revisions_needed}

Please address these points and resubmit the learning path using the `submit_learning_path` tool.
"""
        )
    return {"messages": [revision_msg]}


### -----------------------------------------------------------------
### 6. CONDITIONAL LOGIC
### -----------------------------------------------------------------

def should_continue_iteration(state: CurriculumGraphState) -> str:
    """Router for the iteration controller."""
    print(f"\n===> ❓ DECIDING: Iterate or Assemble?")
    if state['current_skill'] is None:
        print("--- Routing to Architect.")
        return "architect"
    else:
        print("--- Routing to FetchCourseCatalog.")
        return "fetch_course_catalog"

def should_search_web(state: CurriculumGraphState) -> str:
    """Router after fetching from ChromaDB."""
    print(f"\n===> ❓ DECIDING: Found in DB or Search Web?")
    if not state['course_catalog']:
        print("--- No courses found in DB. Routing to WebSearch.")
        return "search_web"
    else:
        print("--- Courses found in DB. Routing to Evaluation.")
        return "evaluate_courses"

def should_route_recommendation(state: CurriculumGraphState) -> str:
    """Router for recommendation branching."""
    print(f"\n===> ❓ DECIDING: Recommend or Suggest?")
    if not state['evaluated_courses']:
        print("--- No courses. Routing to SuggestAlternatives.")
        return "suggest_alternatives"
        
    if not any(c["score"] > 0 for c in state['evaluated_courses']):
        print(f"--- No courses passed evaluation. Routing to SuggestAlternatives.")
        return "suggest_alternatives"
    else:
        print(f"--- Courses passed evaluation. Routing to RecommendEnrollment.")
        return "recommend_enrollment"

def should_call_submit_tool(state: CurriculumGraphState) -> str:
    """Router after the architect runs."""
    print(f"\n===> ❓ DECIDING: Architect Action?")
    if state.get('draft_path'):
        print("--- Architect called tool. Routing to Critic.")
        return "critic"
    else:
        print("--- ❌ ERROR: Architect failed to call tool. Routing to HandleSubmission.")
        return "handle_submission"
    
def should_continue_or_end_critique(state: CurriculumGraphState) -> str:
    """Router for the critic's decision."""
    print(f"\n===> ❓ DECIDING: Critique Result?")
    MAX_REVISIONS = 2
    
    if state["structured_critique"].is_approved:
        print("--- ✅ Critique is APPROVED. Ending workflow.")
        return "__end__"
    
    if state["revision_count"] >= MAX_REVISIONS:
        print(f"--- ❌ Reached max revisions ({MAX_REVISIONS}). Ending.")
        return "__end__"
    
    print("--- ❌ Critique found issues. Routing to PrepareRevision.")
    return "prepare_revision"


### -----------------------------------------------------------------
### 7. BUILD AND COMPILE THE GRAPH
### -----------------------------------------------------------------

print("\n" + "="*80)
print("🏗️  BUILDING COMBINED GRAPH")
print("="*80)

workflow = StateGraph(CurriculumGraphState)

# --- Add All Nodes ---
# WF1 Nodes
workflow.add_node("entry", entry_node)
workflow.add_node("architect", architect_node)
workflow.add_node("handle_submission", handle_submission_node)
workflow.add_node("critic", critic_node)
workflow.add_node("prepare_revision", prepare_revision_node)
# Control Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("iteration_controller", iteration_controller_node)
# WF2 Nodes
workflow.add_node("fetch_course_catalog", fetch_course_catalog)
workflow.add_node("web_search_for_courses", web_search_for_courses)
workflow.add_node("process_and_store_courses", process_and_store_courses)
workflow.add_node("start_evaluation", start_evaluation)
# Parallel Evaluation Nodes
workflow.add_node("check_prerequisites", check_prerequisites)
workflow.add_node("match_interests", match_interests)
workflow.add_node("estimate_workload", estimate_workload)
# Reducer & Branching
workflow.add_node("aggregate_scores", aggregate_scores)
workflow.add_node("recommend_enrollment", recommend_enrollment)
workflow.add_node("suggest_alternatives", suggest_alternatives)

# --- Add Edges ---
workflow.set_entry_point("entry")
workflow.add_edge("entry", "planner")
workflow.add_edge("planner", "iteration_controller")

# Iteration Loop
workflow.add_conditional_edges(
    "iteration_controller",
    should_continue_iteration,
    {
        "fetch_course_catalog": "fetch_course_catalog",
        "architect": "architect"
    }
)

# RAG/Search Branch
workflow.add_conditional_edges(
    "fetch_course_catalog",
    should_search_web,
    {
        "evaluate_courses": "start_evaluation",
        "search_web": "web_search_for_courses"
    }
)

# Web-search branch
workflow.add_edge("web_search_for_courses", "process_and_store_courses")
workflow.add_edge("process_and_store_courses", "check_prerequisites")
workflow.add_edge("process_and_store_courses", "match_interests")
workflow.add_edge("process_and_store_courses", "estimate_workload")

# Original evaluation branch
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

# Architect/Critic Loop
workflow.add_conditional_edges(
    "architect",
    should_call_submit_tool,
    {
        "critic": "handle_submission",
        "handle_submission": "handle_submission"
    }
)

# After handle_submission, decide whether to go to critic or prepare_revision
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

# Compile the graph
app = workflow.compile()

print("--- ✅ Graph compilation complete!")
try:
    app.get_graph().draw_png("combined_workflow_graph_v2.png")
    print("--- ✅ Graph diagram saved to 'combined_workflow_graph_v2.png'")
except ImportError:
    print("--- (Install 'pygraphviz' to visualize the graph: pip install pygraphviz)")


### -----------------------------------------------------------------
### 8. RUN THE COMBINED GRAPH
### -----------------------------------------------------------------

if __name__ == "__main__":
    
    mock_skill_gap = """
- User is strong in Python basics.
- User's resume shows no experience with web frameworks.
- Target Role: 'Full-stack Developer'
- CRITICAL Gaps:
  1. FastAPI (for building the API)
  2. PostgreSQL (for the database)
  3. React (for the frontend)
"""
    
    mock_resume = """
**Satyajeet Das**
- B.Tech Computer Science Student
- Skills: Python, C++, Data Structures, Algorithms, Machine Learning
- Interests: 'Machine Learning', 'Web Development', 'AI'
- Projects: Basic ML model with Scikit-learn (used course 101)
"""
    
    inputs = {
        "skill_gap_analysis": mock_skill_gap,
        "resume_inventory": mock_resume,
    }
    
    config = {"recursion_limit": 100}
    
    try:
        final_state = app.invoke(inputs, config=config)
        
        print("\n" + "="*80)
        print("🏁 COMBINED WORKFLOW COMPLETE")
        print("="*80)
        
        if final_state.get('draft_path'):
            print("\n🎉 FINAL PERSONALIZED LEARNING PATH:\n")
            print(final_state['draft_path'].model_dump_json(indent=2))
        else:
            print("\n⚠️  WARNING: No draft_path was generated.")
            print("Final Critique:", final_state.get('structured_critique'))
        
    except Exception as e:
        print(f"\n❌ ERROR: Workflow failed!")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()