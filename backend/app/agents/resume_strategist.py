import os
import sys
import json
import asyncio
import re
import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional
from functools import lru_cache

import requests
from dotenv import load_dotenv

# Core LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, EmailStr
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# --- 1. Setup and Configuration ---

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
MODEL_DB_URL = os.getenv("MODEL_DB_URL")

# Groq models: llama-3.3-70b-versatile (best), llama-3.1-70b-versatile, mixtral-8x7b-32768
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- 2. Tool Definitions ---

@tool
def get_github_profile(username: str) -> dict:
    """Fetches a user's public profile information from GitHub given their username."""
    print(f"\n⚙️ EXECUTING TOOL: get_github_profile(username='{username}')")
    headers = {"Authorization": f"Bearer {GITHUB_ACCESS_TOKEN}"} if GITHUB_ACCESS_TOKEN else {}
    response = requests.get(f"https://api.github.com/users/{username}", headers=headers, timeout=10)
    response.raise_for_status()
    profile = response.json()
    return {
        "name": profile.get("name"), 
        "bio": profile.get("bio"),
        "location": profile.get("location"), 
        "public_repos": profile.get("public_repos"),
    }

@tool
def get_github_repos(username: str) -> List[dict]:
    """Fetches the top 5 most recently updated repositories for a given GitHub username."""
    print(f"\n⚙️ EXECUTING TOOL: get_github_repos(username='{username}')")
    headers = {"Authorization": f"Bearer {GITHUB_ACCESS_TOKEN}"} if GITHUB_ACCESS_TOKEN else {}
    params = {"sort": "updated", "per_page": 5}
    response = requests.get(
        f"https://api.github.com/users/{username}/repos", 
        headers=headers, 
        params=params,
        timeout=10
    )
    response.raise_for_status()
    repos = response.json()
    return [
        {
            "name": repo.get("name"), 
            "description": repo.get("description"), 
            "language": repo.get("language"),
            "stars": repo.get("stargazers_count", 0)
        }
        for repo in repos
    ]

# --- 3. Pydantic Models (Professional Resume Structure) ---

class ContactInfo(BaseModel):
    name: str = Field(description="The user's full name.")
    email: EmailStr = Field(description="The user's email address.")
    phone: str = Field(description="The user's phone number.")
    location: str = Field(description="The user's city and state, e.g., 'Pune, Maharashtra'.")
    github_username: str = Field(description="The user's GitHub username.")
    # You could add: linkedin_username: Optional[str] = Field(description="The user's LinkedIn username.")

class Education(BaseModel):
    institution: str = Field(description="Name of the university or school, e.g., 'Indian Institute of Information Technology, Raichur'.")
    degree: str = Field(description="Degree, e.g., 'Bachelor of Technology'.")
    field: str = Field(description="Field of study, e.g., 'Computer Science and Engineering'.")
    status: str = Field(description="e.g., 'Currently Pursuing' or 'Graduated May 2025'.")

class Project(BaseModel):
    name: str = Field(description="The name of the project.")
    description_bullets: List[str] = Field(description="2-3 professional, action-oriented bullet points. Start with verbs like 'Engineered', 'Developed', 'Implemented'.")
    tech_stack: List[str] = Field(description="A list of key technologies used.")

class SkillCategory(BaseModel):
    category_name: str = Field(description="e.g., 'Programming Languages', 'Frameworks', 'Databases', 'Tools'.")
    skills: List[str] = Field(description="A list of skills in this category.")

class ResumeJSON(BaseModel):
    """The top-level structure for the professional resume data."""
    contact: ContactInfo
    summary: str = Field(description="A 2-4 sentence professional summary tailored to the target role, highlighting key skills and experience.")
    education: List[Education]
    skills: List[SkillCategory] = Field(description="A list of skill categories.")
    projects: List[Project] = Field(description="A list of the user's top 3-5 most relevant projects.")


# --- 4. State Management ---

class WorkflowStep(Enum):
    """Defines the controlled steps of the resume creation workflow."""
    GREET = auto()
    GET_ROLE = auto()
    GET_GITHUB_USERNAME = auto()
    FETCH_GITHUB_DATA = auto()
    ASK_GAPS = auto()
    GET_EMAIL = auto()
    GET_PHONE = auto()
    QUERY_TEMPLATES = auto()
    SELECT_TEMPLATE = auto()
    GENERATE_JSON = auto()
    GENERATE_LATEX = auto()
    DONE = auto()

@dataclass
class ResumeState:
    """A dataclass to hold all the data and the current step of the conversation."""
    current_step: WorkflowStep = WorkflowStep.GREET
    user_data: Dict[str, Any] = field(default_factory=dict)
    github_data: Optional[Dict[str, Any]] = None
    available_templates: Optional[str] = None
    selected_template: Dict[str, str] = field(default_factory=dict)
    # This will hold the DICT version of the ResumeJSON Pydantic model
    resume_json: Optional[Dict[str, Any]] = None
    last_ai_response: str = ""

# --- 5. Main Application Controller (State Machine) ---

class ResumeBuilder:
    """Manages the state and flow of the resume building process using LCEL."""

    def __init__(self):
        self.state = ResumeState()
        # Groq LLM - much faster than OpenRouter
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.2,
            max_tokens=8000,
            streaming=True,
        )
        # Non-streaming LLM for JSON generation (more reliable)
        self.llm_json = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.1,
            max_tokens=8000,
            streaming=False,
        )
        self.sql_query_tool = None
        self._setup_database()

    def _setup_database(self):
        """Initialize database connection with error handling."""
        try:
            if MODEL_DB_URL:
                db = SQLDatabase.from_uri(MODEL_DB_URL)
                self.sql_query_tool = QuerySQLDataBaseTool(db=db)
                print("✅ SQL Database connected successfully.")
            else:
                print("⚠️ Warning: MODEL_DB_URL not set. Template features disabled.")
        except Exception as e:
            print(f"⚠️ Warning: SQL Database connection failed: {e}. Template features disabled.")

    def _run_generative_step(self, system_prompt: str, user_input: str = None) -> str:
        """Runs a standard conversational step using an LCEL chain and returns the response."""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt), 
            ("human", "{user_input}")
        ])
        chain = prompt_template | self.llm | StrOutputParser()
        
        print("🤖 AI: ", end="", flush=True)
        full_response = ""
        try:
            for chunk in chain.stream({"user_input": user_input or ""}):
                print(chunk, end="", flush=True)
                full_response += chunk
        except Exception as e:
            print(f"\n⚠️ Error during generation: {e}")
            full_response = "I apologize, but I encountered an error. Let's continue."
        
        print()
        self.state.last_ai_response = full_response
        return full_response

    @lru_cache(maxsize=32)
    def _execute_sql_query(self, sql_query: str) -> str:
        """Execute SQL query with caching for identical queries."""
        if not self.sql_query_tool:
            return "SQL database is not available."
        try:
            return self.sql_query_tool.invoke(sql_query)
        except Exception as e:
            return f"Error executing SQL query: {e}"

    def _call_sql_tool_with_approval(self, question: str, sql_query: str) -> str:
        """Explicitly calls the SQL tool with human approval."""
        if not self.sql_query_tool:
            return "SQL database is not available."
        
        print("\n" + "⚠️" * 20)
        print("DATABASE QUERY REQUIRES APPROVAL")
        print(f"   - Purpose: {question}")
        print(f"   - Query: {sql_query}")
        print("⚠️" * 20)
        
        approval = input("Approve this database query? (yes/no): ").strip().lower()
        if approval not in ['yes', 'y']:
            return "User denied database query."
        
        print("✅ Approved - Executing SQL query...")
        return self._execute_sql_query(sql_query)

    @staticmethod
    def _escape_latex(data):
        """
        Recursively escape special LaTeX characters in strings 
        within nested lists and dictionaries.
        """
        if isinstance(data, str):
            return (data.replace('\\', r'\textbackslash{}')
                        .replace('&', r'\&')
                        .replace('%', r'\%')
                        .replace('$', r'\$')
                        .replace('#', r'\#')
                        .replace('_', r'\_')
                        .replace('{', r'\{')
                        .replace('}', r'\}')
                        .replace('~', r'\textasciitilde{}')
                        .replace('^', r'\textasciicircum{}'))
        if isinstance(data, dict):
            # Recurse on values, but not keys
            return {k: ResumeBuilder._escape_latex(v) for k, v in data.items()}
        if isinstance(data, list):
            # Recurse on list items
            return [ResumeBuilder._escape_latex(i) for i in data]
        # Return other types (int, bool, etc.) unchanged
        return data

    def _fetch_github_data_parallel(self, username: str) -> Dict[str, Any]:
        """Fetch GitHub profile and repos in parallel for better performance."""
        print("\n🤖 AI: Fetching your GitHub data (this will be quick)...")
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                profile_future = executor.submit(get_github_profile.invoke, username)
                repos_future = executor.submit(get_github_repos.invoke, username)
                
                profile = profile_future.result(timeout=10)
                repos = repos_future.result(timeout=10)
                
            print("✅ GitHub data fetched successfully.")
            return {"profile": profile, "repos": repos}
            
        except Exception as e:
            print(f"❌ Error fetching GitHub data: {e}. We can proceed without it.")
            return {"error": str(e), "profile": {}, "repos": []}

    def _query_templates_optimized(self, role: str) -> str:
        """Optimized template querying with single comprehensive query."""
        if not self.sql_query_tool:
            return None
        
        # Single optimized query that handles everything
        keyword = role.lower().split(' ')[0]
        
        sql_query = f"""
        WITH matched_templates AS (
            SELECT DISTINCT 
                rt.id,
                rt.name, 
                rt.preview_image,
                rt.latex_src,
                STRING_AGG(t.name, ', ') as tags,
                COUNT(*) OVER() as total_count
            FROM resume_templates rt
            JOIN "_ResumeTemplateToTag" rtt ON rt.id = rtt."A"
            JOIN tags t ON rtt."B" = t.id
            WHERE LOWER(t.name) LIKE '%{keyword}%'
            GROUP BY rt.id, rt.name, rt.preview_image, rt.latex_src
            LIMIT 10
        )
        SELECT * FROM matched_templates
        UNION ALL
        SELECT DISTINCT 
            rt.id,
            rt.name, 
            rt.preview_image,
            rt.latex_src,
            STRING_AGG(t.name, ', ') as tags,
            0 as total_count
        FROM resume_templates rt
        JOIN "_ResumeTemplateToTag" rtt ON rt.id = rtt."A"
        JOIN tags t ON rtt."B" = t.id
        WHERE NOT EXISTS (SELECT 1 FROM matched_templates)
        GROUP BY rt.id, rt.name, rt.preview_image, rt.latex_src
        LIMIT 10;
        """
        
        return self._call_sql_tool_with_approval(
            f"Find templates for '{role}'", 
            sql_query
        )

    #
    # --- JINJA2 RENDER FUNCTION REMOVED ---
    #
    
    def run(self):
        """The main loop that drives the conversation state machine."""
        print_header()
        
        while self.state.current_step != WorkflowStep.DONE:
            step = self.state.current_step
            
            # Handle gap-filling logic
            if step == WorkflowStep.ASK_GAPS:
                if 'email' not in self.state.user_data:
                    self.state.current_step = WorkflowStep.GET_EMAIL
                elif 'phone' not in self.state.user_data:
                    self.state.current_step = WorkflowStep.GET_PHONE
                else:
                    self.state.current_step = WorkflowStep.QUERY_TEMPLATES
                continue
            
            # State machine transitions
            if step == WorkflowStep.GREET:
                self._run_generative_step(
                    "You are a friendly AI Resume Assistant. Greet the user briefly and ask for the target job role they are applying for. Keep it concise."
                )
                user_input = input("\n👤 YOU: ").strip()
                if user_input:
                    self.state.user_data['target_role'] = user_input
                    self.state.current_step = WorkflowStep.GET_GITHUB_USERNAME

            elif step == WorkflowStep.GET_GITHUB_USERNAME:
                self._run_generative_step(
                    "Briefly acknowledge the user's role and ask for their GitHub username. Keep it short.",
                    f"My target role is {self.state.user_data['target_role']}."
                )
                user_input = input("\n👤 YOU: ").strip()
                if user_input:
                    self.state.user_data['github_username'] = user_input
                    self.state.current_step = WorkflowStep.FETCH_GITHUB_DATA

            elif step == WorkflowStep.FETCH_GITHUB_DATA:
                username = self.state.user_data['github_username']
                self.state.github_data = self._fetch_github_data_parallel(username)
                self.state.current_step = WorkflowStep.ASK_GAPS
                continue

            elif step == WorkflowStep.GET_EMAIL or step == WorkflowStep.GET_PHONE:
                if step == WorkflowStep.GET_EMAIL:
                    prompt = "Briefly mention you found their GitHub data. Then ask for their email address. Keep it concise."
                    user_input_prompt = f"Here is my GitHub data: {json.dumps(self.state.github_data)}"
                else:
                    prompt = "Ask for the user's phone number briefly."
                    user_input_prompt = ""
                
                self._run_generative_step(prompt, user_input_prompt)
                user_input = input("\n👤 YOU: ").strip()
                
                if user_input:
                    if step == WorkflowStep.GET_EMAIL:
                        self.state.user_data['email'] = user_input
                    else:
                        self.state.user_data['phone'] = user_input
                
                self.state.current_step = WorkflowStep.ASK_GAPS

            elif step == WorkflowStep.QUERY_TEMPLATES:
                if not self.sql_query_tool:
                    print("\n🤖 AI: Skipping template search as database is not configured.")
                    self.state.current_step = WorkflowStep.GENERATE_JSON
                    continue
                
                role = self.state.user_data.get('target_role', 'technical professional')
                results = self._query_templates_optimized(role)
                
                self.state.available_templates = str(results)
                print(f"\n📋 Templates retrieved successfully.")
                self.state.current_step = WorkflowStep.SELECT_TEMPLATE

            elif step == WorkflowStep.SELECT_TEMPLATE:
                
                #
                # --- NEW FALLBACK (Simple LaTeX, no Jinja) ---
                #
                fallback_latex_template = {
                    "name": "Generic-Fallback",
                    "latex_src": r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\begin{document}

% --- Contact ---
\begin{center}
    {\Huge {contact_name_placeholder}} \\
    {contact_email_placeholder} $\cdot$ {contact_phone_placeholder} $\cdot$ {contact_location_placeholder} \\
    \href{https://github.com/{contact_github_placeholder}}{github.com/{contact_github_placeholder}}
\end{center}

% --- Summary ---
\section*{Summary}
{summary_text_placeholder}

% --- Education ---
\section*{Education}
{education_section_placeholder}
% e.g., \subsection*{University Name} \textit{Degree} \hfill Date

% --- Skills ---
\section*{Technical Skills}
{skills_section_placeholder}
% e.g., \textbf{Languages:} Python, Java

% --- Projects ---
\section*{Projects}
{projects_section_placeholder}
% e.g., \subsection*{Project Name} \begin{itemize} \item ... \end{itemize}

\end{document}"""
                }

                if not self.state.available_templates or self.state.available_templates == "[]":
                    print("\n🤖 AI: No templates found in DB. Using a generic LaTeX template.")
                    self.state.selected_template = fallback_latex_template
                    self.state.current_step = WorkflowStep.GENERATE_JSON
                    continue

                # Parse available templates
                template_list = re.findall(r"\('([^']+)'", str(self.state.available_templates))
                template_names = [t for t in template_list if t and t != 'None']
                
                print(f"\n📋 Available templates: {', '.join(template_names[:5])}")
                
                # AI selects template (fast with Groq)
                self._run_generative_step(
                    f"""Select the BEST template for: {self.state.user_data['target_role']}
Available: {', '.join(template_names[:10])}
Output ONLY the template name, nothing else.""",
                    "Recommend a template"
                )
                
                chosen_template_name = re.sub(r'[^a-zA-Z0-9_-]', '', self.state.last_ai_response.strip())
                print(f"\n🤖 AI: Selected template '{chosen_template_name}'")
                
                try:
                    # Extract from the comprehensive query result
                    result_data = eval(self.state.available_templates)
                    
                    for item in result_data:
                        if item[1] == chosen_template_name:
                            self.state.selected_template = {
                                "name": item[1],
                                "latex_src": item[3] if len(item) > 3 else ""
                            }
                            break
                    
                    if not self.state.selected_template.get('latex_src'):
                        raise ValueError("Template LaTeX source not found")
                    
                    print(f"✅ Template loaded: {self.state.selected_template['name']}")
                    
                except Exception as e:
                    print(f"⚠️ Using generic template due to: {e}")
                    self.state.selected_template = fallback_latex_template
                
                self.state.current_step = WorkflowStep.GENERATE_JSON

            elif step == WorkflowStep.GENERATE_JSON:
                print("\n🤖 AI: Structuring your professional resume data...")
                
                json_gen_prompt = ChatPromptTemplate.from_template(
                    """
You are a professional resume writer and career coach. Your task is to generate a 
structured JSON document to build a world-class resume.

**Target Role:** {target_role}

**Collected User Data:**
GitHub Data: {github_data}
Email: {email}
Phone: {phone}
GitHub Username: {github_username} 
(Note: The user's context in the prompt also mentions Satyajeet Das, IIIT Raichur, and projects like 'GenAI-CustomerServ' and 'Fair-Share'. Use this context.)

**Your Task:**
Create a highly professional resume JSON. You MUST adhere to the Pydantic schema.

1.  **Summary:** Write a powerful, 2-4 sentence summary that directly targets the **{target_role}** role, using the user's skills and projects as proof.
2.  **Education:** Infer education from the context (IIIT Raichur, B.Tech in CS&E).
3.  **Projects:**
    * For each project (from GitHub and context), do NOT just copy the description.
    * You MUST write 2-3 **action-oriented bullet points**.
    * Start each bullet point with a strong verb (e.g., "Engineered...", "Developed...").
    * **Example:**
        * **Weak:** "MNSIT-Number-Prediction: A machine learning model for number prediction."
        * **Strong:**
            * "Developed and trained a Convolutional Neural Network (CNN) using TensorFlow to achieve high accuracy on the MNIST dataset."
            * "Engineered a data pipeline for image preprocessing and augmentation."
4.  **Skills:** Intelligently categorize skills into logical groups.

Output ONLY the valid JSON object.
"""
                )
                
                json_chain = json_gen_prompt | self.llm_json | JsonOutputParser(pydantic_object=ResumeJSON)
                
                try:
                    #
                    # --- THIS IS THE FIX ---
                    # The parser returns a dict, not a Pydantic object.
                    #
                    resume_dict = json_chain.invoke({
                        "target_role": self.state.user_data.get('target_role', 'Software Engineer'),
                        "github_data": json.dumps(self.state.github_data),
                        "email": self.state.user_data.get('email', ''),
                        "phone": self.state.user_data.get('phone', ''),
                        "github_username": self.state.user_data.get('github_username', '')
                    })
                    #
                    # --- ASSIGN THE DICT DIRECTLY ---
                    #
                    self.state.resume_json = resume_dict
                    print("✅ Structured professional data generated successfully.")

                except Exception as e:
                    print(f"\n⚠️ JSON generation failed ({e}). Using fallback data.")
                    # NEW FALLBACK - must match ResumeJSON structure
                    github_profile = self.state.github_data.get("profile", {})
                    github_repos = self.state.github_data.get("repos", [])
                    
                    self.state.resume_json = {
                        "contact": {
                            "name": github_profile.get("name", "Satyajeet Das"),
                            "email": self.state.user_data.get('email', 'email@example.com'),
                            "phone": self.state.user_data.get('phone', '555-1234'),
                            "location": github_profile.get("location", "Pune, Maharashtra"),
                            "github_username": self.state.user_data.get('github_username', 'user')
                        },
                        "summary": f"Proactive and skilled {self.state.user_data.get('target_role', 'Software Engineer')} with experience from technical projects. Eager to apply and grow skills in a professional environment.",
                        "education": [
                            {
                                "institution": "Indian Institute of Information Technology, Raichur",
                                "degree": "Bachelor of Technology",
                                "field": "Computer Science and Engineering",
                                "status": "Currently Pursuing"
                            }
                        ],
                        "skills": [
                            {
                                "category_name": "Languages",
                                "skills": list(set([repo.get("language") for repo in github_repos if repo.get("language")])) or ["Python", "JavaScript", "C++"]
                            },
                            {
                                "category_name": "Tools",
                                "skills": ["Git", "GitHub", "VS Code"]
                            }
                        ],
                        "projects": [
                            {
                                "name": repo.get("name", "Project"),
                                "description_bullets": [
                                    f"Developed a project using {repo.get('language', 'key technologies')}.",
                                    repo.get("description", "Contributed to project development and implementation.")
                                ],
                                "tech_stack": [repo.get("language", "Python")]
                            }
                            for repo in github_repos[:3]
                        ]
                    }
                
                self.state.current_step = WorkflowStep.GENERATE_LATEX

            elif step == WorkflowStep.GENERATE_LATEX:
                print("\n🤖 AI: Generating your LaTeX resume code...")
                output_dir = "resume_output"
                os.makedirs(output_dir, exist_ok=True)
                
                # Get the selected template as a guide
                template_example = self.state.selected_template.get('latex_src', '')
                
                # Escape the JSON data *before* sending it to the LLM
                try:
                    escaped_json_data = self._escape_latex(self.state.resume_json)
                    json_string_input = json.dumps(escaped_json_data, indent=2)
                except Exception as e:
                    print(f"Warning: Could not escape LaTeX data: {e}")
                    json_string_input = json.dumps(self.state.resume_json, indent=2)

                # Define the prompt for the LLM to write LaTeX
                latex_gen_prompt = ChatPromptTemplate.from_template(
                    """
You are an expert LaTeX developer. Your task is to generate a complete, 
professional, and runnable .tex file.

**USER DATA (JSON):**
{resume_json}

**EXAMPLE TEMPLATE STRUCTURE:**
{template_example}

**INSTRUCTIONS:**
1.  Use the "USER DATA" to fill in all relevant sections from the JSON.
2.  Use the "EXAMPLE TEMPLATE STRUCTURE" as a strong guide for the layout, sections, and style.
3.  **Crucially:** You must replace all placeholders (like `{contact_name_placeholder}` or `{summary_text_placeholder}`) with the actual data from the JSON.
4.  Create proper LaTeX for sections, subsections, itemize, etc., based on the JSON structure.
    * For 'contact', fill the header.
    * For 'summary', put the text in the summary section.
    * For 'education', loop through and create entries.
    * For 'skills', loop through categories and list skills.
    * For 'projects', loop through them, creating a subsection for each with bullet points.
5.  **Output ONLY the complete, raw LaTeX code.** Do not include any other text, explanations, or markdown backticks (```latex ... ```).
6.  The output must be a single, valid .tex file starting with `\documentclass`.
"""
                )
                
                # Use the streaming LLM for this creative task
                chain = latex_gen_prompt | self.llm | StrOutputParser()
                
                print("🤖 AI: [Generating LaTeX] ", end="", flush=True)
                complete_resume_latex = ""
                try:
                    for chunk in chain.stream({
                        "resume_json": json_string_input,
                        "template_example": template_example
                    }):
                        print(chunk, end="", flush=True)
                        complete_resume_latex += chunk

                    # Clean the output (just in case LLM adds backticks)
                    complete_resume_latex = complete_resume_latex.strip().strip('```latex').strip('```').strip()
                    
                    if not complete_resume_latex.startswith(r"\documentclass"):
                        raise ValueError("LLM output was not valid LaTeX.")

                    resume_path = os.path.join(output_dir, "resume.tex")
                    with open(resume_path, "w", encoding='utf-8') as f:
                        f.write(complete_resume_latex)
                    
                    print("\n" + "="*20 + " LATEX GENERATION COMPLETE " + "="*20)
                    print(f"✅ Resume saved to '{output_dir}/resume.tex'")
                    print(f"   - Size: {len(complete_resume_latex)} characters")
                    print("\n💡 Compile with: pdflatex resume.tex")
                    print("="*67)

                except Exception as e:
                    print(f"\n⚠️ LaTeX generation failed: {e}. Saving fallback data.")
                    # Fallback is just to save the JSON data
                    json_path = os.path.join(output_dir, "resume_data.json")
                    with open(json_path, "w", encoding='utf-8') as f:
                        json.dump(self.state.resume_json, f, indent=4)
                    print(f"✅ Resume data saved to '{json_path}' as a fallback.")
                
                self.state.current_step = WorkflowStep.DONE
                continue # Skip input()

        print("\n\n✅ Resume generation complete!")

# --- 6. Application Entry Point ---

def print_header():
    """Prints the application header."""
    print("\n" + "📄" * 40)
    print("      RESUME STRATEGIST (Powered by Groq)")
    print("📄" * 40)
    print("⚡ Fast AI-powered resume content generation")

def main():
    """Main function to run the application."""
    if not GROQ_API_KEY:
        print("❌ Error: GROQ_API_KEY not found in .env file.")
        print("💡 Get your free API key at: https://console.groq.com/")
        sys.exit(1)
    
    try:
        builder = ResumeBuilder()
        builder.run()
    except (KeyboardInterrupt, EOFError):
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()