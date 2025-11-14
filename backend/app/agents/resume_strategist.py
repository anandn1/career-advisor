import os
import sys
import json
import re
import ast  # For safely parsing string tuples
import traceback
import concurrent.futures
from typing import List, Dict, Any, Optional, TypedDict

import requests
from dotenv import load_dotenv

# --- Core LangChain imports (New Docs Pattern) ---
from langchain.agents import create_agent, AgentState
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, EmailStr

# --- DB Tool Imports ---
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# --- 1. Setup and Configuration ---

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
MODEL_DB_URL = os.getenv("MODEL_DB_URL")
GROQ_MODEL = "openai/gpt-oss-120b"  # Best for tool calling - fast and accurate

# --- 2. Pydantic Models (For JSON Output) ---
# (These are unchanged. They are used by the generate_resume_json_data tool)

class ContactInfo(BaseModel):
    name: str = Field(description="The user's full name.")
    email: EmailStr = Field(description="The user's email address.")
    phone: str = Field(description="The user's phone number.")
    location: str = Field(description="The user's city and state, e.g., 'Pune, Maharashtra'.")
    github_username: str = Field(description="The user's GitHub username.")

class Education(BaseModel):
    institution: str = Field(description="Name of the university or school.")
    degree: str = Field(description="Degree, e.g., 'Bachelor of Technology'.")
    field: str = Field(description="Field of study, e.g., 'Computer Science and Engineering'.")
    status: str = Field(description="e.g., 'Currently Pursuing' or 'Graduated May 2025'.")
    location: str = Field(default="", description="Location of the institution, e.g., 'Raichur, Karnataka, India'.")
    gpa: str = Field(default="", description="GPA or percentage, e.g., '8.8/10' or '85%'.")

class WorkExperience(BaseModel):
    company: str = Field(description="Name of the company or organization.")
    position: str = Field(description="Job title/position, e.g., 'Full Stack Developer Intern'.")
    location: str = Field(description="Location of the job, e.g., 'Remote, India'.")
    duration: str = Field(description="Duration of employment, e.g., 'March 2025 -- August 2025'.")
    description_bullets: List[str] = Field(description="2-4 professional bullet points describing achievements and responsibilities.")

class Project(BaseModel):
    name: str = Field(description="The name of the project.")
    description_bullets: List[str] = Field(description="2-3 professional, action-oriented bullet points.")
    tech_stack: List[str] = Field(description="A list of key technologies used.")

class SkillCategory(BaseModel):
    category_name: str = Field(description="e.g., 'Programming Languages', 'Frameworks', 'Tools'.")
    skills: List[str] = Field(description="A list of skills in this category.")

class ResumeJSON(BaseModel):
    """The top-level structure for the professional resume data."""
    contact: ContactInfo
    summary: str = Field(description="A 2-4 sentence professional summary tailored to the target role.")
    education: List[Education] = Field(default_factory=list, description="List of educational qualifications.")
    work_experience: List[WorkExperience] = Field(default_factory=list, description="List of work experiences.")
    skills: List[SkillCategory]
    projects: List[Project]


# --- 3. Agent State Definition (New Docs Pattern) ---

class ResumeAgentState(AgentState):
    """
    Defines the custom state for our agent, as per the new docs.
    This TypedDict will be managed by LangGraph under the hood.
    """
    # 'messages' is already included in AgentState
    
    # This will hold all data explicitly provided by the user
    user_data: Dict[str, Any] = Field(default_factory=dict)
    
    # This will hold data fetched from GitHub
    github_data: Dict[str, Any] = Field(default_factory=dict)
    
    # This will hold the structured JSON resume
    resume_json: Optional[Dict[str, Any]] = None
    
    # This will hold the chosen LaTeX template source
    selected_template_latex: Optional[str] = None
    
    # This will hold the *full list* of [('id', 'name', ...)] from the DB
    available_templates: List[tuple] = Field(default_factory=list)


# --- 4. Standalone Agent Tools (New Docs Pattern) ---
# Tools are now standalone functions.
# - To READ state: Add a state key (e.g., `user_data: dict`) to the signature.
# - To WRITE state: Return a dictionary with the state keys to update.

@tool
def get_github_profile_and_repos(username: str) -> dict:
    """
    Fetches a user's GitHub profile AND their top 5 repos in parallel.
    Returns a dictionary to update the 'github_data' and 'user_data' state.
    """
    print(f"\n⚙️ EXECUTING TOOL: get_github_profile_and_repos(username='{username}')")
    headers = {"Authorization": f"Bearer {GITHUB_ACCESS_TOKEN}"} if GITHUB_ACCESS_TOKEN else {}
    
    def fetch_profile():
        response = requests.get(f"https://api.github.com/users/{username}", headers=headers, timeout=10)
        response.raise_for_status()
        profile = response.json()
        return {
            "name": profile.get("name"),
            "bio": profile.get("bio"),
            "location": profile.get("location"),
            "public_repos": profile.get("public_repos"),
        }

    def fetch_repos():
        params = {"sort": "updated", "per_page": 5}
        response = requests.get(
            f"https://api.github.com/users/{username}/repos",
            headers=headers,
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        repos = response.json()
        return [
            {
                "name": repo.get("name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "stars": repo.get("stargazers_count", 0),
            }
            for repo in repos
        ]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            profile_future = executor.submit(fetch_profile)
            repos_future = executor.submit(fetch_repos)
            
            profile_data = profile_future.result()
            repo_data = repos_future.result()
            
        print("✅ GitHub data fetched successfully.")
        
        # This return value will PATCH the ResumeAgentState
        return {
            "github_data": {"profile": profile_data, "repos": repo_data},
            "user_data": {
                "name": profile_data.get("name"),
                "location": profile_data.get("location"),
                "github_username": username,
            }
        }
    except Exception as e:
        print(f"❌ Error fetching GitHub data: {e}.")
        return {"github_data": {"error": str(e)}}


@tool
def save_user_details(
    target_role: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    name: Optional[str] = None,
    location: Optional[str] = None,
    education: Optional[List[dict]] = None,
    work_experience: Optional[List[dict]] = None,
) -> dict:
    """
    Saves or updates user details in the state. Call this ONLY when you receive ACTUAL information from the user.
    
    Parameters:
    - target_role: The job role they're applying for (e.g., "Software Engineer")
    - email: Their actual email address (e.g., "john@example.com") - must be valid format
    - phone: Their actual phone number (e.g., "+1-555-0123")
    - name: Their full name (only if not found from GitHub)
    - location: Their location (e.g., "San Francisco, CA")
    - education: List of education entries, each with: institution, degree, field, status, location (optional), gpa (optional)
    - work_experience: List of work experience entries, each with: company, position, location, duration, description_bullets (list)
    
    CRITICAL RULES:
    - ONLY call with actual values the user just provided in their message
    - NEVER use placeholder values like "your_email" or "example@example.com"
    - Only include parameters you actually have - omit the rest
    - Call this multiple times as you collect different pieces of information
    
    Returns a dictionary to update the 'user_data' state.
    """
    print(f"\n⚙️ EXECUTING TOOL: save_user_details(...)")
    
    # Build the update patch with only provided values
    update_patch = {}
    if target_role: 
        update_patch['target_role'] = target_role
    if email:
        # Basic email validation
        if '@' not in email or '.' not in email.split('@')[-1]:
            print(f"⚠️ Warning: '{email}' doesn't look like a valid email")
        update_patch['email'] = email
    if phone: 
        update_patch['phone'] = phone
    if name: 
        update_patch['name'] = name
    if location: 
        update_patch['location'] = location
    if education is not None:
        update_patch['education'] = education
    if work_experience is not None:
        update_patch['work_experience'] = work_experience
    
    print(f"✅ Saving details: {list(update_patch.keys())}")
    
    # Return a dict to patch the 'user_data' key
    return {"user_data": update_patch}


def query_and_select_template(target_role: str, db_tool: QuerySQLDataBaseTool) -> dict:
    """
    Queries the SQL database for resume templates based on the target role,
    selects the best one, and returns its LaTeX source code to be saved in state.
    """
    print(f"\n⚙️ EXECUTING TOOL: query_and_select_template(role='{target_role}')")
    if not db_tool:
        print("⚠️ DB tool not configured. Using fallback template.")
        return {"selected_template_latex": get_fallback_latex()}

    # 1. Query the DB (using the passed-in tool)
    keyword = target_role.lower().split(' ')[0].strip()
    
    # Simplified query: Just get all templates and their LaTeX source
    # We'll do a simple search for templates with matching tags
    sql_query = f"""
    SELECT DISTINCT rt.id, rt.name, rt.preview_image, rt.latex_src
    FROM resume_templates rt
    LEFT JOIN "_ResumeTemplateToTag" rtt ON rt.id = rtt."A"
    LEFT JOIN tags t ON rtt."B" = t.id
    WHERE LOWER(t.name) LIKE '%{keyword}%' OR t.name IS NULL
    LIMIT 5;
    """
    
    try:
        print("... (executing SQL query) ...")
        result_str = db_tool.invoke(sql_query)
        
        # Debug: Print what we got
        print(f"📊 DB Result (first 200 chars): {str(result_str)[:200]}")
        
        # The QuerySQLDataBaseTool returns a string representation of the result
        # It could be in different formats, let's handle them
        if not result_str or result_str.strip() == "":
            raise ValueError("Empty result from database")
        
        # Try to parse as Python literal
        try:
            parsed_results = ast.literal_eval(result_str)
        except (ValueError, SyntaxError) as parse_error:
            # If that fails, the result might be a simple string description
            print(f"⚠️ Could not parse as literal: {parse_error}")
            print(f"Raw result: {result_str}")
            raise ValueError(f"Could not parse database result: {parse_error}")
        
        if not isinstance(parsed_results, list) or not parsed_results:
            raise ValueError("No templates found or bad result format.")

        # 2. Select the best one (using the first one as "best")
        best_template = parsed_results[0]
        
        # Handle both tuple and dict results
        if isinstance(best_template, tuple):
            template_name = best_template[1] if len(best_template) > 1 else "Unknown"
            template_latex_src = best_template[3] if len(best_template) > 3 else ""
        elif isinstance(best_template, dict):
            template_name = best_template.get('name', 'Unknown')
            template_latex_src = best_template.get('latex_src', '')
        else:
            raise ValueError(f"Unexpected template format: {type(best_template)}")
        
        if not template_latex_src:
            raise ValueError("Template has no LaTeX source")
        
        print(f"✅ Found {len(parsed_results)} templates. Auto-selecting: '{template_name}'")
        
        # 3. Return the patch for the state
        return {
            "available_templates": parsed_results,
            "selected_template_latex": template_latex_src
        }
        
    except Exception as e:
        print(f"⚠️ Error querying/parsing templates: {e}. Using fallback.")
        import traceback
        traceback.print_exc()
        return {"selected_template_latex": get_fallback_latex()}

def get_fallback_latex():
    """Returns a simple example LaTeX resume for one-shot learning."""
    return r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{enumitem}

\begin{document}

% Header
\begin{center}
    {\Huge\bfseries John Doe} \\
    \vspace{0.5em}
    john.doe@example.com $\cdot$ +1-555-0123 $\cdot$ San Francisco, CA \\
    \href{https://github.com/johndoe}{github.com/johndoe}
\end{center}

% Summary
\section*{Summary}
Experienced Software Engineer with 3+ years of expertise in full-stack development, 
specializing in React, Node.js, and cloud technologies. Proven track record of 
delivering scalable applications and leading cross-functional teams.

% Education
\section*{Education}
\textbf{Bachelor of Science in Computer Science} \\
University of California, Berkeley $\cdot$ Graduated May 2020

% Skills
\section*{Technical Skills}
\begin{itemize}[leftmargin=*,label={}]
    \item \textbf{Languages:} Python, JavaScript, TypeScript, Java
    \item \textbf{Frameworks:} React, Node.js, Express, Django
    \item \textbf{Tools:} Git, Docker, AWS, PostgreSQL
\end{itemize}

% Projects
\section*{Projects}
\textbf{E-Commerce Platform} \\
\begin{itemize}[leftmargin=1.5em]
    \item Built a full-stack e-commerce platform serving 10K+ users using React and Node.js
    \item Implemented secure payment processing with Stripe API
    \item \textit{Tech Stack: React, Node.js, PostgreSQL, AWS}
\end{itemize}

\textbf{Real-Time Chat Application} \\
\begin{itemize}[leftmargin=1.5em]
    \item Developed a WebSocket-based chat app with real-time messaging
    \item Integrated user authentication and message encryption
    \item \textit{Tech Stack: Socket.io, Express, MongoDB}
\end{itemize}

\end{document}"""

@tool
def generate_resume_json_data(
    user_data: dict, github_data: dict
) -> dict:
    """
    Generates the structured ResumeJSON data.
    This tool READS 'user_data' and 'github_data' from the state.
    It returns a dictionary to update the 'resume_json' state.
    """
    print(f"\n⚙️ EXECUTING TOOL: generate_resume_json_data(...)")
    
    # LLM for JSON generation
    llm_json = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.1,
        streaming=False,
    )

    json_gen_prompt = ChatPromptTemplate.from_template(
        """
You are a professional resume writer. Your task is to generate a structured JSON 
document (adhering to the Pydantic schema) based on the user's data.

**Target Role:** {target_role}

**Collected User Data:**
GitHub Profile: {github_profile}
GitHub Repos: {github_repos}
Contact Info: {user_contact}
Education: {user_education}

**Your Task:**
Create a highly professional resume JSON.

1.  **Contact:** Use the collected Contact Info.
2.  **Summary:** Write a powerful 2-4 sentence summary targeting the **{target_role}**.
3.  **Education:** Format the user's provided education.
4.  **Projects:** For each project, write 2-3 **action-oriented bullet points** (e.g., "Engineered...", "Developed...").
5.  **Skills:** Intelligently categorize skills from the GitHub data.

Output ONLY the valid JSON object.
"""
    )
    
    json_chain = json_gen_prompt | llm_json | JsonOutputParser(pydantic_object=ResumeJSON)
    
    try:
        contact_info = {
            "name": user_data.get('name', 'Missing Name'),
            "email": user_data.get('email', 'missing@email.com'),
            "phone": user_data.get('phone', 'Missing Phone'),
            "location": user_data.get('location', 'Missing Location'),
            "github_username": user_data.get('github_username', 'Missing GitHub')
        }
        
        resume_dict = json_chain.invoke({
            "target_role": user_data.get('target_role', 'Software Engineer'),
            "github_profile": json.dumps(github_data.get("profile", {})),
            "github_repos": json.dumps(github_data.get("repos", [])),
            "user_contact": json.dumps(contact_info),
            "user_education": json.dumps(user_data.get('education', []))
        })
        
        print("✅ Structured professional data generated successfully.")
        # Return a patch for the state
        return {"resume_json": resume_dict}

    except Exception as e:
        print(f"\n⚠️ JSON generation failed ({e}).")
        # Return a patch for the state
        return {"resume_json": {"error": str(e)}}


@tool
def generate_latex_file(
    resume_json: dict, selected_template_latex: str
) -> str:
    """
    Generates the final .tex file from the 'resume_json' and 'selected_template_latex' state.
    This is the final step and does not return a state patch, just a success message.
    """
    print(f"\n⚙️ EXECUTING TOOL: generate_latex_file(...)")
    
    if not resume_json or "error" in resume_json:
        return "Error: Cannot generate LaTeX. The resume JSON data is missing or invalid."
    if not selected_template_latex:
        return "Error: Cannot generate LaTeX. No template was selected."

    # LLM for LaTeX filling
    llm_latex = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.0,
        streaming=False,
    )
    
    def _escape_latex(data):
        """Recursively escape special LaTeX characters."""
        if isinstance(data, str):
            return (data.replace('\\', r'\textbackslash{}')
                        .replace('&', r'\&').replace('%', r'\%')
                        .replace('$', r'\$').replace('#', r'\#')
                        .replace('_', r'\_').replace('{', r'\{')
                        .replace('}', r'\}').replace('~', r'\textasciitilde{}')
                        .replace('^', r'\textasciicircum{}'))
        if isinstance(data, dict):
            return {k: _escape_latex(v) for k, v in data.items()}
        if isinstance(data, list):
            return [_escape_latex(i) for i in data]
        return data

    try:
        escaped_json_string = json.dumps(_escape_latex(resume_json), indent=2)
    except Exception as e:
        print(f"Warning: Could not escape LaTeX data: {e}")
        escaped_json_string = json.dumps(resume_json, indent=2)

    latex_gen_prompt = ChatPromptTemplate.from_template(
        """
You are an expert LaTeX resume generator. Your task is to create a complete, 
professional .tex file using the provided user data.

**EXAMPLE LATEX RESUME (for reference style/structure):**
{template_example}

**USER DATA (JSON) - Use this to create the resume:**
{resume_json}

**CRITICAL INSTRUCTIONS:**
1.  **PRESERVE ALL CUSTOM COMMANDS**: The example template contains custom LaTeX commands like:
    - \\header{{}} - for the name header
    - \\contactinfo{{}}{{}}{{}}{{}} - for contact information
    - \\entrytitle{{}}{{}}{{}} - for project/work entries
    - \\eduentry{{}}{{}}{{}}{{}} - for education entries
    **YOU MUST USE THESE EXACT COMMANDS** in your output, do not replace them with basic LaTeX!

2.  **PRESERVE ALL PACKAGES AND PREAMBLE**: Copy the entire preamble from the example, including:
    - All \\usepackage declarations (geometry, enumitem, hyperref, xcolor, titlesec, parskip, tcolorbox)
    - All \\newcommand definitions
    - All \\titleformat and \\titlespacing settings
    - All \\setlist configurations
    - The \\hypersetup configuration

3.  **USE THE TEMPLATE STRUCTURE EXACTLY**:
    - Start with the same \\documentclass line
    - Copy all the preamble (packages + custom commands) EXACTLY
    - Use \\header{{NAME}} for the name (not \\begin{{center}})
    - Use \\contactinfo{{location}}{{phone}}{{email}}{{links}} for contact (not manual formatting)
    - Use \\eduentry for education entries
    - Use \\entrytitle for work experience entries
    - Use tcolorbox with tabular for Technical Skills (copy the exact format)
    - Use the same itemize structure with proper nesting

4.  **FILL IN THE USER DATA**: Replace only the content values with data from the JSON:
    - Name from contact.name
    - Email from contact.email  
    - Phone from contact.phone
    - Location from contact.location
    - GitHub/LinkedIn from contact.github
    - Education from education array (use \\eduentry command)
    - Work Experience from work_experience array (use \\entrytitle command for each position, then itemize bullets)
    - Skills from skills object (maintain the tcolorbox table format)
    - Projects from projects array (use \\entrytitle for each project)

5.  **Output ONLY the complete, raw LaTeX code** 
    - No explanations, no markdown backticks, no extra text
    - Must start with \\documentclass
    - All special characters in the data are already escaped
    - The output should compile exactly like the example template
"""
    )
    
    chain = latex_gen_prompt | llm_latex | StrOutputParser()
    
    try:
        print("🤖 AI: [Generating LaTeX code] ...")
        complete_resume_latex = chain.invoke({
            "resume_json": escaped_json_string,
            "template_example": selected_template_latex
        })

        # Clean up the output - remove markdown code fences if present
        complete_resume_latex = complete_resume_latex.strip()
        
        # Remove markdown code blocks
        if complete_resume_latex.startswith('```'):
            lines = complete_resume_latex.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]  # Remove first line
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]  # Remove last line
            complete_resume_latex = '\n'.join(lines).strip()
        
        # Validation - check for required LaTeX document structure
        has_documentclass = r"\documentclass" in complete_resume_latex
        has_begin_document = r"\begin{document}" in complete_resume_latex
        has_end_document = r"\end{document}" in complete_resume_latex
        
        # Require all three essential components
        if not has_documentclass:
            raise ValueError("LaTeX output missing \\documentclass declaration")
        if not has_begin_document:
            raise ValueError("LaTeX output missing \\begin{document}")
        if not has_end_document:
            raise ValueError("LaTeX output missing \\end{document}")
        
        # Additional check: should start near the beginning (allow some whitespace/comments)
        first_500_chars = complete_resume_latex[:500]
        if r"\documentclass" not in first_500_chars:
            print(f"⚠️ Warning: \\documentclass not found in first 500 characters")
            print(f"   First 200 chars: {complete_resume_latex[:200]}")
            # Just warn, don't fail

        output_dir = "resume_output"
        os.makedirs(output_dir, exist_ok=True)
        resume_path = os.path.join(output_dir, "resume.tex")
        
        with open(resume_path, "w", encoding='utf-8') as f:
            f.write(complete_resume_latex)
        
        success_message = (
            f"\n\n{'='*20} LATEX GENERATION COMPLETE {'='*20}\n"
            f"✅ Resume saved to '{output_dir}/resume.tex'\n"
            f"💡 Compile with: pdflatex resume.tex\n"
            f"{'='*67}\n\n"
            "Process is complete. Please inform the user."
        )
        print(success_message)
        return success_message

    except Exception as e:
        print(f"\n⚠️ LaTeX generation failed: {e}.")
        return f"LaTeX generation failed: {e}. Please apologize to the user."


# --- 5. System Prompt (The Agent's "Logic") ---

# This prompt is the new "logic" that replaces the FSM.
# It tells the agent *how* to proceed.
SYSTEM_PROMPT = """
You are a friendly and professional AI Resume Assistant.
Your goal is to build a professional LaTeX resume for the user.
You MUST follow this step-by-step process, using your tools to gather data 
and update the state.

**IMPORTANT: NEVER call tools with placeholder or fake data. Only call tools when you have REAL user data.**

**Your Plan:**
1.  **Greet & Get Role:** Greet the user and ask for their target job role.
2.  **Save Role:** Once you have the ACTUAL role from the user, call `save_user_details` with ONLY the target_role parameter.
3.  **Get GitHub:** Once the role is saved, ask for their GitHub username.
4.  **Fetch GitHub Data:** When you have the ACTUAL username, call `get_github_profile_and_repos` with the real username.
5.  **Fill Gaps:** Review what you know so far. Ask the user for any missing CRITICAL info ONE AT A TIME:
    * `email` - a valid email address
    * `phone` - their phone number
    * If name is missing from GitHub, ask for their full name
6.  **Save Each Gap:** When the user provides REAL information, call `save_user_details` with ONLY the fields you just received.
    - Example: If they just gave you email, call save_user_details(email="their_actual_email@domain.com")
    - DO NOT include fields you don't have yet!
7.  **Collect Education:** Ask the user for their education details:
    * Institution name
    * Degree (e.g., "Bachelor of Technology")
    * Field of study (e.g., "Computer Science")
    * Status (e.g., "Expected August 2027" or "Graduated May 2024")
    * Location (optional)
    * GPA (optional)
    - Save each education entry using save_user_details(education=[{...}])
8.  **Collect Work Experience (Optional):** Ask if they have any work experience. If yes, collect:
    * Company name
    * Position/role
    * Location
    * Duration (e.g., "March 2025 -- August 2025")
    * 2-4 bullet points describing achievements
    - Save using save_user_details(work_experience=[{...}])
    - If they have no work experience, skip this step.
9.  **Query Templates:** Once you have the `target_role`, call `query_and_select_template`. This will find and select a template automatically.
10. **Generate JSON:** After ALL data is collected (role, github, email, phone, education, work_experience) AND the template is selected, you MUST call `generate_resume_json_data`.
11. **Generate LaTeX:** Once the `resume_json` is in the state, you MUST call the final tool, `generate_latex_file`.
12. **Done:** Inform the user the file is complete.

**State Management Rules:**
- You do not see the state directly. You must rely on tool outputs.
- When you call `save_user_details`, the state is updated.
- When you call `get_github_profile_and_repos`, the state is updated.
- You must call `generate_resume_json_data` *before* `generate_latex_file`.
- Do not ask for information you can get from a tool.
- Be polite and guide the user one step at a time.
"""

# --- 6. Application Entry Point ---

def print_header():
    """Prints the application header."""
    print("\n" + "📄" * 40)
    print("      AGENTIC RESUME STRATEGIST (New Docs)")
    print("📄" * 40)
    print("⚡ Powered by create_agent and LangGraph")

def main():
    """Main function to configure and run the agent."""
    if not GROQ_API_KEY:
        print("❌ Error: GROQ_API_KEY not found in .env file.")
        sys.exit(1)
    
    # --- Initialize Tools ---
    
    # 1. Initialize the DB tool
    db_tool = None
    if MODEL_DB_URL:
        try:
            db = SQLDatabase.from_uri(MODEL_DB_URL)
            db_tool = QuerySQLDataBaseTool(db=db)
            print("✅ SQL Database connected successfully.")
        except Exception as e:
            print(f"⚠️ Warning: SQL Database connection failed: {e}. Template features disabled.")
    else:
        print("⚠️ Warning: MODEL_DB_URL not set. Template features disabled.")

    # 2. Create the tool list
    # We "curry" the db_tool into the query_and_select_template tool
    # This is a bit of a hack, as `create_agent` doesn't have a clear "context" pass-through
    # for tools like LangGraph's `with_config`.
    
    # We must wrap the db_tool-dependent function
    @tool
    def query_and_select_template_wrapper(target_role: str) -> dict:
        """
        Queries the SQL database for resume templates based on the target role,
        selects the best one, and returns its LaTeX source code to be saved in state.
        """
        # This wrapper calls the real function, passing the db_tool instance
        return query_and_select_template(target_role, db_tool)

    all_tools = [
        get_github_profile_and_repos,
        save_user_details,
        query_and_select_template_wrapper,
        generate_resume_json_data,
        generate_latex_file,
    ]

    # --- Initialize Model ---
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.2,
    )
    
    # --- Create Agent (New Docs Pattern) ---
    agent = create_agent(
        model=llm,
        tools=all_tools,
        state_schema=ResumeAgentState,
        system_prompt=SYSTEM_PROMPT,
    )
    
    print_header()

    # --- Main Agent Loop ---
    # We start with an empty state
    # The 'messages' list will be our chat history
    
    # We must use `stream` or `invoke` with the full state dict
    
    # Start the conversation
    state = {"messages": [HumanMessage(content="Hi, can you help me build a resume?")]}

    while True:
        try:
            print("🤖 AI: ", end="", flush=True)
            # Use stream to get chunks
            full_response_content = ""
            for chunk in agent.stream(state, stream_mode="values"):
                # chunk is the full ResumeAgentState at that step
                # We only care about the *last* message
                last_message = chunk["messages"][-1]
                
                # Check if it's an AI message and has new content
                if isinstance(last_message, AIMessage) and last_message.content:
                    # This is complex because streaming tokens comes in AIMessage chunks
                    # A simpler way for a console app is to use `invoke`
                    pass
            
            # --- Using invoke for simpler console interaction ---
            # stream is complex. Let's use invoke.
            
            result_state = agent.invoke(state)
            last_ai_message = result_state["messages"][-1]
            
            if not isinstance(last_ai_message, AIMessage):
                print("\nAgent run finished with a tool call. Something is wrong.")
                break

            ai_response = last_ai_message.content
            print(ai_response)
            
            # Check for final step
            if "Process is complete" in ai_response:
                print("\n\n✅ Resume generation complete!")
                break
                
            user_input = input("\n👤 YOU: ").strip()
            if not user_input:
                continue
                
            # Update the state for the next loop
            state = {
                "messages": [
                    # We pass the full history back in
                    *result_state["messages"], 
                    HumanMessage(content=user_input)
                ],
                # We must also pass the *rest* of the state
                "user_data": result_state.get("user_data", {}),
                "github_data": result_state.get("github_data", {}),
                "resume_json": result_state.get("resume_json"),
                "selected_template_latex": result_state.get("selected_template_latex"),
                "available_templates": result_state.get("available_templates", [])
            }

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()