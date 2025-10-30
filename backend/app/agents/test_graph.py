import os
import sys
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

# Core LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# --- 1. Setup and Configuration ---

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
MODEL_DB_URL = os.getenv("MODEL_DB_URL")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

# --- 2. Tool Definitions ---

@tool
def get_github_profile(username: str) -> dict:
    """Fetches a user's public profile information from GitHub given their username."""
    # (Implementation from previous code)
    print(f"\n⚙️ EXECUTING TOOL: get_github_profile(username='{username}')")
    headers = {"Authorization": f"Bearer {GITHUB_ACCESS_TOKEN}"} if GITHUB_ACCESS_TOKEN else {}
    response = requests.get(f"https://api.github.com/users/{username}", headers=headers)
    response.raise_for_status()
    profile = response.json()
    return {
        "name": profile.get("name"), "bio": profile.get("bio"),
        "location": profile.get("location"), "public_repos": profile.get("public_repos"),
    }


@tool
def get_github_repos(username: str) -> List[dict]:
    """Fetches the top 5 most recently updated repositories for a given GitHub username."""
    # (Implementation from previous code)
    print(f"\n⚙️ EXECUTING TOOL: get_github_repos(username='{username}')")
    headers = {"Authorization": f"Bearer {GITHUB_ACCESS_TOKEN}"} if GITHUB_ACCESS_TOKEN else {}
    params = {"sort": "updated", "per_page": 5}
    response = requests.get(f"https://api.github.com/users/{username}/repos", headers=headers, params=params)
    response.raise_for_status()
    repos = response.json()
    return [
        {"name": repo.get("name"), "description": repo.get("description"), "language": repo.get("language")}
        for repo in repos
    ]

# --- 3. State Management ---

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
    SELECT_TEMPLATE = auto() # New Step
    GENERATE_JSON = auto()
    GENERATE_LATEX = auto()  # New Step
    DONE = auto()

@dataclass
class ResumeState:
    """A dataclass to hold all the data and the current step of the conversation."""
    current_step: WorkflowStep = WorkflowStep.GREET
    user_data: Dict[str, Any] = field(default_factory=dict)
    github_data: Optional[Dict[str, Any]] = None
    available_templates: Optional[str] = None # Stores list of matching templates
    selected_template: Dict[str, str] = field(default_factory=dict) # Stores name and latex_src of chosen template
    resume_json: Optional[Dict[str, Any]] = None
    last_ai_response: str = ""  # Store the last AI response

# --- 4. Main Application Controller (State Machine) ---

class ResumeBuilder:
    """Manages the state and flow of the resume building process using LCEL."""

    def __init__(self):
        self.state = ResumeState()
        self.llm = ChatOpenAI(
            openai_api_key=OPENROUTER_API_KEY, openai_api_base=OPENROUTER_API_URL,
            model_name="deepseek/deepseek-chat", temperature=0.3, streaming=True,
        )
        self.sql_query_tool = None
        try:
            if MODEL_DB_URL:
                db = SQLDatabase.from_uri(MODEL_DB_URL)
                self.sql_query_tool = QuerySQLDataBaseTool(db=db)
                print("✅ SQL Database connected successfully.")
            else:
                print("⚠️ Warning: MODEL_DB_URL not set. Template features disabled.")
        except Exception as e:
            print(f"⚠️ Warning: SQL Database connection failed: {e}. Template features disabled.")

    def _run_generative_step(self, system_prompt: str, user_input: str = None):
        # (Implementation from previous code, no changes needed)
        prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{user_input}")])
        chain = prompt_template | self.llm | StrOutputParser()
        print("🤖 AI: ", end="", flush=True)
        full_response = ""
        for chunk in chain.stream({"user_input": user_input or ""}):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()
        # Store the response in state
        self.state.last_ai_response = full_response
        return full_response

    def _call_sql_tool_with_approval(self, question: str, sql_query: str) -> str:
        if not self.sql_query_tool: return "SQL database is not available."
        
        print("\n" + "⚠️" * 20)
        print("DATABASE QUERY REQUIRES APPROVAL")
        print(f"  - Purpose: {question}")
        print(f"  - Query: {sql_query}")
        print("⚠️" * 20)

        approval = input("Approve this database query? (yes/no): ").strip().lower()
        if approval not in ['yes', 'y']: return "User denied database query."

        print("✅ Approved - Executing SQL query...")
        try:
            return self.sql_query_tool.invoke(sql_query)
        except Exception as e:
            return f"Error executing SQL query: {e}"

    def run(self):
        print_header()
        while self.state.current_step != WorkflowStep.DONE:
            step = self.state.current_step
            
            # --- State Transitions (No LLM calls, just logic) ---
            if step == WorkflowStep.ASK_GAPS:
                if 'email' not in self.state.user_data: self.state.current_step = WorkflowStep.GET_EMAIL
                elif 'phone' not in self.state.user_data: self.state.current_step = WorkflowStep.GET_PHONE
                else: self.state.current_step = WorkflowStep.QUERY_TEMPLATES
                continue
            
            # --- Steps that require LLM or User/Tool Interaction ---
            if step == WorkflowStep.GREET:
                self._run_generative_step("You are a friendly AI Resume Assistant. Greet the user and ask for the target job role they are applying for.")
                user_input = input("\n👤 YOU: ").strip()
                if user_input:
                    self.state.user_data['target_role'] = user_input
                    self.state.current_step = WorkflowStep.GET_GITHUB_USERNAME

            elif step == WorkflowStep.GET_GITHUB_USERNAME:
                self._run_generative_step(
                    "Acknowledge the user's role and ask for their GitHub username so you can pull their public project information.",
                    f"My target role is {self.state.user_data['target_role']}."
                )
                user_input = input("\n👤 YOU: ").strip()
                if user_input:
                    self.state.user_data['github_username'] = user_input
                    self.state.current_step = WorkflowStep.FETCH_GITHUB_DATA

            elif step == WorkflowStep.FETCH_GITHUB_DATA:
                print("\n🤖 AI: Great! Fetching your public data from GitHub...")
                # (Implementation from previous code)
                username = self.state.user_data['github_username']
                try:
                    profile = get_github_profile.invoke(username)
                    repos = get_github_repos.invoke(username)
                    self.state.github_data = {"profile": profile, "repos": repos}
                    print("✅ GitHub data fetched successfully.")
                except Exception as e:
                    print(f"❌ Error fetching GitHub data: {e}. We can proceed without it.")
                    self.state.github_data = {"error": str(e)}
                self.state.current_step = WorkflowStep.ASK_GAPS
                continue

            elif step == WorkflowStep.GET_EMAIL or step == WorkflowStep.GET_PHONE:
                # (Combined implementation from previous code)
                if step == WorkflowStep.GET_EMAIL:
                    prompt = "Briefly summarize the fetched GitHub data for the user. Then, ask for their email address."
                    user_input_prompt = f"Here is my GitHub data: {json.dumps(self.state.github_data)}"
                else:
                    prompt = "Ask for the user's phone number."
                    user_input_prompt = ""

                self._run_generative_step(prompt, user_input_prompt)
                user_input = input("\n👤 YOU: ").strip()
                if user_input:
                    if step == WorkflowStep.GET_EMAIL: self.state.user_data['email'] = user_input
                    else: self.state.user_data['phone'] = user_input
                self.state.current_step = WorkflowStep.ASK_GAPS

            elif step == WorkflowStep.QUERY_TEMPLATES:
                if not self.sql_query_tool:
                    print("\n🤖 AI: Skipping template search as database is not configured.")
                    self.state.current_step = WorkflowStep.GENERATE_JSON
                    continue
                
                role = self.state.user_data.get('target_role', 'technical professional').lower()
                # A query that joins templates with tags based on the role
                question = f"Find resume templates tagged with '{role}'"
                sql_query = f"""
                SELECT rt.name, rt.preview_image, t.name as tag_name FROM resume_templates rt
                JOIN "_ResumeTemplateToTag" rtt ON rt.id = rtt."A"
                JOIN tags t ON rtt."B" = t.id
                WHERE lower(t.name) LIKE '%{role.split(' ')[0]}%';
                """
                results = self._call_sql_tool_with_approval(question, sql_query)
                self.state.available_templates = str(results)
                self.state.current_step = WorkflowStep.SELECT_TEMPLATE

            elif step == WorkflowStep.SELECT_TEMPLATE:
                self._run_generative_step(
                    "You are a career advisor. Based on the user's target role and the available templates, recommend the single best template and briefly explain why. Output ONLY the name of the template you choose.",
                    f"Target Role: {self.state.user_data['target_role']}\nAvailable Templates: {self.state.available_templates}"
                )
                chosen_template_name = self.state.last_ai_response.strip().replace("'", "''") # Sanitize for SQL
                
                # Now fetch the full LaTeX source for the chosen template
                question = f"Fetch the LaTeX source for template '{chosen_template_name}'"
                sql_query = f"SELECT name, latex_src FROM resume_templates WHERE name = '{chosen_template_name}';"
                results = self._call_sql_tool_with_approval(question, sql_query)
                
                try:
                    # The result is a string like "[('Template Name', 'LaTeX Code...')]"
                    parsed_result = eval(results)
                    if parsed_result:
                        self.state.selected_template = {"name": parsed_result[0][0], "latex_src": parsed_result[0][1]}
                        print(f"\n🤖 AI: Excellent choice! We will use the '{self.state.selected_template['name']}' template.")
                    else:
                        raise ValueError("Template not found.")
                except Exception:
                    print(f"\n🤖 AI: I couldn't retrieve that template. I'll proceed without a specific one.")
                    self.state.selected_template = {"name": "Generic", "latex_src": "% No template found, generating generic LaTeX."}
                
                self.state.current_step = WorkflowStep.GENERATE_JSON

            elif step == WorkflowStep.GENERATE_JSON:
                print("\n🤖 AI: Now, I'll organize your information into a structured format...")
                
                # Define the schema for the resume JSON
                class ContactInfo(BaseModel):
                    name: str = Field(description="Full name")
                    email: str = Field(description="Email address")
                    phone: str = Field(description="Phone number")
                    location: Optional[str] = Field(default=None, description="City, State/Country")
                
                class Project(BaseModel):
                    name: str = Field(description="Project name")
                    description: str = Field(description="Brief description with impact")
                    tech_stack: List[str] = Field(description="Technologies used")
                
                class ResumeJSON(BaseModel):
                    contact: ContactInfo
                    summary: str = Field(description="Professional summary (2-3 sentences)")
                    skills: Dict[str, List[str]] = Field(description="Categorized skills")
                    projects: List[Project] = Field(description="Key projects")
                
                # Create the prompt for JSON generation
                json_gen_prompt = ChatPromptTemplate.from_template(
                    """You are a resume writer. Generate a structured JSON resume based on the following data:
                    
                    Target Role: {target_role}
                    GitHub Data: {github_data}
                    Email: {email}
                    Phone: {phone}
                    
                    Create a professional resume JSON with:
                    - Contact information
                    - A compelling 2-3 sentence professional summary tailored to the target role
                    - Skills categorized by type (Languages, Frameworks, Tools, etc.)
                    - Top 3-5 projects with impactful descriptions
                    
                    Output ONLY valid JSON matching the schema.
                    """
                )
                
                json_chain = json_gen_prompt | self.llm | JsonOutputParser(pydantic_object=ResumeJSON)
                
                try:
                    final_json = json_chain.invoke({
                        "target_role": self.state.user_data.get('target_role', 'Software Engineer'),
                        "github_data": json.dumps(self.state.github_data),
                        "email": self.state.user_data.get('email', ''),
                        "phone": self.state.user_data.get('phone', '')
                    })
                    self.state.resume_json = final_json
                except Exception as e:
                    print(f"\n⚠️ Warning: JSON generation failed ({e}). Using fallback data.")
                    # Fallback to dummy data
                    self.state.resume_json = {
                        "contact": {
                            "name": self.state.github_data.get('profile', {}).get('name', 'John Doe'),
                            "email": self.state.user_data.get("email", "email@example.com"),
                            "phone": self.state.user_data.get("phone", "123-456-7890"),
                            "location": self.state.github_data.get('profile', {}).get('location', 'Earth')
                        },
                        "summary": "A dynamic and results-driven professional.",
                        "skills": {"Languages": ["Python", "JavaScript"], "Tools": ["Docker", "Git"]},
                        "projects": [{"name": "Project Alpha", "description": "Built X with Y to achieve Z.", "tech_stack": ["Python"]}]
                    }
                
                print("✅ Structured data generated.")
                self.state.current_step = WorkflowStep.GENERATE_LATEX

            elif step == WorkflowStep.GENERATE_LATEX:
                print("\n🤖 AI: Finally, I'll merge your data with the template to create the final LaTeX code.")
                latex_prompt = ChatPromptTemplate.from_template(
                    """You are a LaTeX expert. Your task is to populate the given LaTeX template with the user's resume data (in JSON format).
                    - Replace placeholders like `{{ name }}`, `{{ email }}`, etc., with the correct data.
                    - If the template has loops (e.g., for projects or skills), correctly format the data to fit.
                    - Ensure all special characters in the user's data (like `_`, `%`, `&`) are properly escaped for LaTeX.
                    - Output ONLY the final, complete LaTeX code. Do not include any explanations or markdown formatting.

                    LATEX TEMPLATE:
                    ```latex
                    {latex_template}
                    ```

                    USER RESUME DATA (JSON):
                    ```json
                    {resume_json}
                    ```
                    """
                )
                latex_chain = latex_prompt | self.llm | StrOutputParser()
                final_latex = latex_chain.invoke({
                    "latex_template": self.state.selected_template['latex_src'],
                    "resume_json": json.dumps(self.state.resume_json, indent=2)
                })
                print("\n" + "="*20 + " FINAL LATEX CODE " + "="*20 + "\n")
                print(final_latex)
                print("\n" + "="*58)
                self.state.current_step = WorkflowStep.DONE

        print("\n\n✅ Resume generation process complete!")


# --- 5. Application Entry Point ---

def print_header():
    """Prints the application header."""
    print("\n" + "📄" * 40)
    print("      RESUME STRATEGIST (LCEL State Machine)")
    print("📄" * 40)
    print("I will guide you step-by-step to build your resume.")

def main():
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)
    
    try:
        builder = ResumeBuilder()
        builder.run()
    except (KeyboardInterrupt, EOFError):
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()