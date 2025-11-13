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
    resume_json: Optional[Dict[str, Any]] = None
    last_ai_response: str = ""  # Store the last AI response

# --- 4. Main Application Controller (State Machine) ---

class ResumeBuilder:
    """Manages the state and flow of the resume building process using LCEL."""

    def __init__(self):
        self.state = ResumeState()
        self.llm = ChatOpenAI(
            openai_api_key=OPENROUTER_API_KEY, openai_api_base=OPENROUTER_API_URL,
            model_name="deepseek/deepseek-chat", temperature=0.2, streaming=True,
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

    def _run_generative_step(self, system_prompt: str, user_input: str = None) -> str:
        """Runs a standard conversational step using an LCEL chain and returns the response."""
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
        """Explicitly calls the SQL tool with human approval."""
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

    def _generate_latex_section(self, section_name: str, prompt_template: str, json_data: dict) -> str:
        """Generates a LaTeX snippet for a specific resume section."""
        print(f"   - Generating LaTeX for {section_name} section...")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        def escape_latex(data):
            if isinstance(data, str): return data.replace('&', r'\&').replace('%', r'\%').replace('$', r'\$').replace('#', r'\#').replace('_', r'\_').replace('{', r'\{').replace('}', r'\}').replace('~', r'\textasciitilde{}').replace('^', r'\textasciicircum{}')
            if isinstance(data, dict): return {k: escape_latex(v) for k, v in data.items()}
            if isinstance(data, list): return [escape_latex(i) for i in data]
            return data

        escaped_data = escape_latex(json_data)
        return chain.invoke({"json_data": json.dumps(escaped_data, indent=2)})

    def run(self):
        """The main loop that drives the conversation state machine."""
        print_header()
        while self.state.current_step != WorkflowStep.DONE:
            step = self.state.current_step
            
            if step == WorkflowStep.ASK_GAPS:
                if 'email' not in self.state.user_data: self.state.current_step = WorkflowStep.GET_EMAIL
                elif 'phone' not in self.state.user_data: self.state.current_step = WorkflowStep.GET_PHONE
                else: self.state.current_step = WorkflowStep.QUERY_TEMPLATES
                continue
            
            if step == WorkflowStep.GREET:
                self._run_generative_step("You are a friendly AI Resume Assistant. Greet the user and ask for the target job role they are applying for.")
                user_input = input("\n👤 YOU: ").strip()
                if user_input:
                    self.state.user_data['target_role'] = user_input
                    self.state.current_step = WorkflowStep.GET_GITHUB_USERNAME

            elif step == WorkflowStep.GET_GITHUB_USERNAME:
                self._run_generative_step("Acknowledge the user's role and ask for their GitHub username so you can pull their public project information.", f"My target role is {self.state.user_data['target_role']}.")
                user_input = input("\n👤 YOU: ").strip()
                if user_input:
                    self.state.user_data['github_username'] = user_input
                    self.state.current_step = WorkflowStep.FETCH_GITHUB_DATA

            elif step == WorkflowStep.FETCH_GITHUB_DATA:
                print("\n🤖 AI: Great! Fetching your public data from GitHub...")
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
                if step == WorkflowStep.GET_EMAIL:
                    prompt, user_input_prompt = "Briefly summarize the fetched GitHub data for the user. Then, ask for their email address.", f"Here is my GitHub data: {json.dumps(self.state.github_data)}"
                else:
                    prompt, user_input_prompt = "Ask for the user's phone number.", ""
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
                
                # First, let's see what tags exist
                print("\n🔍 Checking available tags in database...")
                check_tags_query = "SELECT DISTINCT name FROM tags ORDER BY name;"
                tags_result = self._call_sql_tool_with_approval("List all available tags", check_tags_query)
                print(f"Available tags: {tags_result}")
                
                # Now query templates - using a more flexible search
                question = f"Find resume templates with tags related to '{role}'"
                keyword = role.split(' ')[0]  # e.g., "software" from "software engineer"
                
                # Updated query with better logic
                sql_query = f"""
                SELECT DISTINCT 
                    rt.name, 
                    rt.preview_image, 
                    STRING_AGG(t.name, ', ') as tags
                FROM resume_templates rt
                JOIN "_ResumeTemplateToTag" rtt ON rt.id = rtt."A"
                JOIN tags t ON rtt."B" = t.id
                WHERE LOWER(t.name) LIKE '%{keyword}%'
                GROUP BY rt.name, rt.preview_image
                ORDER BY rt.name;
                """
                
                results = self._call_sql_tool_with_approval(question, sql_query)
                
                if not results or results == "[]" or "Error" in results:
                    # Fallback: get ALL templates if no match found
                    print("\n⚠️ No templates found for specific keyword. Fetching all available templates...")
                    fallback_query = """
                    SELECT DISTINCT 
                        rt.name, 
                        rt.preview_image, 
                        STRING_AGG(t.name, ', ') as tags
                    FROM resume_templates rt
                    JOIN "_ResumeTemplateToTag" rtt ON rt.id = rtt."A"
                    JOIN tags t ON rtt."B" = t.id
                    GROUP BY rt.name, rt.preview_image
                    ORDER BY rt.name
                    LIMIT 10;
                    """
                    results = self._call_sql_tool_with_approval("Get all available templates", fallback_query)
                
                self.state.available_templates = str(results)
                print(f"\n📋 Found templates: {self.state.available_templates}")
                self.state.current_step = WorkflowStep.SELECT_TEMPLATE

            elif step == WorkflowStep.SELECT_TEMPLATE:
                # Parse the available templates to show only names
                import re
                template_list = re.findall(r"\('([^']+)'", str(self.state.available_templates))
                template_names = [t for t in template_list if t and t != 'None']
                
                if not template_names:
                    print("\n🤖 AI: No templates found. Using a generic template.")
                    self.state.selected_template = {
                        "name": "Generic", 
                        "latex_src": r"""
\documentclass[11pt,a4paper]{article}
\begin{document}
\section*{Resume}
% Generic template
\end{document}
"""
                    }
                    self.state.current_step = WorkflowStep.GENERATE_JSON
                    continue
                
                # Show available templates to user
                print(f"\n📋 Available templates: {', '.join(template_names[:10])}")
                
                # AI recommends a template
                self._run_generative_step(
                    f"""You are a career advisor. The user is applying for: {self.state.user_data['target_role']}
                    
Available resume templates: {', '.join(template_names[:10])}

Select the BEST template name from the list above. Output ONLY the template name, nothing else.
Example: If templates are ['1', '2', '3'], output just: 2
""",
                    f"Recommend a template for {self.state.user_data['target_role']}"
                )
                
                # Clean up the AI response to get just the template name
                chosen_template_name = self.state.last_ai_response.strip()
                # Remove quotes, periods, extra text
                chosen_template_name = re.sub(r'[^a-zA-Z0-9_-]', '', chosen_template_name)
                
                print(f"\n🤖 AI: I recommend template '{chosen_template_name}'")
                
                # Sanitize for SQL (prevent SQL injection)
                chosen_template_name_safe = chosen_template_name.replace("'", "''")
                
                # Fetch the LaTeX source
                question = f"Fetch the LaTeX source for template '{chosen_template_name}'"
                sql_query = f"SELECT name, latex_src FROM resume_templates WHERE name = '{chosen_template_name_safe}';"
                results = self._call_sql_tool_with_approval(question, sql_query)
                
                try:
                    # Parse the result
                    parsed_result = eval(results) if isinstance(results, str) else results
                    
                    if parsed_result and len(parsed_result) > 0:
                        self.state.selected_template = {
                            "name": parsed_result[0][0], 
                            "latex_src": parsed_result[0][1]
                        }
                        print(f"\n✅ Successfully loaded template: {self.state.selected_template['name']}")
                    else:
                        raise ValueError("Template query returned empty result")
                        
                except Exception as e:
                    print(f"\n⚠️ Error loading template ({e}). Trying first available template...")
                    
                    # Fallback: Just get the first template from the list
                    first_template = template_names[0]
                    sql_query = f"SELECT name, latex_src FROM resume_templates WHERE name = '{first_template}';"
                    results = self._call_sql_tool_with_approval(f"Fetch template '{first_template}'", sql_query)
                    
                    try:
                        parsed_result = eval(results) if isinstance(results, str) else results
                        self.state.selected_template = {
                            "name": parsed_result[0][0], 
                            "latex_src": parsed_result[0][1]
                        }
                        print(f"\n✅ Using fallback template: {self.state.selected_template['name']}")
                    except:
                        # Ultimate fallback
                        print("\n⚠️ Could not retrieve any template. Using generic template.")
                        self.state.selected_template = {
                            "name": "Generic",
                            "latex_src": r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\begin{document}
\section*{Resume}
% Template content will be generated
\end{document}
"""
                        }
    
                self.state.current_step = WorkflowStep.GENERATE_JSON

            elif step == WorkflowStep.GENERATE_JSON:
                print("\n🤖 AI: Now, I'll organize your information into a structured format...")
                class ContactInfo(BaseModel): name: str; email: str; phone: str; location: Optional[str]
                class Project(BaseModel): name: str; description: str; tech_stack: List[str]
                class ResumeJSON(BaseModel): contact: ContactInfo; summary: str; skills: Dict[str, List[str]]; projects: List[Project]
                json_gen_prompt = ChatPromptTemplate.from_template("Generate a structured JSON resume based on the following data:\nTarget Role: {target_role}\nGitHub Data: {github_data}\nEmail: {email}\nPhone: {phone}\n\nCreate a professional resume JSON. Output ONLY valid JSON matching the schema.")
                json_chain = json_gen_prompt | self.llm | JsonOutputParser(pydantic_object=ResumeJSON)
                try:
                    self.state.resume_json = json_chain.invoke({"target_role": self.state.user_data.get('target_role', 'Software Engineer'),"github_data": json.dumps(self.state.github_data),"email": self.state.user_data.get('email', ''), "phone": self.state.user_data.get('phone', '')})
                except Exception as e:
                    print(f"\n⚠️ Warning: JSON generation failed ({e}). Using fallback data.")
                    self.state.resume_json = {"contact": {"name": "John Doe", "email": "johndoe@example.com", "phone": "555-1234", "location": "City, State"}, "summary": "...", "skills": {"Languages": ["Python"]}, "projects": [{"name": "Default Project", "description": "A default project description.", "tech_stack": ["Python"]}]}
                print("✅ Structured data generated.", self.state.resume_json)
                self.state.current_step = WorkflowStep.GENERATE_LATEX

            elif step == WorkflowStep.GENERATE_LATEX:
                print("\n🤖 AI: Using the template as a one-shot learning example to generate your complete resume...")
                output_dir = "resume_output"
                os.makedirs(output_dir, exist_ok=True)
                
                # Get the template LaTeX source as an example
                template_example = self.state.selected_template.get('latex_src', '')
                
                # One-shot learning prompt: use template as example
                one_shot_prompt = """
You are an expert LaTeX resume generator. You will be given:
1. An EXAMPLE LaTeX resume template
2. User's resume data in JSON format

Your task: Generate a COMPLETE LaTeX resume document that:
- Follows the EXACT SAME structure, style, and formatting as the example template
- Replaces ALL placeholder content with the user's actual data
- Maintains all LaTeX commands, packages, and styling from the example
- Ensures all special characters are properly escaped (use \\& for &, \\% for %, etc.)
- Outputs a complete, compilable LaTeX document

EXAMPLE TEMPLATE:
{template_example}

USER'S RESUME DATA (JSON):
{resume_json}

Generate the complete LaTeX resume. Output ONLY the LaTeX code, no explanations.
"""
                
                # Create the prompt template
                prompt = ChatPromptTemplate.from_template(one_shot_prompt)
                chain = prompt | self.llm | StrOutputParser()
                
                print("   - Analyzing template structure...")
                print("   - Populating with your data...")
                print("   - Generating complete LaTeX document...")
                
                try:
                    # Generate the complete resume using one-shot learning
                    complete_resume_latex = chain.invoke({
                        "template_example": template_example,
                        "resume_json": json.dumps(self.state.resume_json, indent=2)
                    })
                    
                    # Save the generated resume
                    resume_path = os.path.join(output_dir, "resume.tex")
                    with open(resume_path, "w", encoding='utf-8') as f:
                        f.write(complete_resume_latex)
                    
                    print("\n" + "="*20 + " LATEX GENERATION COMPLETE " + "="*20 + "\n")
                    print(f"✅ Complete LaTeX resume generated in '{output_dir}' directory.")
                    print(f"   - File: resume.tex")
                    print(f"   - Size: {len(complete_resume_latex)} characters")
                    print("\n📝 Resume Preview (first 500 chars):")
                    print("-" * 60)
                    print(complete_resume_latex[:500] + "...")
                    print("-" * 60)
                    print("\n💡 To create your PDF:")
                    print("   1. Use a LaTeX compiler (MiKTeX, TeX Live, or Overleaf)")
                    print("   2. Compile 'resume.tex' from the 'resume_output' directory")
                    print("   3. Your PDF will be generated!")
                    print("\n" + "="*63)
                    
                except Exception as e:
                    print(f"\n⚠️ Error during LaTeX generation: {e}")
                    print("Falling back to template with basic substitutions...")
                    
                    # Fallback: simple string replacement
                    contact_info = self.state.resume_json.get('contact', {})
                    fallback_latex = template_example.replace('Your Name Here', contact_info.get('name', 'Your Name'))
                    fallback_latex = fallback_latex.replace('example@gmail.com', contact_info.get('email', 'email@example.com'))
                    fallback_latex = fallback_latex.replace('555-1234', contact_info.get('phone', '555-1234'))
                    
                    with open(os.path.join(output_dir, "resume.tex"), "w", encoding='utf-8') as f:
                        f.write(fallback_latex)
                    
                    print(f"✅ Fallback resume saved to '{output_dir}/resume.tex'")
                
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
    """Main function to run the application."""
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