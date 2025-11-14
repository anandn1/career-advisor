"""
Career Advisor Supervisor Orchestrator (Main Entry Point & Router)
"""

import os
import time
import json
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import AIMessage, ToolMessage

# --- 1. Custom Agent & Settings Imports ---
print("Importing shared components and settings...")
try:
    from utils import ResumeParser
    from settings import DB_URL, llm
except ImportError as e:
    print("="*80)
    print(f"[FATAL ERROR] Could not import from 'utils.py' or 'settings.py'.")
    print(f"Details: {e}")
    print("Please make sure all files are in the same directory.")
    print("="*80)
    exit()

# --- 2. Specialist Agent Imports ---
print("Importing specialist agents...")

# --- Tool-based Agents ---
try:
    from job_market_analyst_i import run_job_market_analyst_graph
    print("Successfully imported [Job Market Analyst]")
except ImportError as e:
    print(f"[ERROR] Could not import 'run_job_market_analyst_graph'. {e}")
    def run_job_market_analyst_graph(request: str, user_skills: list, checkpointer) -> str:
        print("[STUB] Job Market Analyst not found.")
        return json.dumps({"error": "Job Market Analyst not found, using stub."})

try:
    from curriculum_architect_i import run_curriculum_architect_graph
    print("Successfully imported [Curriculum Architect]")
except ImportError as e:
    print(f"[ERROR] Could not import 'run_curriculum_architect_graph'. {e}")
    # FIX: Update stub signature to match actual call
    def run_curriculum_architect_graph(inputs: dict, checkpointer) -> str:
        print("[STUB] Curriculum Architect not found.")
        return json.dumps({"error": "Curriculum Architect not found, using stub."})

try:
    from resume_analyzer import run_resume_analyzer_graph
    print("Successfully imported [Resume Analyzer]")
except ImportError as e:
    print(f"[ERROR] Could not import 'run_resume_analyzer_graph'. {e}")
    def run_resume_analyzer_graph(request: str, resume_path: str, checkpointer) -> str:
        print("[STUB] Resume Analyzer not found.")
        return json.dumps({"error": "Resume Analyzer not found, using stub."})

# --- Interactive (Handoff) Agents ---
try:
    from interview_coach_i import run_interactive_interview
    print("Successfully imported [Interview Coach]")
except ImportError as e:
    print(f"[ERROR] Could not import 'run_interactive_interview'. {e}")
    def run_interactive_interview(checkpointer, user_skills: list):
        print(f"[STUB] Interview Coach not available: {e}")

try:
    from resume_strategist_i import run_interactive_resume_builder
    print("Successfully imported [Resume Strategist]")
except ImportError as e:
    print(f"[ERROR] Could not import 'run_interactive_resume_builder'. {e}")
    def run_interactive_resume_builder():
        print(f"[STUB] Resume Strategist not available: {e}")

# ============================================================================
# Supervisor Class
# ============================================================================

class CareerSupervisor:
    """
    The main supervisor class that holds the user's state (skills, resume)
    and orchestrates specialist agents.
    """
    
    def __init__(self, user_skills: list, resume_path: str, checkpointer):
        print("\nInitializing Career Supervisor...")
        if not user_skills:
            print("[WARNING] Supervisor initialized with no user skills.")
        
        self.user_skills = user_skills
        self.resume_path = resume_path
        self.model = llm
        self.checkpointer = checkpointer
        
        # Cache for skill gap analysis (avoids redundant calls)
        self.last_skill_gap_analysis = None
        self.last_target_role = None
        
        # Define tools as closures
        @tool
        def analyze_resume(request: str) -> str:
            """Analyze a user's existing resume."""
            print(f"\n[Supervisor] Calling Resume Analyzer...")
            return run_resume_analyzer_graph(request, self.resume_path, self.checkpointer)

        @tool
        def analyze_job_market(request: str) -> str:
            """Find jobs and analyze market trends."""
            result = run_job_market_analyst_graph(request, self.user_skills, self.checkpointer)
            # Cache the analysis
            try:
                data = json.loads(result)
                self.last_skill_gap_analysis = data.get("skill_gap_analysis", {})
                self.last_target_role = data.get("role", "AI Engineer")
                print(f"[Supervisor] Cached skill gaps for {self.last_target_role}")
            except:
                pass
            return result

        @tool
        def design_learning_plan(request: str) -> str:
            """Create personalized learning plan from skill gaps."""
            print(f"\n[Supervisor] Building learning plan...")
            
            # Use cached gaps from previous job market analysis
            if self.last_skill_gap_analysis and self.last_skill_gap_analysis.get("skill_gaps"):
                skill_gaps = self.last_skill_gap_analysis["skill_gaps"][:5]
                target_role = self.last_target_role or "AI Engineer"
                print(f"[Supervisor] Using cached skill gaps for {target_role}")
            else:
                # Fallback: extract role from request
                target_role = "AI Engineer"
                request_lower = request.lower()
                import re
                for pattern in [
                    r"for\s+(.+?)\s+(?:roles|jobs|career|position|in)",
                    r"in\s+(.+?)\s+(?:domain|field|area)",
                    r"to\s+become\s+(.+?)(?:\s|$)",
                    r"covering\s+(.+?)\s+(?:skills|gap|for)"
                ]:
                    match = re.search(pattern, request_lower)
                    if match:
                        target_role = match.group(1).strip().title()
                        break
                skill_gaps = ["machine learning", "deep learning", "computer vision", 
                            "data analysis", "communication"]
            
            # Build structured inputs
            inputs = {
                "request": f"Build a learning plan for {target_role}",
                "user_skills": self.user_skills,
                "skill_gap_analysis": {"target_role": target_role, "gaps": skill_gaps},
                "resume_inventory": {"skills": self.user_skills, "experience_level": "intermediate"}
            }
            
            return run_curriculum_architect_graph(inputs, self.checkpointer)

        @tool
        def initiate_interview_session(request: str) -> str:
            """Start an interactive mock interview session."""
            print("\n[Supervisor] Initiating handoff to Interview Coach...")
            return json.dumps({"handoff": "interview"})

        @tool
        def initiate_resume_builder_session(request: str) -> str:
            """Start an interactive resume building session."""
            print("\n[Supervisor] Initiating handoff to Resume Strategist...")
            return json.dumps({"handoff": "resume"})
        
        self.tools = [
            analyze_resume,
            analyze_job_market,
            design_learning_plan,
            initiate_interview_session,
            initiate_resume_builder_session
        ]
        
        self.supervisor_agent = self._create_supervisor()
        print("Supervisor is ready.")

    def _create_supervisor(self):
        """Creates the main supervisor agent."""
        from langgraph.prebuilt import create_react_agent
        
        SUPERVISOR_PROMPT = (
            "You are a helpful, expert-level career advisor assistant. "
            "Your job is to coordinate a team of specialist AI agents. You have agents for:\n"
            "1.  **Resume Analysis** (for reviewing existing resumes)\n"
            "2.  **Job Market Analysis** (for finding jobs)\n"
            "3.  **Learning Plan Design** (for creating curriculums)\n"
            "4.  **Interactive Interview Coach** (for mock interviews)\n"
            "5.  **Interactive Resume Builder** (for creating new resumes)\n\n"
            "**Your Task:**\n"
            "1.  For requests to analyze/review/critique an *existing* resume, find jobs, or design a learning plan, use the appropriate tool. These are non-interactive: summarize the JSON response for the user.\n"
            "2.  For requests to *practice* an interview OR *build a new* resume, use the `initiate_interview_session` or `initiate_resume_builder_session` tools.\n"
            "3.  **IMPORTANT Handoff Rule:** When you use an 'initiate' tool, tell the user you are handing them over.\n"
            "   - **Example:** 'I'll hand you to our interview coach now. Type 'exit' to return.'"
        )

        return create_agent(
            self.model,
            tools=self.tools,
            system_prompt=SUPERVISOR_PROMPT,
            checkpointer=self.checkpointer,
        )

    def invoke(self, user_request: str, thread_id: str) -> tuple[str, str]:
        """
        Runs a single turn of the supervisor.
        Returns: (final_ai_response, raw_tool_output)
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = self.supervisor_agent.invoke(
                {"messages": [{"role": "user", "content": user_request}]},
                config,
            )
            
            messages = final_state.get("messages", [])
            final_response = ""
            tool_output = ""
            
            # Extract final AI response
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    if isinstance(msg.content, str):
                        final_response = msg.content
                    else:
                        final_response = str(msg.content)
                    break
            
            # Extract ONLY the most recent ToolMessage (not all of them)
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    if isinstance(msg.content, str):
                        tool_output = msg.content
                    elif isinstance(msg.content, list) and msg.content:
                        tool_output = str(msg.content[0])
                    else:
                        tool_output = str(msg.content)
                    break  # Only take the most recent one
            
            return final_response, tool_output
            
        except Exception as e:
            print(f"\n[Supervisor Error] {e}")
            import traceback
            traceback.print_exc()
            return f"Sorry, an error occurred in the supervisor: {e}", ""

# ============================================================================
# Main Execution (The Router Loop)
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Starting Career Advisor Supervisor...")
    print("="*80)

    # --- Step 1: Get User Skills AND Resume Path (once) ---
    print("\n" + "="*70)
    print("SKILL EXTRACTION FROM RESUME")
    print("="*70)
    
    resume_path = input("\nEnter the full path to your resume (PDF or TXT): ").strip().strip('"\'')
    if not resume_path:
        print("[ERROR] No resume path provided. Exiting.")
        exit()
        
    resume_data = ResumeParser.parse_resume(resume_path)
    user_skills = resume_data.get('skills', [])
    if not user_skills:
        print(f"[ERROR] Could not get skills. Exiting.")
        exit()
        
    print(f"\nSuccessfully extracted {len(user_skills)} skills.")
    
    # --- Step 2: Initialize Supervisor (with 'with' block) ---
    print("\nAttempting to connect to Postgres for checkpointer...")
    try:
        with PostgresSaver.from_conn_string(DB_URL) as db_checkpointer:
            db_checkpointer.setup()
            print("[Main Router] Postgres Checkpointer is ready.")
            
            # Initialize the supervisor
            supervisor = CareerSupervisor(
                user_skills=user_skills,
                resume_path=resume_path,
                checkpointer=db_checkpointer
            )
            
            # Create a persistent thread_id for the supervisor
            supervisor_thread_id = f"supervisor_main_thread_{int(time.time())}"
            print(f"\nSupervisor session started. (Thread: {supervisor_thread_id})")
            print("Type 'exit' to end the session.")
            
            # --- Step 3: Main Router Loop ---
            while True:
                user_request = input("\n[User]: ")
                if user_request.lower() in ["exit", "quit"]:
                    print("[Supervisor]: Goodbye!")
                    break
                
                if not user_request.strip():
                    continue

                print("\n" + "="*80)
                print("[Main Router] Sending to supervisor...")
                
                # Run the supervisor
                final_response, raw_tool_output = supervisor.invoke(
                    user_request, 
                    supervisor_thread_id
                )
                
                # --- ROUTER LOGIC ---
                # Check for handoff signals in the tool output
                if '{"handoff": "interview"}' in raw_tool_output:
                    print(f"\n[Supervisor]: {final_response}")
                    print("\n[Main Router] Handing off to Interview Coach...")
                    run_interactive_interview(
                        checkpointer=db_checkpointer, 
                        user_skills=user_skills
                    )
                    print("\n[Main Router] Control returned to Supervisor.")
                
                elif '{"handoff": "resume"}' in raw_tool_output:
                    print(f"\n[Supervisor]: {final_response}")
                    print("\n[Main Router] Handing off to Resume Strategist...")
                    run_interactive_resume_builder()
                    print("\n[Main Router] Control returned to Supervisor.")
                    
                else:
                    # No handoff, just a normal response
                    print(f"\n[Supervisor]: {final_response}")

    except Exception as e:
        print("="*80)
        print(f"\n[FATAL ERROR] {e}")
        if "Postgres" in str(e):
             print("Could not connect to Postgres DB. Please check settings.py.")
        print("="*80)
        exit()