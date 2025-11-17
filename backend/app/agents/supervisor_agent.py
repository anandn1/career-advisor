# # backend/app/agents/supervisor_agent.py
import os
import sys
import io
import time
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure utf-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set up paths
ROOT = Path(__file__).resolve().parents[3]
AGENTS_DIR = Path(__file__).parent

for path in [str(ROOT), str(AGENTS_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from datetime import datetime
from enum import Enum

# Colors / UI helpers
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @staticmethod
    def header(text: str):
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

    @staticmethod
    def section(text: str):
        print(f"\n{Colors.BOLD}{Colors.BLUE}> {text}{Colors.END}")
        print(f"{Colors.BLUE}{'-'*80}{Colors.END}")

    @staticmethod
    def success(text: str):
        print(f"{Colors.GREEN}SUCCESS: {text}{Colors.END}")

    @staticmethod
    def error(text: str):
        print(f"{Colors.RED}ERROR: {text}{Colors.END}")

    @staticmethod
    def info(text: str):
        print(f"{Colors.YELLOW}INFO: {text}{Colors.END}")

# State Management
class SessionState(Enum):
    INITIALIZED = "initialized"
    INPUT_COLLECTED = "input_collected"
    SKILLS_EXTRACTED = "skills_extracted"
    JOB_ANALYSIS = "job_analysis"
    RESUME_GENERATED = "resume_generated"
    CURRICULUM_READY = "curriculum_ready"
    INTERVIEW_DONE = "interview_done"
    REPORT_GENERATED = "report_generated"
    CLEANED_UP = "cleaned_up"
    ERROR = "error"

class GlobalState:
    def __init__(self):
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = time.time()
        self.state = SessionState.INITIALIZED
        self.state_history: List[Tuple[str, float]] = []
        self.target_role: Optional[str] = None
        self.location: Optional[str] = None
        self.resume_path: Optional[str] = None
        self.github_username: Optional[str] = None
        self.email: Optional[str] = None
        self.phone: Optional[str] = None
        self.user_skills: List[str] = []
        self.job_data: Dict[str, Any] = {}
        self.skill_gaps: List[str] = []
        self.learning_path: Optional[Any] = None
        self.resume_json: Optional[Dict[str, Any]] = None
        self.resume_pdf_path: Optional[str] = None
        self.interview_history: List[Dict[str, Any]] = []
        self.final_report: Optional[str] = None
        self.errors: List[str] = []

    def update_state(self, new_state: SessionState, message: str = ""):
        self.state = new_state
        elapsed = time.time() - self.start_time
        self.state_history.append((new_state.value + (": " + message if message else ""), elapsed))
        Colors.info(f"[{new_state.value}] {message}")

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.update_state(SessionState.ERROR, msg)
        Colors.error(msg)

# Enhanced Import Handler
def safe_import(module_name: str):
    """Enhanced module import with alias support."""
    module_aliases = {
        'interview_agent_v3_interactive': 'interview_coach',
        'interview_agent': 'interview_coach',
    }
    
    actual_name = module_aliases.get(module_name, module_name)
    
    possible_paths = [
        actual_name,
        f"backend.app.agents.{actual_name}",
        f"agents.{actual_name}",
        f"app.agents.{actual_name}",
    ]
    
    for path in possible_paths:
        try:
            if '.' in path:
                parts = path.split('.')
                module_name_only = parts[-1]
                mod = __import__(path, fromlist=[module_name_only])
            else:
                mod = __import__(path, fromlist=['*'])
            Colors.success(f"Successfully imported: {actual_name}")
            return mod
        except ImportError:
            continue
        except Exception as e:
            continue
    
    Colors.info(f"Module '{module_name}' (actual: '{actual_name}') not available")
    return None

# Agent Wrappers

def extract_skills_wrapper(state: GlobalState) -> bool:
    """Extract skills from resume or GitHub"""
    Colors.section("Extracting Skills")
    try:
        skills_mod = safe_import('skills')
        if not skills_mod or not hasattr(skills_mod, 'get_final_skills_data'):
            Colors.info("Skills module not found; skipping extraction.")
            return False

        if state.resume_path:
            Colors.info(f"Extracting skills from resume: {state.resume_path}")
            skills_set = skills_mod.get_final_skills_data(None, state.resume_path)
        elif state.github_username:
            Colors.info(f"Extracting skills from GitHub: {state.github_username}")
            skills_set = skills_mod.get_final_skills_data(state.github_username, None)
        else:
            Colors.info("No resume or GitHub provided.")
            return False

        state.user_skills = sorted(list(skills_set)) if skills_set else []
        Colors.success(f"Extracted {len(state.user_skills)} skills")
        
        if state.user_skills:
            print("\nYour Skills:")
            for i in range(0, len(state.user_skills), 3):
                chunk = state.user_skills[i:i+3]
                print("  " + " | ".join(f"{s:<20}" for s in chunk))
        
        state.update_state(SessionState.SKILLS_EXTRACTED, f"Extracted {len(state.user_skills)}")
        return True
    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Skill extraction failed: {e}")
        return False

def job_market_analysis_wrapper(state: GlobalState) -> bool:
    """Run job market analysis"""
    Colors.section("Running Job Market Analysis")
    try:
        jm = safe_import('job_market_analyst')
        if not jm:
            Colors.info("job_market_analyst module not found; skipping.")
            return False

        agent_state = {
            "messages": [{"role": "user", "content": f"Find {state.target_role or 'Software Engineer'} jobs in {state.location or 'Remote'}"}],
            "role": state.target_role or "Software Engineer",
            "location": state.location or "Remote",
            "user_skills": state.user_skills,
            "job_data": [],
            "skill_gap_data": {},
            "next_step": "search_db"
        }

        if hasattr(jm, 'parse_query'):
            agent_state = jm.parse_query(agent_state)
        if hasattr(jm, 'search_db'):
            agent_state = jm.search_db(agent_state)
        if hasattr(jm, 'scrape_web'):
            agent_state = jm.scrape_web(agent_state)
        if hasattr(jm, 'analyze_skills'):
            agent_state = jm.analyze_skills(agent_state)

        sg = agent_state.get('skill_gap_data', {})
        state.skill_gaps = sg.get('skill_gaps', []) if sg else []
        state.job_data = agent_state.get('job_data', {})
        
        Colors.success("Job market analysis complete")
        state.update_state(SessionState.JOB_ANALYSIS, f"Jobs: {len(state.job_data) if state.job_data else 0}, Gaps: {len(state.skill_gaps)}")
        
        if state.skill_gaps:
            print("\nTop Skill Gaps:")
            for i, g in enumerate(state.skill_gaps[:10], 1):
                print(f"  {i}. {g}")
        
        return True
    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Job analysis failed: {e}")
        return False

def generate_resume_wrapper(state: GlobalState, force: bool = False) -> bool:
    """Always run FULL interactive resume builder from resume_strategist,
    but still return resume_json + resume_inventory for downstream agents."""
    
    Colors.section("Generating Resume (Interactive Mode)")

    try:
        rs_mod = safe_import("resume_strategist")

        # Debug
        print("\nDEBUG resume_strategist module =", rs_mod)
        print("DEBUG module contents =", dir(rs_mod), "\n")

        # Ensure interactive function exists
        if not hasattr(rs_mod, "run_interactive_mode"):
            Colors.error("resume_strategist.py does NOT contain run_interactive_mode()")
            return False

        Colors.info("Launching FULL interactive resume builder...")
        rs_mod.run_interactive_mode()

        # Check if LaTeX was generated
        tex_path = "resume_output/resume.tex"
        if os.path.exists(tex_path):
            Colors.success(f"Interactive resume generated: {tex_path}")
            state.resume_pdf_path = tex_path
        else:
            Colors.error("❌ Interactive mode did NOT generate resume_output/resume.tex")
            return False

        # ============
        # 2️⃣ RETURN REQUIRED DATA FOR SUPERVISOR / CURRICULUM
        # ============

        # Minimal JSON (you can expand later)
        resume_json = {
            "contact": {
                "name": "Candidate",
                "email": "",
                "phone": "",
                "location": ""
            },
            "summary": f"Resume generated interactively for role: {state.target_role}",
            "skills": state.user_skills or [],
            "education": [],
            "work_experience": [],
            "projects": []
        }

        resume_inventory = f"""
Interactive Resume Inventory
----------------------------
Target Role: {state.target_role}
Skills ({len(state.user_skills)}): {', '.join(state.user_skills[:15])}
"""

        state.resume_json = resume_json
        state.resume_inventory = resume_inventory

        Colors.success("Resume JSON + Inventory (fallback) generated!")
        state.update_state(SessionState.RESUME_GENERATED)

        print("\n==================== RESUME INVENTORY ====================\n")
        print(state.resume_inventory)
        print("\n===========================================================\n")

        return True

    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Interactive resume generation failed: {e}")
        return False



def interview_wrapper(state: GlobalState) -> bool:
    """Launch interactive interview WITH USER CONTEXT"""
    Colors.section("Starting Interactive Interview")
    try:
        ia = safe_import('interview_coach')
        if not ia:
            Colors.error("interview_coach.py not found")
            Colors.info("Expected location: backend/app/agents/interview_coach.py")
            return False

        if not hasattr(ia, 'run_interactive_interview'):
            Colors.error("run_interactive_interview function not found in interview_coach")
            available = [x for x in dir(ia) if not x.startswith('_') and callable(getattr(ia, x))]
            Colors.info(f"Available functions: {', '.join(available[:10])}")
            return False

        Colors.header("INTERVIEW COACH - Interactive Session")
        Colors.info("="*70)
        Colors.info("Interview will focus on your skill gaps and target role.")
        print("\n" + "="*70 + "\n")
        
        # BUILD TOPIC LIST FROM USER'S SKILL GAPS
        topic_mapping = {
            'angular': 'Frontend Frameworks',
            'react': 'Frontend Frameworks',
            'vue': 'Frontend Frameworks',
            'aws': 'Cloud & DevOps',
            'azure': 'Cloud & DevOps',
            'docker': 'Cloud & DevOps',
            'kubernetes': 'Cloud & DevOps',
            'python': 'Programming Languages',
            'java': 'Programming Languages',
            'go': 'Programming Languages',
            'javascript': 'Programming Languages',
            'typescript': 'Programming Languages',
            'sql': 'Databases',
            'postgresql': 'Databases',
            'mongodb': 'Databases',
            'system design': 'System Design',
            'api': 'System Design',
            'microservices': 'System Design',
            'css': 'Frontend Basics',
            'html': 'Frontend Basics',
            'bootstrap': 'Frontend Basics',
        }
        
        # Get unique topics from skill gaps
        user_topics = set()
        for gap in (state.skill_gaps or [])[:15]:
            gap_lower = gap.lower()
            for keyword, topic in topic_mapping.items():
                if keyword in gap_lower:
                    user_topics.add(topic)
                    break
        
        # Always include DSA and System Design as fallback
        if not user_topics:
            user_topics = {'Data Structures & Algorithms', 'System Design'}
        else:
            user_topics.add('System Design')
        
        topic_list = list(user_topics)
        
        Colors.info(f"Interview Topics (based on your gaps): {', '.join(topic_list)}")
        Colors.info(f"Target Role: {state.target_role or 'Full Stack Developer'}")
        Colors.info(f"Your Current Skills: {len(state.user_skills)} identified")
        print()
        
        try:
            # PASS CONTEXT TO INTERVIEW
            ia.run_interactive_interview(
                user_skills=state.user_skills,
                skill_gaps=state.skill_gaps,
                target_role=state.target_role,
                suggested_topics=topic_list
            )
            state.update_state(SessionState.INTERVIEW_DONE, "Interview completed")
            Colors.success("\nInterview session completed!")
            return True
        except KeyboardInterrupt:
            Colors.info("\n\nInterview interrupted by user")
            state.update_state(SessionState.INTERVIEW_DONE, "Interview interrupted")
            return False
            
    except Exception as e:
        Colors.error(f"Interview failed: {e}")
        traceback.print_exc()
        state.add_error(f"Interview failed: {e}")
        return False

def assemble_and_save_report(state: GlobalState) -> str:
    """Generate final report"""
    Colors.section("Generating Final Report")
    try:
        report_lines = [
            "="*80,
            "CAREER ADVISORY REPORT".center(80),
            "="*80,
            f"Session: {state.session_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {int(time.time() - state.start_time)}s",
            "",
            "USER PROFILE",
            "-" * 80,
            f"Target Role: {state.target_role or 'Not specified'}",
            f"Location: {state.location or 'Not specified'}",
            f"GitHub: {state.github_username or 'Not provided'}",
            f"Skills Extracted: {len(state.user_skills)}",
            ""
        ]
        
        if state.skill_gaps:
            report_lines.extend([
                "SKILL GAPS IDENTIFIED",
                "-" * 80,
                f"Total Gaps: {len(state.skill_gaps)}",
                ""
            ])
            for i, g in enumerate(state.skill_gaps[:20], 1):
                report_lines.append(f"  {i}. {g}")
            report_lines.append("")
        
        if state.learning_path and getattr(state.learning_path, "recommendations", None):
            report_lines.extend([
                "RECOMMENDED COURSES",
                "-" * 80,
                f"Total Courses: {len(state.learning_path.recommendations)}",
                ""
            ])
            for i, rec in enumerate(state.learning_path.recommendations[:15], 1):
                title = getattr(rec, "course_title", str(rec))
                skill = getattr(rec, "skill_name", "")
                url = getattr(rec, "course_url", "")
                report_lines.append(f"  {i}. {title}")
                report_lines.append(f"     Skill: {skill}")
                if url:
                    report_lines.append(f"     URL: {url}")
                report_lines.append("")
        
        if state.resume_pdf_path:
            report_lines.extend([
                "RESUME",
                "-" * 80,
                f"Generated: {state.resume_pdf_path}",
                ""
            ])
        
        if state.interview_history:
            report_lines.extend([
                "INTERVIEW SUMMARY",
                "-" * 80
            ])
            for i, s in enumerate(state.interview_history, 1):
                report_lines.append(f"  Session {i}: {s.get('summary', s) if isinstance(s, dict) else s}")
            report_lines.append("")
        
        report_lines.extend([
            "NEXT STEPS & TIPS",
            "-" * 80,
            "1. Focus on closing the top 5 skill gaps",
            "2. Complete recommended courses in priority order",
            "3. Practice 2-3 interview questions per topic daily",
            "4. Build projects showcasing new skills",
            "5. Update your resume with new skills and projects",
            "",
            "="*80
        ])
        
        report_text = "\n".join(report_lines)
        fname = f"report_{state.session_id}.txt"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        Colors.success(f"Report saved: {fname}")
        state.final_report = fname
        state.update_state(SessionState.REPORT_GENERATED, f"Report: {fname}")
        return fname
    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Report generation failed: {e}")
        return ""
def curriculum_wrapper(state: GlobalState) -> bool:
    Colors.section("Building Learning Path (Curriculum Architect)")
    try:
        cur = safe_import("curriculum_architect")
        if not cur or not hasattr(cur, "app"):
            Colors.error("Curriculum architect not found.")
            return False

        # filter soft skills
        SOFT_SKILLS = {
            "communication", "teamwork", "collaboration", "problem solving",
            "scrum", "agile", "user experience", "user interface",
            "leadership", "presentation", "time management"
        }

        technical_skills = [
            s for s in state.skill_gaps if s.lower() not in SOFT_SKILLS
        ]

        if not technical_skills:
            technical_skills = state.skill_gaps

        skill_gap_text = "\n".join(technical_skills)

        resume_inventory_text = json.dumps({
            "skills": state.user_skills,
            "target_role": state.target_role,
            "location": state.location,
            "github": state.github_username,
        }, indent=2)

        extra_context = {
            "skills_to_learn": technical_skills,
            "existing_skills": state.user_skills,
            "user_profile": {
                "target_role": state.target_role,
                "location": state.location,
                "experience_level": "Intermediate",
            },
            "job_context": {
                "job_count": len(state.job_data) if isinstance(state.job_data, list) else 0
            }
        }

        graph_input = {
            "skill_gap_analysis": skill_gap_text,
            "resume_inventory": resume_inventory_text,
            "debug_context": extra_context
        }

        print("\n===== DEBUG: Input sent to Curriculum Architect =====")
        print(json.dumps(graph_input, indent=2))
        print("=====================================================\n")

        config = {"recursion_limit": 300}
        Colors.info("Invoking Curriculum Architect...")
        final_state = cur.app.invoke(graph_input, config=config)

        state.learning_path = (
            final_state.get("draft_path")
            or final_state.get("final_path")
            or final_state
        )

        Colors.success("Learning path generated successfully!")
        state.update_state(SessionState.CURRICULUM_READY)

        print("\n================ LEARNING PATH (TOP 20) ================\n")

        path = state.learning_path

        if hasattr(path, "dict"):
            path = path.dict()

        if isinstance(path, dict) and "path" in path:
            path = path["path"]

        if isinstance(path, dict) and "recommendations" in path:
            recs = path["recommendations"][:20]

            for i, r in enumerate(recs, 1):
                print(f"{i}. {r.get('course_title', 'Unknown Course')} "
                      f"({r.get('skill_name', 'Unknown Skill')})")
                url = r.get("course_url")
                if url:
                    print(f"     URL: {url}")
                print()
        else:
            print("⚠ WARNING: Learning path missing 'recommendations' key.\n")

        print("=========================================================\n")
        return True

    except Exception as e:
        traceback.print_exc()
        state.add_error(f"Curriculum generation failed: {e}")
        return False

# Main Loop
def run_supervisor_loop():
    """Main conversational supervisor loop"""
    Colors.header("AGENTIC CAREER ADVISOR - Interactive Terminal")
    Colors.info("Talk naturally with the assistant. Type 'exit' to quit.\n")
    Colors.info("Example commands:")
    print("  - 'Find jobs for backend developer in Bangalore'")
    print("  - 'Generate my resume'")
    print("  - 'Start a mock interview'")
    print("  - 'Show my skills'")
    print("  - 'Create learning path'\n")

    state = GlobalState()
    state.update_state(SessionState.INITIALIZED, "Supervisor started")

    # Initial setup
    try:
        print("\nHello! I'm your AI Career Advisor. Let's get started.\n")
        
        if not state.target_role:
            state.target_role = input("What job role are you targeting? (e.g., Full Stack Developer): ").strip() or None
            if state.target_role:
                Colors.success(f"Target role: {state.target_role}")
        
        if not state.location:
            state.location = input("Preferred location (city or 'Remote'): ").strip() or None
            if state.location:
                Colors.success(f"Location: {state.location}")

        print("\nHow would you like to provide your background?")
        print("  1. Upload resume (PDF/DOCX/TXT)")
        print("  2. Provide GitHub username")
        print("  3. Skip for now")
        
        choice = input("\nSelect (1/2/3): ").strip()
        
        if choice == '1':
            p = input("Full path to resume: ").strip().strip('"\'')
            if p and Path(p).exists():
                state.resume_path = p
                Colors.success(f"Using resume: {p}")
                extract_skills_wrapper(state)
            else:
                Colors.error("Path not found. Skipping.")
        elif choice == '2':
            gh = input("GitHub username or URL: ").strip()
            if gh:
                if gh.startswith("https://github.com/"):
                    gh = gh.rstrip('/').split('/')[-1]
                state.github_username = gh
                Colors.success(f"GitHub username: {gh}")
                extract_skills_wrapper(state)
        else:
            Colors.info("Proceeding without resume/GitHub for now.")
        
        state.update_state(SessionState.INPUT_COLLECTED, "User input collected")
        
        # Run initial job analysis if we have the basics
        if state.target_role and state.user_skills:
            Colors.info("\nRunning initial job market analysis...")
            job_market_analysis_wrapper(state)

    except KeyboardInterrupt:
        print("\n\nSetup interrupted. Exiting.")
        return

    # Main conversation loop
    print("\n" + "="*70)
    print("READY! You can now chat naturally with me.".center(70))
    print("="*70 + "\n")
    
    while True:
        try:
            user_input = input("\nYOU: ").strip()
            if not user_input:
                continue

            lower = user_input.lower()

            # Exit command
            if lower in ('exit', 'quit', 'q', 'bye'):
                print("\nGoodbye! Saving your session...")
                state.update_state(SessionState.CLEANED_UP, "User exited")
                break

            # Interview commands
            if any(word in lower for word in ['interview', 'mock', 'practice', 'prep']):
                Colors.info("Starting interview session...")
                if not state.skill_gaps:
                    Colors.info("Running job analysis first to identify topics...")
                    job_market_analysis_wrapper(state)
                interview_wrapper(state)
                continue

            # Job analysis commands
            if any(word in lower for word in ['job', 'jobs', 'opening', 'openings', 'market', 'search']):
                Colors.info("Running job market analysis...")
                job_market_analysis_wrapper(state)
                continue

            # Learning path commands
            if any(word in lower for word in ['learn', 'course', 'courses', 'curriculum', 'path', 'recommend']):
                Colors.info("Generating personalized learning path...")
                if not state.skill_gaps:
                    Colors.info("Need to identify skill gaps first...")
                    job_market_analysis_wrapper(state)
                curriculum_wrapper(state)
                continue

            # Resume commands
            if any(word in lower for word in ['resume', 'cv', 'build resume', 'create resume']):
                Colors.info("Generating resume...")
                generate_resume_wrapper(state, force=True)
                continue

            # Skills commands
            if any(word in lower for word in ['skill', 'skills', 'show skills', 'my skills']):
                if not state.user_skills:
                    Colors.info("No skills extracted yet. Extracting now...")
                    extract_skills_wrapper(state)
                else:
                    print("\nYour Extracted Skills:")
                    for i in range(0, len(state.user_skills), 3):
                        chunk = state.user_skills[i:i+3]
                        print("  " + " | ".join(f"{s:<20}" for s in chunk))
                continue

            # Report commands
            if any(word in lower for word in ['report', 'summary', 'final']):
                Colors.info("Assembling final report...")
                assemble_and_save_report(state)
                continue

            # Default response
            Colors.info("I can help you with:")
            print("  - 'start interview' - Practice mock interviews")
            print("  - 'find jobs' - Analyze job market for your role")
            print("  - 'learning path' - Get recommended courses")
            print("  - 'generate resume' - Build professional resume")
            print("  - 'show skills' - View your extracted skills")
            print("  - 'report' - Generate final summary report")
            print("  - 'exit' - End session and save report")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            Colors.error(f"Unexpected error: {e}")
            traceback.print_exc()
            Colors.info("You can continue or type 'exit' to quit.")

    # Save final report on exit
    try:
        fname = assemble_and_save_report(state)
        Colors.success(f"\nFinal report saved: {fname}")
        print(f"\nSession complete! Duration: {int(time.time() - state.start_time)}s")
    except Exception as e:
        Colors.error(f"Failed to save final report: {e}")


if __name__ == "__main__":
    run_supervisor_loop()