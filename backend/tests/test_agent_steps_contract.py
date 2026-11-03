"""The frontend draws agent progress from step names in frontend/src/lib/agent-steps.json.

These tests fail when a graph node or a handler progress step is renamed, added or removed
without updating that file (the UI would otherwise silently stop showing progress).
"""

import inspect
import json
import re

import pytest

from app.agents.curriculum_architect import get_curriculum_graph
from app.agents.job_market_analyst import get_job_market_graph
from app.core.config import REPO_ROOT
from app.services import run_handlers

STEPS_FILE = REPO_ROOT / "frontend" / "src" / "lib" / "agent-steps.json"
GRAPH_EDGES = {"__start__", "__end__"}

pytestmark = pytest.mark.skipif(not STEPS_FILE.exists(), reason="frontend not checked out")


def frontend_steps(kind: str) -> set[str]:
    stops = json.loads(STEPS_FILE.read_text())[kind]
    return {step for steps in stops.values() for step in steps}


def graph_nodes(graph) -> set[str]:
    return set(graph.get_graph().nodes) - GRAPH_EDGES


def handler_progress_steps() -> set[str]:
    return set(re.findall(r'progress\(\s*"([a-z_]+)"', inspect.getsource(run_handlers)))


def test_job_market_steps_match_the_graph():
    assert frontend_steps("job_market") == graph_nodes(get_job_market_graph())


def test_learning_path_steps_match_the_graph():
    assert frontend_steps("learning_path") == graph_nodes(get_curriculum_graph())


def test_handler_progress_steps_match_the_frontend():
    kinds = ("skills_from_resume", "skills_from_github", "resume", "resume_render", "resume_review")
    assert set().union(*(frontend_steps(kind) for kind in kinds)) == handler_progress_steps()
