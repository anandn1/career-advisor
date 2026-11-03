"""Running compiled LangGraph graphs while reporting each finished step."""

from collections.abc import Callable

from langgraph.graph.state import CompiledStateGraph

StepCallback = Callable[[str], None]


def run_graph(
    graph: CompiledStateGraph,
    state: dict,
    config: dict | None = None,
    on_step: StepCallback | None = None,
) -> dict:
    """Invoke `graph` and return its final state, calling `on_step(node)` after each node."""
    final_state = state
    for mode, chunk in graph.stream(state, config=config, stream_mode=["updates", "values"]):
        if mode == "values":
            final_state = chunk
        elif on_step is not None:
            for node in chunk:
                on_step(node)
    return final_state
