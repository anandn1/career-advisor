"""Results files and a Markdown summary: one row per task, one column per model."""

import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from evals.core import DATA_DIR, DECIDED_BY_APP, RATE_LIMITED, CaseResult
from evals.models import PRICES_PER_MILLION, Usage

COST_UNIT = 1000  # cost is shown per 1,000 runs of a case


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=DATA_DIR
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_metadata(
    models: list[str], tasks: list[str], repeats: int, judge_models: list[str]
) -> dict:
    """What a reader needs to reproduce a run, including which version of the labels it used."""
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "uncommitted_changes": bool(_git("status", "--porcelain")),
        "models": models,
        "tasks": tasks,
        "repeats": repeats,
        "judge_models": judge_models,
        "prices_per_million": PRICES_PER_MILLION,
        "data_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            for path in sorted(DATA_DIR.glob("*.json"))
        },
    }


def save(results: list[CaseResult], metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"metadata": metadata, "results": [asdict(r) for r in results]}
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def load(path: Path) -> tuple[list[CaseResult], dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, list):  # files from before metadata was recorded
        document = {"metadata": {}, "results": document}
    rows = document["results"]
    results = [CaseResult(**{**row, "usage": Usage(**row["usage"])}) for row in rows]
    return results, document["metadata"]


def _score_range(scored: list[CaseResult]) -> str:
    """Mean score; with repeats, also the lowest and highest per-run mean."""
    by_run: dict[int, list[float]] = defaultdict(list)
    for case in scored:
        by_run[case.run].append(case.score)
    run_means = [mean(scores) for scores in by_run.values()]
    overall = f"{mean(c.score for c in scored):.2f}"
    if len(run_means) < 2:
        return overall
    return f"{overall} [{min(run_means):.2f}-{max(run_means):.2f}]"


def _is_rate_limited(case: CaseResult) -> bool:
    return bool(case.error and case.error.startswith(RATE_LIMITED))


def _is_decided_by_app(case: CaseResult) -> bool:
    return case.error == DECIDED_BY_APP


def _cell(cases: list[CaseResult]) -> str:
    scored = [c for c in cases if c.score is not None]
    rate_limited = sum(_is_rate_limited(c) for c in cases)
    by_app = sum(_is_decided_by_app(c) for c in cases)
    score = _score_range(scored) if scored else "n/a"
    passed = sum(c.passed for c in scored)
    errors = sum(c.error is not None for c in scored)
    # Cost and time per case the model actually ran: refused calls and cases the app decided
    # cost nothing and take no time, so counting them would make a model look cheap and fast.
    ran = [c for c in cases if not (_is_rate_limited(c) or _is_decided_by_app(c))] or cases
    costs = [c.cost for c in ran if c.cost is not None]
    cost = f"${mean(costs) * COST_UNIT:.2f}" if costs else "-"
    seconds = mean(c.usage.seconds for c in ran)
    error_note = (
        (f", {errors} err" if errors else "")
        + (f", {rate_limited} rate-limited" if rate_limited else "")
        + (f", {by_app} decided by the app" if by_app else "")
    )
    return f"{score} ({passed}/{len(scored)}{error_note}) {cost} {seconds:.1f}s"


def summary_table(results: list[CaseResult]) -> str:
    by_task: dict[str, dict[str, list[CaseResult]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        by_task[result.task][result.model].append(result)
    models = list(dict.fromkeys(r.model for r in results))
    lines = [
        "Each cell: mean score [range of per-run means] (passed/scored, errors) "
        f"cost per {COST_UNIT:,} runs, seconds per case. Unjudged, rate-limited and "
        "app-decided cases are left out of the score.",
        "",
        "| Task | " + " | ".join(models) + " |",
        "| :--- |" + " :--- |" * len(models),
    ]
    for task, per_model in by_task.items():
        cells = [_cell(per_model[m]) if per_model.get(m) else "-" for m in models]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
