"""Run the model evals: uv run python -m evals --models openai/gpt-oss-20b@low --tasks all

Calls Groq (and Gemini for judged tasks), so it costs quota and is not part of pytest.
"""

import argparse
import atexit
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
EMBEDDING_CACHE = EVALS_DIR / ".cache" / "embedding-model"
RESULTS_DIR = EVALS_DIR / "results"


def _isolate_data_dir() -> None:
    """Fresh vector stores per run (so earlier runs cannot leak in), shared model download."""
    data_dir = tempfile.mkdtemp(prefix="career-evals-")
    atexit.register(shutil.rmtree, data_dir, ignore_errors=True)
    os.environ["DATA_DIR"] = data_dir
    os.environ["EMBEDDING_MODEL_DIR"] = str(EMBEDDING_CACHE)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _new_results_path(prefix: str = "") -> Path:
    """A results file name that no earlier run has used (runs can start in the same second)."""
    stem = f"{prefix}{datetime.now():%Y%m%d-%H%M%S}"
    path, counter = RESULTS_DIR / f"{stem}.json", 1
    while path.exists():
        counter += 1
        path = RESULTS_DIR / f"{stem}-{counter}.json"
    return path


def _parse_args(task_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--rejudge",
        type=Path,
        metavar="RESULTS_FILE",
        help="judge the cases in a results file that are not judged yet, and save it",
    )
    mode.add_argument(
        "--rescore",
        type=Path,
        metavar="RESULTS_FILE",
        help="apply the current pass rule to the judged cases in a results file, without "
        "calling the judge, and save it",
    )
    mode.add_argument(
        "--baselines",
        action="store_true",
        help="print the best score a constant answer gets on each classification task",
    )
    mode.add_argument(
        "--calibrate-judge",
        action="store_true",
        help="check the judge scores good outputs high and deliberately bad ones low",
    )
    mode.add_argument(
        "--models",
        help="comma-separated Groq models, optionally with @reasoning_effort, "
        "e.g. openai/gpt-oss-20b@low,qwen/qwen3.8-27b",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help=f"comma-separated, or 'all': {', '.join(task_names)}",
    )
    parser.add_argument(
        "--judge-models", help="comma-separated Gemini models for judged tasks, tried in order"
    )
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=1,
        help="run every case this many times per model to see run-to-run variation; "
        "only the first run is judged",
    )
    parser.add_argument(
        "--defer-judging",
        action="store_true",
        help="skip the judge now (e.g. its daily quota is used up); finish with --rejudge",
    )
    parser.add_argument(
        "--latex-image",
        help="Docker image with TeX to compile generated resumes in when pdflatex is missing, "
        "e.g. career-advisor-backend",
    )
    return parser.parse_args()


def main() -> int:
    _isolate_data_dir()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # Imported after the environment is set, because settings are read on first use.
    from evals import report
    from evals.core import TaskContext
    from evals.judge import JUDGE_MODELS, get_judge
    from evals.models import ModelSpec, UsageRecorder, use_model
    from evals.tasks import curriculum, interview, resume, skills

    all_tasks = {**skills.TASKS, **interview.TASKS, **curriculum.TASKS, **resume.TASKS}
    args = _parse_args(list(all_tasks))
    names = list(all_tasks) if args.tasks == "all" else args.tasks.split(",")
    unknown = [name for name in names if name not in all_tasks]
    if unknown:
        print(f"Unknown tasks: {', '.join(unknown)}", file=sys.stderr)
        return 2

    if args.rescore:
        return _rescore(args.rescore)
    if args.baselines:
        from evals.baselines import floors_table

        print(floors_table())
        return 0

    judge = get_judge(tuple(args.judge_models.split(",")) if args.judge_models else JUDGE_MODELS)
    if judge is None and (args.rejudge or args.calibrate_judge):
        print("Set GEMINI_API_KEY to use the judge.", file=sys.stderr)
        return 2
    if args.rejudge:
        return _rejudge(args.rejudge, judge)
    if args.calibrate_judge:
        return _calibrate(judge)

    specs = [ModelSpec.parse(text) for text in args.models.split(",")]
    metadata = report.run_metadata(
        [spec.label for spec in specs],
        names,
        args.repeats,
        list(judge.models) if judge and not args.defer_judging else [],
    )
    results = []
    for spec in specs:
        recorder = UsageRecorder()
        with use_model(spec, recorder):
            for run in range(1, args.repeats + 1):
                run_judge = judge if run == 1 and not args.defer_judging else None
                context = TaskContext(spec, recorder, run_judge, args.latex_image, run)
                for name in names:
                    print(f"{spec.label} run {run}: {name}", flush=True)
                    results.extend(all_tasks[name](context))

    path = _new_results_path()
    report.save(results, metadata, path)
    _print_summary(results, path)
    return 0


def _rejudge(path: Path, judge) -> int:
    from evals import report
    from evals.core import rejudge

    results, metadata = report.load(path)
    judged = rejudge(results, judge)
    print(f"Judged {judged} more cases.")
    if judged:
        used = metadata.get("judge_models", [])
        metadata["judge_models"] = list(dict.fromkeys([*used, *judge.models]))
    report.save(results, metadata, path)
    _print_summary(results, path)
    return 0


def _rescore(path: Path) -> int:
    from evals import report
    from evals.core import rescore

    results, metadata = report.load(path)
    print(f"{rescore(results)} cases changed between passed and failed.")
    report.save(results, metadata, path)
    _print_summary(results, path)
    return 0


def _calibrate(judge) -> int:
    import json
    from dataclasses import asdict

    from evals.calibration import calibrate, calibration_table

    results = calibrate(judge)
    path = _new_results_path("judge-calibration-")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(r), "passed": r.passed} for r in results]
    path.write_text(json.dumps({"judge_models": judge.models, "pairs": rows}, indent=2))
    print(calibration_table(results))
    print(f"\nDetails: {path}")
    return 0 if results and all(r.passed for r in results) else 1


def _print_summary(results, path: Path) -> None:
    from evals import report

    print()
    print(report.summary_table(results))
    print(f"\nDetails: {path}")


if __name__ == "__main__":
    sys.exit(main())
