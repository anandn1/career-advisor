"""Resume strategist (draft + LaTeX) and resume reviewer."""

import json
import re
import subprocess

from app.agents.resume_reviewer import review_resume
from app.agents.resume_strategist import ResumeProfile, generate_resume
from app.db.seed_templates import TEMPLATES_DIR
from app.services import latex
from evals.core import CaseResult, JudgeRequest, Outcome, TaskContext, load_data

COMPILE_TIMEOUT_SECONDS = 180
DOCKER_ERROR = 125  # docker's own exit code (daemon down, image missing), not the container's
# A percentage or multiplier stated as a result: "30%", "30 percent", "3x", "3×", "$3\times$".
_METRIC = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b|x\b|×|\$?\\times)", re.IGNORECASE)
COMPILE_SCRIPT = (
    "import sys; from app.services.latex import compile_pdf; compile_pdf(sys.stdin.read())"
)
REVIEW_ITEMS = range(3, 6)  # the reviewer is asked for 3-5 strengths, weaknesses, suggestions

RESUME_CRITERIA = [
    "Faithful: no employer, title, date, degree, number or achievement beyond the input profile",
    "Tailored to the target role: summary and skill order put the most relevant experience first",
    "Bullets are concise, start with an action and keep every concrete detail from the profile",
]
REVIEW_CRITERIA = [
    "Every point is grounded in the resume text; nothing is invented about the candidate",
    "Specific to the target role rather than generic resume advice",
    "Suggestions are concrete and actionable",
    "Honest about the gaps between the resume and the role",
]


_COMMENT = re.compile(r"(?<!\\)%.*")  # to the end of the line; "\\%" is a percent sign
_FORMATTING = re.compile(r"\\[a-zA-Z]+\*?\{([^{}]*)\}")  # \\textbf{75} -> 75
_LATEX_SPACE = re.compile(r"~|\\[,;: ]")  # 75~\\% and 75\\,\\% are "75 %"
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")  # 1,000 -> 1000


def _plain_text(source: str) -> str:
    """What the reader sees, roughly: the body without comments, escapes undone.

    Lets names like 'R\\&D' or '40\\%' be found, and keeps a commented-out template line
    from counting as text that is in the resume.
    """
    body = source.split("\\begin{document}", 1)[-1]
    body = _COMMENT.sub("", body)
    return re.sub(r"\\([%&$#_{}])", r"\1", body)


def _metric_text(text: str) -> str:
    """Plain text with formatting commands, LaTeX spaces and thousands separators removed."""
    previous = None
    while previous != text:  # nested formatting, e.g. \\textbf{\\emph{75}}
        previous, text = text, _FORMATTING.sub(r"\1", text)
    return _THOUSANDS.sub("", _LATEX_SPACE.sub(" ", text))


def invented_metrics(text: str, known_numbers: list[str]) -> list[str]:
    """Percentages and multipliers whose number is not in the profile ('40.0%' matches '40')."""
    known = {float(number) for number in known_numbers}
    return sorted({n for n in _METRIC.findall(_metric_text(text)) if float(n) not in known})


def _compile(source: str, latex_image: str | None) -> str:
    """'ok', 'failed: ...', or 'not checked' when no TeX (or Docker) is available."""
    try:
        if latex.is_available():
            latex.compile_pdf(source)
            return "ok"
        if latex_image is None:
            return "not checked"
        subprocess.run(
            ["docker", "run", "--rm", "-i", "--network", "none", "--entrypoint", "python"]
            + [latex_image, "-c", COMPILE_SCRIPT],
            input=source,
            text=True,
            capture_output=True,
            check=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
        )
        return "ok"
    except FileNotFoundError:  # no docker on this machine
        return "not checked"
    except subprocess.TimeoutExpired:
        return f"failed: no result within {COMPILE_TIMEOUT_SECONDS}s"
    except subprocess.CalledProcessError as error:
        if error.returncode == DOCKER_ERROR or "LatexUnavailableError" in (error.stderr or ""):
            return "not checked"  # no such image, or no TeX in it: not the model's fault
        return f"failed: {(error.stderr or '')[-300:]}"
    except latex.LatexCompileError as error:
        return f"failed: {error.log_tail[-300:]}"


def resume_request(case: dict, resume_body: str) -> JudgeRequest:
    """The judge reads the finished resume (the LaTeX), which is what the user gets."""
    return JudgeRequest(
        "Write a resume as LaTeX, tailored to the target role, using only the given profile.",
        json.dumps(case["profile"], indent=2),
        resume_body,
        RESUME_CRITERIA,
    )


def review_request(case: dict, review_json: str) -> JudgeRequest:
    return JudgeRequest(
        f"Review this resume for the role '{case['target_role']}'. "
        f"User's request: {case['request'] or 'none'}.",
        resume_text(case["resume"]),
        review_json,
        REVIEW_CRITERIA,
    )


def resume_generation(ctx: TaskContext) -> list[CaseResult]:
    def run(case: dict) -> Outcome:
        template = case["template"]
        template_source = (
            (TEMPLATES_DIR / f"{template}.tex").read_text(encoding="utf-8") if template else None
        )
        result = generate_resume(ResumeProfile.model_validate(case["profile"]), template_source)
        text = _plain_text(result.latex)
        dropped = [fact for fact in case["must_keep"] if fact not in text]
        invented = invented_metrics(text, case["known_numbers"])
        leaked = [marker for marker in case["template_markers"] if marker in text]
        compiled = _compile(result.latex, ctx.latex_image)
        checks = [not dropped, not invented, not leaked]
        if compiled != "not checked":
            checks.append(compiled == "ok")
        details = {
            "dropped_facts": dropped,
            "invented_metrics": invented,
            "leaked_template_text": leaked,
            "compiled": compiled,
            "latex": result.latex,
        }
        request = resume_request(case, result.latex.split("\\begin{document}", 1)[-1])
        return ctx.judged_outcome(request, details, Outcome(sum(checks) / len(checks), all(checks)))

    return [
        ctx.run_case("resume_generation", case["id"], lambda case=case: run(case))
        for case in load_data("resume")["generation"]
    ]


def resume_text(reference: str) -> str:
    """'skills.json#backend-engineer' -> that case's text."""
    file_name, _, case_id = reference.partition("#")
    cases = load_data(file_name.removesuffix(".json"))["cases"]
    return next(case["text"] for case in cases if case["id"] == case_id)


def resume_review(ctx: TaskContext) -> list[CaseResult]:
    def run(case: dict) -> Outcome:
        text = resume_text(case["resume"])
        review = review_resume(text, case["target_role"], case["request"])
        sizes = [len(review.strengths), len(review.weaknesses), len(review.suggestions)]
        sizes_ok = all(size in REVIEW_ITEMS for size in sizes)
        request = review_request(case, review.model_dump_json(indent=2))
        objective = Outcome(float(sizes_ok), sizes_ok)
        return ctx.judged_outcome(request, {"item_counts": sizes}, objective)

    return [
        ctx.run_case("resume_review", case["id"], lambda case=case: run(case))
        for case in load_data("resume")["review"]
    ]


TASKS = {"resume_generation": resume_generation, "resume_review": resume_review}
