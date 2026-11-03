"""Compiles LaTeX source to PDF with pdflatex in a locked-down subprocess."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

COMPILE_TIMEOUT_SECONDS = 60
# pdflatex runs twice so cross-references and page layout settle.
COMPILE_PASSES = 2

# The source is LLM-generated, so treat it as untrusted: no shell escape and
# no reading or writing files outside the working directory.
_SANDBOX_ENV = {"shell_escape": "f", "openin_any": "p", "openout_any": "p"}
_INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "TEXMFHOME", "TEXMFVAR", "TEXMFCONFIG")
LOG_TAIL_CHARS = 2000


class LatexUnavailableError(RuntimeError):
    """pdflatex is not installed (expected outside the Docker image)."""


class LatexCompileError(RuntimeError):
    def __init__(self, reason: str, log_tail: str = ""):
        super().__init__(f"pdflatex failed: {reason}")
        self.log_tail = log_tail


def is_available() -> bool:
    return shutil.which("pdflatex") is not None


def _environment(work_dir: Path) -> dict[str, str]:
    """Only what TeX needs: never pass our API keys through to the subprocess."""
    inherited = {key: os.environ[key] for key in _INHERITED_ENV if key in os.environ}
    return {**inherited, "HOME": str(work_dir), **_SANDBOX_ENV}


def _run_pdflatex(work_dir: Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                "doc.tex",
            ],
            cwd=work_dir,
            env=_environment(work_dir),
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise LatexCompileError(f"timed out after {COMPILE_TIMEOUT_SECONDS}s") from error


def compile_pdf(latex_source: str) -> bytes:
    """Return the PDF bytes for `latex_source`."""
    if not is_available():
        raise LatexUnavailableError("pdflatex is not installed")
    with tempfile.TemporaryDirectory(prefix="resume-") as tmp:
        work_dir = Path(tmp)
        (work_dir / "doc.tex").write_text(latex_source, encoding="utf-8")
        for _ in range(COMPILE_PASSES):
            result = _run_pdflatex(work_dir)
            if result.returncode != 0:
                raise LatexCompileError("compilation errors", result.stdout[-LOG_TAIL_CHARS:])
        pdf = work_dir / "doc.pdf"
        if not pdf.exists():
            raise LatexCompileError("no PDF was produced", result.stdout[-LOG_TAIL_CHARS:])
        return pdf.read_bytes()
