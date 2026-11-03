import subprocess

import pytest
from langchain_core.runnables import RunnableLambda

from app.agents import resume_strategist as rs
from app.services import latex

# --- Escaping and cleanup of LLM output ---


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("\\", r"\textbackslash{}"),
        ("plain text", "plain text"),
        ("", ""),
    ],
)
def test_each_special_character_is_escaped(raw, escaped):
    assert rs.escape_latex(raw) == escaped


def test_backslash_is_escaped_only_once():
    assert rs.escape_latex("\\&") == r"\textbackslash{}\&"


def test_escaping_reaches_nested_values():
    data = {"a": ["R&D", {"b": "50%"}], "n": 3, "none": None}
    assert rs.escape_latex(data) == {
        "a": [r"R\&D", {"b": r"50\%"}],
        "n": 3,
        "none": None,
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```latex\n\\documentclass{a}\n```", "\\documentclass{a}"),
        (
            "Sure!\n```latex\n\\documentclass{a}\n```\nHope it helps",
            "\\documentclass{a}",
        ),
        ("```\n\\documentclass{a}", "\\documentclass{a}"),
        ("Here you go: \\documentclass{a} body", "\\documentclass{a} body"),
        ("  \\documentclass{a}  ", "\\documentclass{a}"),
        ("no latex at all", "no latex at all"),
    ],
)
def test_code_fences_and_prose_are_removed(text, expected):
    assert rs.strip_code_fences(text) == expected


VALID = "\\documentclass{article}\n\\begin{document}\nHi\n\\end{document}"


def test_complete_document_is_valid():
    assert rs.validate_latex(VALID) == VALID


@pytest.mark.parametrize("marker", rs.REQUIRED_LATEX_MARKERS)
def test_missing_marker_is_invalid(marker):
    with pytest.raises(rs.InvalidLatexError, match=marker.replace("\\", "\\\\")):
        rs.validate_latex(VALID.replace(marker, ""))


def test_commented_out_markers_do_not_count():
    commented = "\n".join(f"% {line}" for line in VALID.splitlines())
    with pytest.raises(rs.InvalidLatexError):
        rs.validate_latex(commented)


def test_escaped_percent_is_not_a_comment():
    document = "\\documentclass{a} 50\\% \\begin{document}\\end{document}"
    assert rs.validate_latex(document) == document


def test_bundled_default_template_is_valid():
    rs.validate_latex(rs.default_template())


@pytest.mark.parametrize(
    ("length", "expected_length"),
    [
        (rs.MAX_BULLET_LENGTH, rs.MAX_BULLET_LENGTH),
        (rs.MAX_BULLET_LENGTH + 1, rs.MAX_BULLET_LENGTH),
    ],
)
def test_long_bullets_are_shortened_to_the_limit(length, expected_length):
    assert len(rs._shorten("x" * length)) == expected_length


def test_shortened_bullets_end_with_an_ellipsis():
    assert rs._shorten("x" * (rs.MAX_BULLET_LENGTH + 1)).endswith("...")


def test_repo_without_description_gets_a_default_bullet():
    repo = {
        "name": "tool",
        "description": "",
        "language": "",
        "html_url": "https://g/tool",
    }
    project = rs.projects_from_repos([repo])[0]
    assert (
        project.description_bullets[0] == "Developed the tool repository."
        and project.tech_stack == []
    )


def test_profile_without_github_makes_no_requests(monkeypatch):
    monkeypatch.setattr(rs.github, "fetch_profile_and_repos", lambda *a: pytest.fail("fetched"))
    data = rs._profile_with_github(rs.ResumeProfile(target_role="Dev", skills=["go"]))
    assert data["skills"] == ["go"]


def test_github_fills_only_missing_contact_fields(monkeypatch):
    github_data = {
        "profile": {
            "name": "GH Name",
            "location": "GH City",
            "html_url": "https://github.com/x",
        },
        "repos": [{"name": "r", "description": "d", "language": "Rust", "html_url": "u"}],
    }
    monkeypatch.setattr(rs.github, "fetch_profile_and_repos", lambda *a: github_data)
    profile = rs.ResumeProfile(
        target_role="Dev",
        github_username="x",
        contact=rs.ContactInfo(name="Real Name"),
        skills=["go"],
    )
    data = rs._profile_with_github(profile)
    assert (data["contact"]["name"], data["contact"]["location"], data["skills"]) == (
        "Real Name",
        "GH City",
        ["go", "Rust"],
    )


# --- Compiling ---


@pytest.fixture
def fake_pdflatex(monkeypatch):
    """pdflatex 'is installed'; `behaviour` decides what each run does."""
    calls = []

    def run(behaviour):
        def fake_run(command, cwd, env, **kwargs):
            calls.append({"command": command, "cwd": cwd, "env": env, "kwargs": kwargs})
            return behaviour(cwd)

        monkeypatch.setattr(latex.shutil, "which", lambda name: "/usr/bin/pdflatex")
        monkeypatch.setattr(latex.subprocess, "run", fake_run)
        return calls

    return run


def succeed(cwd):
    (cwd / "doc.pdf").write_bytes(b"%PDF-1.7 fake")
    return subprocess.CompletedProcess([], 0, stdout="ok")


def test_compiled_pdf_bytes_are_returned(fake_pdflatex):
    fake_pdflatex(succeed)
    assert latex.compile_pdf(VALID) == b"%PDF-1.7 fake"


def test_pdflatex_runs_twice(fake_pdflatex):
    calls = fake_pdflatex(succeed)
    latex.compile_pdf(VALID)
    assert len(calls) == latex.COMPILE_PASSES


def test_api_keys_are_not_passed_to_pdflatex(fake_pdflatex, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    calls = fake_pdflatex(succeed)
    latex.compile_pdf(VALID)
    assert "GROQ_API_KEY" not in calls[0]["env"]


def test_pdflatex_runs_sandboxed(fake_pdflatex):
    calls = fake_pdflatex(succeed)
    latex.compile_pdf(VALID)
    env = calls[0]["env"]
    assert (env["openin_any"], env["openout_any"], env["shell_escape"]) == (
        "p",
        "p",
        "f",
    )


def test_pdflatex_home_is_its_temp_dir(fake_pdflatex):
    calls = fake_pdflatex(succeed)
    latex.compile_pdf(VALID)
    assert calls[0]["env"]["HOME"] == str(calls[0]["cwd"])


def test_compile_errors_include_the_log(fake_pdflatex):
    fake_pdflatex(lambda cwd: subprocess.CompletedProcess([], 1, stdout="! Undefined control"))
    with pytest.raises(latex.LatexCompileError) as error:
        latex.compile_pdf(VALID)
    assert "Undefined control" in error.value.log_tail


def test_success_without_a_pdf_is_a_compile_error(fake_pdflatex):
    fake_pdflatex(lambda cwd: subprocess.CompletedProcess([], 0, stdout="No pages of output."))
    with pytest.raises(latex.LatexCompileError, match="no PDF"):
        latex.compile_pdf(VALID)


def test_timeout_is_a_compile_error(fake_pdflatex):
    def hang(cwd):
        raise subprocess.TimeoutExpired("pdflatex", latex.COMPILE_TIMEOUT_SECONDS)

    fake_pdflatex(hang)
    with pytest.raises(latex.LatexCompileError, match="timed out"):
        latex.compile_pdf(VALID)


def test_missing_pdflatex_is_reported(monkeypatch):
    monkeypatch.setattr(latex.shutil, "which", lambda name: None)
    with pytest.raises(latex.LatexUnavailableError):
        latex.compile_pdf(VALID)


def test_pdflatex_command_is_locked_down(fake_pdflatex):
    calls = fake_pdflatex(succeed)
    latex.compile_pdf(VALID)
    flags = {"-no-shell-escape", "-halt-on-error", "-interaction=nonstopmode"}
    assert flags <= set(calls[0]["command"])


def test_pdflatex_runs_with_a_timeout(fake_pdflatex):
    calls = fake_pdflatex(succeed)
    latex.compile_pdf(VALID)
    assert calls[0]["kwargs"]["timeout"] == latex.COMPILE_TIMEOUT_SECONDS


def test_failure_on_the_second_pass_is_reported(fake_pdflatex):
    runs = []

    def second_pass_fails(cwd):
        runs.append(cwd)
        if len(runs) == 2:
            return subprocess.CompletedProcess([], 1, stdout="! Emergency stop")
        return succeed(cwd)

    fake_pdflatex(second_pass_fails)
    with pytest.raises(latex.LatexCompileError):
        latex.compile_pdf(VALID)


class RecordingResumeLLM(RunnableLambda):
    """Fake resume LLM: replies with `reply` (or `draft` when structured) and records prompts."""

    def __init__(self, reply="", draft=None):
        self.prompts = []
        self.reply = reply
        self.draft = draft
        super().__init__(self._respond)

    def _respond(self, prompt):
        self.prompts.append(prompt.to_string())
        return self.reply

    def _structured(self, prompt):
        self.prompts.append(prompt.to_string())
        return self.draft

    def with_structured_output(self, schema):
        return RunnableLambda(self._structured)


def draft(**contact):
    return rs.ResumeDraft(contact=rs.ContactInfo(**contact), summary="Cut costs by 50%")


def test_render_latex_escapes_the_resume_data(monkeypatch):
    llm = RecordingResumeLLM(reply=f"```latex\n{VALID}\n```")
    monkeypatch.setattr(rs, "get_resume_llm", lambda: llm)
    rs.render_latex(draft(name="R&D Lead"), "TEMPLATE")
    assert r"R\\&D Lead" in llm.prompts[0] and r"50\\%" in llm.prompts[0]


def test_render_latex_returns_the_cleaned_document(monkeypatch):
    monkeypatch.setattr(
        rs, "get_resume_llm", lambda: RecordingResumeLLM(reply=f"```\n{VALID}\n```")
    )
    assert rs.render_latex(draft(), "TEMPLATE") == VALID


def test_render_latex_rejects_incomplete_output(monkeypatch):
    monkeypatch.setattr(rs, "get_resume_llm", lambda: RecordingResumeLLM(reply="Sorry, I can't."))
    with pytest.raises(rs.InvalidLatexError):
        rs.render_latex(draft(), "TEMPLATE")


def test_render_latex_includes_the_template(monkeypatch):
    llm = RecordingResumeLLM(reply=VALID)
    monkeypatch.setattr(rs, "get_resume_llm", lambda: llm)
    rs.render_latex(draft(), "MY-REFERENCE-TEMPLATE")
    assert "MY-REFERENCE-TEMPLATE" in llm.prompts[0]


def test_generate_resume_uses_the_default_template_when_none_given(monkeypatch):
    llm = RecordingResumeLLM(reply=VALID, draft=draft(name="Ada"))
    monkeypatch.setattr(rs, "get_resume_llm", lambda: llm)
    result = rs.generate_resume(rs.ResumeProfile(target_role="Dev"))
    assert result.latex == VALID and rs.default_template()[:200] in llm.prompts[1]


def test_draft_prompt_names_the_target_role(monkeypatch):
    llm = RecordingResumeLLM(draft=draft())
    monkeypatch.setattr(rs, "get_resume_llm", lambda: llm)
    rs.draft_resume(rs.ResumeProfile(target_role="Site Reliability Engineer"))
    assert '"Site Reliability Engineer"' in llm.prompts[0]


def test_github_repos_become_projects(monkeypatch):
    github_data = {
        "profile": {},
        "repos": [{"name": "r", "description": "d", "language": "Rust", "html_url": "u"}],
    }
    monkeypatch.setattr(rs.github, "fetch_profile_and_repos", lambda *a: github_data)
    data = rs._profile_with_github(rs.ResumeProfile(target_role="Dev", github_username="x"))
    assert [p["name"] for p in data["projects"]] == ["r"]


def test_github_profile_url_is_fetched_as_a_contact_link(monkeypatch):
    github_data = {"profile": {"html_url": "https://github.com/x"}, "repos": []}
    monkeypatch.setattr(rs.github, "fetch_profile_and_repos", lambda *a: github_data)
    data = rs._profile_with_github(rs.ResumeProfile(target_role="Dev", github_username="x"))
    assert data["contact"]["github"] == "https://github.com/x"


@pytest.mark.parametrize(
    ("value", "username"),
    [("octocat", "octocat"), ("https://github.com/octocat", "octocat"), ("", None), (None, None)],
)
def test_resume_profile_normalizes_the_github_username(value, username):
    assert rs.ResumeProfile(target_role="Dev", github_username=value).github_username == username


def test_resume_profile_rejects_a_bad_github_username():
    with pytest.raises(ValueError):
        rs.ResumeProfile(target_role="Dev", github_username="../etc/passwd")


def test_line_break_before_percent_starts_a_comment():
    commented = "\\documentclass{a}\\begin{document} x \\\\% \\end{document}"
    with pytest.raises(rs.InvalidLatexError, match="end"):
        rs.validate_latex(commented)


def test_users_own_github_link_is_kept(monkeypatch):
    github_data = {"profile": {"html_url": "https://github.com/from-api"}, "repos": []}
    monkeypatch.setattr(rs.github, "fetch_profile_and_repos", lambda *a: github_data)
    profile = rs.ResumeProfile(
        target_role="Dev", github_username="x", contact=rs.ContactInfo(github="https://mine.dev")
    )
    assert rs._profile_with_github(profile)["contact"]["github"] == "https://mine.dev"


def test_repo_languages_already_in_skills_are_not_repeated(monkeypatch):
    github_data = {
        "profile": {},
        "repos": [{"name": "r", "description": "", "language": "Go", "html_url": "u"}],
    }
    monkeypatch.setattr(rs.github, "fetch_profile_and_repos", lambda *a: github_data)
    profile = rs.ResumeProfile(target_role="Dev", github_username="x", skills=["Go", "SQL"])
    assert rs._profile_with_github(profile)["skills"] == ["Go", "SQL"]
