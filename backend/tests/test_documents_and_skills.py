import pytest
from conftest import FakeReply
from docx import Document as DocxDocument

from app.agents import skills
from app.services import documents

# --- Reading resume files ---


def test_txt_text_is_read(tmp_path):
    path = tmp_path / "cv.txt"
    path.write_text("Python and Go", encoding="utf-8")
    assert documents.read_document_text(path) == "Python and Go"


def test_invalid_utf8_in_txt_is_replaced(tmp_path):
    path = tmp_path / "cv.txt"
    path.write_bytes(b"Python \xff Go")
    assert documents.read_document_text(path) == "Python � Go"


def test_docx_paragraphs_and_table_cells_are_read(tmp_path):
    path = tmp_path / "cv.docx"
    doc = DocxDocument()
    doc.add_paragraph("Experienced engineer")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Skills"
    table.cell(0, 1).text = "Kubernetes, Terraform"
    doc.save(path)
    assert (
        documents.read_document_text(path) == "Experienced engineer\nSkills\nKubernetes, Terraform"
    )


@pytest.mark.parametrize("name", ["cv.pdf", "cv.docx"])
def test_corrupt_files_raise_a_document_error(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"this is not really a " + name.encode())
    with pytest.raises(documents.UnreadableDocumentError):
        documents.read_document_text(path)


@pytest.mark.parametrize("name", ["cv.doc", "cv.rtf", "cv", "cv.PDF.exe"])
def test_unsupported_types_are_rejected(tmp_path, name):
    path = tmp_path / name
    path.write_text("x")
    with pytest.raises(documents.UnsupportedDocumentError):
        documents.read_document_text(path)


def test_suffix_check_ignores_case(tmp_path):
    path = tmp_path / "CV.TXT"
    path.write_text("Rust")
    assert documents.read_document_text(path) == "Rust"


def test_document_errors_share_one_base_class():
    assert issubclass(documents.UnsupportedDocumentError, documents.DocumentError)
    assert issubclass(documents.UnreadableDocumentError, documents.DocumentError)


# --- Skill extraction ---


def test_llm_skill_list_is_parsed_and_deduplicated(monkeypatch, fake_llm):
    monkeypatch.setattr(skills, "get_llm", lambda: fake_llm(" Python, 'Go', python ,, \"SQL\" "))
    assert skills.extract_skills_from_text("resume") == ["Go", "Python", "SQL"]


def test_llm_failure_falls_back_to_the_skill_matcher(monkeypatch):
    class BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("rate limited")

    monkeypatch.setattr(skills, "get_llm", lambda: BrokenLLM())
    assert skills.extract_skills_from_text("Built APIs with Python and Docker") == [
        "docker",
        "python",
    ]


def test_github_skills_use_languages_and_readme(monkeypatch, fake_llm):
    llm = fake_llm(FakeReply("Rust, WebAssembly"))
    monkeypatch.setattr(skills, "get_llm", lambda: llm)
    monkeypatch.setattr(skills.github, "fetch_repos", lambda name: [{"languages_url": "x"}])
    monkeypatch.setattr(skills.github, "fetch_repo_languages", lambda repos: ["Rust"])
    monkeypatch.setattr(skills.github, "fetch_profile_readme", lambda name: "I love WASM")
    assert skills.extract_skills_from_github("octocat") == ["Rust", "WebAssembly"]


def test_github_llm_prompt_includes_languages_and_readme(monkeypatch, fake_llm):
    llm = fake_llm("Rust")
    monkeypatch.setattr(skills, "get_llm", lambda: llm)
    monkeypatch.setattr(skills.github, "fetch_repos", lambda name: [])
    monkeypatch.setattr(skills.github, "fetch_repo_languages", lambda repos: ["Rust", "Go"])
    monkeypatch.setattr(skills.github, "fetch_profile_readme", lambda name: "README text")
    skills.extract_skills_from_github("octocat")
    prompt = llm.prompts[0][1].content
    assert '"Rust, Go"' in prompt and '"README text"' in prompt


def test_github_llm_failure_falls_back_to_languages(monkeypatch):
    class BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("down")

    monkeypatch.setattr(skills, "get_llm", lambda: BrokenLLM())
    monkeypatch.setattr(skills.github, "fetch_repos", lambda name: [])
    monkeypatch.setattr(skills.github, "fetch_repo_languages", lambda repos: ["Go"])
    monkeypatch.setattr(skills.github, "fetch_profile_readme", lambda name: None)
    assert skills.extract_skills_from_github("octocat") == ["Go"]


def test_invalid_github_username_is_rejected_before_any_call(monkeypatch):
    monkeypatch.setattr(skills.github.requests, "get", lambda *a, **k: pytest.fail("request"))
    with pytest.raises(skills.github.InvalidUsernameError):
        skills.extract_skills_from_github("../etc")


def test_resume_file_text_is_sent_to_the_llm(monkeypatch, fake_llm, tmp_path):
    path = tmp_path / "cv.txt"
    path.write_text("Kubernetes wizard")
    llm = fake_llm("Kubernetes")
    monkeypatch.setattr(skills, "get_llm", lambda: llm)
    assert skills.extract_skills_from_resume(path) == ["Kubernetes"]
    assert "Kubernetes wizard" in llm.prompts[0][1].content


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF whose page shows `text` in Helvetica, with a correct xref table."""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += b"%d 0 obj\n" % number + obj + b"\nendobj\n"
    xref = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    body += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    body += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return body


def test_pdf_text_is_read(tmp_path):
    path = tmp_path / "cv.pdf"
    path.write_bytes(minimal_pdf("Python and Kubernetes"))
    assert "Python and Kubernetes" in documents.read_document_text(path)


def test_pypdf_is_used_when_pdfplumber_finds_no_text(tmp_path, monkeypatch):
    path = tmp_path / "cv.pdf"
    path.write_bytes(minimal_pdf("Rust developer"))

    class BlankPdf:
        pages = [type("Page", (), {"extract_text": lambda self: ""})()]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(documents.pdfplumber, "open", lambda path: BlankPdf())
    assert "Rust developer" in documents.read_document_text(path)


def test_empty_docx_table_cells_are_skipped(tmp_path):
    path = tmp_path / "cv.docx"
    doc = DocxDocument()
    table = doc.add_table(rows=1, cols=3)
    table.cell(0, 0).text = "Go"
    table.cell(0, 2).text = "SQL"
    doc.save(path)
    assert documents.read_document_text(path) == "Go\nSQL"


def test_missing_txt_file_raises_a_document_error(tmp_path):
    with pytest.raises(documents.UnreadableDocumentError):
        documents.read_document_text(tmp_path / "missing.txt")


def test_empty_llm_reply_falls_back_to_the_skill_matcher(monkeypatch, fake_llm):
    monkeypatch.setattr(skills, "get_llm", lambda: fake_llm(""))
    assert skills.extract_skills_from_text("Strong Python skills") == ["python"]


def test_empty_llm_reply_for_github_falls_back_to_languages(monkeypatch, fake_llm):
    monkeypatch.setattr(skills, "get_llm", lambda: fake_llm(" , "))
    monkeypatch.setattr(skills.github, "fetch_repos", lambda name: [])
    monkeypatch.setattr(skills.github, "fetch_repo_languages", lambda repos: ["Go"])
    monkeypatch.setattr(skills.github, "fetch_profile_readme", lambda name: None)
    assert skills.extract_skills_from_github("octocat") == ["Go"]
