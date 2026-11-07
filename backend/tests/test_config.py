import pytest
from pydantic import ValidationError

from app.core import checkpointer, config, resources
from app.core.config import Settings


def settings(**values):
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize(
    ("url", "normalized"),
    [
        ("postgres://u:p@h/db", "postgresql://u:p@h/db"),
        ("postgresql://u:p@h/db", "postgresql://u:p@h/db"),
    ],
)
def test_database_url_scheme_is_normalized(url, normalized, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", url)
    assert settings().database_url == normalized


@pytest.mark.parametrize(
    ("url", "sqlalchemy_url"),
    [
        ("postgres://u:p%40ss@h/db", "postgresql+psycopg://u:p%40ss@h/db"),
        ("postgresql://h/db", "postgresql+psycopg://h/db"),
        ("postgresql+psycopg://h/db", "postgresql+psycopg://h/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_sqlalchemy_url_uses_psycopg_3(url, sqlalchemy_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", url)
    assert settings().sqlalchemy_url == sqlalchemy_url


def test_db_url_is_accepted_as_an_alias(monkeypatch):
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setenv("DB_URL", "postgresql://alias/db")
    assert settings().database_url == "postgresql://alias/db"


def test_missing_required_settings_fail_fast(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(ValidationError, match="groq_api_key"):
        settings()


@pytest.mark.parametrize(
    ("raw", "origins"),
    [
        ("http://a,http://b", ["http://a", "http://b"]),
        (" http://a , ,http://b ", ["http://a", "http://b"]),
        ('["http://a", "http://b"]', ["http://a", "http://b"]),
        ("http://only", ["http://only"]),
    ],
)
def test_cors_origins_accept_csv_or_json(raw, origins, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", raw)
    assert settings().cors_origins == origins


def test_cors_origins_default_to_local_frontend():
    assert settings().cors_origins == ["http://localhost:3000"]


def test_empty_optional_keys_count_as_unset(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "")
    assert settings().cohere_api_key is None


def test_data_paths_live_under_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    s = settings()
    assert {
        s.market_chroma_dir.parent,
        s.courses_chroma_dir.parent,
        s.interview_chroma_dir.parent,
        s.output_dir.parent,
    } == {tmp_path}


def test_secrets_are_not_shown_in_repr():
    assert "test-groq-key" not in repr(settings())


def test_checkpointer_pool_is_closed_when_setup_fails(monkeypatch):
    closed = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed.append(True)

    class Saver:
        def __init__(self, pool):
            pass

        def setup(self):
            raise ConnectionError("database down")

    monkeypatch.setattr(checkpointer, "ConnectionPool", Pool)
    monkeypatch.setattr(checkpointer, "PostgresSaver", Saver)
    with pytest.raises(ConnectionError):
        checkpointer.get_checkpointer()
    assert closed == [True]


def test_database_password_is_not_shown_in_repr(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:hunter2@h/db")
    assert "hunter2" not in repr(settings())


def test_each_store_has_its_own_directory():
    s = settings()
    directories = [
        s.market_chroma_dir,
        s.courses_chroma_dir,
        s.interview_chroma_dir,
        s.output_dir,
    ]
    assert len(set(directories)) == len(directories)


def test_embedding_model_lives_under_data_dir_by_default(monkeypatch, tmp_path):
    """The folder name is spelled out: renaming it would orphan models already downloaded."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert settings().embedding_model_dir == tmp_path / "models" / "embeddinggemma-300m-onnx"


def test_embedding_model_dir_can_point_outside_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_MODEL_DIR", str(tmp_path / "baked-model"))
    assert settings().embedding_model_dir == tmp_path / "baked-model"


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "spaces"])
def test_a_blank_embedding_model_dir_falls_back_to_the_default(monkeypatch, tmp_path, blank):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_MODEL_DIR", blank)
    assert settings().embedding_model_dir == tmp_path / config.EMBEDDING_MODEL_SUBDIR


def test_a_padded_embedding_model_dir_is_trimmed(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_MODEL_DIR", f"  {tmp_path / 'baked-model'}  ")
    assert settings().embedding_model_dir == tmp_path / "baked-model"


def test_the_default_models_are_the_ones_the_evals_chose():
    chosen = settings()
    assert (chosen.llm_model, chosen.llm_reasoning_effort) == (
        "openai/gpt-oss-120b",
        "medium",
    )
    assert (chosen.resume_llm_model, chosen.resume_llm_reasoning_effort) == (
        "openai/gpt-oss-20b",
        "low",
    )


@pytest.mark.parametrize("raw", ["none", "None", " NONE "])
def test_none_sends_no_reasoning_effort(raw, monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", raw)
    monkeypatch.setenv("RESUME_LLM_REASONING_EFFORT", raw)
    chosen = settings()
    assert (chosen.llm_reasoning_effort, chosen.resume_llm_reasoning_effort) == (
        None,
        None,
    )


@pytest.mark.parametrize("raw", ["", "   "])
@pytest.mark.parametrize(
    ("field", "default"),
    [("llm_reasoning_effort", "medium"), ("resume_llm_reasoning_effort", "low")],
)
def test_an_empty_or_blank_reasoning_effort_keeps_the_default(raw, field, default, monkeypatch):
    monkeypatch.setenv(field.upper(), raw)
    assert getattr(settings(), field) == default


@pytest.mark.parametrize("raw", ["", "  "])
def test_an_empty_or_blank_model_keeps_the_default(raw, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", raw)
    monkeypatch.setenv("RESUME_LLM_MODEL", raw)
    chosen = settings()
    assert (chosen.llm_model, chosen.resume_llm_model) == (
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    )


def test_a_padded_model_name_is_trimmed(monkeypatch):
    monkeypatch.setenv("RESUME_LLM_MODEL", " openai/gpt-oss-20b \n")
    assert settings().resume_llm_model == "openai/gpt-oss-20b"


@pytest.mark.parametrize("raw", ["maximum", "minimal", "off", "0"])
@pytest.mark.parametrize("field", ["llm_reasoning_effort", "resume_llm_reasoning_effort"])
def test_an_unknown_reasoning_effort_fails_fast_and_names_the_setting(raw, field, monkeypatch):
    monkeypatch.setenv(field.upper(), raw)
    with pytest.raises(ValidationError) as error:
        settings()
    assert [e["loc"] for e in error.value.errors()] == [(field,)]


@pytest.mark.parametrize(
    ("raw", "effort"), [("High", "high"), (" low", "low"), ("MEDIUM", "medium")]
)
def test_reasoning_effort_ignores_case_and_spaces(raw, effort, monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", raw)
    assert settings().llm_reasoning_effort == effort


def test_the_general_llm_uses_the_configured_model_and_effort(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some/model")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    llm = resources.get_llm()
    assert (llm.model_name, llm.reasoning_effort) == ("some/model", "high")


def test_the_resume_llm_uses_its_own_model_effort_and_output_budget(monkeypatch):
    monkeypatch.setenv("RESUME_LLM_MODEL", "other/model")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("RESUME_LLM_REASONING_EFFORT", "high")
    llm = resources.get_resume_llm()
    assert (llm.model_name, llm.reasoning_effort) == ("other/model", "high")
    assert resources.get_llm().reasoning_effort == "low"
    assert llm.max_tokens == resources.RESUME_MAX_TOKENS


def test_none_reaches_both_llms_as_no_effort(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    monkeypatch.setenv("RESUME_LLM_REASONING_EFFORT", "none")
    assert resources.get_llm().reasoning_effort is None
    assert resources.get_resume_llm().reasoning_effort is None


def test_both_llms_get_the_key_from_settings_not_the_process_environment(monkeypatch):
    loaded = config.get_settings()
    monkeypatch.delenv("GROQ_API_KEY")  # as when the key is only in a dotenv file
    monkeypatch.setattr(resources, "get_settings", lambda: loaded)
    for factory in (resources.get_llm, resources.get_resume_llm):
        assert factory().groq_api_key.get_secret_value() == "test-groq-key"


def test_the_general_llm_is_deterministic_and_uncapped_and_the_resume_llm_is_not():
    general, resume = resources.get_llm(), resources.get_resume_llm()
    assert general.temperature < 1e-6  # langchain-groq stores 0 as 1e-08
    assert general.max_tokens is None
    assert resume.temperature == pytest.approx(0.1)
