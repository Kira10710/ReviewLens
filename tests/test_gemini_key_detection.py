"""Regression tests for Gemini API-key detection and the safe diagnostic.

Background (see README): the UI reported "Gemini is unavailable" because the viewed app instance had no key in ITS OWN
environment (a variable set in another PowerShell window is invisible to it), no secrets.toml existed where Streamlit
looks, and the UI did not say which instance or source was checked. A real key never appears in these tests.
"""
import json
import subprocess
import sys

import pytest

from src import ai_insights as ai
from src import config

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(config.ROOT / "app.py")
KEY = "AIzaSyFAKEKEYFORTESTS0123456789abcdefgh"


def write_secrets(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ environment variable
def test_environment_variable_is_detected_and_reports_its_source(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    st = ai.resolve_api_key()
    assert st.found and st.key == KEY and st.source == "environment variable"
    assert ai.get_api_key() == KEY


def test_environment_is_read_at_call_time_not_import_time(monkeypatch):
    assert ai.get_api_key() is None
    monkeypatch.setenv("GEMINI_API_KEY", KEY)                     # set AFTER the module was imported
    assert ai.get_api_key() == KEY
    monkeypatch.delenv("GEMINI_API_KEY")
    assert ai.get_api_key() is None                               # and removal is noticed too (nothing cached)


@pytest.mark.parametrize("raw", [f"  {KEY}  ", f'"{KEY}"', f"'{KEY}'", f"{KEY}\r\n"])
def test_whitespace_and_surrounding_quotes_are_stripped(monkeypatch, raw):
    monkeypatch.setenv("GEMINI_API_KEY", raw)
    assert ai.get_api_key() == KEY


def test_empty_environment_variable_counts_as_missing_and_is_explained(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert ai.get_api_key() is None
    d = ai.key_diagnostics()
    assert d["details"]["env"] == {"present": True, "empty": True, "placeholder": False}
    assert any("exists but is empty" in p for p in d["problems"])


@pytest.mark.parametrize("placeholder", ["your-gemini-api-key-here", "YOUR_API_KEY", "<paste key>", "changeme"])
def test_placeholder_values_are_not_treated_as_keys(monkeypatch, placeholder):
    monkeypatch.setenv("GEMINI_API_KEY", placeholder)
    assert ai.get_api_key() is None
    assert any("placeholder" in p for p in ai.key_diagnostics()["problems"])


# ------------------------------------------------------------------ missing key
def test_missing_key_everywhere():
    assert ai.get_api_key() is None and ai.get_api_key({}) is None and ai.get_api_key(None) is None
    d = ai.key_diagnostics()
    assert d["found"] is False and d["source"] is None and d["key_length"] is None
    assert d["checked_in_order"] == ["environment variable", "Streamlit secrets", "project .streamlit/secrets.toml"]
    assert any("different PowerShell window" in p for p in d["problems"])


# ------------------------------------------------------------------ Streamlit secrets
def test_streamlit_secrets_detected_with_source():
    st = ai.resolve_api_key({"GEMINI_API_KEY": KEY})
    assert st.found and st.source == "Streamlit secrets"


def test_secrets_lookup_is_case_insensitive_and_one_section_deep():
    assert ai.get_api_key({"gemini_api_key": KEY}) == KEY
    assert ai.get_api_key({"general": {"GEMINI_API_KEY": KEY}}) == KEY          # key placed under a [section]
    assert ai.get_api_key({"other": "x", "general": {"nothing": 1}}) is None


def test_broken_secrets_object_reports_error_type_without_crashing():
    class Broken:
        def get(self, *a):
            raise RuntimeError("secrets file unreadable")

        def keys(self):
            raise RuntimeError("secrets file unreadable")

    assert ai.get_api_key(Broken()) is None
    d = ai.key_diagnostics(Broken())
    assert d["details"]["streamlit_secrets"]["error"] == "RuntimeError"
    assert any("TOML syntax" in p for p in d["problems"])


def test_placeholder_in_secrets_is_flagged():
    assert ai.get_api_key({"GEMINI_API_KEY": "your-gemini-api-key-here"}) is None
    assert any("secrets.toml.example" in p for p in ai.key_diagnostics({"GEMINI_API_KEY": "your-gemini-api-key-here"})["problems"])


# ------------------------------------------------------------------ project secrets file (launch folder independent)
def test_project_secrets_file_is_used_even_when_streamlit_secrets_are_empty(monkeypatch, tmp_path):
    f = write_secrets(tmp_path / "proj" / ".streamlit" / "secrets.toml", f'GEMINI_API_KEY = "{KEY}"\n')
    monkeypatch.setattr(ai, "PROJECT_SECRETS", f)
    monkeypatch.chdir(tmp_path)                                  # app launched from another folder: Streamlit's own
    st = ai.resolve_api_key({})                                  # secrets are empty, the project file still works
    assert st.found and st.source == "project .streamlit/secrets.toml"


def test_malformed_project_secrets_is_reported_not_fatal(monkeypatch, tmp_path):
    f = write_secrets(tmp_path / "secrets.toml", 'GEMINI_API_KEY = "unterminated\n')
    monkeypatch.setattr(ai, "PROJECT_SECRETS", f)
    assert ai.get_api_key() is None
    d = ai.key_diagnostics()
    assert d["details"]["project_secrets_file"]["exists"] and d["details"]["project_secrets_file"]["error"]
    assert any("could not be parsed" in p for p in d["problems"])


# ------------------------------------------------------------------ precedence
def test_precedence_environment_then_streamlit_secrets_then_project_file(monkeypatch, tmp_path):
    proj = write_secrets(tmp_path / "secrets.toml", 'GEMINI_API_KEY = "project-file-key"\n')
    monkeypatch.setattr(ai, "PROJECT_SECRETS", proj)
    assert ai.resolve_api_key({}).source == "project .streamlit/secrets.toml"
    assert ai.get_api_key({}) == "project-file-key"
    assert ai.resolve_api_key({"GEMINI_API_KEY": "secrets-key"}).source == "Streamlit secrets"
    assert ai.get_api_key({"GEMINI_API_KEY": "secrets-key"}) == "secrets-key"
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert ai.resolve_api_key({"GEMINI_API_KEY": "secrets-key"}).source == "environment variable"
    assert ai.get_api_key({"GEMINI_API_KEY": "secrets-key"}) == "env-key"
    monkeypatch.setenv("GEMINI_API_KEY", "your-gemini-api-key-here")           # a placeholder does not shadow a real key
    assert ai.get_api_key({"GEMINI_API_KEY": "secrets-key"}) == "secrets-key"


# ------------------------------------------------------------------ the key never leaks
def test_diagnostics_and_status_never_contain_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    st = ai.resolve_api_key({"GEMINI_API_KEY": KEY})
    d = ai.key_diagnostics({"GEMINI_API_KEY": KEY}, port=8501)
    assert KEY not in repr(st) and KEY not in str(st) and KEY not in json.dumps(d)
    assert d["found"] and d["key_length"] == len(KEY) and d["process"]["port"] == 8501


def test_command_line_check_reports_source_without_printing_the_key():
    env = {"GEMINI_API_KEY": KEY, "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    out = subprocess.run([sys.executable, "-m", "src.ai_insights", "--check"], cwd=str(config.ROOT), env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and KEY not in out.stdout + out.stderr
    report = json.loads(out.stdout.split("\n\nNote:")[0])
    assert report["found"] is True and report["source"] == "environment variable"


# ------------------------------------------------------------------ the real app (st.secrets and environment)
def sidebar_text(at):
    return " ".join(m.value for m in at.sidebar.markdown)


def test_app_without_any_key_explains_what_was_checked():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, at.exception
    assert "Gemini: off (optional)" in sidebar_text(at)
    at.sidebar.radio[0].set_value("Aspect Insights").run()
    assert not at.exception


def test_app_detects_key_from_real_streamlit_secrets_and_never_shows_it():
    at = AppTest.from_file(APP, default_timeout=90)
    at.secrets["GEMINI_API_KEY"] = KEY
    at.run()
    assert not at.exception, at.exception
    assert "Gemini: configured (Streamlit secrets)" in sidebar_text(at) and KEY not in sidebar_text(at)


def test_app_detects_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert "Gemini: configured (environment variable)" in sidebar_text(at) and KEY not in sidebar_text(at)


def test_app_environment_wins_over_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    at = AppTest.from_file(APP, default_timeout=90)
    at.secrets["GEMINI_API_KEY"] = "other-secrets-key-value"
    at.run()
    assert "configured (environment variable)" in sidebar_text(at)


def test_unavailable_message_has_a_check_panel_that_hides_the_key():
    from src import sentiment_model as sm
    if not sm.MODEL_FILE.exists():
        pytest.skip("models not trained")
    from src.pipeline import analyze_reviews, load_analyzers
    from src.review_preprocessing import clean_and_deduplicate_reviews
    clean, _ = clean_and_deduplicate_reviews([{"review_id": "A", "review_text": "The display is excellent.", "rating": 5}])
    rev, asp = analyze_reviews(clean, *load_analyzers())
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.session_state["clean_df"], at.session_state["review_df"], at.session_state["aspect_df"] = clean, rev, asp
    at.session_state["origin"] = {"kind": "dataset_import", "filename": "f.csv", "n_products": 1}
    at.sidebar.radio[0].set_value("Aspect Insights").run()
    assert not at.exception, at.exception
    assert any("Gemini is unavailable" in w.value and "configuration check" in w.value for w in at.warning)
    assert any("Gemini configuration check" in e.label for e in at.expander)
    assert any("Key detected:** no" in m.value for m in at.markdown)


# ------------------------------------------------------------------ launch modes (verified by hand against real instances)
import os
import textwrap
from datetime import datetime


def _run_python(code, env_extra, cwd):
    env = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PATH": os.environ.get("PATH", ""), **env_extra}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=60)


DETECT = f"""
import sys; sys.path.insert(0, r"{config.ROOT}")
from src import ai_insights as a
s = a.resolve_api_key()
print(s.found, s.source)
"""


@pytest.mark.parametrize("value", [KEY, f'  "{KEY}"  ', f"'{KEY}'"])
def test_real_process_detects_variable_when_launched_from_another_folder(value):
    out = _run_python(DETECT, {"GEMINI_API_KEY": value}, cwd=config.ROOT.parent)       # e.g. run from C:\Prakrut\NLP
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["True", "environment", "variable"] and KEY not in out.stdout + out.stderr


@pytest.mark.skipif(os.name != "nt", reason="Windows environment variable names are case-insensitive")
def test_lowercase_variable_name_is_found_on_windows():
    out = _run_python(DETECT, {"gemini_api_key": KEY}, cwd=config.ROOT)
    assert out.stdout.split()[:1] == ["True"]


def test_process_without_the_variable_reports_not_found():
    out = _run_python(DETECT, {}, cwd=config.ROOT)
    assert out.stdout.split() == ["False", "None"]


def test_app_detects_environment_variable_when_started_from_a_different_working_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.chdir(tmp_path)                                  # Streamlit's own ./.streamlit lookup finds nothing here
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, at.exception
    assert "Gemini: configured (environment variable)" in sidebar_text(at) and KEY not in sidebar_text(at)


def test_the_ui_uses_one_detection_path_for_sidebar_and_ai_section(monkeypatch):
    """Sidebar status and the AI Insights section both derive from the same ai_insights.resolve_api_key call."""
    import re
    src = (config.ROOT / "app.py").read_text(encoding="utf-8")
    assert not re.search(r"os\.(environ|getenv)", src)           # app.py never reads the environment itself
    assert src.count("ai_insights.get_api_key") == 1 and src.count("ai_insights.key_diagnostics") == 1


# ------------------------------------------------------------------ extra diagnostics
def test_process_start_time_is_a_real_past_timestamp():
    t = ai.process_start_time()
    assert t is not None
    started = datetime.fromisoformat(t)
    assert started <= datetime.now(started.tzinfo) and (datetime.now(started.tzinfo) - started).total_seconds() < 24 * 3600


def test_similar_variable_names_are_reported_by_name_only(monkeypatch):
    monkeypatch.setenv("GEMINI_KEY", "SECRET-VALUE-123")
    monkeypatch.setenv("GOOGLE_API_KEY", "ANOTHER-SECRET-456")
    d = ai.key_diagnostics()
    assert {"GEMINI_KEY", "GOOGLE_API_KEY"} <= set(d["similar_env_names"])
    assert any("similar names" in p and "GEMINI_KEY" in p for p in d["problems"])
    blob = json.dumps(d)
    assert "SECRET-VALUE-123" not in blob and "ANOTHER-SECRET-456" not in blob      # names only, never values


def test_not_found_message_includes_when_the_process_started():
    d = ai.key_diagnostics()
    assert d["process"]["started"] and any("started at" in p for p in d["problems"])


def test_similar_names_not_flagged_when_the_real_variable_exists(monkeypatch):
    monkeypatch.setenv("GEMINI_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    d = ai.key_diagnostics()
    assert d["found"] and not any("similar names" in p for p in d["problems"])
