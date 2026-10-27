"""Streamlit tests for the AI Insights section. Gemini is mocked via the SDK client factory; no real API calls."""
import json
import re

import httpx
import pytest
from google.genai import errors as genai_errors

from src import ai_insights as ai
from src import config
from src import sentiment_model as sm
from src.pipeline import export_aspects, export_cleaned, export_predictions

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(config.ROOT / "app.py")
KEY = "AIzaSyFAKEKEYFORTESTS0123456789abcdefgh"


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeClient:
    def __init__(self, raises=None):
        self.calls, self.raises, self.models = [], raises, self

    def generate_content(self, model, contents, config):
        self.calls.append(contents)
        if self.raises:
            raise self.raises
        payload = json.loads(re.search(r"<review_data>\n(.*)\n</review_data>", contents, re.S).group(1))
        ids = [e["id"] for e in payload["excerpts"]]
        return FakeResponse(json.dumps({
            "consensus": "Customers praise the display but find battery life disappointing.",
            "aspects": [{"aspect": "Display", "praised": "Clear screen.", "criticized": "", "support": "limited",
                         "evidence_ids": ids[:1]}],
            "pros": [{"point": "Display quality", "evidence_ids": ids[:1]}],
            "cons": [{"point": "Battery drains fast", "evidence_ids": ids[:1]}],
            "recommendation": {"suited_for": "Screen-focused buyers", "consider_alternatives_if": "Battery matters most"},
            "confidence": {"level": "low", "rationale": "Very few reviews.", "conflicting_opinions": []}}))


@pytest.fixture
def seeded(monkeypatch):
    if not sm.MODEL_FILE.exists():
        pytest.skip("models not trained")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from src.pipeline import analyze_reviews, load_analyzers
    from src.review_preprocessing import clean_and_deduplicate_reviews
    clean, _ = clean_and_deduplicate_reviews([    # TEST FIXTURE records
        {"review_id": "F1", "review_text": "The display is excellent, but the battery drains too quickly.", "rating": 3,
         "product_id": "B000000001"},
        {"review_id": "F2", "review_text": "Terrible keyboard and awful build quality.", "rating": 1, "product_id": "B000000001"},
        {"review_id": "F3", "review_text": "Love the sound and the price is great.", "rating": 5, "product_id": "B000000001"},
        {"review_id": "F4", "review_text": "Nothing special to say about this one.", "rating": 4, "product_id": "B000000001"}],
        source="dataset_import")
    rev, asp = analyze_reviews(clean, *load_analyzers())
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.session_state["clean_df"], at.session_state["review_df"], at.session_state["aspect_df"] = clean, rev, asp
    at.session_state["origin"] = {"kind": "dataset_import", "filename": "fixture.csv", "product_title": "Fixture Product",
                                  "product_id": "B000000001", "n_products": 1, "collected": 4, "status": "test",
                                  "loaded_at": "now"}
    return at


def to_insights(at):
    at.sidebar.radio[0].set_value("Aspect Insights").run()
    assert not at.exception, at.exception
    return at


def button(at, prefix):
    return [b for b in at.button if b.label.startswith(prefix)][0]


# ------------------------------------------------------------------ without a key
def test_without_key_the_section_explains_and_everything_else_works(seeded):
    at = to_insights(seeded)
    assert any("Gemini is unavailable" in w.value for w in at.warning)
    assert button(at, "Generate AI Insights").disabled
    assert any("Gemini: off (optional)" in m.value for m in at.sidebar.markdown)
    assert any(m.label == "Excerpts sent to Gemini" and m.value == "4" for m in at.metric)
    assert any(s.value for s in at.subheader)                                    # existing charts/sections still render
    df, asp = at.session_state["review_df"], at.session_state["aspect_df"]
    assert export_cleaned(df) and export_predictions(df) and export_aspects(asp)  # exports do not depend on Gemini
    for page in ("Overview", "Review Analysis", "Data Export", "Model Evaluation"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, at.exception)


def test_disclosure_and_limits_are_visible(seeded):
    at = to_insights(seeded)
    text = " ".join(c.value for c in at.caption)
    assert "sent to Google" in text and "Limits in force" in text and "no usernames" in text


# ------------------------------------------------------------------ with a (fake) key
def test_generate_shows_labelled_ai_output_and_caches_across_reruns(seeded, monkeypatch):
    client = FakeClient()
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(ai, "_default_client_factory", lambda key, timeout: client)
    at = to_insights(seeded)
    assert any("Gemini: configured" in m.value and KEY not in m.value for m in at.sidebar.markdown)
    button(at, "Generate AI Insights").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "AI-generated by Gemini" in html and "4 reviews analyzed" in html and "4 excerpts supplied" in html
    assert "Overall customer consensus" in html and "Aspect-wise findings" in html and "Buying recommendation" in html
    assert "not live-scraped" in html and KEY not in html and len(client.calls) == 1
    # a rerun with no click must not call the API again
    at.sidebar.radio[0].set_value("Data Export").run()
    to_insights(at)
    assert len(client.calls) == 1 and any("AI-generated by Gemini" in m.value for m in at.markdown)
    # the explicit regenerate button forces a new call
    button(at, "Regenerate AI Insights").click().run()
    assert len(client.calls) == 2


def test_prompt_sent_to_gemini_comes_from_the_loaded_reviews(seeded, monkeypatch):
    client = FakeClient()
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(ai, "_default_client_factory", lambda key, timeout: client)
    at = to_insights(seeded)
    button(at, "Generate AI Insights").click().run()
    payload = json.loads(re.search(r"<review_data>\n(.*)\n</review_data>", client.calls[0], re.S).group(1))
    assert payload["product"] == "Fixture Product" and "not live-scraped" in payload["source"]
    assert payload["statistics"]["reviews_analyzed"] == 4
    assert {e["text"] for e in payload["excerpts"]} >= {"The display is excellent, but the battery drains too quickly."}


@pytest.mark.parametrize("exc,needle", [
    (genai_errors.APIError(429, {"error": {"message": "Quota", "status": "RESOURCE_EXHAUSTED"}}), "quota or rate limit"),
    (httpx.ConnectError("dns failed"), "network error"),
    (httpx.ReadTimeout("slow"), "did not respond in time"),
])
def test_api_failure_is_shown_helpfully_and_app_keeps_working(seeded, monkeypatch, exc, needle):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(ai, "_default_client_factory", lambda key, timeout: FakeClient(raises=exc))
    at = to_insights(seeded)
    button(at, "Generate AI Insights").click().run()
    assert not at.exception, at.exception
    assert any(needle in w.value for w in at.warning)
    assert at.session_state["ai_cache"] == {} and KEY not in " ".join(w.value for w in at.warning)
    at.sidebar.radio[0].set_value("Review Analysis").run()                       # rest of the app unaffected
    assert not at.exception


def test_changing_the_reviews_clears_the_old_summary(seeded, monkeypatch):
    client = FakeClient()
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(ai, "_default_client_factory", lambda key, timeout: client)
    at = to_insights(seeded)
    button(at, "Generate AI Insights").click().run()
    assert any("AI-generated by Gemini" in m.value for m in at.markdown)
    df, asp = at.session_state["review_df"], at.session_state["aspect_df"]
    at.session_state["review_df"] = df.iloc[:3].copy()
    at.session_state["aspect_df"] = asp[asp["review_id"].isin(df.iloc[:3]["review_id"])]
    at.run()
    assert not any("AI-generated by Gemini" in m.value for m in at.markdown)
    assert any("changed since the last summary" in i.value for i in at.info)
    assert len(client.calls) == 1                                                # no automatic call on change


def test_mixed_products_are_refused(seeded, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(ai, "_default_client_factory", lambda key, timeout: FakeClient())
    at = to_insights(seeded)
    df = at.session_state["review_df"].copy()
    df.loc[df.index[0], "product_id"] = "B000000002"
    at.session_state["review_df"] = df
    at.run()
    assert any("different products" in w.value for w in at.warning) and button(at, "Generate AI Insights").disabled


def test_ai_section_cannot_run_without_loaded_reviews():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Aspect Insights").run()
    assert not at.exception and any("No reviews loaded" in i.value for i in at.info)
