"""Pipeline tests. Review texts here are TEST FIXTURES written for the tests; they are not
scraped data and no metric is derived from them."""
import io
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src import config, html_parser as hp
from src import sentiment_model as sm
from src.pipeline import (aspect_breakdown, analyze_reviews, export_aspects, export_cleaned,
                          export_predictions, top_terms)
from src.review_preprocessing import clean_and_deduplicate_reviews, load_reviews_csv

FIX = Path(__file__).parent / "fixtures"

FIXTURE_REVIEWS = [   # TEST FIXTURE records
    {"review_id": "T1", "review_title": "Mixed feelings", "review_text":
        "The display is excellent, but the battery drains too quickly.", "rating": 3, "source": "csv_upload"},
    {"review_id": "T2", "review_text": "Terrible build quality and the keyboard feels cheap.", "rating": 1,
     "source": "csv_upload"},
    {"review_id": "T3", "review_text": "Love the sound and the price is great.", "rating": 5,
     "source": "csv_upload"},
    {"review_id": "T4", "review_text": "My cousin visited yesterday and we ate pizza.", "rating": 4,
     "source": "csv_upload"},
    {"review_text": "Battery lasts all day.", "rating": 4, "source": "csv_upload"},                  # no id
    {"review_text": "  Battery   lasts all day. ", "rating": 4, "source": "csv_upload"},   # same content -> dup
    {"review_text": "   "},
]


@pytest.fixture(scope="module")
def analyzed(trained_models):
    sent, aspects = trained_models
    clean, stats = clean_and_deduplicate_reviews(FIXTURE_REVIEWS, source="csv_upload")
    reviews, aspect_df = analyze_reviews(clean, sent, aspects)
    return clean, stats, reviews, aspect_df


@pytest.fixture(scope="module")
def trained_models():
    from src.aspect_sentiment import AspectSentimentAnalyzer
    if not sm.MODEL_FILE.exists() or not config.SELECTED_MODEL_FILE.exists():
        pytest.skip("Models not trained.")
    return sm.load_review_model(), AspectSentimentAnalyzer()


# ------------------------------------------------------------------ model loading
def test_missing_review_model_raises_clear_error(tmp_path):
    with pytest.raises(sm.ModelNotAvailable, match="not found"):
        sm.load_review_model(tmp_path / "nope.joblib")


def test_corrupt_review_model_raises_clear_error(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"this is not a joblib file")
    with pytest.raises(sm.ModelNotAvailable, match="Could not load"):
        sm.load_review_model(bad)


def test_missing_benchmark_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(sm, "BENCH_DIR", tmp_path)
    with pytest.raises(sm.ModelNotAvailable, match="not found"):
        sm.load_benchmark("test")


def test_saved_review_model_loads(trained_models):
    sent, _ = trained_models
    assert set(sent.classes) == {"negative", "neutral", "positive"}
    assert sent.meta.get("train_examples", 0) > 0


# ------------------------------------------------------------------ sentiment output structure
def test_sentiment_prediction_structure(trained_models):
    sent, _ = trained_models
    out = sent.predict(["Absolutely wonderful, works perfectly.", "Awful. It broke after one day, not worth it."])
    assert [o["label"] for o in out] == ["positive", "negative"]
    for o in out:
        assert set(o) == {"label", "score_type", "confidence", "scores"}
        assert set(o["scores"]) == {"negative", "neutral", "positive"}
        if o["score_type"] == "probability":
            assert abs(sum(o["scores"].values()) - 1) < 1e-6
        else:
            assert o["confidence"] is None


def test_sentiment_predict_empty_list(trained_models):
    assert trained_models[0].predict([]) == []


def test_negation_is_not_destroyed(trained_models):
    sent = trained_models[0]
    neg, pos = sent.predict(["This is not good at all", "This is really good"])
    assert neg["label"] != "positive" and pos["label"] == "positive"


# ------------------------------------------------------------------ aspects in the pipeline
def test_end_to_end_fixture_records(analyzed):
    clean, stats, reviews, aspects = analyzed
    assert stats == {"input_records": 7, "dropped_empty": 1, "duplicates_removed": 1, "retained": 5}
    assert len(reviews) == 5
    for col in ("review_sentiment", "aspect_summary", "aspect_overall", "final_label", "n_aspects", "rating"):
        assert col in reviews.columns
    assert list(reviews["rating"])[:3] == [3.0, 1.0, 5.0]
    assert {"review_id", "aspect", "aspect_context", "sentiment", "confidence", "score_type"} <= set(aspects.columns)
    assert set(aspects["review_id"]) <= set(reviews["review_id"])
    assert aspects["sentiment"].isin(["positive", "negative", "neutral"]).all()


def test_mixed_aspects_in_one_review(analyzed):
    _, _, reviews, aspects = analyzed
    rid = reviews.loc[reviews["review_id"] == "T1", "review_id"].iloc[0]
    got = dict(zip(aspects.loc[aspects["review_id"] == rid, "aspect"],
                   aspects.loc[aspects["review_id"] == rid, "sentiment"]))
    assert got == {"Display": "positive", "Battery": "negative"}
    row = reviews[reviews["review_id"] == rid].iloc[0]
    assert row["aspect_overall"] == "Mixed" and row["n_aspects"] == 2 and row["final_label_basis"] == "aspects"
    ctx = aspects[(aspects["review_id"] == rid) & (aspects["aspect"] == "Battery")]["aspect_context"].iloc[0]
    assert "excellent" not in ctx            # context isolates the contrasting clause


def test_multiple_aspects_in_one_review(analyzed):
    _, _, reviews, aspects = analyzed
    asp = aspects[aspects["review_id"] == "T2"]
    assert {"Build Quality", "Keyboard"} <= set(asp["aspect"])
    assert (asp["sentiment"] == "negative").all()


def test_no_aspect_falls_back_to_whole_review_classifier(analyzed):
    _, _, reviews, aspects = analyzed
    row = reviews[reviews["review_id"] == "T4"].iloc[0]
    assert row["n_aspects"] == 0 and row["aspect_overall"] == "No aspects detected"
    assert row["final_label"] == row["review_sentiment"].capitalize()
    assert "no aspect" in row["final_label_basis"]
    assert "T4" not in set(aspects["review_id"])


def test_star_rating_is_not_used_as_sentiment(trained_models):
    sent, aspects = trained_models
    df, _ = clean_and_deduplicate_reviews([
        {"review_id": "a", "review_text": "Terrible. Broke in a day, complete waste of money.", "rating": 5},
        {"review_id": "b", "review_text": "Fantastic, works perfectly and I love it.", "rating": 1}])
    out, _ = analyze_reviews(df, sent, aspects)
    assert list(out["review_sentiment"]) == ["negative", "positive"]      # opposite of the star ratings


def test_empty_input_returns_empty_frames(trained_models):
    out, asp = analyze_reviews(pd.DataFrame(), *trained_models)
    assert out.empty and asp.empty


# ------------------------------------------------------------------ exports
def test_csv_exports_roundtrip(analyzed):
    clean, _, reviews, aspects = analyzed
    c = pd.read_csv(io.BytesIO(export_cleaned(reviews)))
    assert {"review_id", "review_text", "cleaned_text", "source", "scraped_at"} <= set(c.columns)
    assert len(c) == len(reviews) and set(c["source"].dropna()) <= {"csv_upload"}
    p = pd.read_csv(io.BytesIO(export_predictions(reviews)))
    assert {"review_sentiment", "aspect_summary", "final_label", "source"} <= set(p.columns)
    a = pd.read_csv(io.BytesIO(export_aspects(aspects)))
    assert len(a) == len(aspects) and {"aspect", "sentiment", "aspect_context", "review_id"} <= set(a.columns)


def test_export_of_empty_frames_has_headers():
    assert export_aspects(pd.DataFrame()).decode().startswith("review_id")


# ------------------------------------------------------------------ insights helpers
def test_aspect_breakdown_and_top_terms(analyzed):
    _, _, reviews, aspects = analyzed
    b = aspect_breakdown(aspects)
    assert {"aspect", "negative", "neutral", "positive", "total"} == set(b.columns)
    assert b["total"].sum() == len(aspects)
    terms = top_terms(reviews, "negative", n=5, min_count=1)
    assert {"term", "reviews", "reviews_pct"} == set(terms.columns)
    assert aspect_breakdown(pd.DataFrame()).empty


# ------------------------------------------------------------------ scrape fixtures -> pipeline
def test_parsed_fixture_html_flows_through_pipeline(trained_models):
    html = (FIX / "review_page_synthetic.html").read_text(encoding="utf-8")
    assert hp.classify_page(html, 200) == hp.OK
    recs = hp.parse_review_page(html, "https://www.amazon.com/product-reviews/B000000001", "B000000001")
    for r in recs:
        r["source"] = "amazon_scrape"
    clean, stats = clean_and_deduplicate_reviews(recs)
    assert stats["duplicates_removed"] == 1
    reviews, aspects = analyze_reviews(clean, *trained_models)
    first = aspects[aspects["review_id"] == "R1AAAAAAAAAAA1"]
    assert dict(zip(first["aspect"], first["sentiment"])) == {"Display": "positive", "Battery": "negative"}
    assert set(reviews["source"]) == {"amazon_scrape"}
