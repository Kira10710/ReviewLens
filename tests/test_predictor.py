import json

import pandas as pd
import pytest

from src import config
from src.predictor import ModelNotAvailable, load_model, overall_label, read_reviews_csv

ASPECT_KEYS = {"aspect", "matched_text", "sentence", "context", "sentiment", "score_type",
               "confidence", "scores"}


# ------------------------------------------------------------------ overall label
@pytest.mark.parametrize("labels,expected", [
    ([], "No aspects detected"),
    (["positive", "neutral"], "Positive"),
    (["negative"], "Negative"),
    (["neutral", "neutral"], "Neutral"),
    (["positive", "negative", "neutral"], "Mixed"),
])
def test_overall_label(labels, expected):
    assert overall_label(labels) == expected


# ------------------------------------------------------------------ model loading
def test_saved_models_load_and_predict():
    for name, path in config.MODEL_FILES.items():
        if not path.exists():
            pytest.skip("models not trained")
        model = load_model(name)
        assert set(model.classes_) == set(config.LABELS)
        X = pd.DataFrame([{"sentence": "The screen is great.", "aspect": "screen", "start": 4, "end": 10}])
        assert model.predict(X)[0] in config.LABELS


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(ModelNotAvailable):
        load_model(path=tmp_path / "missing.joblib")


def test_metrics_file_is_real_evaluation():
    if not config.METRICS_FILE.exists():
        pytest.skip("not evaluated")
    m = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
    assert m["test_examples"] == 638 and m["train_examples"] == 2313
    for res in m["models"].values():
        cm = res["confusion_matrix"]
        assert sum(map(sum, cm)) == m["test_examples"]
        assert 0 < res["macro_f1"] <= 1


# ------------------------------------------------------------------ prediction structure
def test_prediction_output_structure(trained_predictor):
    res = trained_predictor.analyze_review("The keyboard is excellent.")
    assert set(res) == {"review", "aspects", "overall", "n_aspects", "message"}
    assert res["n_aspects"] == 1
    a = res["aspects"][0]
    assert set(a) == ASPECT_KEYS
    assert a["sentiment"] in config.LABELS
    assert set(a["scores"]) == set(config.LABELS)


def test_empty_review(trained_predictor):
    for text in ["", "   ", None]:
        res = trained_predictor.analyze_review(text)
        assert res["aspects"] == [] and res["overall"] == "No aspects detected"
        assert res["message"]


def test_unknown_aspect_message(trained_predictor):
    res = trained_predictor.analyze_review("My cousin visited yesterday and we ate pizza.")
    assert res["n_aspects"] == 0 and "No known product aspect" in res["message"]


def test_positive_and_negative_in_same_review(logreg_predictor):
    res = logreg_predictor.analyze_review(
        "The battery life is amazing, but the screen is terrible.")
    by_aspect = {a["aspect"]: a["sentiment"] for a in res["aspects"]}
    assert by_aspect == {"Battery": "positive", "Display": "negative"}
    assert res["overall"] == "Mixed"


def test_probabilities_only_for_logreg(logreg_predictor, trained_predictor):
    a = logreg_predictor.analyze_review("Great keyboard.")["aspects"][0]
    assert a["score_type"] == "probability"
    assert abs(sum(a["scores"].values()) - 1) < 1e-6 and 0 <= a["confidence"] <= 1
    b = trained_predictor.analyze_review("Great keyboard.")["aspects"][0]
    if b["score_type"] == "svm_margin":
        assert b["confidence"] is None


def test_batch_outputs(trained_predictor):
    reviews, aspects = trained_predictor.analyze_batch(
        pd.Series(["Great screen but awful battery.", "Nothing special to say."]))
    assert len(reviews) == 2 and list(reviews["n_aspects"]) == [2, 0]
    assert set(aspects["review_id"]) == {0}
    assert {"review_id", "aspect", "sentiment", "context"} <= set(aspects.columns)


def test_analyze_csv_end_to_end(trained_predictor):
    sample = (config.DATA_DIR / "samples" / "sample_reviews.csv").read_bytes()
    reviews, aspects, info = trained_predictor.analyze_csv(sample)
    assert info["rows_total"] == 10 and info["rows_missing_text"] == 1
    assert len(reviews) == 9 and reviews.columns.is_unique
    assert "input_review_id" in reviews.columns       # clashing input column kept, renamed
    assert (reviews["overall"] == "No aspects detected").sum() == 1
    assert len(aspects) == reviews["n_aspects"].sum()
    assert reviews.to_csv(index=False)                # downloadable


# ------------------------------------------------------------------ CSV validation
def test_csv_valid_with_missing_values():
    data = b"id,Review_Text\n1,Great screen\n2,\n3,  \n4,Bad keyboard\n"
    df, col, info = read_reviews_csv(data)
    assert col == "Review_Text"            # case-insensitive column match
    assert info["rows_total"] == 4 and info["rows_missing_text"] == 2 and len(df) == 2


def test_csv_missing_column():
    with pytest.raises(ValueError, match="not found"):
        read_reviews_csv(b"text\nhello\n")


def test_csv_custom_column():
    df, col, _ = read_reviews_csv(b"comment\nnice screen\n", "comment")
    assert col == "comment" and len(df) == 1


@pytest.mark.parametrize("payload,msg", [
    (b"", "empty"),
    (b"review_text\n", "no rows"),
    (b"review_text\n\n\n", "no rows|no non-empty"),
    (b'review_text,x\n"unterminated,1\n', "parse|no non-empty|not found"),
    (b"\x00\x01\x02binary", "not found|parse|no rows"),
])
def test_csv_malformed(payload, msg):
    with pytest.raises(ValueError, match=msg):
        read_reviews_csv(payload)
