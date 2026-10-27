"""The analysis limit chosen on the Import Dataset page must be exactly what reaches the pipeline.

Review texts below are generated for the tests (not real customer reviews).
"""
import csv

import pytest

from src import config
from src import pipeline as pipeline_module
from src import sentiment_model as sm
from src.dataset_loader import (LIMIT_CHOICES, limit_label, load_dataset_reviews, loader_limit, planned_count,
                                scan_products)
from src.review_preprocessing import MAX_UPLOAD_ROWS

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(config.ROOT / "app.py")
KEY_A, KEY_B = "PRODA", "PRODB"
N_A, N_B = 150, 30


def build_csv(path):
    """Product A: 150 unique reviews, with an exact duplicate row after every 10th review and a few empty-text
    rows, all inside the first 100 reviews. Product B: 30 unique reviews."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "asins", "reviews.text", "reviews.rating", "reviews.date", "reviews.id"])
        for i in range(N_A):
            row = [KEY_A, "Fixture Product A", "B00000000A", f"Fixture review {i}: the battery and screen are fine.",
                   str(1 + i % 5), "2017-01-13T00:00:00.000Z", ""]
            w.writerow(row)
            if i % 10 == 9:
                w.writerow(row)                                                   # exact duplicate row
            if i % 25 == 0:
                w.writerow([KEY_A, "Fixture Product A", "B00000000A", "", "3", "", ""])   # empty text
        for i in range(N_B):
            w.writerow([KEY_B, "Fixture Product B", "B00000000B", f"Fixture B review {i}: price is good.", "4",
                        "2017-02-01T00:00:00Z", ""])
    return path


@pytest.fixture
def csv_path(tmp_path):
    return build_csv(tmp_path / "many.csv")


# ------------------------------------------------------------------ loader
def test_scan_reports_exact_available_unique_reviews(csv_path):
    products, info = scan_products(csv_path)
    by = products.set_index("product_key")
    assert by.loc[KEY_A, "unique_reviews"] == N_A and by.loc[KEY_B, "unique_reviews"] == N_B
    assert by.loc[KEY_A, "reviews_with_text"] == N_A + 15          # includes the 15 duplicate rows
    assert products.iloc[0]["product_key"] == KEY_A                   # sorted by unique reviews
    df, linfo = load_dataset_reviews(csv_path, [KEY_A], max_rows=10 ** 6)
    assert len(df) == by.loc[KEY_A, "unique_reviews"] and linfo["truncated"] is False   # scan == load


@pytest.mark.parametrize("limit", [20, 100, 500, 1000, 2000])
def test_loader_returns_exactly_the_limit_when_enough_exist(csv_path, limit):
    df, info = load_dataset_reviews(csv_path, [KEY_A], max_rows=limit)
    assert len(df) == min(limit, N_A) and info["retained"] == len(df)
    assert info["truncated"] is (limit < N_A) and info["requested_limit"] == limit


def test_duplicates_do_not_consume_the_limit(csv_path):
    """Regression: the old loader counted raw rows, so 100 requested gave ~90 after de-duplication."""
    df, info = load_dataset_reviews(csv_path, [KEY_A], max_rows=100)
    assert len(df) == 100 and df["review_id"].is_unique and df["cleaned_text"].is_unique
    assert info["duplicates_removed"] >= 9 and info["dropped_empty_text"] >= 3


def test_limit_above_available_returns_all_available(csv_path):
    df, info = load_dataset_reviews(csv_path, [KEY_B], max_rows=100)
    assert len(df) == N_B and info["truncated"] is False


def test_limit_helpers():
    assert LIMIT_CHOICES == (20, 100, 500, 1000, 2000, "max")
    assert [loader_limit(c) for c in LIMIT_CHOICES] == [20, 100, 500, 1000, 2000, MAX_UPLOAD_ROWS]
    assert planned_count(100, 150) == 100 and planned_count(100, 30) == 30 and planned_count("max", 150) == 150
    assert planned_count("max", 10 ** 6) == MAX_UPLOAD_ROWS and planned_count(2000, 10 ** 6) == 2000
    assert planned_count(100, 0) == 0
    assert "up to" in limit_label("max") and limit_label(1000) == "1,000"


# ------------------------------------------------------------------ UI -> pipeline
@pytest.fixture
def import_app(tmp_path, monkeypatch):
    if not sm.MODEL_FILE.exists():
        pytest.skip("models not trained")
    d = tmp_path / "Dataset"
    d.mkdir()
    build_csv(d / "many.csv")
    monkeypatch.setattr(config, "DATASET_DIR", d)
    seen = []
    real = pipeline_module.analyze_reviews

    def spy(reviews, *a, **k):
        seen.append(len(reviews))
        return real(reviews, *a, **k)

    monkeypatch.setattr(pipeline_module, "analyze_reviews", spy)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Import Dataset").run()
    [s for s in at.selectbox if s.label == "Product"][0].set_value(KEY_A).run()
    return at, seen


def metric_values(at):
    return {m.label: m.value for m in at.metric}


def click_load(at):
    [b for b in at.button if b.label == "Load and analyze"][0].click().run()
    assert not at.exception, at.exception


def test_selecting_100_sends_100_reviews_to_the_pipeline(import_app):
    at, seen = import_app
    at.radio(key="analysis_limit").set_value(100).run()
    vals = metric_values(at)
    assert vals["Available (unique valid reviews)"] == str(N_A) and vals["Will be analyzed"] == "100"
    click_load(at)
    assert seen == [100]                                       # exactly 100 reached analyze_reviews
    assert len(at.session_state["review_df"]) == 100 and len(at.session_state["clean_df"]) == 100
    assert any(f"Analyzed 100 of {N_A} available" in s.value for s in at.success)
    at.sidebar.radio[0].set_value("Review Analysis").run()
    assert any("100 of 100 reviews" in c.value for c in at.caption)


@pytest.mark.parametrize("choice,expected", [(20, 20), (500, N_A), ("max", N_A)])
def test_other_limits_reach_the_pipeline(import_app, choice, expected):
    at, seen = import_app
    at.radio(key="analysis_limit").set_value(choice).run()
    assert metric_values(at)["Will be analyzed"] == str(expected)
    click_load(at)
    assert seen == [expected] and len(at.session_state["review_df"]) == expected


def test_limit_above_available_shows_actual_count(import_app):
    at, seen = import_app
    [s for s in at.selectbox if s.label == "Product"][0].set_value(KEY_B).run()
    at.radio(key="analysis_limit").set_value(100).run()
    vals = metric_values(at)
    assert vals["Available (unique valid reviews)"] == str(N_B) and vals["Will be analyzed"] == str(N_B)
    click_load(at)
    assert seen == [N_B]


def test_nothing_is_analyzed_before_a_product_is_chosen(tmp_path, monkeypatch):
    d = tmp_path / "Dataset"
    d.mkdir()
    build_csv(d / "many.csv")
    monkeypatch.setattr(config, "DATASET_DIR", d)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Import Dataset").run()
    assert [b for b in at.button if b.label == "Load and analyze"][0].disabled
    assert metric_values(at)["Will be analyzed"] == "-"


def test_export_preview_is_labelled_and_downloads_hold_everything(import_app):
    at, seen = import_app
    at.radio(key="analysis_limit").set_value(100).run()
    click_load(at)
    at.sidebar.radio[0].set_value("Data Export").run()
    assert any("Preview: first 20 of 100 analyzed reviews" in c.value for c in at.caption)
    from src.pipeline import export_cleaned
    assert export_cleaned(at.session_state["review_df"]).decode().count("\n") >= 100


# ------------------------------------------------------------------ real files
@pytest.mark.real
def test_real_file_limit_100_gives_100():
    path = config.DATASET_DIR / "1429_1.csv"
    if not path.exists():
        pytest.skip("dataset file not present")
    products, _ = scan_products(path)
    top = products.iloc[0]
    assert top["unique_reviews"] >= 100
    df, info = load_dataset_reviews(path, [top["product_key"]], max_rows=100)
    assert len(df) == 100 and info["truncated"] is True


@pytest.mark.real
def test_real_file_old_shortfall_is_gone():
    """May19's top product has 8,343 rows with many duplicates; a 5,000 limit used to return 4,265."""
    path = config.DATASET_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
    if not path.exists():
        pytest.skip("dataset file not present")
    products, _ = scan_products(path)
    top = products.iloc[0]
    df, info = load_dataset_reviews(path, [top["product_key"]], max_rows=5000)
    assert len(df) == min(5000, int(top["unique_reviews"])) == 5000
