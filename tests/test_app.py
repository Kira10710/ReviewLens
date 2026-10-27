"""Headless Streamlit smoke tests. Session state is seeded with TEST FIXTURE reviews."""
import pytest

from src import config
from src import sentiment_model as sm

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(config.ROOT / "app.py")
PAGES = ["Overview", "Scrape Amazon Reviews", "Review Analysis", "Aspect Insights", "Data Export",
         "Model Evaluation"]


def fresh():
    return AppTest.from_file(APP, default_timeout=90).run()


@pytest.mark.parametrize("page", PAGES)
def test_empty_state_pages_render(page):
    at = fresh()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


def test_empty_state_message_shown():
    at = fresh()
    at.sidebar.radio[0].set_value("Review Analysis").run()
    assert any("No reviews loaded" in i.value for i in at.info)


def test_invalid_url_shows_error_and_no_scrape():
    at = fresh()
    at.sidebar.radio[0].set_value("Scrape Amazon Reviews").run()
    at.text_input(key="url_input").set_value("https://example.com/dp/B000000001").run()
    assert any("not a supported Amazon storefront" in e.value for e in at.error)
    assert at.session_state["scrape_result"] is None


@pytest.fixture
def seeded():
    if not sm.MODEL_FILE.exists():
        pytest.skip("models not trained")
    from src.pipeline import analyze_reviews, load_analyzers
    from src.review_preprocessing import clean_and_deduplicate_reviews
    clean, _ = clean_and_deduplicate_reviews([   # TEST FIXTURE records
        {"review_id": "F1", "review_text": "The display is excellent, but the battery drains too quickly.", "rating": 3},
        {"review_id": "F2", "review_text": "Terrible keyboard and awful build quality.", "rating": 1},
        {"review_id": "F3", "review_text": "My cousin visited yesterday and we ate pizza."}], source="csv_upload")
    rev, asp = analyze_reviews(clean, *load_analyzers())
    at = fresh()
    at.session_state["clean_df"], at.session_state["review_df"], at.session_state["aspect_df"] = clean, rev, asp
    at.session_state["origin"] = {"kind": "csv_upload", "filename": "fixture.csv", "collected": 3,
                                  "status": "test", "loaded_at": "now"}
    return at


@pytest.mark.parametrize("page", ["Overview", "Review Analysis", "Aspect Insights", "Data Export"])
def test_pages_render_with_results(seeded, page):
    seeded.sidebar.radio[0].set_value(page).run()
    assert not seeded.exception, seeded.exception


def test_overview_shows_real_counts(seeded):
    seeded.sidebar.radio[0].set_value("Overview").run()
    values = {m.label: m.value for m in seeded.metric}
    assert values["Unique reviews retained"] == "3" and values["Aspect mentions detected"] == "4"
    assert any("User-uploaded CSV" in m.value for m in seeded.markdown)


def test_analysis_filters(seeded):
    seeded.sidebar.radio[0].set_value("Review Analysis").run()
    seeded.multiselect[2].set_value(["Battery"]).run()    # aspect filter
    assert not seeded.exception
    assert any("1 of 3 reviews" in c.value for c in seeded.caption)


# ------------------------------------------------------------------ Import Dataset mode
def _import_app(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    d = tmp_path / "Dataset"
    d.mkdir()
    shutil.copy(Path(__file__).parent / "fixtures" / "datafiniti_sample.csv", d / "datafiniti_sample.csv")
    monkeypatch.setattr(config, "DATASET_DIR", d)
    at = fresh()
    at.sidebar.radio[0].set_value("Import Dataset").run()
    return at


def test_import_page_renders_and_requires_product_choice(tmp_path, monkeypatch):
    at = _import_app(tmp_path, monkeypatch)
    assert not at.exception, at.exception
    assert any("several products" in m.value for m in at.markdown)
    assert any("Header row verified" in s.value for s in at.success)
    load_btn = [b for b in at.button if b.label == "Load and analyze"][0]
    assert load_btn.disabled                                  # nothing selected -> cannot mix products
    assert at.session_state["review_df"] is None


def test_import_flow_loads_selected_product_only(tmp_path, monkeypatch):
    if not sm.MODEL_FILE.exists():
        pytest.skip("models not trained")
    at = _import_app(tmp_path, monkeypatch)
    prod = [s for s in at.selectbox if s.label == "Product"][0]
    prod.set_value("FXP1AAAAAAAAAAAAAAAA").run()
    [b for b in at.button if b.label == "Load and analyze"][0].click().run()
    assert not at.exception, at.exception
    df = at.session_state["review_df"]
    assert len(df) == 3 and set(df["source"]) == {"dataset_import"} and set(df["product_id"]) == {"B000000001"}
    assert at.session_state["origin"]["kind"] == "dataset_import"
    at.sidebar.radio[0].set_value("Overview").run()
    assert any("Imported dataset (not live-scraped)" in m.value for m in at.markdown)
    assert not any("Live Amazon scrape" in m.value for m in at.markdown)
    assert at.session_state["scrape_result"] is None          # scraping mode untouched
