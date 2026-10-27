"""Regression tests for: "Required column 'review_text' not found. Columns present: ['AVqVGZNv...', '2017-03-03T16:...'".

Root cause (reproduced): a Datafiniti CSV *without a header line* was sent to the generic uploader, which treats
row 1 as the header, so data values were reported as column names. Dataset-style and header-less files are now routed
through src/dataset_loader.py and never reach the generic ``review_text`` loader.
"""
import csv
import io
import itertools
from pathlib import Path

import pytest

from src import config
from src import csv_diagnostics
from src.dataset_loader import (DATAFINITI_LAYOUTS, DatasetSchemaError, HeaderlessDatasetError, classify_uploaded_csv,
                                infer_headerless_layout, inspect_csv_schema, load_dataset_reviews, scan_products)
from src.predictor import read_reviews_csv
from src.review_preprocessing import columns_look_like_data, describe_columns, load_reviews_csv

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(config.ROOT / "app.py")
FIX = Path(__file__).parent / "fixtures"
SAMPLE, HEADERLESS = FIX / "datafiniti_sample.csv", FIX / "datafiniti_headerless.csv"
REAL = config.DATASET_DIR
REAL_FILES = {
    "Datafiniti 24-column (2017-18 export)": REAL / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
    "Datafiniti 24-column (May 2019 export)": REAL / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv",
    "Datafiniti 21-column (1429_1 export)": REAL / "1429_1.csv",
}
csv.field_size_limit(2 ** 31 - 1)


def real_slice(path, n=5, header=False) -> bytes:
    """First ``n`` data rows of a real dataset file, optionally with the header line removed."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        head = next(r)
        rows = list(itertools.islice(r, n))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if header:
        w.writerow(head)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ------------------------------------------------------------------ the exact failing input
@pytest.mark.real
def test_reproduced_failure_headerless_kindle_data_is_now_a_clear_message():
    path = REAL_FILES["Datafiniti 24-column (2017-18 export)"]
    if not path.exists():
        pytest.skip("dataset file not present")
    data = real_slice(path, 5, header=False)
    assert data.startswith(b"AVqVGZNvQMlgsOJE6eUY,2017-03-03T16:56:05Z")        # the first cells quoted in the report
    with pytest.raises(ValueError) as ei:
        load_reviews_csv(data)                                                   # the call the app's uploader makes
    msg = str(ei.value)
    assert "NO header row" in msg and "Import Dataset" in msg
    assert "bbystatic" not in msg and len(msg) < 700                              # no longer dumps image URLs
    with pytest.raises(ValueError, match="NO header row"):
        read_reviews_csv(data)                                                   # legacy reader too


def test_headerless_fixture_never_reaches_generic_loader_message():
    with pytest.raises(ValueError, match="NO header row"):
        load_reviews_csv(HEADERLESS.read_bytes())


def test_generic_csv_without_review_text_column_keeps_normal_message():
    with pytest.raises(ValueError, match="Required column 'review_text' not found"):
        load_reviews_csv(b"comment,stars\nnice,5\n")


def test_columns_look_like_data_and_describe_columns():
    assert columns_look_like_data(["AVqVGZNvQMlgsOJE6eUY", "2017-03-03T16:56:05Z"])
    assert columns_look_like_data(["x", "https://example.invalid/a.jpg"]) and columns_look_like_data(["B00ZV9PXP2"])
    assert not columns_look_like_data(["id", "reviews.text", "name"])
    text = describe_columns(["a" * 200, "b", "c", "d", "e", "f", "g", "h"], 3)
    assert len(text) < 120 and "total" in text


# ------------------------------------------------------------------ routing decision
def test_classify_uploads():
    assert classify_uploaded_csv(SAMPLE.read_bytes())["kind"] == "dataset"               # reviews.text + id
    k = classify_uploaded_csv(HEADERLESS.read_bytes())
    assert k["kind"] == "headerless" and k["suggested_layout"] == "Datafiniti 24-column (May 2019 export)"
    assert classify_uploaded_csv(b"review_text,rating\nfine,5\n")["kind"] == "generic"
    assert classify_uploaded_csv(b"review_text\nfine\n")["kind"] == "generic"            # one column is not 'headerless'
    assert classify_uploaded_csv(b"Comment;Stars\nnice;5\n")["kind"] == "generic"
    assert classify_uploaded_csv(b"")["kind"] == "error"


# ------------------------------------------------------------------ explicit layout application
def test_headerless_requires_explicit_opt_in_and_names_the_layout():
    with pytest.raises(HeaderlessDatasetError) as ei:
        scan_products(HEADERLESS)
    assert ei.value.suggested_layout == "Datafiniti 24-column (May 2019 export)" and ei.value.looks_like_data
    assert "Apply it explicitly" in str(ei.value)
    products, info = scan_products(HEADERLESS, infer_layout=True)
    assert info["layout_applied"] == "Datafiniti 24-column (May 2019 export)" and info["n_products"] == 3


def test_headerless_fixture_loads_with_reviews_text_normalised():
    df, info = load_dataset_reviews(HEADERLESS, ["FXP1AAAAAAAAAAAAAAAA"], infer_layout=True)
    assert len(df) == 3 and info["layout_applied"].startswith("Datafiniti 24-column")
    assert df.iloc[0]["cleaned_text"] == "The display is excellent, but the battery drains too quickly."
    assert df.iloc[0]["rating"] == 5.0 and df.iloc[0]["product_title"].startswith("FIXTURE Tablet")


def test_unknown_headerless_data_is_never_guessed(tmp_path):
    p = tmp_path / "odd.csv"
    p.write_text("AVxxxxxxxxxxxxxxxxxx,2017-03-03T16:56:05Z,some text,5\nAVyyyyyyyyyyyyyyyyyy,2017-03-04T16:56:05Z,more,4\n",
                 encoding="utf-8")
    with pytest.raises(HeaderlessDatasetError) as ei:
        inspect_csv_schema(p)
    assert ei.value.suggested_layout is None
    with pytest.raises(DatasetSchemaError, match="will not guess"):
        inspect_csv_schema(p, infer_layout=True)


def test_layout_inference_rejects_mismatching_rows():
    cols = DATAFINITI_LAYOUTS["Datafiniti 24-column (May 2019 export)"]
    row = [""] * len(cols)
    assert infer_headerless_layout([row]) is None                       # empty/invalid values match nothing
    assert infer_headerless_layout([["a", "b"]]) is None and infer_headerless_layout([]) is None


@pytest.mark.real
@pytest.mark.parametrize("layout", list(REAL_FILES))
def test_each_real_file_without_header_infers_its_own_layout(layout):
    path = REAL_FILES[layout]
    if not path.exists():
        pytest.skip("dataset file not present")
    rows = [r for r in csv.reader(io.StringIO(real_slice(path, 5).decode("utf-8")))]
    name, cols = infer_headerless_layout(rows)
    assert name == layout
    with open(path, newline="", encoding="utf-8-sig") as f:
        assert next(csv.reader(f)) == cols                               # inferred names == the real header


@pytest.mark.real
def test_real_headerless_slice_loads_through_dataset_loader():
    path = REAL_FILES["Datafiniti 24-column (2017-18 export)"]
    if not path.exists():
        pytest.skip("dataset file not present")
    data = real_slice(path, 40, header=False)
    assert classify_uploaded_csv(data)["kind"] == "headerless"
    products, info = scan_products(data, infer_layout=True)
    top = products.iloc[0]
    df, _ = load_dataset_reviews(data, [top["product_key"]], infer_layout=True)
    assert len(df) == top["unique_reviews"] > 0 and df["cleaned_text"].str.len().gt(0).all()
    assert set(df["source"]) == {"dataset_import"} and df["rating"].between(1, 5).all()


# ------------------------------------------------------------------ real files in the Dataset folder load end to end
@pytest.mark.real
@pytest.mark.parametrize("layout", list(REAL_FILES))
def test_dataset_folder_files_load_through_dataset_loader(layout):
    path = REAL_FILES[layout]
    if not path.exists():
        pytest.skip("dataset file not present")
    info = inspect_csv_schema(path)
    assert info["header_present"] and info["mapping"]["review_text"] == "reviews.text"
    products, _ = scan_products(path)
    df, linfo = load_dataset_reviews(path, [products.iloc[0]["product_key"]], max_rows=100)
    assert len(df) == 100 and df["cleaned_text"].notna().all() and set(df["source"]) == {"dataset_import"}


# ------------------------------------------------------------------ the Import Dataset page
def _page(tmp_path, monkeypatch, src_file, name):
    d = tmp_path / "Dataset"
    d.mkdir()
    (d / name).write_bytes(Path(src_file).read_bytes())
    monkeypatch.setattr(config, "DATASET_DIR", d)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Import Dataset").run()
    return at


def test_import_page_headerless_file_shows_clear_error_and_opt_in(tmp_path, monkeypatch):
    at = _page(tmp_path, monkeypatch, HEADERLESS, "headerless.csv")
    assert not at.exception, at.exception
    assert any("No header row detected" in e.value and "Required column" not in e.value for e in at.error)
    box = [c for c in at.checkbox if "Apply the verified column layout" in c.label][0]
    assert box.value is False and not [s for s in at.selectbox if s.label == "Product"]   # nothing loads by default
    box.check().run()
    assert not at.exception, at.exception
    assert any("No header row in the file" in w.value for w in at.warning)
    assert [s for s in at.selectbox if s.label == "Product"]


def test_import_page_with_header_file_end_to_end(tmp_path, monkeypatch):
    from src import sentiment_model as sm
    if not sm.MODEL_FILE.exists():
        pytest.skip("models not trained")
    at = _page(tmp_path, monkeypatch, SAMPLE, "datafiniti_sample.csv")
    assert not at.exception and not at.error
    [s for s in at.selectbox if s.label == "Product"][0].set_value("FXP1AAAAAAAAAAAAAAAA").run()
    [b for b in at.button if b.label == "Load and analyze"][0].click().run()
    assert not at.exception, at.exception
    df = at.session_state["review_df"]
    assert len(df) == 3 and set(df["source"]) == {"dataset_import"} and df["review_sentiment"].notna().all()


@pytest.mark.real
def test_import_page_real_dataset_folder_end_to_end(monkeypatch):
    from src import sentiment_model as sm
    path = REAL_FILES["Datafiniti 24-column (2017-18 export)"]
    if not path.exists() or not sm.MODEL_FILE.exists():
        pytest.skip("dataset file or models not present")
    at = AppTest.from_file(APP, default_timeout=300).run()              # real DATASET_DIR, nothing patched
    at.sidebar.radio[0].set_value("Import Dataset").run()
    [s for s in at.selectbox if s.label == "Dataset file"][0].set_value(path.name).run()
    assert not at.exception and not [e for e in at.error if "Required column" in e.value]
    prod = [s for s in at.selectbox if s.label == "Product"][0]
    prod.set_value(prod.options[0]).run() if False else None
    from src.dataset_loader import scan_products as sp
    top = sp(path)[0].iloc[0]["product_key"]
    [s for s in at.selectbox if s.label == "Product"][0].set_value(top).run()
    at.radio(key="analysis_limit").set_value(100).run()
    [b for b in at.button if b.label == "Load and analyze"][0].click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["review_df"]) == 100


# ------------------------------------------------------------------ diagnostics are opt-in
def test_diagnostics_only_log_when_enabled(tmp_path, monkeypatch, capsys):
    log = tmp_path / "diag.log"
    monkeypatch.setattr(csv_diagnostics, "LOG_FILE", log)
    monkeypatch.delenv("REVIEWLENS_DEBUG_CSV", raising=False)
    with pytest.raises(ValueError):
        load_reviews_csv(b"a,b\n1,2\n", source_name="upload:x.csv")
    assert not log.exists()
    monkeypatch.setenv("REVIEWLENS_DEBUG_CSV", "1")
    with pytest.raises(ValueError):
        load_reviews_csv(b"a,b\n1,2\n", source_name="upload:x.csv")
    line = log.read_text(encoding="utf-8")
    assert "load_reviews_csv" in line and "upload:x.csv" in line and "header" in line and "['a', 'b']" in line
