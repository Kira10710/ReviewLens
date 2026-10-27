"""Imported-dataset tests.

``tests/fixtures/datafiniti_sample.csv`` is a SYNTHETIC fixture that uses the *actual* Datafiniti column
layout (24 columns, UTF-8 BOM, quoted fields, an embedded newline, a malformed row). Its review texts are
invented for the tests and are not real customer reviews. Tests marked ``real`` run against the files in
the project's Dataset folder and are skipped when those files are absent.
"""
import shutil
from pathlib import Path

import pytest

from src import config
from src.dataset_loader import (SOURCE_DATASET, DatasetSchemaError, inspect_csv_schema, load_dataset_reviews,
                                map_columns, scan_products)
from src.review_preprocessing import load_reviews_csv, parse_date

FIX = Path(__file__).parent / "fixtures"
SAMPLE = FIX / "datafiniti_sample.csv"
HEADERLESS = FIX / "datafiniti_headerless.csv"
P1, P2, P3 = "FXP1AAAAAAAAAAAAAAAA", "FXP2BBBBBBBBBBBBBBBB", "FXP3CCCCCCCCCCCCCCCC"

DATAFINITI_COLUMNS = ["id", "dateAdded", "dateUpdated", "name", "asins", "brand", "categories", "primaryCategories",
                      "imageURLs", "keys", "manufacturer", "manufacturerNumber", "reviews.date", "reviews.dateSeen",
                      "reviews.didPurchase", "reviews.doRecommend", "reviews.id", "reviews.numHelpful",
                      "reviews.rating", "reviews.sourceURLs", "reviews.text", "reviews.title", "reviews.username",
                      "sourceURLs"]


# ------------------------------------------------------------------ schema inspection / mapping
def test_header_is_verified_not_assumed():
    info = inspect_csv_schema(SAMPLE)
    assert info["header_present"] is True and info["headerless"] is False
    assert info["bom"] is True and info["encoding"] == "utf-8-sig" and info["delimiter"] == ","
    assert info["n_columns"] == 24 and info["columns"][0] == "id"      # BOM not glued to the first name
    assert info["header_matches"] >= 10


def test_column_mapping_to_internal_schema():
    m = map_columns(DATAFINITI_COLUMNS)
    assert m["review_text"] == "reviews.text" and m["rating"] == "reviews.rating"
    assert m["review_title"] == "reviews.title" and m["review_date"] == "reviews.date"
    assert m["product_key"] == "id" and m["asins"] == "asins" and m["product_title"] == "name"
    assert m["categories"] == "primaryCategories"
    assert "reviews.username" not in m.values()                       # personal column never mapped


def test_mapping_is_case_insensitive_and_generic_names_work():
    assert map_columns(["Review_Text", "STARS", "Product_Title"]) == {
        "review_text": "Review_Text", "rating": "STARS", "product_title": "Product_Title"}


def test_headerless_file_is_detected_and_refused_without_names():
    with pytest.raises(DatasetSchemaError, match="No header row detected"):
        inspect_csv_schema(HEADERLESS)


def test_headerless_file_loads_only_with_explicit_names():
    products, info = scan_products(HEADERLESS, column_names=DATAFINITI_COLUMNS)
    assert info["headerless"] is True and info["rows_scanned"] == 10 and len(products) == 3
    with pytest.raises(DatasetSchemaError, match="names but the file has"):
        inspect_csv_schema(HEADERLESS, column_names=["a", "b"])


def test_metadata_only_file_is_reported_not_pretended(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text("id,name,asins,brand,categories,imageURLs\nX1,Widget,B000000009,Acme,Tools,http://x.invalid/a.jpg\n",
                 encoding="utf-8")
    with pytest.raises(DatasetSchemaError, match="No review-text column"):
        scan_products(p)


def test_empty_file_and_garbage(tmp_path):
    (tmp_path / "e.csv").write_bytes(b"")
    with pytest.raises(DatasetSchemaError, match="empty"):
        inspect_csv_schema(tmp_path / "e.csv")
    (tmp_path / "g.csv").write_bytes(b"\x00\x01\x02 binary stuff\n\x03\x04\n")
    with pytest.raises(DatasetSchemaError):
        scan_products(tmp_path / "g.csv")


def test_semicolon_delimiter_and_generic_columns(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_text("review_text;rating;product_title\nGreat screen;5;Phone A\nBad battery;1;Phone A\nOk;3;Phone B\n",
                 encoding="utf-8")
    products, info = scan_products(p)
    assert info["delimiter"] == ";" and set(products["product_title"]) == {"Phone A", "Phone B"}
    df, _ = load_dataset_reviews(p, ["Phone A"])
    assert list(df["rating"]) == [5.0, 1.0]


# ------------------------------------------------------------------ scanning products
def test_scan_products_counts_and_malformed_rows():
    products, info = scan_products(SAMPLE)
    assert info["rows_scanned"] == 10 and info["malformed_rows"] == 1 and info["n_products"] == 3
    by = products.set_index("product_key")
    assert by.loc[P1, "rows"] == 5 and by.loc[P1, "reviews_with_text"] == 4 and by.loc[P1, "reviews_with_rating"] == 4
    assert by.loc[P2, "reviews_with_text"] == 3 and by.loc[P3, "reviews_with_text"] == 0
    assert by.loc[P2, "product_title"] == "FIXTURE Alkaline Batteries (36 Count)"   # most common non-empty name
    assert by.loc[P2, "asin"] == "B000000002,B000000003"
    assert products.iloc[0]["product_key"] == P1                                    # sorted by review count


# ------------------------------------------------------------------ loading reviews
def test_products_are_not_mixed_by_default():
    with pytest.raises(DatasetSchemaError, match="multiple products"):
        load_dataset_reviews(SAMPLE)
    df, info = load_dataset_reviews(SAMPLE, [P1])
    assert set(df["product_title"]) == {'FIXTURE Tablet 8" Display, Wi-Fi'}
    assert set(df["product_id"]) == {"B000000001"}


def test_load_one_product_maps_fields_and_cleans():
    df, info = load_dataset_reviews(SAMPLE, [P1])
    assert info["retained"] == 3 and info["duplicates_removed"] == 1 and info["dropped_empty_text"] == 1
    assert info["retained_with_rating"] == 2 and info["retained_without_rating"] == 1
    assert info["malformed_rows_skipped"] == 1 and info["truncated"] is False
    assert set(df["source"]) == {SOURCE_DATASET} and set(df["source_dataset"]) == {"datafiniti_sample.csv"}
    first = df.iloc[0]
    assert first["review_id"] == "r1" and first["rating"] == 5.0 and first["review_date"] == "2017-01-13"
    assert first["verified_purchase"] is True and first["review_title"] == "Mixed"
    assert first["num_helpful"] == "3" and first["brand"] == "FixtureBrand"
    multi = df[df["review_title"] == "Bad"].iloc[0]
    assert "\n" in multi["review_text"] and "\n" not in multi["cleaned_text"]      # original kept, cleaned flattened
    assert multi["cleaned_text"] == 'Terrible keyboard, "cheap" feel, and awful build quality.'
    norate = df[df["rating"].isna()].iloc[0]
    assert norate["rating"] is None and norate["review_date"] == "2017-02-01"        # nothing invented


def test_personal_columns_are_never_loaded():
    df, _ = load_dataset_reviews(SAMPLE, [P1, P2])
    assert not any("username" in c.lower() for c in df.columns)
    assert not df.astype(str).apply(lambda col: col.str.contains("fixture_user")).any().any()


def test_multiple_selected_products_and_multi_asin():
    df, info = load_dataset_reviews(SAMPLE, [P1, P2])
    assert info["retained"] == 6
    b = df[df["product_id"] == "B000000002"]
    assert len(b) == 3 and set(b["asins_all"]) == {"B000000002,B000000003"}


def test_row_cap_truncates_and_is_reported():
    df, info = load_dataset_reviews(SAMPLE, [P1], max_rows=2)
    assert len(df) == 2 and info["truncated"] is True


def test_dedup_is_product_scoped(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("id,name,asins,reviews.text,reviews.rating\nA,Prod A,B000000001,Same text,5\n"
                 "B,Prod B,B000000002,Same text,5\nA,Prod A,B000000001,Same text,5\n", encoding="utf-8")
    df, info = load_dataset_reviews(p, ["A", "B"])
    assert len(df) == 2 and info["duplicates_removed"] == 1          # kept once per product


def test_selection_without_valid_reviews_is_an_error():
    with pytest.raises(DatasetSchemaError, match="No valid reviews"):
        load_dataset_reviews(SAMPLE, [P3])


def test_bytes_source_supported_for_uploads():
    products, info = scan_products(SAMPLE.read_bytes())
    assert info["n_products"] == 3
    df, linfo = load_dataset_reviews(SAMPLE.read_bytes(), [P2], source_name="upload.csv")
    assert set(df["source_dataset"]) == {"upload.csv"}


# ------------------------------------------------------------------ ISO dates and the legacy uploader hint
@pytest.mark.parametrize("value,expected", [("2017-01-13T00:00:00.000Z", "2017-01-13"), ("2017-01-13T00:00:00Z", "2017-01-13"),
                                            ("2017-01-13 08:30:00", "2017-01-13"), ("2017-13-45T00:00:00Z", None)])
def test_iso_timestamps_parse(value, expected):
    assert parse_date(value) == expected


def test_small_uploader_points_datafiniti_files_to_import_page():
    with pytest.raises(ValueError, match="Import Dataset"):
        load_reviews_csv(SAMPLE.read_bytes())


# ------------------------------------------------------------------ real dataset files (skipped if absent)
REAL = config.DATASET_DIR
F_1429 = REAL / "1429_1.csv"
F_DF = REAL / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
F_MAY = REAL / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"


@pytest.mark.real
@pytest.mark.parametrize("path,bom", [(F_1429, False), (F_DF, True), (F_MAY, True)])
def test_real_files_have_verified_headers_and_mapping(path, bom):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    info = inspect_csv_schema(path)
    assert info["header_present"] and info["bom"] is bom and info["delimiter"] == ","
    m = info["mapping"]
    assert m["review_text"] == "reviews.text" and m["rating"] == "reviews.rating"
    assert m["product_key"] == "id" and m["product_title"] == "name"
    assert info["columns"][0] == "id"


@pytest.mark.real
def test_real_5000_row_file_counts():
    if not F_DF.exists():
        pytest.skip("dataset file not present")
    products, info = scan_products(F_DF)
    assert info["rows_scanned"] == 5000 and info["malformed_rows"] == 0 and info["n_products"] == 24
    assert int(products["reviews_with_text"].sum()) == 5000 and int(products["reviews_with_rating"].sum()) == 5000
    top = products.iloc[0]
    df, linfo = load_dataset_reviews(F_DF, [top["product_key"]])
    assert len(df) == top["reviews_with_text"] == 797 and df["rating"].between(1, 5).all()
    assert df["review_date"].notna().all() and set(df["source"]) == {SOURCE_DATASET}


@pytest.mark.real
@pytest.mark.parametrize("path,rows,products,with_text", [(F_1429, 34660, 42, 34658), (F_MAY, 28332, 65, 28331)])
def test_real_large_files_stream_and_count(path, rows, products, with_text):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    prods, info = scan_products(path)
    assert info["rows_scanned"] == rows and info["malformed_rows"] == 0 and info["n_products"] == products
    assert int(prods["reviews_with_text"].sum()) == with_text
    df, linfo = load_dataset_reviews(path, [prods.iloc[0]["product_key"]], max_rows=300)
    assert len(df) <= 300 and df["cleaned_text"].str.len().gt(0).all()
