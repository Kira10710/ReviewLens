import io

import pytest

from src.review_preprocessing import (SCHEMA_COLUMNS, build_model_text, clean_and_deduplicate_reviews,
                                      clean_text, load_reviews_csv, parse_date, parse_rating)


def test_clean_text_entities_whitespace_and_negation_kept():
    raw = "  It&#39;s   &amp;amp; NOT good at all\x00 \n\n never again  "
    out = clean_text(raw)
    assert out == "It's & NOT good at all never again"
    assert "NOT" in out and "never" in out


def test_clean_text_handles_missing_and_markup():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""
    assert clean_text("Nice <b>screen</b><br>really") == "Nice screen really"
    assert clean_text("Great product... Read more") == "Great product"


def test_build_model_text():
    assert build_model_text("Great", "Works well.") == "Great. Works well."
    assert build_model_text("Great!", "Works.") == "Great! Works."
    assert build_model_text(None, "Only body") == "Only body"
    assert build_model_text("Only title", "") == "Only title"


@pytest.mark.parametrize("value,expected", [("4.0 out of 5 stars", 4.0), (5, 5.0), ("3,5", 3.5), ("0", None),
                                            ("9", None), ("abc", None), (None, None), (float("nan"), None)])
def test_parse_rating(value, expected):
    assert parse_rating(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("Reviewed in the United States on March 3, 2024", "2024-03-03"),
    ("Reviewed in India on 12 January 2023", "2023-01-12"),
    ("2024-05-06", "2024-05-06"), ("yesterday-ish", None), (None, None), ("", None)])
def test_parse_date(value, expected):
    assert parse_date(value) == expected


def test_dedup_by_id_and_by_content_and_empty_drop():
    recs = [
        {"review_id": "R1", "review_text": "Same text", "rating": 5},
        {"review_id": "R1", "review_text": "Same text (re-fetched)", "rating": 5},       # same id -> dup
        {"review_text": "No id here", "review_title": "T", "rating": "4"},
        {"review_text": "  no   ID here ", "review_title": "t", "rating": 4},          # same content -> dup
        {"review_text": "   "},                                                          # empty -> dropped
        {"review_text": None},
        "not a dict",
    ]
    df, stats = clean_and_deduplicate_reviews(recs, source="csv_upload")
    assert stats == {"input_records": 7, "dropped_empty": 3, "duplicates_removed": 2, "retained": 2}
    assert list(df.columns) == SCHEMA_COLUMNS
    assert df["review_id"].notna().all() and df["review_id"].is_unique
    assert set(df["source"]) == {"csv_upload"}
    assert df.loc[0, "review_text"] == "Same text"      # original text preserved


def test_missing_metadata_is_none_not_invented():
    df, _ = clean_and_deduplicate_reviews([{"review_text": "Only text"}])
    row = df.iloc[0]
    assert row["rating"] is None and row["review_date"] is None and row["verified_purchase"] is None
    assert row["review_title"] is None and row["scraped_at"] is None


def test_original_text_preserved_alongside_cleaned():
    df, _ = clean_and_deduplicate_reviews([{"review_text": "It&#39;s   great"}])
    assert df.loc[0, "review_text"] == "It&#39;s   great" and df.loc[0, "cleaned_text"] == "It's great"


# ------------------------------------------------------------------ CSV upload validation
def test_csv_valid_with_optional_columns():
    data = b"Review_Text,Rating,review_title,review_date\nGreat screen,5,Love it,2024-01-02\n,3,,\nBad battery,1,,\n"
    df, info = load_reviews_csv(data)
    assert len(df) == 2 and info["rows_dropped_empty"] == 1
    assert df.loc[0, "rating"] == 5.0 and df.loc[0, "review_date"] == "2024-01-02"
    assert set(df["source"]) == {"csv_upload"}


def test_csv_minimum_schema_only():
    df, info = load_reviews_csv(b"review_text\nhello world\n")
    assert len(df) == 1 and df.loc[0, "rating"] is None


def test_csv_custom_text_column():
    df, _ = load_reviews_csv(b"comment\nnice screen\n", "comment")
    assert len(df) == 1


def test_csv_duplicates_removed():
    df, info = load_reviews_csv(b"review_text\nsame\nsame\nother\n")
    assert len(df) == 2 and info["duplicates_removed"] == 1


@pytest.mark.parametrize("payload,msg", [
    (b"", "empty"),
    (b"review_text\n", "no rows"),
    (b"review_text\n\n   \n", "no rows|no non-empty"),
    (b"text\nhello\n", "not found"),
    (b'review_text,x\n"unterminated,1\n', "parse|no non-empty|not found"),
    (b"\x00\x01\x02binary", "not found|parse|no rows"),
])
def test_csv_malformed(payload, msg):
    with pytest.raises(ValueError, match=msg):
        load_reviews_csv(payload)


def test_csv_accepts_file_like_object():
    df, _ = load_reviews_csv(io.BytesIO(b"review_text\nfine\n"))
    assert len(df) == 1
