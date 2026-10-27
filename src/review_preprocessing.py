"""Internal review schema, cleaning, de-duplication and CSV loading.

Both scraped Amazon records and uploaded CSV rows go through the same functions
here, so downstream analysis never depends on where a review came from.
Missing metadata (rating, date, verified flag...) stays ``None``; nothing is invented.
"""
from __future__ import annotations

import hashlib
import html
import io
import re
import unicodedata
from datetime import datetime

import pandas as pd

SCHEMA_COLUMNS = [
    "review_id", "product_id", "product_url", "source_url", "review_title", "review_text",
    "cleaned_text", "rating", "review_date", "review_date_raw", "verified_purchase",
    "source", "scraped_at",
]
SOURCE_SCRAPE = "amazon_scrape"
SOURCE_UPLOAD = "csv_upload"

DEFAULT_TEXT_COLUMN = "review_text"
MAX_UPLOAD_ROWS = 5000
MAX_REVIEW_CHARS = 5000

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                            "–": "-", "—": "-", " ": " ", "​": ""})
_READ_MORE_RE = re.compile(r"\s*(?:\.\.\.\s*)?Read more\s*$", re.IGNORECASE)
_DATE_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y", "%m/%d/%Y",
                 "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S")


def clean_text(value) -> str:
    """Normalise a review string for display and analysis.

    Decodes HTML entities, fixes encoding artefacts via NFKC, straightens quotes,
    strips control characters and collapses whitespace. Case, punctuation and
    negation words are preserved. ``None``/NaN become ''.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    for _ in range(2):  # handles double-escaped entities such as &amp;amp;
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    text = re.sub(r"<[^>]+>", " ", text)           # stray tags left in CSV exports
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = _CONTROL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _READ_MORE_RE.sub("", text).strip()


def build_model_text(title, body) -> str:
    """Text fed to the models: title (if any) followed by the body."""
    title, body = clean_text(title), clean_text(body)
    if title and body:
        sep = " " if title[-1] in ".!?" else ". "
        return f"{title}{sep}{body}"[:MAX_REVIEW_CHARS]
    return (title or body)[:MAX_REVIEW_CHARS]


def parse_rating(value) -> float | None:
    """Return a rating in [1, 5] or None. Accepts 4, '4.0', '4.0 out of 5 stars'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(value))
    if not m:
        return None
    rating = float(m.group(0).replace(",", "."))
    return rating if 1.0 <= rating <= 5.0 else None


def parse_date(value) -> str | None:
    """Return an ISO date (YYYY-MM-DD) or None when the text is not a recognised date."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = clean_text(value)
    if not text:
        return None
    text = re.sub(r"^.*?\bon\s+", "", text) if re.match(r"(?i)reviewed", text) else text
    iso = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$", text)
    if iso:                                    # e.g. 2017-01-13T00:00:00.000Z (Datafiniti exports)
        text = iso.group(1)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _norm_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if s in {"true", "yes", "y", "1", "verified purchase", "verified"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return None


def _dedup_key(row: dict) -> str:
    rid, pid = row.get("review_id"), row.get("product_id") or ""
    if rid:
        return f"id:{pid}:{rid}" if pid else f"id:{rid}"
    basis = "|".join(str(row.get(k) or "").lower() for k in
                     ("product_id", "review_title", "cleaned_text", "review_date", "rating"))
    return "h:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()


def record_key(rec: dict) -> str | None:
    """De-duplication key of a raw record, computed exactly as ``clean_and_deduplicate_reviews`` does.

    Returns None when the record has neither text nor title. Lets streaming code (dataset scan and
    load) count and drop duplicates *before* applying a row limit, so the limit counts unique reviews.
    """
    cleaned = clean_text(rec.get("review_text"))
    title = clean_text(rec.get("review_title")) or None
    if not cleaned and not title:
        return None
    return _dedup_key({
        "review_id": clean_text(rec.get("review_id")) or None,
        "product_id": clean_text(rec.get("product_id")) or None,
        "review_title": title, "cleaned_text": cleaned[:MAX_REVIEW_CHARS],
        "rating": parse_rating(rec.get("rating")), "review_date": parse_date(rec.get("review_date")),
    })


def clean_and_deduplicate_reviews(reviews, source: str | None = None,
                                  extra_fields: tuple = ()) -> tuple[pd.DataFrame, dict]:
    """Normalise records into the internal schema and drop duplicates.

    ``reviews`` is an iterable of dicts (or a DataFrame). Records lacking usable text
    are dropped and counted. ``extra_fields`` names additional input keys to carry through unchanged
    (appended after the schema columns). Duplicates are detected by ``review_id`` when present,
    otherwise by a hash of title + text + date + rating. Returns (DataFrame, stats).
    """
    records = reviews.to_dict("records") if isinstance(reviews, pd.DataFrame) else list(reviews or [])
    stats = {"input_records": len(records), "dropped_empty": 0, "duplicates_removed": 0, "retained": 0}
    seen, rows = set(), []
    for rec in records:
        if not isinstance(rec, dict):
            stats["dropped_empty"] += 1
            continue
        raw_text = rec.get("review_text")
        cleaned = clean_text(raw_text)
        title = clean_text(rec.get("review_title")) or None
        if not cleaned and not title:
            stats["dropped_empty"] += 1
            continue
        date_raw = rec.get("review_date_raw") or rec.get("review_date")
        row = {
            "review_id": clean_text(rec.get("review_id")) or None,
            "product_id": clean_text(rec.get("product_id")) or None,
            "product_url": rec.get("product_url") or None,
            "source_url": rec.get("source_url") or None,
            "review_title": title,
            "review_text": None if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text))
            else str(raw_text),
            "cleaned_text": cleaned[:MAX_REVIEW_CHARS],
            "rating": parse_rating(rec.get("rating")),
            "review_date": parse_date(rec.get("review_date")),
            "review_date_raw": clean_text(date_raw) or None,
            "verified_purchase": _norm_bool(rec.get("verified_purchase")),
            "source": rec.get("source") or source,
            "scraped_at": rec.get("scraped_at") or None,
        }
        for extra in extra_fields:
            val = rec.get(extra)
            row[extra] = None if val is None or (isinstance(val, float) and pd.isna(val)) else val
        key = _dedup_key(row)
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        if not row["review_id"]:
            row["review_id"] = "gen-" + key.split(":", 1)[1][:12] if key.startswith("h:") else key
        rows.append(row)
    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS + list(extra_fields)).astype(object)
    df = df.where(pd.notna(df), None).reset_index(drop=True)
    stats["retained"] = len(df)
    return df, stats


_DATA_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_DATA_ASIN = re.compile(r"^B0[A-Z0-9]{8}$")


def columns_look_like_data(columns) -> bool:
    """True if 'column names' are really a data row (ISO timestamps, URLs or ASINs): the header line is missing."""
    for c in columns:
        c = str(c).strip()
        if _DATA_ISO.match(c) or c.lower().startswith(("http://", "https://")) or _DATA_ASIN.match(c):
            return True
    return False


def describe_columns(columns, n: int = 6, width: int = 40) -> str:
    """Short, safe rendering of a column list (never dumps long image-URL cells)."""
    cols = [str(c) for c in columns]
    shown = [c if len(c) <= width else c[:width] + "..." for c in cols[:n]]
    return f"{shown}" + (f" (+{len(cols) - n} more, {len(cols)} total)" if len(cols) > n else "")


def _find_column(columns, wanted: str):
    lookup = {str(c).strip().lower(): c for c in columns}
    return lookup.get(wanted.strip().lower())


def load_reviews_csv(file, text_column: str | None = None,
                     source_name: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Validate and load an uploaded CSV into the internal schema.

    Required column: ``review_text`` (or the name passed in ``text_column``).
    Optional: ``rating``, ``review_title``, ``review_date``, ``review_id``,
    ``product_id``, ``verified_purchase``. Raises ValueError with a user-friendly
    message for unreadable files, missing columns or no usable rows.
    """
    try:
        if isinstance(file, (bytes, bytearray)):
            file = io.BytesIO(file)
        raw = pd.read_csv(file, dtype=str, keep_default_na=True, on_bad_lines="error",
                          encoding_errors="replace")
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Could not parse the file as CSV: {exc}")
    if raw.empty:
        raise ValueError("The CSV has a header but no rows.")
    text_col = _find_column(raw.columns, text_column or DEFAULT_TEXT_COLUMN)
    if text_col is None:
        from src.csv_diagnostics import log_csv_failure
        log_csv_failure("review_preprocessing.load_reviews_csv", source_name or file,
                        {"reader": "pandas.read_csv", "header": "infer (row 0)", "sep": ",",
                         "encoding": "utf-8 + errors=replace", "text_column_wanted": text_column or DEFAULT_TEXT_COLUMN},
                        raw.columns)
        if columns_look_like_data(raw.columns):
            raise ValueError(
                "This file appears to have NO header row: the first row contains data, not column names "
                f"(first cells: {describe_columns(raw.columns)}). Add a header line, or, for a Datafiniti-style "
                "review export, use the 'Import Dataset' page, which can apply a verified column layout.")
        hint = ""
        if any(str(c).strip().lower() in ("reviews.text", "reviews.rating") for c in raw.columns):
            hint = (" This looks like a Datafiniti-style dataset (reviews.text / reviews.rating). Use the "
                    "'Import Dataset' page, which maps these columns and lets you pick a product.")
        raise ValueError(f"Required column '{text_column or DEFAULT_TEXT_COLUMN}' not found. "
                         f"Columns present: {describe_columns(raw.columns, 12)}.{hint}")
    n_rows = len(raw)
    truncated = n_rows > MAX_UPLOAD_ROWS
    raw = raw.head(MAX_UPLOAD_ROWS)
    records = []
    for _, r in raw.iterrows():
        rec = {"review_text": r[text_col], "source": SOURCE_UPLOAD}
        for field in ("review_title", "rating", "review_date", "review_id", "product_id",
                      "verified_purchase"):
            col = _find_column(raw.columns, field)
            if col is not None:
                rec[field] = r[col]
        records.append(rec)
    df, stats = clean_and_deduplicate_reviews(records, source=SOURCE_UPLOAD)
    if df.empty:
        raise ValueError(f"Column '{text_col}' contains no non-empty reviews.")
    info = {"rows_in_file": n_rows, "rows_processed": min(n_rows, MAX_UPLOAD_ROWS),
            "rows_dropped_empty": stats["dropped_empty"],
            "duplicates_removed": stats["duplicates_removed"], "retained": stats["retained"],
            "truncated": truncated, "optional_columns_found": [
                f for f in ("review_title", "rating", "review_date", "review_id") if
                _find_column(raw.columns, f) is not None]}
    return df, info
