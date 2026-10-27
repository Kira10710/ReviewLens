"""Imported-dataset mode: load Datafiniti-style Amazon review exports (and similar CSVs).

This is a *separate mode* from live scraping. Records are tagged ``source = dataset_import``
and are never presented as scraped.

Design:
* ``inspect_csv_schema`` verifies encoding, delimiter and whether a header row really exists
  (by matching known column names). It never blindly assumes ``header=None``.
* Source columns are mapped to ReviewLens's internal schema by ``COLUMN_ALIASES``.
* Files are streamed row by row with the ``csv`` module and only the needed columns are kept, so
  a 250 MB file is never copied into memory. Rows whose field count does not match the header
  (malformed) are skipped and counted.
* A dataset can hold many products; ``scan_products`` lists them and ``load_dataset_reviews`` loads
  only the products the caller selects. Nothing is mixed unless the caller explicitly asks.
* Personal fields (reviews.username, userCity, userProvince) are never loaded.
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.review_preprocessing import (MAX_UPLOAD_ROWS, clean_and_deduplicate_reviews, clean_text,
                                      columns_look_like_data, describe_columns, record_key)

SOURCE_DATASET = "dataset_import"
csv.field_size_limit(2 ** 31 - 1)

# internal field -> candidate source column names, in priority order (matched case-insensitively)
COLUMN_ALIASES: dict[str, list[str]] = {
    "review_text": ["reviews.text", "review_text", "reviewtext", "review_body", "reviewbody", "review", "text",
                    "content", "comment"],
    "review_title": ["reviews.title", "review_title", "reviewtitle", "summary", "headline", "title"],
    "rating": ["reviews.rating", "rating", "star_rating", "stars", "overall", "score"],
    "review_date": ["reviews.date", "review_date", "reviewdate", "reviewtime", "date"],
    "review_id": ["reviews.id", "review_id", "reviewid"],
    "verified_purchase": ["reviews.didpurchase", "verified_purchase", "verified"],
    "product_key": ["id", "product_key"],                         # Datafiniti: one id per product record
    "asins": ["asins", "asin", "product_id", "productid"],
    "product_title": ["name", "product_title", "product_name", "producttitle"],
    "brand": ["brand"],
    "num_helpful": ["reviews.numhelpful", "num_helpful", "helpful_votes"],
    "do_recommend": ["reviews.dorecommend", "do_recommend"],
    "categories": ["primarycategories", "categories"],
}
PERSONAL_COLUMNS = {"reviews.username", "reviews.usercity", "reviews.userprovince", "username", "user_name"}
EXTRA_FIELDS = ("product_title", "brand", "categories", "num_helpful", "do_recommend", "asins_all",
                "source_dataset")
_ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_ASIN = re.compile(r"^[A-Z0-9]{10}$")


class DatasetSchemaError(ValueError):
    """The file's structure cannot be mapped to reviews (reported verbatim to the user)."""


class HeaderlessDatasetError(DatasetSchemaError):
    """No header row. ``suggested_layout`` names a known layout whose data fingerprint matches, if any."""

    def __init__(self, message: str, suggested_layout: str | None = None, looks_like_data: bool = False):
        super().__init__(message)
        self.suggested_layout = suggested_layout
        self.looks_like_data = looks_like_data


_D24A = ["id", "dateAdded", "dateUpdated", "name", "asins", "brand", "categories", "primaryCategories", "imageURLs",
         "keys", "manufacturer", "manufacturerNumber", "reviews.date", "reviews.dateAdded", "reviews.dateSeen",
         "reviews.doRecommend", "reviews.id", "reviews.numHelpful", "reviews.rating", "reviews.sourceURLs",
         "reviews.text", "reviews.title", "reviews.username", "sourceURLs"]
_D24B = ["id", "dateAdded", "dateUpdated", "name", "asins", "brand", "categories", "primaryCategories", "imageURLs",
         "keys", "manufacturer", "manufacturerNumber", "reviews.date", "reviews.dateSeen", "reviews.didPurchase",
         "reviews.doRecommend", "reviews.id", "reviews.numHelpful", "reviews.rating", "reviews.sourceURLs",
         "reviews.text", "reviews.title", "reviews.username", "sourceURLs"]
_D21 = ["id", "name", "asins", "brand", "categories", "keys", "manufacturer", "reviews.date", "reviews.dateAdded",
        "reviews.dateSeen", "reviews.didPurchase", "reviews.doRecommend", "reviews.id", "reviews.numHelpful",
        "reviews.rating", "reviews.sourceURLs", "reviews.text", "reviews.title", "reviews.userCity",
        "reviews.userProvince", "reviews.username"]
# Known Datafiniti Amazon-review export layouts. Used ONLY for files without a header row, and only when
# the user explicitly applies one AND the sample rows match its value patterns (see infer_headerless_layout).
DATAFINITI_LAYOUTS: dict[str, list[str]] = {
    "Datafiniti 24-column (2017-18 export)": _D24A,
    "Datafiniti 24-column (May 2019 export)": _D24B,
    "Datafiniti 21-column (1429_1 export)": _D21,
}
_ASIN_LIST = re.compile(r"^[A-Z0-9]{10}(,[A-Z0-9]{10})*$")
_ISO_FIRST = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_BOOLISH = {"", "true", "false", "yes", "no", "0", "1"}


def infer_headerless_layout(sample_rows) -> tuple[str, list[str]] | None:
    """Return (layout name, column names) if every sample row matches a known Datafiniti layout.

    Checks column count, ASIN pattern in ``asins``, non-empty ``id``, ISO dates, 1-5 ratings, boolean-like flags,
    at least one non-empty review text. Returns None when nothing matches: no guessing.
    """
    rows = [r for r in sample_rows if r]
    if not rows:
        return None
    for name, cols in DATAFINITI_LAYOUTS.items():
        if any(len(r) != len(cols) for r in rows):
            continue
        ix = {c: i for i, c in enumerate(cols)}
        ok = True
        for r in rows:
            g = lambda c: r[ix[c]].strip()
            if not _ASIN_LIST.match(g("asins")) or not g("id"):
                ok = False
            if g("reviews.rating") not in {"", "1", "2", "3", "4", "5", "1.0", "2.0", "3.0", "4.0", "5.0"}:
                ok = False
            if g("reviews.date") and not _ISO_FIRST.match(g("reviews.date")):
                ok = False
            seen = g("reviews.dateSeen").split(",")[0].strip()
            if not _ISO_FIRST.match(seen):                      # dateSeen is always filled in these exports
                ok = False
            if "reviews.didPurchase" in ix and g("reviews.didPurchase").lower() not in _BOOLISH:
                ok = False
            if g("reviews.doRecommend").lower() not in _BOOLISH:
                ok = False
        if ok and any(r[ix["reviews.text"]].strip() for r in rows):
            return name, list(cols)
    return None


# ---------------------------------------------------------------------------- opening
@contextmanager
def _open_text(source, encoding: str = "utf-8-sig"):
    """Yield a text stream for a path or bytes. 'utf-8-sig' transparently drops a BOM."""
    if isinstance(source, (bytes, bytearray)):
        stream = io.TextIOWrapper(io.BytesIO(bytes(source)), encoding=encoding, errors="replace", newline="")
    else:
        stream = open(source, "r", encoding=encoding, errors="replace", newline="")
    try:
        yield stream
    finally:
        stream.close()


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


# ---------------------------------------------------------------------------- schema
def _alias_lookup(columns) -> dict[str, int]:
    """Lower-cased column name -> index (first occurrence wins)."""
    out: dict[str, int] = {}
    for i, c in enumerate(columns):
        out.setdefault(str(c).strip().lower(), i)
    return out


def map_columns(columns) -> dict[str, str]:
    """Return {internal_field: actual source column name} for fields found in ``columns``."""
    lookup = _alias_lookup(columns)
    mapping = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                mapping[field] = list(columns)[lookup[alias]].strip()
                break
    return mapping


def _header_score(first_row) -> int:
    known = {a for aliases in COLUMN_ALIASES.values() for a in aliases} | PERSONAL_COLUMNS
    return sum(1 for c in first_row if str(c).strip().lower() in known)


def _looks_like_data(first_row) -> bool:
    return any(_ISO_TS.match(c.strip()) or c.strip().lower().startswith(("http://", "https://"))
               or _ASIN.match(c.strip()) for c in first_row)


def inspect_csv_schema(source, column_names: list[str] | None = None, infer_layout: bool = False) -> dict:
    """Inspect a CSV without loading it: encoding, delimiter, header verdict, column mapping.

    ``header_present`` is decided from evidence: at least two first-row cells must be known
    column names. If they are not and the first row looks like data (timestamps, URLs, ASINs),
    the file is reported as headerless (``HeaderlessDatasetError``); supply ``column_names`` to load such a file, or
    pass ``infer_layout=True`` to apply a known Datafiniti layout *if* the sample rows match its value patterns.
    """
    if isinstance(source, (bytes, bytearray)):
        head = bytes(source[:200_000])
        size = len(source)
    else:
        with open(source, "rb") as f:
            head = f.read(200_000)
        size = Path(source).stat().st_size
    bom = head.startswith(b"\xef\xbb\xbf")
    try:
        head.decode("utf-8-sig")
        encoding = "utf-8-sig" if bom else "utf-8"
    except UnicodeDecodeError as exc:
        if exc.start < len(head) - 4:          # a real error, not a multibyte char cut by the 200 KB read
            encoding = "unknown (not valid UTF-8; non-UTF-8 bytes will be replaced)"
        else:
            encoding = "utf-8-sig" if bom else "utf-8"
    text = head.decode("utf-8-sig", errors="replace")
    delimiter = _sniff_delimiter(text[:20_000])
    with _open_text(source) as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            first = next(reader)
        except StopIteration:
            raise DatasetSchemaError("The file is empty.")
        previews = [first]
        for _ in range(4):
            try:
                previews.append(next(reader))
            except StopIteration:
                break
    score = _header_score(first)
    header_present = score >= 2
    info = {"encoding": encoding, "bom": bom, "delimiter": delimiter, "size_bytes": size,
            "n_columns": len(first), "header_present": header_present, "header_matches": score,
            "first_rows": [[c[:60] for c in r[:8]] for r in previews[:3]]}
    if header_present:
        info["columns"] = [c.strip() for c in first]
        info["header_evidence"] = f"{score} first-row cells are recognised column names"
    elif column_names:
        if len(column_names) != len(first):
            raise DatasetSchemaError(f"column_names has {len(column_names)} names but the file has {len(first)} columns.")
        info["columns"] = list(column_names)
        info["header_evidence"] = "no header row detected; caller supplied column_names"
    else:
        looks = _looks_like_data(first)
        suggestion = infer_headerless_layout(previews)
        if infer_layout and suggestion:
            info["columns"] = suggestion[1]
            info["layout_applied"] = suggestion[0]
            info["header_evidence"] = (f"no header row; applied verified layout '{suggestion[0]}' (all "
                                       f"{len(previews)} sample rows match its value patterns)")
        else:
            reason = ("the first row looks like data (timestamps, URLs or ASINs)" if looks
                      else "no recognised column names were found in the first row")
            msg = (f"No header row detected: {reason}. First-row cells: {describe_columns(first)}. ")
            if infer_layout:
                msg += "No known Datafiniti layout matches this data, so ReviewLens will not guess column roles."
            elif suggestion:
                msg += (f"The data matches the known layout '{suggestion[0]}'. Apply it explicitly to continue "
                        "(ReviewLens never does this silently), or add a header row.")
            else:
                msg += ("Add a header row, or load the file with explicit column_names. ReviewLens will not guess "
                        "column roles.")
            raise HeaderlessDatasetError(msg, suggestion[0] if suggestion else None, looks)
    info["mapping"] = map_columns(info["columns"])
    info["headerless"] = not header_present
    return info


def _require_mapping(info: dict) -> dict[str, str]:
    m = info["mapping"]
    if "review_text" not in m:
        raise DatasetSchemaError(
            "No review-text column found. Looked for: " + ", ".join(COLUMN_ALIASES["review_text"][:5])
            + f". Columns present: {describe_columns(info['columns'], 12)}. If this file only contains product metadata "
            "(no review text), there is nothing to analyze.")
    return m


# ---------------------------------------------------------------------------- streaming core
def _row_iter(source, info: dict):
    """Yield (row_number, row) for data rows of the right width; count malformed ones in a dict."""
    stats = {"rows_scanned": 0, "malformed_rows": 0}
    width = info["n_columns"]

    def gen():
        with _open_text(source) as f:
            reader = csv.reader(f, delimiter=info["delimiter"])
            if not info["headerless"]:
                next(reader, None)
            for row in reader:
                if not row:
                    continue
                stats["rows_scanned"] += 1
                if len(row) != width:
                    stats["malformed_rows"] += 1
                    continue
                yield stats["rows_scanned"], row
    return gen(), stats


def _index(info: dict, field: str):
    col = info["mapping"].get(field)
    return info["columns"].index(col) if col is not None else None


def _product_group_field(info: dict) -> str:
    for field in ("product_key", "asins", "product_title"):
        if field in info["mapping"]:
            return field
    return ""


def scan_products(source, column_names: list[str] | None = None,
                  infer_layout: bool = False) -> tuple[pd.DataFrame, dict]:
    """One streaming pass: list the products in the file with their review counts.

    Returns (products_df, file_info). products_df columns: product_key, product_title, asin,
    brand, rows, reviews_with_text, reviews_with_rating, unique_reviews. ``unique_reviews`` is the exact
    number of distinct valid reviews (text present, de-duplicated with the same key the loader uses), i.e.
    how many reviews can actually be analyzed. Only a few columns and one 64-bit hash per review are kept.
    """
    info = inspect_csv_schema(source, column_names, infer_layout)
    _require_mapping(info)
    gi = _index(info, _product_group_field(info)) if _product_group_field(info) else None
    ti, ri = _index(info, "review_text"), _index(info, "rating")
    ni, ai, bi = _index(info, "product_title"), _index(info, "asins"), _index(info, "brand")
    rows, with_text, with_rating = Counter(), Counter(), Counter()
    names, asins, brands = defaultdict(Counter), {}, {}
    seen = defaultdict(set)
    ii, di, tti = _index(info, "review_id"), _index(info, "review_date"), _index(info, "review_title")
    it, stats = _row_iter(source, info)
    for _, row in it:
        key = row[gi].strip() if gi is not None else "(all)"
        rows[key] += 1
        if clean_text(row[ti]):
            with_text[key] += 1
            rec_key = record_key({
                "review_id": row[ii] if ii is not None else None,
                "product_id": (row[ai].split(",")[0] if ai is not None else None),
                "review_title": row[tti] if tti is not None else None, "review_text": row[ti],
                "rating": row[ri] if ri is not None else None,
                "review_date": row[di] if di is not None else None})
            seen[key].add(hash(rec_key))
        if ri is not None and row[ri].strip():
            with_rating[key] += 1
        if ni is not None and row[ni].strip():
            names[key][row[ni].strip()] += 1
        if ai is not None and key not in asins and row[ai].strip():
            asins[key] = row[ai].strip()
        if bi is not None and key not in brands and row[bi].strip():
            brands[key] = row[bi].strip()
    products = pd.DataFrame([{
        "product_key": k,
        "product_title": names[k].most_common(1)[0][0] if names[k] else "(untitled)",
        "asin": asins.get(k, ""), "brand": brands.get(k, ""), "rows": rows[k],
        "reviews_with_text": with_text[k], "reviews_with_rating": with_rating[k],
        "unique_reviews": len(seen[k]),
    } for k in rows]).sort_values("unique_reviews", ascending=False).reset_index(drop=True)
    info.update(stats)
    info["n_products"] = len(products)
    info["product_field"] = _product_group_field(info)
    return products, info


def load_dataset_reviews(source, product_keys: list[str] | None = None, max_rows: int = MAX_UPLOAD_ROWS,
                         column_names: list[str] | None = None, source_name: str | None = None,
                         allow_all_products: bool = False, infer_layout: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load reviews for the selected products into the internal schema.

    ``product_keys`` is required unless the file has no product column or ``allow_all_products`` is
    True, so reviews of different products are never mixed by accident. At most ``max_rows`` rows are
    loaded, in file order (``info['truncated']`` says if the cap was hit). Missing ratings stay empty.
    """
    info = inspect_csv_schema(source, column_names, infer_layout)
    mapping = _require_mapping(info)
    group_field = _product_group_field(info)
    if group_field and not product_keys and not allow_all_products:
        raise DatasetSchemaError("This file contains multiple products. Select at least one product, or "
                                 "explicitly allow all products.")
    wanted = set(product_keys or [])
    gi = _index(info, group_field) if group_field else None
    idx = {f: _index(info, f) for f in COLUMN_ALIASES}
    name = source_name or (Path(source).name if not isinstance(source, (bytes, bytearray)) else "uploaded file")
    records, matched, truncated, no_text, stream_dups = [], 0, False, 0, 0
    seen: set[int] = set()
    it, stats = _row_iter(source, info)
    for row_no, row in it:
        if gi is not None and wanted and row[gi].strip() not in wanted:
            continue
        matched += 1

        def get(field):
            i = idx.get(field)
            return row[i] if i is not None else None

        if not clean_text(get("review_text")):      # a valid review needs text; title-only rows are skipped
            no_text += 1
            continue
        asins_all = (get("asins") or "").strip()
        rec_key = hash(record_key({"review_id": get("review_id"), "product_id": asins_all.split(",")[0],
                                   "review_title": get("review_title"), "review_text": get("review_text"),
                                   "rating": get("rating"), "review_date": get("review_date")}))
        if rec_key in seen:                          # duplicates never consume the limit
            stream_dups += 1
            continue
        if len(records) >= max_rows:                 # a further *unique* valid review exists beyond the limit
            truncated = True
            break
        seen.add(rec_key)
        records.append({
            "review_id": get("review_id") or None,
            "product_id": asins_all.split(",")[0].strip() or None,
            "product_url": None,
            "source_url": None,
            "review_title": get("review_title"),
            "review_text": get("review_text"),
            "rating": get("rating"),
            "review_date": get("review_date"),
            "verified_purchase": get("verified_purchase"),
            "source": SOURCE_DATASET,
            "scraped_at": None,
            "product_title": (get("product_title") or "").strip() or None,
            "brand": (get("brand") or "").strip() or None,
            "categories": (get("categories") or "")[:200] or None,
            "num_helpful": get("num_helpful"),
            "do_recommend": get("do_recommend"),
            "asins_all": asins_all or None,
            "source_dataset": name,
        })
    df, cstats = clean_and_deduplicate_reviews(records, source=SOURCE_DATASET, extra_fields=EXTRA_FIELDS)
    # De-duplication is product-scoped (see review_preprocessing._dedup_key), so identical text on two
    # different products is never merged.
    info.update({"rows_scanned_until_stop": stats["rows_scanned"], "malformed_rows_skipped": stats["malformed_rows"],
                 "rows_matched": matched, "truncated": truncated, "requested_limit": max_rows,
                 "max_rows": max_rows, "dropped_empty_text": no_text + cstats["dropped_empty"],
                 "duplicates_removed": stream_dups + cstats["duplicates_removed"], "retained": cstats["retained"],
                 "retained_with_rating": int(df["rating"].notna().sum()) if len(df) else 0,
                 "retained_without_rating": int(df["rating"].isna().sum()) if len(df) else 0,
                 "selected_products": sorted(wanted)})
    if df.empty:
        raise DatasetSchemaError("No valid reviews (non-empty review text) were found for the selection.")
    return df, info


def classify_uploaded_csv(data: bytes) -> dict:
    """Decide how an uploaded CSV must be handled *before* any generic loader sees it.

    kind = "dataset"   : header present and dataset-style (product key column, or a dotted export-style text column such as
                         ``reviews.text``): route through ``scan_products`` / ``load_dataset_reviews``.
           "headerless": the first row is data (timestamps/URLs/ASINs): never pass it to a header-based reader.
           "generic"   : an ordinary review CSV (e.g. a ``review_text`` column): use ``load_reviews_csv``.
           "error"     : unreadable (empty file ...).
    """
    try:
        info = inspect_csv_schema(data)
    except HeaderlessDatasetError as exc:
        if exc.looks_like_data:
            return {"kind": "headerless", "error": str(exc), "suggested_layout": exc.suggested_layout}
        return {"kind": "generic"}
    except DatasetSchemaError as exc:
        return {"kind": "error", "error": str(exc)}
    m = info["mapping"]
    if "review_text" in m and ("product_key" in m or "." in m["review_text"]):
        return {"kind": "dataset", "mapping": m}
    return {"kind": "generic"}


LIMIT_CHOICES = (20, 100, 500, 1000, 2000, "max")


def limit_label(choice) -> str:
    return f"Maximum available (up to {MAX_UPLOAD_ROWS:,})" if choice == "max" else f"{int(choice):,}"


def loader_limit(choice, cap: int = MAX_UPLOAD_ROWS) -> int:
    """The ``max_rows`` value handed to ``load_dataset_reviews`` for a UI choice ("max" or a number)."""
    return cap if choice == "max" else min(int(choice), cap)


def planned_count(choice, available: int, cap: int = MAX_UPLOAD_ROWS) -> int:
    """How many reviews will actually be analyzed: the choice, never above the cap or what exists."""
    return max(0, min(loader_limit(choice, cap), int(available)))


def format_mapping(info: dict) -> str:
    """Human-readable 'source column -> internal field' summary."""
    internal = {"review_text": "review_text", "review_title": "review_title", "rating": "rating",
                "review_date": "review_date", "review_id": "review_id", "verified_purchase": "verified_purchase",
                "product_key": "product grouping key", "asins": "product_id (first ASIN)",
                "product_title": "product_title", "brand": "brand", "num_helpful": "num_helpful",
                "do_recommend": "do_recommend", "categories": "categories"}
    return "; ".join(f"`{src}` → {internal[f]}" for f, src in info["mapping"].items())
