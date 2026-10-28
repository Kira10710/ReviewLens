"""ReviewLens: Amazon review collection + aspect-based sentiment dashboard.

Launch: streamlit run app.py
Every number shown is computed from the records loaded in this session or read from
reports/*.json (written by src/evaluation.py and src/evaluate.py). Nothing is hard-coded.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src import ai_insights, config
from src.amazon_scraper import (ALLOWED_AMAZON_DOMAINS, MAX_PAGES_HARD_LIMIT, InvalidAmazonURL,
                                scrape_product_reviews, validate_amazon_url)
from src.pipeline import (analyze_reviews, aspect_breakdown, export_aspects, export_cleaned,
                          export_predictions, top_terms)
from src.dataset_loader import (LIMIT_CHOICES, SOURCE_DATASET, DatasetSchemaError, HeaderlessDatasetError,
                                classify_uploaded_csv, format_mapping, limit_label, load_dataset_reviews,
                                loader_limit, planned_count, scan_products)
from src.review_preprocessing import MAX_UPLOAD_ROWS, SOURCE_SCRAPE, SOURCE_UPLOAD, load_reviews_csv

try:  # charts are optional: the app must keep working without them
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:  # pragma: no cover
    PLOTLY_OK = False

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

SENT_COLORS = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#94a3b8"}
ACCENT = "#6d28d9"
PAGES = ["Overview", "Scrape Amazon Reviews", "Import Dataset", "Review Analysis", "Aspect Insights", "Data Export",
         "Model Evaluation"]
PAGE_ICONS = {"Overview": "dashboard", "Scrape Amazon Reviews": "travel_explore", "Import Dataset": "upload_file",
              "Review Analysis": "table_view", "Aspect Insights": "insights", "Data Export": "download",
              "Model Evaluation": "fact_check"}

st.markdown("""
<style>
:root {--rl-accent: #6d28d9; --rl-accent-soft: #f3efff; --rl-border: #e5e7eb; --rl-text: #1f2937; --rl-muted: #6b7280;}
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em; color: var(--rl-text);}
.rl-hero {background: linear-gradient(135deg, #f3efff 0%, #e8eefc 100%); border: 1px solid #ddd6fe;
  border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;}
.rl-hero h1 {margin: 0; font-size: 1.8rem; color: #2e1065;} .rl-hero p {margin: .35rem 0 0; color: #4b5563;}
.rl-tag {display: inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
  background: var(--rl-accent-soft); border: 1px solid #c4b5fd; color: #5b21b6; margin-right: .4rem;}
.rl-card {background: #ffffff; border: 1px solid var(--rl-border); border-radius: 12px; padding: .9rem 1.1rem;
  margin: .5rem 0; box-shadow: 0 1px 2px rgba(16, 24, 40, .05);}
.rl-card h4 {margin: 0 0 .35rem; font-size: 1rem; color: var(--rl-text);} .rl-muted {color: var(--rl-muted); font-size: .85rem;}
.rl-ai {border-left: 3px solid #7c3aed;} .rl-pro {border-left: 3px solid #10b981;} .rl-con {border-left: 3px solid #ef4444;}
.rl-ref {display: inline-block; padding: 0 .45rem; margin-right: .25rem; border-radius: 6px; font-size: .75rem;
  background: var(--rl-accent-soft); color: #5b21b6;}
div[data-testid="stMetric"] {background: #ffffff; border: 1px solid var(--rl-border); border-radius: 12px;
  padding: .7rem .9rem; box-shadow: 0 1px 2px rgba(16, 24, 40, .05);}

/* ---------------- sidebar ---------------- */
section[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid var(--rl-border);}
section[data-testid="stSidebar"] > div {padding-top: .5rem;}
.rl-brand {display: flex; align-items: center; gap: .65rem; padding: .35rem .25rem 1rem; border-bottom: 1px solid var(--rl-border);
  margin-bottom: 1rem;}
.rl-logo {width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #7c3aed, #4f46e5); color: #fff; font-weight: 700; font-size: .95rem; letter-spacing: .02em;
  box-shadow: 0 2px 6px rgba(109, 40, 217, .35);}
.rl-brand-name {font-size: 1.15rem; font-weight: 700; color: #111827; line-height: 1.15;}
.rl-brand-sub {font-size: .72rem; color: var(--rl-muted); line-height: 1.2;}
.rl-sec {font-size: .68rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #9ca3af;
  margin: .25rem .35rem .4rem;}
section[data-testid="stSidebar"] [role="radiogroup"] {gap: .15rem;}
section[data-testid="stSidebar"] [role="radiogroup"] label {width: 100%; padding: .5rem .7rem; border-radius: 9px;
  border-left: 3px solid transparent; cursor: pointer; transition: background .12s;}
section[data-testid="stSidebar"] [role="radiogroup"] label > div > div:first-child {display: none;}
section[data-testid="stSidebar"] [role="radiogroup"] label p {font-size: .92rem; font-weight: 500; color: #374151; margin: 0;}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: #f3f4f6;}
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {background: var(--rl-accent-soft);
  border-left-color: var(--rl-accent);}
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p {color: #4c1d95; font-weight: 650;}
.rl-status {border: 1px solid var(--rl-border); border-radius: 10px; padding: .55rem .7rem; margin-bottom: .4rem; background: #fafafa;}
.rl-status-row {display: flex; align-items: center; justify-content: space-between; font-size: .8rem; color: #374151; padding: .12rem 0;}
.rl-dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: .45rem; background: #9ca3af;}
.rl-dot.ok {background: #10b981;} .rl-dot.warn {background: #f59e0b;} .rl-dot.bad {background: #ef4444;}
.rl-foot {font-size: .7rem; color: #9ca3af; padding: .6rem .3rem 0; line-height: 1.4;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- state and helpers
def init_state():
    defaults = {"clean_df": None, "review_df": None, "aspect_df": None, "origin": None,
                "scrape_result": None, "csv_info": None, "scraping": False, "pending_url": "",
                "pending_pages": 3}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_resource(show_spinner="Loading models...")
def _load_analyzers_cached():
    from src.pipeline import load_analyzers
    return load_analyzers()


def get_models():
    """Return ((sentiment_model, aspect_analyzer), None) or (None, error message)."""
    try:
        return _load_analyzers_cached(), None
    except Exception as exc:
        try:
            _load_analyzers_cached.clear()
        except Exception:
            pass
        return None, str(exc)


def show_chart(fig, height=340):
    if not PLOTLY_OK:
        return
    try:
        fig.update_layout(template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:  # visual enhancements must never break the page
        st.info(f"Chart unavailable ({exc}).")


def analyze_and_store(clean_df: pd.DataFrame, origin: dict) -> bool:
    models, err = get_models()
    if models is None:
        st.error(f"Reviews were loaded but cannot be analyzed because a model is missing: {err}")
        st.session_state.clean_df, st.session_state.review_df, st.session_state.aspect_df = clean_df, None, None
        st.session_state.origin = origin
        return False
    with st.spinner(f"Analyzing {len(clean_df)} reviews..."):
        review_df, aspect_df = analyze_reviews(clean_df, *models)
    st.session_state.clean_df, st.session_state.review_df = clean_df, review_df
    st.session_state.aspect_df, st.session_state.origin = aspect_df, origin
    return True


def have_results() -> bool:
    df = st.session_state.review_df
    return df is not None and len(df) > 0


def empty_state(page_hint=True):
    st.info("No reviews loaded yet. Use **Scrape Amazon Reviews** (live scraping, or a small CSV upload) or "
            "**Import Dataset** (a downloaded review dataset such as a Datafiniti export). These are separate modes.")


def source_badge() -> str:
    o = st.session_state.origin or {}
    if o.get("kind") == SOURCE_SCRAPE:
        return '<span class="rl-tag">Live Amazon scrape</span>'
    if o.get("kind") == SOURCE_DATASET:
        return '<span class="rl-tag">Imported dataset (not live-scraped)</span>'
    if o.get("kind") == SOURCE_UPLOAD:
        return '<span class="rl-tag">User-uploaded CSV (not scraped)</span>'
    return ""


# ---------------------------------------------------------------- pages
def page_overview():
    st.markdown("""<div class="rl-hero"><h1>ReviewLens</h1>
<p>Paste an Amazon product URL, collect the customer reviews Amazon makes publicly accessible, then see
whole-review sentiment and <b>aspect-level</b> sentiment (battery, screen, price, ...) in one dashboard.
If Amazon blocks access, upload a CSV of reviews you obtained legitimately and use the same pipeline.</p></div>""",
                unsafe_allow_html=True)
    if not have_results():
        o = st.session_state.origin
        if o and st.session_state.clean_df is not None:
            st.warning("Reviews are loaded but not analyzed (model missing). See the error above or run "
                       "`python -m src.sentiment_model`.")
        else:
            empty_state()
        st.caption("Status: idle. Nothing has been collected in this session.")
        return
    df, asp, o = st.session_state.review_df, st.session_state.aspect_df, st.session_state.origin
    st.markdown(source_badge(), unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Reviews collected / uploaded", o.get("collected", len(df)))
    c[1].metric("Unique reviews retained", len(df))
    c[2].metric("Aspect mentions detected", len(asp))
    c[3].metric("Reviews with no known aspect", int((df["n_aspects"] == 0).sum()))
    pid = o.get("product_id")
    if o.get("kind") == SOURCE_DATASET:
        st.markdown(f"**Product:** {o.get('product_title', '-')}  \n**ASIN:** `{pid or '-'}`  \n"
                    f"**Dataset file:** `{o.get('filename', '-')}` (offline data, no source URL)")
        if o.get("available") is not None:
            st.caption(f"Analyzed {len(df):,} of {o['available']:,} available unique reviews in the selection "
                       "(limit chosen on the Import Dataset page; first reviews in file order).")
        if o.get("n_products", 1) > 1:
            st.warning(f"{o['n_products']} products are combined in this view; per-product comparisons are "
                       "not meaningful.")
    elif pid:
        st.markdown(f"**Product ID (ASIN):** `{pid}`  \n**Source URL:** {o.get('url', '-')}")
    else:
        st.markdown(f"**Source:** uploaded file `{o.get('filename', '-')}`")
    st.caption(f"Status: {o.get('status', 'loaded')}. Loaded {o.get('loaded_at', '')}.")
    counts = df["review_sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Whole-review sentiment (model predictions)")
        if PLOTLY_OK:
            fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=SENT_COLORS,
                         labels={"x": "predicted sentiment", "y": "reviews"})
            fig.update_layout(showlegend=False)
            show_chart(fig, 300)
        st.dataframe(counts.rename_axis("sentiment").reset_index(name="reviews"), hide_index=True)
    with right:
        st.subheader("Aspect-level sentiment")
        b = aspect_breakdown(asp)
        if b.empty:
            st.info("No known product aspects were detected in these reviews.")
        else:
            st.dataframe(b.head(8), hide_index=True, width="stretch")
    st.caption("Sentiment comes from a TF-IDF classifier trained on a public benchmark. Star ratings are shown "
               "separately and are never used as labels.")


def page_scrape():
    st.header("Scrape Amazon Reviews")
    tab_scrape, tab_csv = st.tabs(["Amazon product URL", "Upload CSV instead"])
    with tab_scrape:
        st.markdown(
            "ReviewLens requests only pages Amazon's `robots.txt` allows, identifies itself honestly, waits between "
            "requests, uses no login or cookies, and **stops at the first CAPTCHA, sign-in wall, access-denied "
            "or throttling response**. Amazon often serves only a handful of reviews on a product page and may "
            "restrict the full review list, so the number collected can be small or zero.")
        url = st.text_input("Amazon product URL", key="url_input", disabled=st.session_state.scraping,
                            placeholder="https://www.amazon.com/dp/B0XXXXXXXXX")
        pages = st.slider("Maximum pages to request (product page + review pages)", 1, MAX_PAGES_HARD_LIMIT, 3,
                          key="pages_input", disabled=st.session_state.scraping)
        try:
            if url.strip():
                v = validate_amazon_url(url)
                st.success(f"Valid Amazon URL. Product identifier (ASIN): `{v.asin}` on `{v.host}`")
        except InvalidAmazonURL as exc:
            st.error(str(exc))
        if st.button("Start scraping", type="primary", disabled=st.session_state.scraping or not url.strip()):
            st.session_state.scraping = True
            st.session_state.pending_url, st.session_state.pending_pages = url, pages
            st.rerun()
        if st.session_state.scraping:
            run_scrape(st.session_state.pending_url, st.session_state.pending_pages)
        show_last_scrape()
    with tab_csv:
        page_upload()


def run_scrape(url: str, pages: int):
    bar, msg = st.progress(0.0), st.empty()

    def progress(text, frac):
        bar.progress(min(max(frac, 0.0), 1.0))
        msg.write(text)

    try:
        with st.status("Scraping (will stop at the first block)...", expanded=True) as status:
            result = scrape_product_reviews(url, pages, progress=progress)
            status.update(label=f"Scrape finished: {result.status}",
                          state="complete" if result.ok else "error")
        st.session_state.scrape_result = result
        if result.ok:
            origin = {"kind": SOURCE_SCRAPE, "product_id": result.asin, "url": result.canonical_url,
                      "collected": result.stats["reviews_extracted"], "status": f"scrape {result.status}",
                      "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            analyze_and_store(result.reviews, origin)
    finally:
        st.session_state.scraping = False
        bar.empty()
        msg.empty()


def show_last_scrape():
    res = st.session_state.scrape_result
    if res is None:
        return
    st.subheader("Last scrape result")
    if res.ok:
        (st.success if res.status == "ok" else st.warning)(res.message)
    else:
        st.error(f"**{res.status.replace('_', ' ').title()}.** {res.message}")
        st.info("No reviews were fabricated or substituted. You can upload a CSV of reviews you obtained "
                "legitimately in the **Upload CSV instead** tab.")
    s = res.stats
    c = st.columns(5)
    c[0].metric("Pages attempted", s["pages_attempted"])
    c[1].metric("Pages parsed", s["pages_parsed"])
    c[2].metric("Reviews extracted", s["reviews_extracted"])
    c[3].metric("Duplicates removed", s["duplicates_removed"])
    c[4].metric("Reviews retained", s["retained"])
    if res.asin:
        st.markdown(f"Product ID (ASIN): `{res.asin}` · {res.canonical_url}")
    if res.pages:
        with st.expander("Page-by-page log"):
            st.dataframe(pd.DataFrame([p.__dict__ for p in res.pages]), hide_index=True, width="stretch")


def page_upload():
    st.markdown(f"Upload a CSV with a **`review_text`** column. Optional columns: `rating` (1-5), `review_title`, "
                f"`review_date`, `review_id`, `product_id`. Up to {MAX_UPLOAD_ROWS} rows are processed. "
                "These records are labelled **uploaded data, not scraped Amazon data**.")
    up = st.file_uploader("Reviews CSV", type=["csv"], key="csv_upload")
    col = st.text_input("Review text column name", value="review_text", key="csv_col")
    if up is not None:
        kind = classify_uploaded_csv(up.getvalue())
        if kind["kind"] in ("dataset", "headerless"):
            st.info("This file looks like a review-dataset export (for example `reviews.text` / `reviews.rating` "
                    "columns, or a header-less export). It is handled by the **dataset importer** below, which maps "
                    "the columns and lets you choose a product. It is not treated as live-scraped data.")
            render_dataset_import(up.getvalue(), (up.name, up.size), up.name, "scrape_upload")
            return
        if kind["kind"] == "error":
            st.error(kind["error"])
            return
    if up is not None and st.button("Analyze uploaded reviews", type="primary"):
        try:
            clean_df, info = load_reviews_csv(up.getvalue(), col, source_name=f"upload:{up.name}")
        except ValueError as exc:
            st.error(str(exc))
            return
        origin = {"kind": SOURCE_UPLOAD, "filename": up.name, "collected": info["rows_processed"],
                  "status": "uploaded CSV analyzed", "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        st.session_state.csv_info = info
        if analyze_and_store(clean_df, origin):
            st.success(f"Analyzed {info['retained']} unique reviews from {info['rows_in_file']} rows "
                       f"({info['rows_dropped_empty']} empty skipped, {info['duplicates_removed']} duplicates removed).")
    info = st.session_state.csv_info
    if info and (st.session_state.origin or {}).get("kind") == SOURCE_UPLOAD:
        st.caption(f"Loaded: {info}")


@st.cache_data(show_spinner="Scanning dataset (streaming, low memory)...")
def cached_scan(source, _fingerprint, infer_layout=False):
    return scan_products(source, infer_layout=infer_layout)


def page_import():
    st.header("Import Dataset")
    st.markdown(
        "**Separate from live scraping.** Use this mode for a review dataset you already have, such as a Datafiniti "
        "Amazon export. Records are tagged `dataset_import` and are never described as scraped. Column names are "
        "mapped automatically (for example `reviews.text` is the review text, `reviews.rating` the rating). "
        "Reviewer usernames and locations are never loaded.")
    folder = config.DATASET_DIR
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    src_choice = st.radio("Source", ["File in the Dataset folder", "Upload a CSV (max 20 MB)"], horizontal=True)
    source, fingerprint, label = None, None, None
    if src_choice.startswith("File"):
        if not files:
            st.warning(f"No CSV files found in `{folder}`. Put dataset files there, or use the upload option.")
        else:
            name = st.selectbox("Dataset file", [f.name for f in files],
                                format_func=lambda n: f"{n}  ({(folder / n).stat().st_size / 1e6:.0f} MB)")
            path = folder / name
            stt = path.stat()
            source, fingerprint, label = path, (stt.st_size, stt.st_mtime_ns), name
    else:
        up = st.file_uploader("Dataset CSV", type=["csv"], key="dataset_upload")
        if up is not None:
            source, fingerprint, label = up.getvalue(), up.name, up.name
    if source is None:
        return
    render_dataset_import(source, fingerprint, label, "folder" if src_choice.startswith("File") else "dataset_upload")


def render_dataset_import(source, fingerprint, label, key_prefix):
    """Shared import UI (Dataset folder, Import page upload, and dataset-style uploads on the Scrape page).

    Everything goes through src/dataset_loader.py, so source columns such as ``reviews.text`` are normalised to the
    internal ``review_text`` field and no raw dataset file is handed to the generic review_text loader.
    """
    infer = False
    try:
        products, info = cached_scan(source, fingerprint, False)
    except HeaderlessDatasetError as exc:
        st.error(f"Could not read this file: {exc}")
        if not exc.suggested_layout:
            return
        apply = st.checkbox(f"Apply the verified column layout '{exc.suggested_layout}' to this header-less file",
                            key=f"{key_prefix}_apply_layout")
        if not apply:
            return
        try:
            products, info = cached_scan(source, fingerprint, True)
        except (DatasetSchemaError, OSError) as exc2:
            st.error(f"Could not read this dataset: {exc2}")
            return
        infer = True
    except (DatasetSchemaError, OSError) as exc:
        st.error(f"Could not read this dataset: {exc}")
        return
    summary = (f"Encoding: {info['encoding']}; delimiter: `{info['delimiter']}`; {info['rows_scanned']:,} data rows; "
               f"{info['malformed_rows']:,} malformed rows skipped; {info['n_products']} product(s).")
    if info.get("layout_applied"):
        st.warning(f"No header row in the file. {info['header_evidence']}. {summary}")
    else:
        st.success(f"Header row verified ({info['header_evidence']}). {summary}")
    with st.expander("Column mapping used", expanded=False):
        st.markdown(format_mapping(info))
        st.caption("Source columns not listed are ignored.")
    labels = {r.product_key: f"{r.product_title[:70]}  |  {r.asin[:24]}  |  {r.unique_reviews:,} unique reviews"
              for r in products.itertuples()}
    if info["n_products"] > 1:
        st.markdown("**This dataset contains several products. Choose which one to analyze** (reviews of different "
                    "products are not mixed unless you explicitly add more).")
        st.dataframe(products[["product_title", "asin", "brand", "rows", "unique_reviews", "reviews_with_rating"]],
                     hide_index=True, width="stretch", height=240)
        pick = st.selectbox("Product", list(labels), index=None, format_func=lambda k: labels[k],
                            placeholder="Select a product...")
        extra = (st.multiselect("Add more products (optional)", [k for k in labels if k != pick],
                                format_func=lambda k: labels[k]) if pick else [])
    else:
        pick, extra = products.iloc[0]["product_key"], []
    keys = ([pick] + extra) if pick else []
    available = int(products[products["product_key"].isin(keys)]["unique_reviews"].sum()) if keys else 0
    choice = st.radio("Analysis limit (reviews to analyze)", list(LIMIT_CHOICES), index=2, horizontal=True,
                      format_func=limit_label, key="analysis_limit")
    will = planned_count(choice, available)
    m = st.columns(4)
    m[0].metric("Available (unique valid reviews)", f"{available:,}" if keys else "-")
    m[1].metric("Selected limit", limit_label(choice).replace("Maximum available ", "max "))
    m[2].metric("Will be analyzed", f"{will:,}" if keys else "-")
    m[3].metric("Safety cap", f"{MAX_UPLOAD_ROWS:,}")
    if keys and available > will:
        st.caption(f"{available - will:,} available reviews will not be analyzed with this limit. Reviews are taken in "
                   "file order, not randomly; duplicates and empty-text rows never count towards the limit.")
    elif keys:
        st.caption("All available reviews for this selection will be analyzed.")
    if st.button("Load and analyze", type="primary", disabled=not keys):
        try:
            df, linfo = load_dataset_reviews(source, keys, max_rows=loader_limit(choice), source_name=label,
                                             infer_layout=infer)
        except DatasetSchemaError as exc:
            st.error(str(exc))
            return
        first = products[products["product_key"] == pick].iloc[0]
        origin = {"kind": SOURCE_DATASET, "filename": label, "product_id": first["asin"].split(",")[0] or None,
                  "product_title": first["product_title"], "n_products": len(keys),
                  "collected": linfo["retained"], "available": available, "limit": choice,
                  "status": "imported dataset analyzed",
                  "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        st.session_state.csv_info = linfo
        if analyze_and_store(df, origin):
            note = " (limit reached; first reviews in file order)" if linfo["truncated"] else ""
            if linfo["retained"] != will:
                st.warning(f"Planned {will:,} reviews but loaded {linfo['retained']:,}; please report this.")
            st.success(f"Analyzed {linfo['retained']:,} of {available:,} available unique reviews{note}. "
                       f"{linfo['dropped_empty_text']} empty-text and "
                       f"{linfo['duplicates_removed']} duplicate rows removed; {linfo['retained_without_rating']} "
                       "without a rating (kept, rating left empty).")



def page_analysis():
    st.header("Review Analysis")
    if not have_results():
        empty_state()
        return
    df, asp = st.session_state.review_df, st.session_state.aspect_df
    st.markdown(source_badge(), unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([2, 1.2, 1.2, 1.5])
    query = f1.text_input("Search review text")
    sents = f2.multiselect("Whole-review sentiment", ["positive", "neutral", "negative"])
    ratings = f3.multiselect("Star rating", [1.0, 2.0, 3.0, 4.0, 5.0],
                             format_func=lambda r: f"{r:.0f} stars")
    aspects = f4.multiselect("Detected aspect", sorted(asp["aspect"].unique()) if len(asp) else [])
    view = df
    if query:
        view = view[view["model_text"].str.contains(query, case=False, regex=False, na=False)]
    if sents:
        view = view[view["review_sentiment"].isin(sents)]
    if ratings:
        view = view[view["rating"].isin(ratings)]
    if aspects:
        ids = set(asp[asp["aspect"].isin(aspects)]["review_id"])
        view = view[view["review_id"].isin(ids)]
    st.caption(f"{len(view)} of {len(df)} reviews")
    cols = ["review_id", "review_title", "cleaned_text", "rating", "review_date", "review_sentiment",
            "aspect_summary", "final_label"]
    st.dataframe(view[cols], hide_index=True, width="stretch", height=360)
    if view.empty:
        return
    st.subheader("Review detail")
    rid = st.selectbox("Select a review", list(view["review_id"]),
                       format_func=lambda r: f"{r}: {str(view.loc[view['review_id'] == r, 'model_text'].iloc[0])[:70]}")
    row = view[view["review_id"] == rid].iloc[0]
    st.markdown("**Original text**")
    st.write(((row["review_title"] + " - ") if row["review_title"] else "") + str(row["review_text"] or ""))
    c = st.columns(4)
    c[0].metric("Star rating (reviewer)", "-" if pd.isna(row["rating"]) or row["rating"] is None else f"{row['rating']:.0f}")
    conf = row["review_sentiment_confidence"]
    c[1].metric("Whole-review sentiment", row["review_sentiment"])
    c[2].metric("Aspect-based overall", row["aspect_overall"])
    c[3].metric("Final label", row["final_label"])
    if row["review_sentiment_score_type"] == "probability" and conf is not None and not pd.isna(conf):
        st.caption(f"Whole-review class probability: {conf:.2f} (raw model output, not a calibrated confidence). "
                   f"Final label basis: {row['final_label_basis']}.")
    a = asp[asp["review_id"] == rid]
    if a.empty:
        st.info("No known product aspect was detected in this review, so the final label falls back to the "
                "whole-review classifier.")
    else:
        st.dataframe(a[["aspect", "matched_text", "sentiment", "confidence", "aspect_context"]],
                     hide_index=True, width="stretch")


def page_insights():
    st.header("Aspect Insights")
    if not have_results():
        empty_state()
        return
    df, asp = st.session_state.review_df, st.session_state.aspect_df
    st.markdown(source_badge(), unsafe_allow_html=True)
    st.info("Charts are computed from the reviews loaded in this session and are **model predictions**, not ground "
            "truth. Aspect detection is rule-based vocabulary matching.")
    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment distribution")
        counts = df["review_sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        if PLOTLY_OK:
            show_chart(px.pie(names=counts.index, values=counts.values, hole=0.5, color=counts.index,
                              color_discrete_map=SENT_COLORS), 300)
    b = aspect_breakdown(asp)
    with right:
        st.subheader("Most mentioned aspects")
        if b.empty:
            st.info("No known aspects were detected.")
        elif PLOTLY_OK:
            show_chart(px.bar(b.head(10), x="total", y="aspect", orientation="h",
                              color_discrete_sequence=[ACCENT]).update_yaxes(autorange="reversed"), 300)
    if b.empty:
        render_ai_insights()
        return
    st.subheader("Aspect-level sentiment breakdown")
    if PLOTLY_OK:
        long = b.melt(id_vars=["aspect", "total"], value_vars=["positive", "neutral", "negative"],
                      var_name="sentiment", value_name="mentions")
        show_chart(px.bar(long, y="aspect", x="mentions", color="sentiment", orientation="h",
                          color_discrete_map=SENT_COLORS, category_orders={"aspect": list(b["aspect"])}),
                   max(300, 32 * len(b) + 80))
    st.subheader("Most frequently negative aspects")
    neg = b[b["negative"] > 0].sort_values(["negative", "total"], ascending=False).copy()
    neg["negative_share"] = (neg["negative"] / neg["total"]).round(2)
    st.dataframe(neg[["aspect", "negative", "total", "negative_share"]].head(10), hide_index=True, width="stretch")

    st.subheader("Inspect the reviews behind a chart")
    c = st.columns(2)
    pick_a = c[0].selectbox("Aspect", list(b["aspect"]))
    pick_s = c[1].selectbox("Sentiment", ["negative", "neutral", "positive"])
    sel = asp[(asp["aspect"] == pick_a) & (asp["sentiment"] == pick_s)]
    st.caption(f"{len(sel)} mention(s)")
    if len(sel):
        shown = sel.merge(df[["review_id", "cleaned_text", "rating"]], on="review_id", how="left")
        st.dataframe(shown[["review_id", "matched_text", "aspect_context", "rating", "cleaned_text"]],
                     hide_index=True, width="stretch")

    st.subheader("Common terms in negative vs positive reviews")
    st.caption("Plain word/phrase frequency within reviews of each predicted sentiment, after stop-word removal. "
               "This is **not** topic modelling.")
    t1, t2 = st.columns(2)
    for col, label in ((t1, "negative"), (t2, "positive")):
        with col:
            st.markdown(f"**{label.title()} reviews**")
            terms = top_terms(df, label)
            if terms.empty:
                st.write("Not enough reviews for repeated terms.")
            else:
                st.dataframe(terms, hide_index=True, width="stretch")
    term = st.text_input("Show reviews containing a term")
    if term:
        hit = df[df["model_text"].str.contains(term, case=False, regex=False, na=False)]
        st.caption(f"{len(hit)} review(s)")
        st.dataframe(hit[["review_id", "cleaned_text", "rating", "review_sentiment"]], hide_index=True, width="stretch")
    render_ai_insights()


def gemini_key() -> str | None:
    """GEMINI_API_KEY from the environment or Streamlit secrets (never displayed)."""
    try:
        secrets = st.secrets
    except Exception:
        secrets = None
    return ai_insights.get_api_key(secrets)


def gemini_diagnostics() -> dict:
    """Key detection report for THIS app process (never contains the key)."""
    try:
        secrets = st.secrets
    except Exception:
        secrets = None
    try:
        port = st.get_option("server.port")
    except Exception:
        port = None
    return ai_insights.key_diagnostics(secrets, port=port)


def render_gemini_check():
    d = gemini_diagnostics()
    with st.expander("Gemini configuration check (this app instance)"):
        p = d["process"]
        st.markdown(f"**Key detected:** {'yes' if d['found'] else 'no'}"
                    + (f" · **source:** {d['source']}" if d["found"] else "")
                    + (f" · length {d['key_length']} characters" if d["found"] else ""))
        st.caption(f"This page is served by process {p['pid']} on port {p['port']}, started at {p['started']} "
                   f"from `{p['cwd']}`. Environment variables set after that time are not visible to it. "
                   "If you configured the key in another terminal, make sure you are viewing the instance you started "
                   "there (check the port in that terminal's 'Local URL').")
        st.markdown("Checked, in order: " + ", ".join(d["checked_in_order"]) + ".")
        st.dataframe(pd.DataFrame(d["secrets_files"]), hide_index=True, width="stretch")
        for problem in d["problems"]:
            st.warning(problem)
        st.caption("The key itself is never displayed or logged.")


def _esc(text) -> str:
    return html.escape(str(text or ""), quote=True)


def _refs(ids) -> str:
    return "".join(f'<span class="rl-ref">{_esc(i)}</span>' for i in ids)


def render_ai_insights():
    """Optional Gemini summary grounded in the already-computed analysis (no models are re-run)."""
    df, asp, origin = st.session_state.review_df, st.session_state.aspect_df, st.session_state.origin or {}
    st.divider()
    st.subheader("AI Insights (Gemini)")
    st.caption("An optional natural-language summary written by Gemini from a small, representative set of review "
               "excerpts plus ReviewLens's own statistics. **When you click Generate, the selected review excerpts "
               "(text, star rating, ReviewLens sentiment and aspect labels; no usernames, locations or links) are sent "
               "to Google's Gemini API.** The existing sentiment, aspect and export features do not use Gemini.")
    key = gemini_key()
    multi = ai_insights.check_single_product(df)
    if origin.get("n_products", 1) > 1 and not multi:
        multi = "Several products were combined when loading. AI Insights summarize one product at a time."
    settings = config.ai_settings()
    with st.expander("Evidence settings"):
        n_ex = st.number_input("Maximum review excerpts sent to Gemini", 6, 60, min(max(settings["max_excerpts"], 6), 60),
                               step=2, key="ai_max_excerpts",
                               help="Lower values use less API quota. Excerpts are chosen across aspects and sentiments.")
        st.caption(f"Model: `{settings['model']}` (set GEMINI_MODEL to change) · excerpt length limit "
                   f"{settings['excerpt_chars']} characters · input size limit {settings['max_input_chars']:,} characters.")
    label = origin.get("product_title") or origin.get("product_id") or origin.get("filename") or ""
    evidence = ai_insights.build_evidence(df, asp, product_label=label, source_kind=origin.get("kind", "dataset_import"),
                                          max_excerpts=int(n_ex))
    lim = evidence.limits
    c = st.columns(4)
    c[0].metric("Reviews analyzed", f"{lim['reviews_analyzed']:,}")
    c[1].metric("Excerpts sent to Gemini", f"{lim['excerpts_supplied']:,}")
    c[2].metric("Excerpts shortened", lim.get("excerpts_truncated", 0))
    c[3].metric("Dropped for size limit", lim.get("excerpts_dropped_for_size", 0))
    st.caption(f"Limits in force: at most {lim.get('max_excerpts', '-')} excerpts, {lim.get('excerpt_chars', '-')} characters "
               f"each, {lim.get('max_input_chars', 0):,} characters in total. Only these excerpts are sent, not the dataset.")
    if not key:
        st.warning("Gemini is unavailable: no usable `GEMINI_API_KEY` found in the environment or Streamlit secrets. "
                   "Open **Gemini configuration check** below to see what this app instance checked. Everything else "
                   "works without it.")
    render_gemini_check()
    if multi:
        st.warning(multi)
    fp = evidence.fingerprint(settings["model"]) if not evidence.empty else None
    cache = st.session_state.setdefault("ai_cache", {})
    cached = fp in cache if fp else False
    clicked = st.button("Regenerate AI Insights" if cached else "Generate AI Insights", type="primary",
                        disabled=not key or bool(multi) or evidence.empty, key="ai_generate")
    if clicked:
        with st.spinner("Asking Gemini to summarize the selected excerpts..."):
            result = ai_insights.get_or_generate(cache, evidence, key, model=settings["model"], force=cached)
        st.session_state.ai_last = {"fp": fp, "result": result}
    last = st.session_state.get("ai_last")
    if not last or last["fp"] != fp:
        if cached and not clicked:
            last = {"fp": fp, "result": cache[fp]}
        elif last and last["fp"] != fp:
            st.info("The reviews or settings changed since the last summary, so it was cleared. Generate again for the "
                    "current selection.")
            return
        else:
            return
    result = last["result"]
    if not result.ok:
        (st.warning if result.status in ("quota", "timeout", "network", "model_unavailable") else st.error)(result.message)
        return
    render_ai_result(result, evidence, origin)


def render_ai_result(result, evidence, origin):
    ins = result.insights
    src = ai_insights.SOURCE_DESCRIPTIONS.get(origin.get("kind", ""), "the analyzed reviews")
    st.markdown(f'<span class="rl-tag">AI-generated by Gemini ({_esc(result.model)})</span>'
                f'<span class="rl-tag">{evidence.limits["reviews_analyzed"]:,} reviews analyzed</span>'
                f'<span class="rl-tag">{evidence.limits["excerpts_supplied"]} excerpts supplied</span>'
                f'<span class="rl-tag">{_esc(src)}</span>', unsafe_allow_html=True)
    st.caption("AI-generated insights based on the analyzed reviews. They can contain mistakes; check the cited "
               "excerpts. Statistics below the summary are computed by ReviewLens, not by Gemini."
               + (" (Cached result for identical reviews and settings.)" if result.from_cache else ""))
    for w in result.warnings:
        st.warning(w)
    st.markdown(f'<div class="rl-card rl-ai"><h4>Overall customer consensus</h4>{_esc(ins["consensus"])}</div>',
                unsafe_allow_html=True)
    if ins["aspects"]:
        st.markdown("**Aspect-wise findings**")
        for a in ins["aspects"]:
            st.markdown(f'<div class="rl-card"><h4>{_esc(a["aspect"])} <span class="rl-muted">· evidence: '
                        f'{_esc(a["support"])}</span></h4>'
                        + (f'<div><b>Praised:</b> {_esc(a["praised"])}</div>' if a["praised"] else "")
                        + (f'<div><b>Criticized:</b> {_esc(a["criticized"])}</div>' if a["criticized"] else "")
                        + f'<div style="margin-top:.35rem">{_refs(a["evidence_ids"])}</div></div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    for col, title, items, cls in ((left, "Pros", ins["pros"], "rl-pro"), (right, "Cons", ins["cons"], "rl-con")):
        with col:
            st.markdown(f"**{title}**")
            if not items:
                st.caption("No evidence-supported points.")
            for it in items:
                st.markdown(f'<div class="rl-card {cls}">{_esc(it["point"])}<div style="margin-top:.3rem">'
                            f'{_refs(it["evidence_ids"])}</div></div>', unsafe_allow_html=True)
    rec = ins["recommendation"]
    st.markdown('<div class="rl-card rl-ai"><h4>Buying recommendation</h4>'
                + (f'<div><b>May suit:</b> {_esc(rec["suited_for"])}</div>' if rec["suited_for"] else "")
                + (f'<div><b>Consider alternatives if:</b> {_esc(rec["consider_alternatives_if"])}</div>'
                   if rec["consider_alternatives_if"] else "") + "</div>", unsafe_allow_html=True)
    conf = ins["confidence"]
    st.markdown(f'<div class="rl-card"><h4>Confidence and evidence: {_esc(conf["level"])}</h4>{_esc(conf["rationale"])}'
                + "".join(f'<div class="rl-muted">Conflicting view: {_esc(c)}</div>' for c in conf["conflicting_opinions"])
                + "</div>", unsafe_allow_html=True)
    with st.expander("Statistics computed by ReviewLens (not by Gemini)"):
        s = evidence.stats
        st.dataframe(pd.DataFrame([{"sentiment": k, "reviews": v, "percent": s["whole_review_sentiment_percent"].get(k)}
                                   for k, v in s["whole_review_sentiment_counts"].items()]), hide_index=True)
        if s["aspects_detected"]:
            st.dataframe(pd.DataFrame(s["aspects_detected"]), hide_index=True, width="stretch")
    with st.expander("Evidence: review excerpts supplied to Gemini"):
        st.dataframe(pd.DataFrame([{"id": e["id"], "whole-review sentiment": e["whole_review_sentiment"],
                                    "stars": e["star_rating"], "aspects": "; ".join(f"{a['aspect']}: {a['sentiment']}"
                                                                                  for a in e["aspects"]),
                                    "text": e["text"]} for e in evidence.excerpts]), hide_index=True, width="stretch")


def page_export():
    st.header("Data Export")
    if not have_results():
        empty_state()
        return
    df, asp = st.session_state.review_df, st.session_state.aspect_df
    st.markdown(source_badge(), unsafe_allow_html=True)
    st.caption("Exports contain only the records loaded in this session. The `source` column marks each row as "
               f"`{SOURCE_SCRAPE}` or `{SOURCE_UPLOAD}`. Scraped files can contain public review text; handle and "
               "share them in line with Amazon's terms and applicable privacy rules.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    c = st.columns(3)
    c[0].download_button("Cleaned reviews CSV", export_cleaned(df), f"reviewlens_cleaned_{stamp}.csv", "text/csv")
    c[1].download_button("Review-level predictions CSV", export_predictions(df),
                         f"reviewlens_review_sentiment_{stamp}.csv", "text/csv")
    c[2].download_button("Aspect-level predictions CSV", export_aspects(asp),
                         f"reviewlens_aspect_sentiment_{stamp}.csv", "text/csv")
    st.caption(f"Preview: first {min(20, len(df))} of {len(df):,} analyzed reviews. The CSV downloads above "
               f"contain all {len(df):,}.")
    st.dataframe(df[["review_id", "source", "cleaned_text", "review_sentiment", "aspect_summary"]].head(20),
                 hide_index=True, width="stretch")


def page_evaluation():
    st.header("Model Evaluation")
    st.info("Two separate benchmarks. **Neither** measures end-to-end aspect extraction on Amazon reviews, and none "
            "of your scraped/uploaded reviews are used here.")
    rev = load_json(config.REPORTS_DIR / "review_sentiment_metrics.json")
    st.subheader("1. Whole-review sentiment")
    if not rev:
        st.warning("Not evaluated yet. Run `python -m src.sentiment_model` then `python -m src.evaluation`.")
    else:
        st.caption(f"{rev['model']} · official held-out test split of {rev['test_examples']} reviews · "
                   f"trained on {rev['train_examples']} · {rev['label_caveat']}")
        c = st.columns(4)
        c[0].metric("Accuracy", f"{rev['accuracy']:.3f}")
        c[1].metric("Macro F1", f"{rev['macro_f1']:.3f}")
        c[2].metric("Weighted F1", f"{rev['weighted_f1']:.3f}")
        c[3].metric("Majority baseline acc.", f"{rev['majority_baseline_accuracy']:.3f}")
        left, right = st.columns(2)
        with left:
            labels = ["negative", "neutral", "positive"]
            if PLOTLY_OK:
                fig = go.Figure(go.Heatmap(z=rev["confusion_matrix"], x=labels, y=labels, text=rev["confusion_matrix"],
                                           texttemplate="%{text}", colorscale=[[0, "#f5f3ff"], [1, ACCENT]],
                                           showscale=False))
                fig.update_xaxes(title="predicted").update_yaxes(title="true", autorange="reversed")
                show_chart(fig, 320)
        with right:
            st.dataframe(pd.DataFrame(rev["per_class"]).T.round(4).reset_index(names="class"), hide_index=True)
            st.caption(f"Inference {rev['inference_ms_per_review']:.3f} ms/review; fit "
                       f"{(rev.get('fit_seconds') or 0):.1f}s. Neutral (3-star) is the weakest class.")
    asp = load_json(config.METRICS_FILE)
    st.subheader("2. Aspect sentiment (gold aspect terms, SemEval-2014 laptops)")
    if not asp:
        st.warning("Not evaluated yet. Run `python -m src.train` then `python -m src.evaluate`.")
    else:
        m = asp["models"][asp["selected_model"]]
        st.caption(f"{m['display_name']} · {asp['test_examples']} test aspect terms · gold aspects supplied, so this "
                   "measures aspect sentiment classification only, in the **laptop** domain.")
        c = st.columns(3)
        c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        c[1].metric("Macro F1", f"{m['macro_f1']:.3f}")
        c[2].metric("Weighted F1", f"{m['weighted_f1']:.3f}")


# ---------------------------------------------------------------- main
init_state()
def sidebar_status():
    """Compact status panel: models, Gemini, and what data is currently loaded."""
    _, model_err = get_models()
    models = ("bad", "Models: error") if model_err else ("ok", "Models: ready")
    _gd = gemini_diagnostics()
    gemini = ("ok", f"Gemini: configured ({_gd['source']})") if _gd["found"] else ("warn", "Gemini: off (optional)")
    rows = [models, gemini]
    o, df = st.session_state.origin or {}, st.session_state.review_df
    if df is not None and len(df):
        kind = {"amazon_scrape": "Live scrape", "dataset_import": "Imported dataset", "csv_upload": "Uploaded CSV"}
        rows.append(("ok", f"Data: {len(df):,} reviews"))
        rows.append(("ok", kind.get(o.get("kind"), "Source loaded")))
    else:
        rows.append(("warn", "Data: none loaded"))
    html_rows = "".join(f'<div class="rl-status-row"><span><span class="rl-dot {c}"></span>{_esc(t)}</span></div>'
                        for c, t in rows)
    st.markdown(f'<div class="rl-sec">Status</div><div class="rl-status">{html_rows}</div>', unsafe_allow_html=True)
    if model_err:
        st.error(f"Model problem: {model_err}")
        with st.expander("Model diagnostics", expanded=False):
            from src.sentiment_model import model_diagnostics
            d = model_diagnostics()
            st.markdown(f"**CWD:** `{d['cwd']}`  \n"
                        f"**Root:** `{d['root']}`  \n"
                        f"**Model path:** `{d['model_file']}`  \n"
                        f"**File exists:** `{'yes' if d['model_file_exists'] else 'no'}`")
            if d.get("model_file_size"):
                st.caption(f"File size: {d['model_file_size']:,} bytes")
            if d.get("files_in_models_dir"):
                st.caption(f"Files in models/: {', '.join(d['files_in_models_dir'])}")
            if st.button("Retry loading models", key="retry_models_btn"):
                try:
                    _load_analyzers_cached.clear()
                except Exception:
                    pass
                st.rerun()


with st.sidebar:
    st.markdown('<div class="rl-brand"><div class="rl-logo">RL</div><div><div class="rl-brand-name">ReviewLens</div>'
                '<div class="rl-brand-sub">Review intelligence</div></div></div>'
                '<div class="rl-sec">Workspace</div>', unsafe_allow_html=True)
    page = st.radio("Section", PAGES, label_visibility="collapsed",
                    format_func=lambda p: f":material/{PAGE_ICONS[p]}: {p}")
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    sidebar_status()
    st.markdown('<div class="rl-foot">Aspect extraction is rule-based. Allowed storefronts: '
                + _esc(", ".join(ALLOWED_AMAZON_DOMAINS[:4])) + ", ...</div>", unsafe_allow_html=True)

{"Overview": page_overview, "Scrape Amazon Reviews": page_scrape, "Import Dataset": page_import,
 "Review Analysis": page_analysis,
 "Aspect Insights": page_insights, "Data Export": page_export, "Model Evaluation": page_evaluation}[page]()

st.markdown("---")
st.caption("ReviewLens · aspect extraction is rule-based; aspect sentiment uses a model trained on SemEval-2014 laptop "
           "reviews; whole-review sentiment uses a model trained on star-derived Amazon benchmark labels.")
