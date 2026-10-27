"""End-to-end analysis: cleaned review records -> sentiment + aspects -> export frames.

    records (scraped or uploaded) -> review_preprocessing -> whole-review sentiment
                                                          -> aspect extraction (rules)
                                                          -> aspect sentiment (trained)
                                                          -> review table, aspect table
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from src.aspect_sentiment import ASPECT_COLUMNS, AspectSentimentAnalyzer, empty_aspect_frame
from src.review_preprocessing import build_model_text
from src.sentiment_model import ReviewSentimentModel

NO_ASPECTS = "No aspects detected"


def load_analyzers():
    """Load both trained models. Raises ModelNotAvailable-style errors with clear messages."""
    from src.sentiment_model import load_review_model
    return load_review_model(), AspectSentimentAnalyzer()


def analyze_reviews(reviews: pd.DataFrame, sentiment_model: ReviewSentimentModel,
                    aspect_analyzer: AspectSentimentAnalyzer) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze a DataFrame in the internal schema.

    Returns (review_df, aspect_df). ``review_df`` keeps every input column and adds:
    model_text, review_sentiment, review_sentiment_confidence, review_sentiment_score_type,
    aspect_summary, n_aspects, aspect_overall, final_label, final_label_basis.
    Whole-review sentiment is the whole-review classifier; ``aspect_overall`` aggregates the
    aspect-level labels only; ``final_label`` is ``aspect_overall`` when aspects were found and
    otherwise falls back to the whole-review label (``final_label_basis`` says which).
    The star ``rating`` column is passed through untouched and never used as a label.
    """
    if reviews is None or reviews.empty:
        return pd.DataFrame(), empty_aspect_frame()
    df = reviews.reset_index(drop=True).copy()
    texts = [build_model_text(t, b) for t, b in zip(df["review_title"], df["cleaned_text"])]
    df["model_text"] = texts
    preds = sentiment_model.predict(texts)
    df["review_sentiment"] = [p["label"] for p in preds]
    df["review_sentiment_confidence"] = [p["confidence"] for p in preds]
    df["review_sentiment_score_type"] = [p["score_type"] for p in preds]

    aspect_rows, summaries, counts, overall = [], [], [], []
    for row, text in zip(df.itertuples(index=False), texts):
        rows = aspect_analyzer.analyze_review(row.review_id, text, row.product_id, row.source)
        aspect_rows.extend(rows)
        summaries.append("; ".join(f"{r['aspect']}: {r['sentiment']}" for r in rows))
        counts.append(len(rows))
        overall.append(aspect_analyzer.aspect_overall([r["sentiment"] for r in rows]))
    df["aspect_summary"] = summaries
    df["n_aspects"] = counts
    df["aspect_overall"] = overall
    has = df["n_aspects"] > 0
    df["final_label"] = df["aspect_overall"].where(has, df["review_sentiment"].str.capitalize())
    df["final_label_basis"] = ["aspects" if h else "whole-review classifier (no aspect detected)" for h in has]
    aspects = pd.DataFrame(aspect_rows, columns=ASPECT_COLUMNS) if aspect_rows else empty_aspect_frame()
    return df, aspects


# ---------------------------------------------------------------------------- exports
CLEANED_EXPORT_COLUMNS = ["review_id", "product_id", "product_url", "source_url", "review_title",
                          "review_text", "cleaned_text", "rating", "review_date", "review_date_raw",
                          "verified_purchase", "source", "scraped_at"]
REVIEW_EXPORT_COLUMNS = ["review_id", "product_id", "source", "review_title", "cleaned_text", "rating",
                         "review_date", "review_sentiment", "review_sentiment_confidence",
                         "review_sentiment_score_type", "aspect_summary", "n_aspects", "aspect_overall",
                         "final_label", "final_label_basis"]


def export_cleaned(review_df: pd.DataFrame) -> bytes:
    return review_df.reindex(columns=CLEANED_EXPORT_COLUMNS).to_csv(index=False).encode("utf-8")


def export_predictions(review_df: pd.DataFrame) -> bytes:
    return review_df.reindex(columns=REVIEW_EXPORT_COLUMNS).to_csv(index=False).encode("utf-8")


def export_aspects(aspect_df: pd.DataFrame) -> bytes:
    return aspect_df.reindex(columns=ASPECT_COLUMNS).to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------- frequency themes
_STOP = set("""a an the and or but if so of to in on at for with from by as is are was were be been being it its it's
this that these those i me my we our you your he she they them their there here have has had do does did done
will would can could should may might just very really also too than then when what which who whom how why
about into over under out up down off again more most some any all each other such only own same both few
get got gets one two product item bought buy use used using like im ive dont doesnt didnt""".split())
_TOKEN = re.compile(r"[a-z][a-z']+")


def top_terms(review_df: pd.DataFrame, sentiment: str, n: int = 15, min_count: int = 2) -> pd.DataFrame:
    """Most frequent unigrams/bigrams in reviews with the given whole-review sentiment.

    Plain frequency counts after stop-word removal. This is NOT topic modelling.
    """
    if review_df is None or review_df.empty:
        return pd.DataFrame(columns=["term", "reviews", "reviews_pct"])
    sub = review_df[review_df["review_sentiment"] == sentiment]
    counter: Counter = Counter()
    for text in sub["model_text"]:
        toks = [t for t in _TOKEN.findall(str(text).lower().replace("’", "'"))]
        grams = {t for t in toks if t not in _STOP and len(t) > 2}
        grams |= {f"{a} {b}" for a, b in zip(toks, toks[1:]) if a not in _STOP and b not in _STOP}
        counter.update(grams)
    rows = [(t, c) for t, c in counter.most_common() if c >= min_count][:n]
    out = pd.DataFrame(rows, columns=["term", "reviews"])
    out["reviews_pct"] = (100 * out["reviews"] / max(len(sub), 1)).round(1)
    return out


def aspect_breakdown(aspect_df: pd.DataFrame) -> pd.DataFrame:
    """Counts of positive / negative / neutral mentions per aspect."""
    if aspect_df is None or aspect_df.empty:
        return pd.DataFrame(columns=["aspect", "negative", "neutral", "positive", "total"])
    t = aspect_df.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
    t = t.reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    t["total"] = t.sum(axis=1)
    return t.sort_values("total", ascending=False).reset_index()
