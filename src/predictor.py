"""End-to-end prediction: rule-based aspect extraction -> trained aspect sentiment model.

For every extracted aspect the classifier receives the sentence and the
aspect's character offsets, i.e. exactly the same input format as during
training on gold aspects. The whole-review sentiment is never used as a
substitute for aspect sentiment; the overall label is derived from the
aspect-level predictions only.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.aspect_extractor import AspectExtractor
from src.feature_builder import scores_for
from src.preprocessing import local_context, normalize_text

DEFAULT_TEXT_COLUMN = "review_text"
MAX_REVIEW_CHARS = 5000
MAX_BATCH_ROWS = 5000


class ModelNotAvailable(RuntimeError):
    pass


def selected_model_name() -> str:
    if not config.SELECTED_MODEL_FILE.exists():
        raise ModelNotAvailable("No trained model found. Run `python -m src.train` first.")
    with open(config.SELECTED_MODEL_FILE, encoding="utf-8") as f:
        return json.load(f)["model"]


def load_model(name: str | None = None, path: str | Path | None = None):
    target = Path(path) if path else config.MODEL_FILES[name or selected_model_name()]
    if not target.exists():
        candidates = [
            config.ROOT / target if not target.is_absolute() else target,
            Path.cwd() / "models" / target.name,
            config.MODELS_DIR / target.name,
        ]
        for c in candidates:
            if c.exists():
                target = c.resolve()
                break
    if not target.exists():
        raise ModelNotAvailable(f"Model file not found: {target}. Run `python -m src.train`.")
    return joblib.load(target)


def overall_label(sentiments: list[str]) -> str:
    """Positive / Negative / Neutral / Mixed from aspect-level labels."""
    s = set(sentiments)
    if not s:
        return "No aspects detected"
    if "positive" in s and "negative" in s:
        return "Mixed"
    if "positive" in s:
        return "Positive"
    if "negative" in s:
        return "Negative"
    return "Neutral"


class ReviewPredictor:
    def __init__(self, model=None, model_name: str | None = None,
                 extractor: AspectExtractor | None = None):
        self.model_name = model_name or (None if model is not None else selected_model_name())
        self.model = model if model is not None else load_model(self.model_name)
        self.extractor = extractor or AspectExtractor()
        self.classes = list(self.model.classes_)
        ctx = self.model.named_steps["context"]
        self._ctx_params = dict(window=ctx.window, clip_clause=ctx.clip_clause)

    def analyze_review(self, text) -> dict:
        clean = normalize_text(text)
        result = {"review": clean, "aspects": [], "overall": "No aspects detected",
                  "n_aspects": 0, "message": ""}
        if not clean:
            result["message"] = "The review is empty."
            return result
        if len(clean) > MAX_REVIEW_CHARS:
            clean = clean[:MAX_REVIEW_CHARS]
            result["review"] = clean
            result["message"] = f"Review truncated to {MAX_REVIEW_CHARS} characters. "
        matches = self.extractor.extract(clean)
        if not matches:
            result["message"] += ("No known product aspect was found. The extractor only recognises "
                                  "phrases in config/aspect_vocabulary.json (e.g. battery, screen, "
                                  "keyboard, price).")
            return result

        X = pd.DataFrame([{"sentence": m.sentence, "aspect": m.matched_text,
                           "start": m.start, "end": m.end} for m in matches])
        preds = self.model.predict(X)
        scores, score_type = scores_for(self.model, X)
        for m, pred, row in zip(matches, preds, np.atleast_2d(scores)):
            _, display_ctx = local_context(m.sentence, m.start, m.end, **self._ctx_params)
            score_map = {c: float(v) for c, v in zip(self.classes, row)}
            result["aspects"].append({
                "aspect": m.aspect,
                "matched_text": m.matched_text,
                "sentence": m.sentence.strip(),
                "context": display_ctx,
                "sentiment": str(pred),
                "score_type": score_type,
                "confidence": score_map[str(pred)] if score_type == "probability" else None,
                "scores": score_map,
            })
        result["n_aspects"] = len(result["aspects"])
        result["overall"] = overall_label([a["sentiment"] for a in result["aspects"]])
        return result

    def analyze_batch(self, reviews: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (review-level DataFrame, aspect-level DataFrame)."""
        review_rows, aspect_rows = [], []
        for idx, text in enumerate(reviews.tolist()):
            res = self.analyze_review(text)
            counts = pd.Series([a["sentiment"] for a in res["aspects"]]).value_counts()
            review_rows.append({
                "review_id": idx, "review_text": res["review"], "overall": res["overall"],
                "n_aspects": res["n_aspects"],
                "aspects": "; ".join(f"{a['aspect']}:{a['sentiment']}" for a in res["aspects"]),
                "n_positive": int(counts.get("positive", 0)),
                "n_negative": int(counts.get("negative", 0)),
                "n_neutral": int(counts.get("neutral", 0)),
                "note": res["message"].strip(),
            })
            for a in res["aspects"]:
                aspect_rows.append({"review_id": idx, "aspect": a["aspect"],
                                    "matched_text": a["matched_text"], "sentiment": a["sentiment"],
                                    "confidence": a["confidence"], "score_type": a["score_type"],
                                    "context": a["context"]})
        aspect_cols = ["review_id", "aspect", "matched_text", "sentiment", "confidence",
                       "score_type", "context"]
        return pd.DataFrame(review_rows), pd.DataFrame(aspect_rows, columns=aspect_cols)

    def analyze_csv(self, file, text_column: str | None = None):
        """Validate an uploaded CSV and analyze it.

        Returns (review_df, aspect_df, info); other input columns are kept on the
        review-level output (renamed ``input_<name>`` if they clash).
        """
        df, col, info = read_reviews_csv(file, text_column)
        reviews, aspects = self.analyze_batch(df[col])
        extra = df[[c for c in df.columns if c != col]].reset_index(drop=True)
        extra.columns = [f"input_{c}" if c in reviews.columns else c for c in extra.columns]
        return pd.concat([extra, reviews], axis=1), aspects, info


def read_reviews_csv(file, text_column: str | None = None) -> tuple[pd.DataFrame, str, dict]:
    """Read and validate an uploaded CSV.

    Returns (df, text_column, info). Raises ValueError with a user-friendly
    message for unreadable files, missing columns or no usable rows.
    """
    try:
        if isinstance(file, (bytes, bytearray)):
            file = io.BytesIO(file)
        df = pd.read_csv(file, dtype=str, keep_default_na=True, on_bad_lines="error",
                         encoding_errors="replace")
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Could not parse the file as CSV: {exc}")
    if df.empty:
        raise ValueError("The CSV has a header but no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {c.lower(): c for c in df.columns}
    wanted = (text_column or DEFAULT_TEXT_COLUMN).strip().lower()
    if wanted not in lookup:
        from src.csv_diagnostics import log_csv_failure
        log_csv_failure("predictor.read_reviews_csv (legacy)", file,
                        {"reader": "pandas.read_csv", "header": "infer (row 0)", "sep": ",",
                         "encoding": "utf-8 + errors=replace"}, df.columns)
        from src.review_preprocessing import columns_look_like_data, describe_columns
        if columns_look_like_data(df.columns):
            raise ValueError("This file appears to have NO header row: the first row contains data, not column "
                             f"names (first cells: {describe_columns(df.columns)}).")
        raise ValueError(f"Required column '{text_column or DEFAULT_TEXT_COLUMN}' not found. "
                         f"Columns present: {describe_columns(df.columns, 12)}")
    col = lookup[wanted]
    n_total = len(df)
    text = df[col].map(normalize_text)
    df = df.loc[text != ""].copy()
    df[col] = text[text != ""]
    n_missing = n_total - len(df)
    if df.empty:
        raise ValueError(f"Column '{col}' contains no non-empty reviews.")
    truncated = len(df) > MAX_BATCH_ROWS
    df = df.head(MAX_BATCH_ROWS).reset_index(drop=True)
    return df, col, {"rows_total": n_total, "rows_missing_text": n_missing,
                     "rows_used": len(df), "truncated": truncated}
