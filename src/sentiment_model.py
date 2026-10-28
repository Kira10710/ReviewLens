"""Whole-review sentiment model (positive / neutral / negative).

Training data: Amazon Reviews Multi (English) from the public SetFit mirror, with the
official train / validation / test splits. Its labels are derived from the reviewer's
1-5 star rating (1-2 -> negative, 3 -> neutral, 4-5 -> positive), so they are a *proxy*
for expressed sentiment, not independent human sentiment annotation. Scraped or
uploaded reviews are NEVER used for training.

Run: ``python -m src.sentiment_model``  (trains, selects on validation, saves model)
then ``python -m src.evaluation``       (scores the held-out test split).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config

LABELS = ["negative", "neutral", "positive"]
STAR_TO_LABEL = {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"}  # SetFit labels are 0-4
BENCH_DIR = config.RAW_DIR / "amazon_reviews_multi_en"
BENCH_BASE_URL = "https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en/resolve/main/{split}.jsonl"
MODEL_FILE = config.MODELS_DIR / "review_sentiment.joblib"
META_FILE = config.MODELS_DIR / "review_sentiment_meta.json"


class ModelNotAvailable(RuntimeError):
    pass


def download_benchmark() -> None:
    import urllib.request
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        path = BENCH_DIR / f"{split}.jsonl"
        if not path.exists():
            urllib.request.urlretrieve(BENCH_BASE_URL.format(split=split), path)


def load_benchmark(split: str) -> pd.DataFrame:
    path = BENCH_DIR / f"{split}.jsonl"
    if not path.exists():
        raise ModelNotAvailable(f"Benchmark file {path} not found. Run `python -m src.sentiment_model` "
                                "(it downloads the data) or see the README manual setup.")
    df = pd.read_json(path, lines=True)
    df["text"] = df["text"].astype(str)
    df["label_name"] = df["label"].map(STAR_TO_LABEL)
    return df.dropna(subset=["label_name"]).reset_index(drop=True)


def _vectorizer() -> TfidfVectorizer:
    # No stop-word removal and a token pattern that keeps contractions, so negations
    # ("not", "n't", "never", "no") remain available to the model.
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w[\w']*\b", dtype=np.float32)


def train_review_model(verbose: bool = True) -> dict:
    download_benchmark()
    train, val = load_benchmark("train"), load_benchmark("validation")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    vec = _vectorizer().fit(train["text"])            # fitted on training split only
    Xtr, Xva = vec.transform(train["text"]), vec.transform(val["text"])
    vec_s = time.perf_counter() - t0

    candidates = {
        "logreg": [LogisticRegression(C=c, class_weight="balanced", max_iter=400) for c in (1.0, 4.0)],
        "linsvm": [LinearSVC(C=c, class_weight="balanced") for c in (0.1, 0.3)],
    }
    results, fitted = {}, {}
    for name, clfs in candidates.items():
        best = None
        for clf in clfs:
            t0 = time.perf_counter()
            clf.fit(Xtr, train["label_name"])
            fit_s = time.perf_counter() - t0
            f1 = f1_score(val["label_name"], clf.predict(Xva), average="macro")
            if verbose:
                print(f"[train] {name} C={clf.C}: validation macro-F1={f1:.4f} (fit {fit_s:.1f}s)", flush=True)
            if best is None or f1 > best["val_macro_f1"]:
                best = {"C": clf.C, "val_macro_f1": float(f1), "fit_seconds": fit_s}
                fitted[name] = clf
        results[name] = best
    selected = max(results, key=lambda k: results[k]["val_macro_f1"])
    pipeline = Pipeline([("tfidf", vec), ("clf", fitted[selected])])
    joblib.dump(pipeline, MODEL_FILE)
    meta = {
        "selected": selected,
        "display_name": {"logreg": "TF-IDF + Logistic Regression", "linsvm": "TF-IDF + Linear SVM"}[selected],
        "selection_rule": "Highest macro-F1 on the official validation split (test split not used).",
        "candidates": results,
        "vectorizer_fit_seconds": vec_s,
        "train_examples": int(len(train)), "validation_examples": int(len(val)),
        "training_data": {
            "name": "Amazon Reviews Multi (English), SetFit mirror",
            "url": "https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en",
            "label_mapping": "stars 1-2 -> negative, 3 -> neutral, 4-5 -> positive (star-derived proxy labels)",
            "train_label_counts": train["label_name"].value_counts().to_dict(),
        },
        "has_probabilities": hasattr(fitted[selected], "predict_proba"),
        "model_file": MODEL_FILE.name,
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"[train] selected {meta['display_name']} (validation macro-F1 "
              f"{results[selected]['val_macro_f1']:.4f}); saved {MODEL_FILE}")
    return meta


class ReviewSentimentModel:
    """Loaded whole-review classifier with a uniform prediction format."""

    def __init__(self, pipeline: Pipeline, meta: dict | None = None):
        self.pipeline = pipeline
        self.meta = meta or {}
        self.classes = [str(c) for c in pipeline.classes_]
        self.has_probabilities = hasattr(pipeline.named_steps["clf"], "predict_proba")

    def predict(self, texts) -> list[dict]:
        texts = ["" if t is None else str(t) for t in texts]
        if not texts:
            return []
        preds = self.pipeline.predict(texts)
        if self.has_probabilities:
            scores, score_type = self.pipeline.predict_proba(texts), "probability"
        else:
            scores, score_type = self.pipeline.decision_function(texts), "svm_margin"
        out = []
        for pred, row in zip(preds, np.atleast_2d(scores)):
            score_map = {c: float(v) for c, v in zip(self.classes, row)}
            out.append({"label": str(pred), "score_type": score_type,
                        "confidence": score_map[str(pred)] if score_type == "probability" else None,
                        "scores": score_map})
        return out


def resolve_model_path(path: str | Path | None = None) -> Path:
    """Resolve review sentiment model path against candidate locations and working directories."""
    if path is not None:
        p = Path(path)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p.resolve()
        if (config.ROOT / p).exists():
            return (config.ROOT / p).resolve()
        if (config.MODELS_DIR / p.name).exists():
            return (config.MODELS_DIR / p.name).resolve()
        return p if p.is_absolute() else (config.MODELS_DIR / p.name)

    candidates = [
        MODEL_FILE,
        Path.cwd() / "models" / "review_sentiment.joblib",
        Path(__file__).resolve().parent.parent / "models" / "review_sentiment.joblib",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return MODEL_FILE


def model_diagnostics(path: str | Path | None = None) -> dict:
    """Safe diagnostics for model resolution and filesystem status. Never exposes secrets or contents."""
    cwd = Path.cwd()
    root = config.ROOT
    models_dir = config.MODELS_DIR
    resolved_path = resolve_model_path(path)

    models_dir_exists = models_dir.exists()
    files_in_models = []
    if models_dir_exists:
        try:
            files_in_models = sorted([f.name for f in models_dir.iterdir() if f.is_file()])
        except Exception:
            files_in_models = []

    return {
        "cwd": str(cwd),
        "root": str(root),
        "models_dir": str(models_dir),
        "models_dir_exists": models_dir_exists,
        "model_file": str(resolved_path),
        "model_file_exists": resolved_path.exists(),
        "model_file_size": resolved_path.stat().st_size if resolved_path.exists() else None,
        "files_in_models_dir": files_in_models,
    }


def load_review_model(path: str | Path | None = None) -> ReviewSentimentModel:
    path = resolve_model_path(path)
    if not path.exists():
        raise ModelNotAvailable(f"Review sentiment model not found at {path}. "
                                "Train it with `python -m src.sentiment_model`.")
    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise ModelNotAvailable(f"Could not load {path}: {exc}") from exc
    meta = {}
    meta_path = path.parent / "review_sentiment_meta.json"
    if not meta_path.exists():
        meta_path = config.MODELS_DIR / "review_sentiment_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ReviewSentimentModel(pipeline, meta)


if __name__ == "__main__":
    train_review_model()
