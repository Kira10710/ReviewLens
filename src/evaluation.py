"""Held-out evaluation of the whole-review sentiment model on the official test split.

Run: ``python -m src.evaluation`` -> reports/review_sentiment_metrics.json

Caveat recorded in the output: the benchmark labels are derived from star ratings, so
these scores measure agreement with the star-derived label, not with human-annotated
sentiment. They say nothing about aspect extraction or aspect-level sentiment (see
reports/metrics.json for the separate SemEval aspect benchmark).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from src import config
from src.sentiment_model import BENCH_DIR, LABELS, META_FILE, load_benchmark, load_review_model

METRICS_FILE = config.REPORTS_DIR / "review_sentiment_metrics.json"


def evaluate_review_model(verbose: bool = True) -> dict:
    model = load_review_model()
    test = load_benchmark("test")
    train = load_benchmark("train")
    t0 = time.perf_counter()
    pred = model.pipeline.predict(test["text"])
    infer_s = time.perf_counter() - t0
    y = test["label_name"]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average="macro", zero_division=0)
    meta = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}
    overlap = len(set(train["text"]) & set(test["text"]))
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_type": "whole-review 3-class sentiment on the official held-out test split",
        "label_caveat": "Labels are derived from star ratings (1-2 neg, 3 neutral, 4-5 pos); this is a proxy "
                        "for sentiment, not human-annotated sentiment.",
        "model": meta.get("display_name", "unknown"),
        "selection_rule": meta.get("selection_rule"),
        "validation_macro_f1": meta.get("candidates", {}).get(meta.get("selected", ""), {}).get("val_macro_f1"),
        "train_examples": meta.get("train_examples"),
        "validation_examples": meta.get("validation_examples"),
        "test_examples": int(len(test)),
        "test_label_counts": {k: int(v) for k, v in y.value_counts().items()},
        "train_label_counts": meta.get("training_data", {}).get("train_label_counts"),
        "train_test_identical_texts": overlap,
        "majority_baseline_accuracy": float(y.value_counts(normalize=True).max()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "weighted_f1": float(f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(s[i])} for i, lab in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "classification_report": classification_report(y, pred, labels=LABELS, digits=4, zero_division=0),
        "fit_seconds": meta.get("candidates", {}).get(meta.get("selected", ""), {}).get("fit_seconds"),
        "inference_ms_per_review": 1000 * infer_s / max(len(test), 1),
        "benchmark_files": sorted(p_.name for p_ in BENCH_DIR.glob("*.jsonl")),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if verbose:
        print(f"[eval] {out['model']}: acc={out['accuracy']:.4f} macroF1={out['macro_f1']:.4f} "
              f"weightedF1={out['weighted_f1']:.4f} on {out['test_examples']} test reviews")
        print(out["classification_report"])
        print(np.array(out["confusion_matrix"]))
    return out


if __name__ == "__main__":
    evaluate_review_model()
