"""Evaluate the trained models on the official SemEval-2014 laptop test split.

These are GOLD-ASPECT results: the gold aspect terms and offsets are given to
the classifier, so they measure aspect sentiment classification only, not the
end-to-end pipeline (which must also extract aspects).

Run: ``python -m src.evaluate``
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from src import config
from src.data_loader import load_split


def evaluate_model(pipeline, X, y) -> dict:
    t0 = time.perf_counter()
    pred = pipeline.predict(X)
    infer_s = time.perf_counter() - t0
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=config.LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, labels=config.LABELS,
                                                    average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "weighted_f1": float(f1_score(y, pred, labels=config.LABELS, average="weighted",
                                      zero_division=0)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(s[i])}
                      for i, lab in enumerate(config.LABELS)},
        "confusion_matrix": confusion_matrix(y, pred, labels=config.LABELS).tolist(),
        "classification_report": classification_report(y, pred, labels=config.LABELS,
                                                       digits=4, zero_division=0),
        "inference_seconds_total": infer_s,
        "inference_ms_per_example": 1000 * infer_s / max(len(X), 1),
        "n_test": int(len(X)),
    }


def evaluate_all(verbose: bool = True) -> dict:
    if not config.CV_FILE.exists():
        raise FileNotFoundError("No training summary found. Run `python -m src.train` first.")
    with open(config.CV_FILE, encoding="utf-8") as f:
        cv = json.load(f)
    test_df, n_conflict = load_split("test")
    X, y = test_df, test_df["polarity"]

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_type": "gold-aspect sentiment classification on official test split",
        "labels": config.LABELS,
        "train_examples": cv["train_examples"],
        "train_conflict_excluded": cv["conflict_excluded"],
        "test_examples": int(len(test_df)),
        "test_conflict_excluded": n_conflict,
        "test_label_counts": {k: int(v) for k, v in y.value_counts().items()},
        "majority_baseline_accuracy": float(y.value_counts(normalize=True).max()),
        "selected_model": cv["selected_model"],
        "selection_rule": cv["selection_rule"],
        "models": {},
    }
    for name, info in cv["models"].items():
        pipe = joblib.load(config.MODEL_FILES[name])
        res = evaluate_model(pipe, X, y)
        res.update({
            "display_name": info["display_name"],
            "best_params": info["best_params"],
            "cv_macro_f1_mean": info["cv_macro_f1_mean"],
            "cv_macro_f1_std": info["cv_macro_f1_std"],
            "train_seconds": info["final_fit_seconds"],
            "tuning_seconds": info["tuning_seconds"],
        })
        out["models"][name] = res
        report_path = config.REPORTS_DIR / f"classification_report_{name}.txt"
        report_path.write_text(
            f"{info['display_name']} - official test split (gold aspects)\n\n"
            + res["classification_report"]
            + "\nConfusion matrix (rows=true, cols=pred, order=" + ", ".join(config.LABELS) + ")\n"
            + str(np.array(res["confusion_matrix"])) + "\n",
            encoding="utf-8",
        )
        if verbose:
            print(f"[eval] {info['display_name']}: acc={res['accuracy']:.4f} "
                  f"macroF1={res['macro_f1']:.4f} weightedF1={res['weighted_f1']:.4f}")
    best_on_test = max(out["models"], key=lambda k: out["models"][k]["macro_f1"])
    out["test_best_model"] = best_on_test
    out["selection_agrees_with_test"] = best_on_test == out["selected_model"]
    with open(config.METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if verbose:
        print(f"[eval] selected (by CV): {out['selected_model']}; best on test: {best_on_test}")
        print(f"[eval] wrote {config.METRICS_FILE}")
    return out


if __name__ == "__main__":
    evaluate_all()
