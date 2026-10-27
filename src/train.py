"""Train both baselines on the official training split only.

Hyper-parameters are tuned with stratified 5-fold cross-validation inside the
training split (macro-F1). The test split is never loaded here. Model selection
uses the mean cross-validated macro-F1 (held-out validation folds).

Run: ``python -m src.train``
"""
from __future__ import annotations

import json
import time

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config
from src.data_loader import load_split
from src.feature_builder import PARAM_GRIDS, build_pipeline


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if hasattr(v, "item"):
        return v.item()
    return v


def train_all(n_jobs: int = -1, verbose: bool = True) -> dict:
    train_df, n_conflict = load_split("train")
    X, y = train_df, train_df["polarity"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {"train_examples": len(train_df), "conflict_excluded": n_conflict,
               "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on training split",
               "scoring": "f1_macro", "models": {}}
    for name in ["logreg", "linsvm"]:
        if verbose:
            print(f"[train] tuning {config.MODEL_DISPLAY[name]} ...", flush=True)
        search = GridSearchCV(build_pipeline(name), PARAM_GRIDS[name], scoring="f1_macro",
                              cv=cv, n_jobs=n_jobs, refit=False)
        t0 = time.perf_counter()
        search.fit(X, y)
        tune_s = time.perf_counter() - t0

        # Final fit on the full training split with the best parameters (timed).
        best = build_pipeline(name).set_params(**search.best_params_)
        t0 = time.perf_counter()
        best.fit(X, y)
        fit_s = time.perf_counter() - t0
        joblib.dump(best, config.MODEL_FILES[name])

        i = search.best_index_
        summary["models"][name] = {
            "display_name": config.MODEL_DISPLAY[name],
            "best_params": _jsonable(search.best_params_),
            "cv_macro_f1_mean": float(search.cv_results_["mean_test_score"][i]),
            "cv_macro_f1_std": float(search.cv_results_["std_test_score"][i]),
            "grid_size": len(search.cv_results_["params"]),
            "tuning_seconds": tune_s,
            "final_fit_seconds": fit_s,
            "model_file": config.MODEL_FILES[name].name,
            "vocabulary_size": int(sum(len(t.vocabulary_) for _, t, _ in
                                       best.named_steps["tfidf"].transformers_
                                       if hasattr(t, "vocabulary_"))),
        }
        if verbose:
            m = summary["models"][name]
            print(f"  best CV macro-F1 = {m['cv_macro_f1_mean']:.4f} +/- {m['cv_macro_f1_std']:.4f}"
                  f"  params={m['best_params']}  fit={fit_s:.2f}s", flush=True)

    selected = max(summary["models"], key=lambda k: summary["models"][k]["cv_macro_f1_mean"])
    summary["selected_model"] = selected
    summary["selection_rule"] = ("Highest mean 5-fold cross-validated macro-F1 on the training "
                                 "split (test split not used for selection).")
    with open(config.CV_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(config.SELECTED_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump({"model": selected, "display_name": config.MODEL_DISPLAY[selected],
                   "file": config.MODEL_FILES[selected].name,
                   "selection_rule": summary["selection_rule"]}, f, indent=2)
    if verbose:
        print(f"[train] selected: {config.MODEL_DISPLAY[selected]}")
    return summary


if __name__ == "__main__":
    train_all()
