"""Tests for robust model path resolution and startup diagnostics across working directories."""
import os
from pathlib import Path
import pytest

from src import config
from src import sentiment_model as sm
from src import predictor


def test_resolve_model_path_default():
    resolved = sm.resolve_model_path()
    assert isinstance(resolved, Path)
    assert resolved.name == "review_sentiment.joblib"
    assert resolved.exists()


def test_resolve_model_path_from_different_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path.cwd() == tmp_path
    resolved = sm.resolve_model_path()
    assert resolved.exists()
    assert resolved.name == "review_sentiment.joblib"
    # Verify model actually loads from different cwd
    model = sm.load_review_model()
    assert set(model.classes) == {"negative", "neutral", "positive"}


def test_resolve_relative_path_from_different_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = sm.resolve_model_path("models/review_sentiment.joblib")
    assert resolved.exists()
    assert resolved.name == "review_sentiment.joblib"


def test_resolve_explicit_nonexistent_path(tmp_path):
    missing = tmp_path / "nonexistent_model.joblib"
    resolved = sm.resolve_model_path(missing)
    assert resolved == missing
    assert not resolved.exists()
    with pytest.raises(sm.ModelNotAvailable, match="not found"):
        sm.load_review_model(missing)


def test_model_diagnostics_structure_and_safety():
    diag = sm.model_diagnostics()
    assert isinstance(diag, dict)
    expected_keys = {
        "cwd", "root", "models_dir", "models_dir_exists",
        "model_file", "model_file_exists", "model_file_size", "files_in_models_dir"
    }
    assert expected_keys.issubset(diag.keys())
    assert diag["model_file_exists"] is True
    assert isinstance(diag["files_in_models_dir"], list)
    assert "review_sentiment.joblib" in diag["files_in_models_dir"]
    # Verify no raw model or secrets in diagnostics
    for k, v in diag.items():
        if isinstance(v, str):
            assert "AIza" not in v
            assert "pipeline" not in v.lower()


def test_predictor_load_model_from_different_cwd(tmp_path, monkeypatch):
    if not config.SELECTED_MODEL_FILE.exists():
        pytest.skip("aspect model not trained")
    model_name = predictor.selected_model_name()
    target_file = config.MODEL_FILES[model_name]
    if not target_file.exists():
        pytest.skip(f"{target_file} not found")
    monkeypatch.chdir(tmp_path)
    model = predictor.load_model(model_name)
    assert hasattr(model, "predict")
