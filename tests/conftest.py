import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


@pytest.fixture(scope="session")
def trained_predictor():
    """Predictor backed by the saved, selected model (skips if not trained)."""
    if not config.SELECTED_MODEL_FILE.exists():
        pytest.skip("Models not trained; run `python -m src.train` first.")
    from src.predictor import ReviewPredictor
    return ReviewPredictor()


@pytest.fixture(scope="session")
def logreg_predictor():
    if not config.MODEL_FILES["logreg"].exists():
        pytest.skip("Logistic Regression model not trained.")
    from src.predictor import ReviewPredictor
    return ReviewPredictor(model_name="logreg")


@pytest.fixture(autouse=True)
def _isolate_gemini_config(monkeypatch, tmp_path):
    """Tests must never see the developer's real GEMINI_API_KEY or project secrets file."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from src import ai_insights
    monkeypatch.setattr(ai_insights, "PROJECT_SECRETS", tmp_path / "no_such_dir" / "secrets.toml")
