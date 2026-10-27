"""Central paths and constants for ReviewLens."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "config"
DATASET_DIR = ROOT / "Dataset"      # user-supplied review datasets (not committed)

TRAIN_XML = RAW_DIR / "Laptop_Train_v2.xml"
TEST_XML = RAW_DIR / "Laptops_Test_Gold.xml"

# Public mirrors of the official SemEval-2014 Task 4 laptop files. The official
# portal (alt.qcri.org) requires a manual META-SHARE download, so these mirrors
# are used for automation and verified by SHA-256 + published sentence/term counts.
DATASET_SOURCES = {
    "train": {
        "path": TRAIN_XML,
        "url": "https://raw.githubusercontent.com/davidsbatista/Aspect-Based-Sentiment-Analysis/"
               "master/datasets/ABSA-SemEval2014/Laptop_Train_v2.xml",
        "sha256": "efec2ed73b4efb83b47c697114d800149c558118884afa80362a6e975ba7248c",
        "expected_sentences": 3045,
        "expected_terms": 2358,
    },
    "test": {
        "path": TEST_XML,
        "url": "https://raw.githubusercontent.com/AlexYangLi/ABSA_Keras/master/raw_data/"
               "semeval14_laptop/Laptops_Test_Gold.xml",
        "sha256": "aa3a30c9dd7e0eacf9b69e898eb4118a36c1b4f391a30a08dfc41a38d8b8745b",
        "expected_sentences": 800,
        "expected_terms": 654,
    },
}
LEXICON_DIR = RAW_DIR / "opinion_lexicon"
LEXICON_SOURCE = {
    "name": "Hu & Liu (2004) Opinion Lexicon",
    "url": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/opinion_lexicon.zip",
    "files": {"positive-words.txt": "9f643e85bcd7eb32e680f05df3d87fdd3277d6626122b3efec94f3ee3790b596",
              "negative-words.txt": "f80f5cfe3a62e7e07768eca9435b9ba514dbd5fc0eecf181e80b16f8f4578b34"},
}
OFFICIAL_PAGE ="https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools"

LABELS = ["negative", "neutral", "positive"]
ALL_POLARITIES = LABELS + ["conflict"]

MODEL_FILES = {
    "logreg": MODELS_DIR / "tfidf_logreg.joblib",
    "linsvm": MODELS_DIR / "tfidf_linearsvc.joblib",
}
MODEL_DISPLAY = {"logreg": "TF-IDF + Logistic Regression", "linsvm": "TF-IDF + Linear SVM"}
SELECTED_MODEL_FILE = MODELS_DIR / "selected_model.json"
METRICS_FILE = REPORTS_DIR / "metrics.json"
CV_FILE = REPORTS_DIR / "cv_results.json"
DATASET_STATS_FILE = REPORTS_DIR / "dataset_stats.json"
ASPECT_VOCAB_FILE = CONFIG_DIR / "aspect_vocabulary.json"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------- Gemini AI Insights (optional)
# All values can be overridden with environment variables; the API key is read ONLY from GEMINI_API_KEY
# (environment or Streamlit secrets) and is never stored in code, logs or the UI.
import os as _os


def ai_settings() -> dict:
    """Current AI Insights settings (read at call time so environment changes take effect)."""
    def _num(name, default, cast):
        try:
            return cast(_os.environ.get(name, default))
        except (TypeError, ValueError):
            return cast(default)
    return {
        "model": _os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
        "timeout_seconds": _num("GEMINI_TIMEOUT_SECONDS", 45, float),
        "max_excerpts": _num("AI_MAX_EXCERPTS", 24, int),
        "max_input_chars": _num("AI_MAX_INPUT_CHARS", 24000, int),
        "excerpt_chars": _num("AI_EXCERPT_CHARS", 600, int),
    }
