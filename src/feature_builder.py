"""Feature construction for aspect-context sentiment classification.

Input to every model is a DataFrame with columns ``sentence``, ``aspect``,
``start`` and ``end`` (character offsets of the aspect inside ``sentence``).
``AspectContextBuilder`` turns each row into three text views, which are then
vectorised by separate TF-IDF vectorizers fitted on training data only:

* ``context`` - tokens in a window around the aspect (optionally clipped to its
  clause, negation-marked, aspect replaced by a placeholder, optional NEAR_
  copies of the words closest to the aspect),
* ``sentence`` - the whole sentence (negation-marked), as weak global evidence,
* ``aspect`` - the aspect term itself (captures per-aspect label priors),
* ``lexicon`` - LEX_POS / LEX_NEG tokens for context words found in the Hu & Liu
  opinion lexicon (polarity flipped under negation). This view generalises to
  opinion words that are rare in the 2.3k training examples; its weight is a
  hyper-parameter, so cross-validation can switch it off (weight 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.data_loader import load_lexicon
from src.preprocessing import local_context, mark_negation, tokenize

REQUIRED_COLUMNS = ["sentence", "aspect", "start", "end"]


class AspectContextBuilder(BaseEstimator, TransformerMixin):
    """Stateless transformer: raw aspect rows -> text views for TF-IDF."""

    def __init__(self, window: int = 6, clip_clause: bool = True, negation: bool = True,
                 near: int = 0):
        self.window = window
        self.clip_clause = clip_clause
        self.negation = negation
        self.near = near

    def fit(self, X, y=None):
        # The lexicon is copied into the fitted object so saved models are self-contained.
        self.lexicon_ = load_lexicon()
        return self

    def _lexicon_tokens(self, context_tokens: list[str]) -> str:
        out = []
        for tok in context_tokens:
            is_near = tok.startswith("NEAR_")
            tok = tok[5:] if is_near else tok
            negated = tok.startswith("NEG_")
            word = tok[4:] if negated else tok
            suffix = "_NEAR" if is_near else ""
            if word in self.lexicon_["positive"]:
                out.append(("LEX_NEG" if negated else "LEX_POS") + suffix)
            elif word in self.lexicon_["negative"]:
                out.append(("LEX_POS" if negated else "LEX_NEG") + suffix)
        return " ".join(out) or "LEX_NONE"

    def transform(self, X):
        missing = [c for c in REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns for feature building: {missing}")
        if not hasattr(self, "lexicon_"):
            self.fit(X)
        ctx, sent, asp, lex = [], [], [], []
        for row in X[REQUIRED_COLUMNS].itertuples(index=False):
            c, _ = local_context(row.sentence, int(row.start), int(row.end),
                                 self.window, self.clip_clause, self.negation, self.near)
            toks = tokenize(row.sentence)
            ctx.append(c)
            lex.append(self._lexicon_tokens(c.split()))
            sent.append(" ".join(mark_negation(toks) if self.negation else toks))
            asp.append(" ".join(tokenize(row.aspect)))
        return pd.DataFrame({"context": ctx, "sentence": sent, "aspect": asp, "lexicon": lex})


def _identity_tokens(s: str) -> list[str]:
    return s.split()


def _tfidf(ngram=(1, 2), min_df=1):
    return TfidfVectorizer(tokenizer=_identity_tokens, preprocessor=None, lowercase=False,
                           token_pattern=None, ngram_range=ngram, min_df=min_df,
                           sublinear_tf=True)


WEIGHT_OPTIONS = [
    {"context": 1.0, "sentence": 0.5, "aspect": 0.5, "lexicon": 0.0},
    {"context": 1.0, "sentence": 0.5, "aspect": 0.5, "lexicon": 0.5},
    {"context": 1.0, "sentence": 0.5, "aspect": 0.5, "lexicon": 1.0},
]


def build_pipeline(model: str) -> Pipeline:
    """Create an untrained pipeline for 'logreg' or 'linsvm'."""
    vectors = ColumnTransformer([
        ("context", _tfidf((1, 2)), "context"),
        ("sentence", _tfidf((1, 1), min_df=2), "sentence"),
        ("aspect", _tfidf((1, 1)), "aspect"),
        # log-scaled counts of LEX_POS / LEX_NEG (no idf, no length normalisation)
        ("lexicon", TfidfVectorizer(tokenizer=_identity_tokens, lowercase=False, token_pattern=None,
                                    use_idf=False, norm=None, sublinear_tf=True), "lexicon"),
    ], transformer_weights=dict(WEIGHT_OPTIONS[1]))
    if model == "logreg":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                 random_state=config.RANDOM_STATE)
    elif model == "linsvm":
        clf = LinearSVC(C=0.3, class_weight="balanced", random_state=config.RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model '{model}'")
    return Pipeline([("context", AspectContextBuilder()), ("tfidf", vectors), ("clf", clf)])


PARAM_GRIDS = {
    "logreg": {
        "context__window": [4, 8, 0],
        "context__clip_clause": [True, False],
        "context__near": [0, 3],
        "tfidf__transformer_weights": WEIGHT_OPTIONS,
        "clf__C": [0.5, 2.0, 8.0],
    },
    "linsvm": {
        "context__window": [4, 8, 0],
        "context__clip_clause": [True, False],
        "context__near": [0, 3],
        "tfidf__transformer_weights": WEIGHT_OPTIONS,
        "clf__C": [0.05, 0.2, 1.0],
    },
}


def scores_for(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return (score matrix, score_type). Probabilities when the classifier has them,
    otherwise raw decision-function margins (which are NOT probabilities)."""
    if hasattr(pipeline, "predict_proba") and hasattr(pipeline.named_steps["clf"], "predict_proba"):
        return pipeline.predict_proba(X), "probability"
    return pipeline.decision_function(X), "svm_margin"
