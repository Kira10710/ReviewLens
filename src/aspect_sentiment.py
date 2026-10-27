"""Aspect-level sentiment for reviews.

Aspect extraction: rule-based vocabulary matching (``src/aspect_extractor.py``), NOT a
learned model. Aspect sentiment: the supervised TF-IDF classifier trained on the SemEval-2014
Task 4 *laptop* aspect-term benchmark (``src/train.py``), applied to each aspect's own clause
context. That training domain is laptops, so accuracy on other product types is unmeasured
and expected to be lower.
"""
from __future__ import annotations

import pandas as pd

from src.predictor import ReviewPredictor, overall_label

ASPECT_COLUMNS = ["review_id", "product_id", "source", "aspect", "matched_text", "aspect_context",
                  "sentence", "sentiment", "confidence", "score_type"]


class AspectSentimentAnalyzer:
    def __init__(self, predictor: ReviewPredictor | None = None):
        self.predictor = predictor or ReviewPredictor()
        self.model_name = self.predictor.model_name

    def analyze_text(self, text) -> dict:
        """Return the ReviewPredictor result dict for one review text."""
        return self.predictor.analyze_review(text)

    def analyze_review(self, review_id: str, text, product_id=None, source=None) -> list[dict]:
        rows = []
        for a in self.analyze_text(text)["aspects"]:
            rows.append({
                "review_id": review_id, "product_id": product_id, "source": source,
                "aspect": a["aspect"], "matched_text": a["matched_text"],
                "aspect_context": a["context"], "sentence": a["sentence"],
                "sentiment": a["sentiment"], "confidence": a["confidence"],
                "score_type": a["score_type"],
            })
        return rows

    @staticmethod
    def aspect_overall(sentiments: list[str]) -> str:
        """Documented aggregation: any positive + any negative -> Mixed; otherwise the
        single polarity present; all-neutral -> Neutral; none -> 'No aspects detected'."""
        return overall_label(sentiments)


def empty_aspect_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ASPECT_COLUMNS)
