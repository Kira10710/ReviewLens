"""Run the hand-written demo reviews through the full pipeline with both models.

Writes reports/demo_predictions.json. Run: ``python -m scripts.demo_reviews``
"""
import json

from src import config
from src.predictor import ReviewPredictor

DEMO_REVIEWS = [
    "The battery life is amazing, easily lasting ten hours, but the screen is too dim outdoors.",
    "Keyboard feels cheap and the touchpad is unresponsive. Terrible build quality for this price.",
    "Great performance for gaming and the display is gorgeous. Fans get loud though.",
    "The laptop comes with Windows 11 preinstalled and a 512GB SSD.",
    "Customer service was rude and refused to honour the warranty.",
    "I love the design and it is very light, however the speakers sound tinny.",
    "My cousin visited yesterday and we ate pizza.",
]


def main():
    out = {}
    for name in config.MODEL_FILES:
        p = ReviewPredictor(model_name=name)
        out[name] = [p.analyze_review(r) for r in DEMO_REVIEWS]
    with open(config.REPORTS_DIR / "demo_predictions.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for i, review in enumerate(DEMO_REVIEWS):
        print(f"\n[{i + 1}] {review}")
        for name in config.MODEL_FILES:
            r = out[name][i]
            aspects = ", ".join(
                f"{a['aspect']}('{a['matched_text']}')={a['sentiment']}"
                + (f" {a['confidence']:.2f}" if a["confidence"] is not None else "")
                for a in r["aspects"]) or r["message"][:60]
            print(f"   {name:7s} overall={r['overall']:<20s} {aspects}")


if __name__ == "__main__":
    main()
