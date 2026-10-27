import json

import pytest

from src.aspect_extractor import AspectExtractor, load_vocabulary
from src.preprocessing import clause_span, local_context, mark_negation


@pytest.fixture(scope="module")
def ex():
    return AspectExtractor()


def aspects(ex, text):
    return [(m.aspect, m.matched_text.lower()) for m in ex.extract(text)]


def test_empty_and_none_input(ex):
    assert ex.extract("") == []
    assert ex.extract("   \n ") == []
    assert ex.extract(None) == []
    assert ex.extract(float("nan")) == []


def test_multiword_aspect_preferred_over_single_word(ex):
    found = aspects(ex, "The battery life is excellent.")
    assert found == [("Battery", "battery life")]   # not also a separate "battery" match


def test_multiple_aspects_in_one_sentence(ex):
    found = aspects(ex, "The keyboard is great, the touchpad is awful and the price is fair.")
    assert [a for a, _ in found] == ["Keyboard", "Touchpad", "Price"]


def test_word_boundaries(ex):
    # "os" must not match inside "most"/"cost", "ram" not inside "program"/"frame"
    found = aspects(ex, "Most of the frame is cheap plastic.")
    assert all(a not in ("Software", "Performance") for a, _ in found)
    assert aspects(ex, "This program crashes") == []


def test_case_plural_and_hyphen(ex):
    assert ("Display", "screens") in aspects(ex, "Both SCREENS look sharp")
    assert ("Touchpad", "track-pad") in aspects(ex, "The track-pad is smooth")


def test_unknown_aspects_return_nothing(ex):
    assert ex.extract("My cousin visited yesterday and we ate pizza.") == []


def test_no_duplicate_aspect_per_sentence(ex):
    found = aspects(ex, "The screen is bright and the display is sharp.")
    assert [a for a, _ in found] == ["Display"]


def test_same_aspect_in_different_sentences_kept(ex):
    found = ex.extract("The screen is bright. Later the screen flickered.")
    assert [m.sentence_index for m in found] == [0, 1]


def test_offsets_point_to_match(ex):
    for m in ex.extract("Love it. The fan noise is loud but the keyboard is nice."):
        assert m.sentence[m.start:m.end] == m.matched_text


def test_custom_vocabulary(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"_comment": "x", "Lens": ["zoom lens", "lens"]}), encoding="utf-8")
    ex = AspectExtractor(load_vocabulary(p))
    assert aspects(ex, "The zoom lens is sharp") == [("Lens", "zoom lens")]


def test_clause_context_separates_contrasting_opinions():
    s = "The battery is great but the screen is dim."
    i = s.index("screen")
    lo, hi = clause_span(s, i, i + 6)
    assert "great" not in s[lo:hi] and "dim" in s[lo:hi]
    ctx, display = local_context(s, i, i + 6, window=8)
    assert "ASPECT_TERM" in ctx and "great" not in ctx


def test_negation_marking():
    assert mark_negation(["not", "good", ",", "great"]) == ["not", "NEG_good", ",", "great"]
