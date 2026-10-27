"""Rule-based aspect extraction from a configurable vocabulary.

This is NOT a learned model: it is dictionary phrase matching with word
boundaries, longest-match-first, and no overlapping matches. It only finds
aspects whose wording appears in ``config/aspect_vocabulary.json``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config
from src.preprocessing import normalize_text, split_sentences


@dataclass
class AspectMatch:
    aspect: str          # canonical aspect name, e.g. "Battery"
    matched_text: str    # exact surface text in the review, e.g. "battery life"
    sentence: str        # sentence containing the match
    start: int           # character offsets inside ``sentence``
    end: int
    sentence_index: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_vocabulary(path: str | Path | None = None) -> dict[str, list[str]]:
    path = Path(path) if path else config.ASPECT_VOCAB_FILE
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {k: [p for p in v if isinstance(p, str) and p.strip()]
             for k, v in raw.items() if not k.startswith("_") and isinstance(v, list)}
    if not vocab:
        raise ValueError(f"Aspect vocabulary at {path} is empty")
    return vocab


def _phrase_pattern(phrase: str) -> str:
    words = [re.escape(w) for w in re.split(r"[\s\-]+", phrase.strip().lower()) if w]
    body = r"[\s\-]+".join(words)
    # Optional plural on the last word; lookarounds give word boundaries that
    # also work for phrases ending in non-word characters.
    return rf"(?<![\w-])(?:{body})(?:e?s)?(?![\w-])"


class AspectExtractor:
    def __init__(self, vocabulary: dict[str, list[str]] | None = None):
        self.vocabulary = vocabulary if vocabulary is not None else load_vocabulary()
        entries = [(phrase.lower(), aspect) for aspect, phrases in self.vocabulary.items()
                   for phrase in phrases]
        # Longest phrases first so "battery life" wins over "battery".
        entries.sort(key=lambda e: (-len(e[0]), e[0]))
        self._patterns = [(re.compile(_phrase_pattern(p), re.IGNORECASE), a) for p, a in entries]

    def extract(self, text) -> list[AspectMatch]:
        """Return aspect matches ordered by position.

        Overlapping matches are resolved in favour of the longest phrase, and a
        canonical aspect is reported at most once per sentence.
        """
        text = normalize_text(text)
        if not text:
            return []
        results: list[AspectMatch] = []
        for s_idx, (s0, s1) in enumerate(split_sentences(text)):
            sentence = text[s0:s1]
            taken: list[tuple[int, int]] = []
            found: list[AspectMatch] = []
            for pattern, aspect in self._patterns:
                for m in pattern.finditer(sentence):
                    a, b = m.span()
                    if any(a < y and x < b for x, y in taken):
                        continue
                    taken.append((a, b))
                    found.append(AspectMatch(aspect, m.group(0), sentence, a, b, s_idx))
            found.sort(key=lambda m: m.start)
            seen = set()
            for m in found:
                if m.aspect not in seen:
                    seen.add(m.aspect)
                    results.append(m)
        return results
