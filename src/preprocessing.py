"""Text utilities: tokenisation, negation marking, clause and context extraction.

The same functions are used for benchmark training (gold aspect offsets) and
for end-to-end prediction on new reviews, so the model always sees inputs built
the same way.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]")
NEGATIONS = {
    "not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "cannot",
    "without", "hardly", "barely", "isn't", "wasn't", "aren't", "weren't", "don't",
    "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't",
    "hasn't", "haven't", "hadn't", "n't",
}
PUNCT = {".", ",", "!", "?", ";", ":"}
# Words that typically start a new opinion clause ("great screen but poor battery").
CLAUSE_BREAK_RE = re.compile(
    r"\s*(?:;|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bwhereas\b|\bwhile\b|\byet\b|"
    r"\bexcept\b|\bunfortunately\b)\s*",
    flags=re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def normalize_text(text) -> str:
    """Collapse whitespace and coerce non-strings (None/NaN) to ''."""
    if text is None or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


def mark_negation(tokens: list[str]) -> list[str]:
    """Prefix tokens following a negation word with NEG_ until punctuation."""
    out, negated = [], False
    for tok in tokens:
        if tok in PUNCT:
            negated = False
            out.append(tok)
            continue
        out.append(f"NEG_{tok}" if negated else tok)
        if tok in NEGATIONS or tok.endswith("n't"):
            negated = True
    return out


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of sentences in ``text``."""
    spans = []
    for m in SENTENCE_RE.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        if s < e:
            spans.append((s, e))
    return spans


def clause_span(sentence: str, start: int, end: int) -> tuple[int, int]:
    """Character span of the clause containing [start, end).

    Clauses are delimited by contrastive connectives and semicolons. If the aspect
    itself overlaps a delimiter the whole sentence is returned.
    """
    left, right = 0, len(sentence)
    for m in CLAUSE_BREAK_RE.finditer(sentence):
        if m.end() <= start:
            left = m.end()
        elif m.start() >= end:
            right = m.start()
            break
    return left, right


def local_context(sentence: str, start: int, end: int, window: int = 6,
                  clip_clause: bool = True, negation: bool = True,
                  near: int = 0) -> tuple[str, str]:
    """Build the classifier context for one aspect occurrence.

    Returns (context_tokens, display_text): ``context_tokens`` is the processed
    token string fed to the model, with the aspect replaced by a placeholder and
    up to ``window`` tokens kept on each side (window <= 0 keeps the whole span);
    ``display_text`` is the raw text span used, for showing to users.
    """
    lo, hi = clause_span(sentence, start, end) if clip_clause else (0, len(sentence))
    left_text, right_text = sentence[lo:start], sentence[end:hi]
    left = tokenize(left_text)
    right = tokenize(right_text)
    if window and window > 0:
        left, right = left[-window:], right[:window]
    toks = left + ["ASPECT_TERM"] + right
    if negation:
        toks = mark_negation(toks)
    if near and near > 0:
        # Extra copies of words within ``near`` tokens of the aspect, so the
        # bag-of-words model can tell close opinion words from distant ones.
        a = len(left)
        close = toks[max(0, a - near):a] + toks[a + 1:a + 1 + near]
        toks = toks + [f"NEAR_{t}" for t in close if t not in PUNCT]
    display = sentence[lo:hi].strip(" ,;")
    return " ".join(toks), display
