"""Download, verify and parse the SemEval-2014 Task 4 laptop XML files.

Run ``python -m src.data_loader`` to download (if missing), validate and write
processed CSVs plus dataset statistics.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src import config

RECORD_COLUMNS = ["sentence_id", "sentence", "aspect", "polarity", "start", "end"]


class DatasetError(RuntimeError):
    """Raised when the benchmark data is missing or invalid."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(force: bool = False) -> dict:
    """Download missing raw files from the mirrors and verify their checksums."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    status = {}
    for split, src in config.DATASET_SOURCES.items():
        path = src["path"]
        if force or not path.exists():
            try:
                urllib.request.urlretrieve(src["url"], path)
            except Exception as exc:  # network failure: report, never substitute data
                raise DatasetError(
                    f"Could not download the {split} file ({exc}). Follow the manual setup "
                    f"steps in README.md and place the file at {path}."
                ) from exc
        digest = sha256_of(path)
        status[split] = {"path": str(path), "sha256": digest, "checksum_ok": digest == src["sha256"]}
    return status


def download_lexicon() -> None:
    """Fetch the Hu & Liu opinion lexicon (NLTK data mirror) if it is missing."""
    import zipfile
    lex_dir = config.LEXICON_DIR
    if all((lex_dir / name).exists() for name in config.LEXICON_SOURCE["files"]):
        return
    lex_dir.mkdir(parents=True, exist_ok=True)
    tmp = lex_dir / "opinion_lexicon.zip"
    try:
        urllib.request.urlretrieve(config.LEXICON_SOURCE["url"], tmp)
    except Exception as exc:
        raise DatasetError(f"Could not download the opinion lexicon ({exc}); see README.md.") from exc
    with zipfile.ZipFile(tmp) as zf:
        for name in config.LEXICON_SOURCE["files"]:
            (lex_dir / name).write_bytes(zf.read(f"opinion_lexicon/{name}"))
    tmp.unlink()


def load_lexicon() -> dict[str, set[str]]:
    """Return {'positive': set, 'negative': set} from the opinion lexicon files."""
    download_lexicon()
    out = {}
    for polarity in ("positive", "negative"):
        path = config.LEXICON_DIR / f"{polarity}-words.txt"
        words = path.read_text(encoding="utf-8", errors="replace").splitlines()
        out[polarity] = {w.strip().lower() for w in words if w.strip() and not w.startswith(";")}
    return out


def parse_semeval_xml(path: str | Path) -> pd.DataFrame:
    """Parse a SemEval-2014 ABSA XML file into one row per aspect-term occurrence.

    Sentences without aspect terms produce no rows. Raises DatasetError on
    missing or malformed files.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise DatasetError(f"Malformed XML in {path}: {exc}") from exc
    if root.tag != "sentences":
        raise DatasetError(f"Unexpected root element <{root.tag}> in {path}; expected <sentences>")

    rows = []
    for sent in root.iter("sentence"):
        text_el = sent.find("text")
        if text_el is None or text_el.text is None:
            continue
        text = text_el.text
        for term in sent.iter("aspectTerm"):
            polarity = term.get("polarity")
            if polarity is None:
                raise DatasetError(
                    f"Aspect term without polarity in sentence {sent.get('id')} of {path}; "
                    "this looks like an unlabeled (phase A/B) file, not the gold file."
                )
            start, end = int(term.get("from")), int(term.get("to"))
            rows.append({
                "sentence_id": sent.get("id"),
                "sentence": text,
                "aspect": term.get("term"),
                "polarity": polarity.strip().lower(),
                "start": start,
                "end": end,
            })
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


def count_sentences(path: str | Path) -> int:
    return sum(1 for _ in ET.parse(path).getroot().iter("sentence"))


def validate_offsets(df: pd.DataFrame) -> int:
    """Return the number of rows whose character offsets do not match the term text."""
    bad = 0
    for row in df.itertuples():
        if row.sentence[row.start:row.end] != row.aspect:
            bad += 1
    return bad


def to_three_class(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop 'conflict' aspects for the 3-class experiment. Returns (df, n_excluded)."""
    mask = df["polarity"] == "conflict"
    unknown = set(df["polarity"]) - set(config.ALL_POLARITIES)
    if unknown:
        raise DatasetError(f"Unknown polarity labels: {sorted(unknown)}")
    return df.loc[~mask].reset_index(drop=True), int(mask.sum())


def load_split(split: str) -> tuple[pd.DataFrame, int]:
    """Load a split ('train' or 'test') as a 3-class DataFrame and conflict count."""
    df = parse_semeval_xml(config.DATASET_SOURCES[split]["path"])
    return to_three_class(df)


def prepare_dataset() -> dict:
    """Validate raw files, write processed CSVs and a provenance/statistics report."""
    checks = download_dataset()
    stats = {"source_page": config.OFFICIAL_PAGE, "splits": {}}
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for split, src in config.DATASET_SOURCES.items():
        full = parse_semeval_xml(src["path"])
        n_sent = count_sentences(src["path"])
        three, n_conflict = to_three_class(full)
        three.to_csv(config.PROCESSED_DIR / f"{split}_aspects.csv", index=False)
        stats["splits"][split] = {
            "file": src["path"].name,
            "mirror_url": src["url"],
            "sha256": checks[split]["sha256"],
            "checksum_matches_recorded": checks[split]["checksum_ok"],
            "sentences": n_sent,
            "aspect_terms_all_labels": len(full),
            "matches_published_counts": (n_sent == src["expected_sentences"]
                                         and len(full) == src["expected_terms"]),
            "label_counts_all": full["polarity"].value_counts().to_dict(),
            "conflict_excluded": n_conflict,
            "aspect_terms_three_class": len(three),
            "label_counts_three_class": three["polarity"].value_counts().to_dict(),
            "offset_mismatches": validate_offsets(full),
            "sentences_with_aspects": int(full["sentence_id"].nunique()),
        }
    tr_s = set(parse_semeval_xml(config.TRAIN_XML)["sentence"])
    te_s = set(parse_semeval_xml(config.TEST_XML)["sentence"])
    stats["train_test_identical_sentences"] = len(tr_s & te_s)
    lex = load_lexicon()
    stats["lexicon"] = {
        "name": config.LEXICON_SOURCE["name"], "url": config.LEXICON_SOURCE["url"],
        "positive_words": len(lex["positive"]), "negative_words": len(lex["negative"]),
        "checksums_match_recorded": all(
            sha256_of(config.LEXICON_DIR / n) == h for n, h in config.LEXICON_SOURCE["files"].items()),
        "use": "Optional feature view (counts of lexicon words in the aspect context); "
               "whether it is used is decided by cross-validation on the training split.",
    }
    stats["preprocessing"] = [
        "One example per annotated aspect-term occurrence (gold aspect terms and offsets).",
        "Polarity 'conflict' excluded from the 3-class experiment (count recorded per split).",
        "Sentences without aspect terms are not used for sentiment classification.",
        "No deduplication or text normalisation is applied to the raw files; lowercasing and "
        "negation marking happen inside the model pipeline.",
    ]
    with open(config.DATASET_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    print(json.dumps(prepare_dataset(), indent=2))
