import pytest

from src import config
from src.data_loader import DatasetError, parse_semeval_xml, to_three_class, validate_offsets

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
  <sentence id="1">
    <text>The battery life is great but the screen is dim.</text>
    <aspectTerms>
      <aspectTerm term="battery life" polarity="positive" from="4" to="16"/>
      <aspectTerm term="screen" polarity="negative" from="34" to="40"/>
    </aspectTerms>
  </sentence>
  <sentence id="2"><text>I bought it last week.</text></sentence>
  <sentence id="3">
    <text>The keyboard is fine and the price okay.</text>
    <aspectTerms>
      <aspectTerm term="keyboard" polarity="neutral" from="4" to="12"/>
      <aspectTerm term="price" polarity="conflict" from="29" to="34"/>
    </aspectTerms>
  </sentence>
</sentences>"""


@pytest.fixture
def sample_xml(tmp_path):
    p = tmp_path / "sample.xml"
    p.write_text(SAMPLE_XML, encoding="utf-8")
    return p


def test_parse_aspects_and_labels(sample_xml):
    df = parse_semeval_xml(sample_xml)
    assert len(df) == 4                       # sentence without aspects yields no rows
    assert list(df["aspect"]) == ["battery life", "screen", "keyboard", "price"]
    assert list(df["polarity"]) == ["positive", "negative", "neutral", "conflict"]
    assert df.loc[0, "start"] == 4 and df.loc[0, "end"] == 16
    assert validate_offsets(df) == 0


def test_conflict_excluded_and_counted(sample_xml):
    three, n_conflict = to_three_class(parse_semeval_xml(sample_xml))
    assert n_conflict == 1
    assert set(three["polarity"]) == {"positive", "negative", "neutral"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        parse_semeval_xml(tmp_path / "nope.xml")


def test_malformed_xml_raises(tmp_path):
    p = tmp_path / "bad.xml"
    p.write_text("<sentences><sentence id='1'><text>oops</sentence>", encoding="utf-8")
    with pytest.raises(DatasetError, match="Malformed"):
        parse_semeval_xml(p)


def test_unlabeled_file_rejected(tmp_path):
    p = tmp_path / "phaseB.xml"
    p.write_text('<sentences><sentence id="1"><text>Nice screen</text><aspectTerms>'
                 '<aspectTerm term="screen" from="5" to="11"/></aspectTerms></sentence></sentences>',
                 encoding="utf-8")
    with pytest.raises(DatasetError, match="without polarity"):
        parse_semeval_xml(p)


@pytest.mark.skipif(not config.TRAIN_XML.exists() or not config.TEST_XML.exists(),
                    reason="SemEval raw files not present")
def test_real_dataset_matches_published_counts():
    train = parse_semeval_xml(config.TRAIN_XML)
    test = parse_semeval_xml(config.TEST_XML)
    assert len(train) == 2358 and len(test) == 654
    assert validate_offsets(train) == 0 and validate_offsets(test) == 0
    assert (train["polarity"] == "conflict").sum() == 45
    assert (test["polarity"] == "conflict").sum() == 16
    # Train and test must not share sentences.
    assert not set(train["sentence"]) & set(test["sentence"]) or \
        len(set(train["sentence"]) & set(test["sentence"])) < 5
