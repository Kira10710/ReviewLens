from pathlib import Path

import pytest

from src import html_parser as hp

FIX = Path(__file__).parent / "fixtures"


def fx(name):
    return (FIX / name).read_text(encoding="utf-8")


def test_parse_review_list_fixture_fields():
    recs = hp.parse_review_page(fx("review_page_synthetic.html"),
                                "https://www.amazon.com/product-reviews/B000000001", "B000000001")
    assert len(recs) == 5                      # duplicates are removed later, not by the parser
    r = recs[0]
    assert r["review_id"] == "R1AAAAAAAAAAA1" and r["product_id"] == "B000000001"
    assert r["rating"] == 5.0
    assert r["review_title"] == "Fixture title one"     # rating <i> removed from title
    assert "display is excellent" in r["review_text"] and "Second line" in r["review_text"]
    assert r["verified_purchase"] is True
    assert r["review_date_raw"].startswith("Reviewed in the United States")
    assert r["source_url"].endswith("B000000001")


def test_missing_fields_stay_none():
    recs = hp.parse_review_page(fx("review_page_synthetic.html"), "u", "B000000001")
    second, body_only = recs[1], recs[3]
    assert second["review_date"] is None and second["verified_purchase"] is None
    assert body_only["rating"] is None and body_only["review_title"] is None
    assert body_only["review_text"] == "Body only, no title and no rating."


def test_product_page_layout_and_read_more_removed():
    recs = hp.parse_review_page(fx("product_page_synthetic.html"), "u", "B000000001")
    assert len(recs) == 1 and recs[0]["rating"] == 4.5
    assert recs[0]["review_text"] == "Delivery was quick and the screen is sharp."


@pytest.mark.parametrize("html", ["", "   ", None, "<html></html>", "not html at all"])
def test_empty_and_garbage_html_yield_no_records(html):
    assert hp.parse_review_page(html, "u", "B000000001") == []


def test_malformed_html_does_not_crash():
    recs = hp.parse_review_page(fx("malformed.html"), "u", "B000000001")
    assert len(recs) == 1 and recs[0]["review_text"].startswith("Unclosed")


@pytest.mark.parametrize("fixture,status,expected", [
    ("review_page_synthetic.html", 200, hp.OK),
    ("product_page_synthetic.html", 200, hp.OK),
    ("captcha.html", 200, hp.CAPTCHA),
    ("captcha.html", 503, hp.CAPTCHA),
    ("signin.html", 200, hp.SIGNIN),
    ("access_denied.html", 403, hp.ACCESS_DENIED),
    ("not_found.html", 404, hp.NOT_FOUND),
    ("no_reviews.html", 200, hp.NO_REVIEWS),
    ("js_shell.html", 200, hp.UNRECOGNIZED),
])
def test_classify_page(fixture, status, expected):
    assert hp.classify_page(fx(fixture), status) == expected


@pytest.mark.parametrize("status,expected", [(429, hp.THROTTLED), (503, hp.THROTTLED), (403, hp.ACCESS_DENIED),
                                             (404, hp.NOT_FOUND), (200, hp.EMPTY)])
def test_empty_body_statuses(status, expected):
    assert hp.classify_page("", status) == expected


def test_throttle_and_server_error_with_html_body():
    assert hp.classify_page("<html><title>x</title></html>", 429) == hp.THROTTLED
    assert hp.classify_page("<html><title>x</title></html>", 500) == hp.SERVER_ERROR


def test_blocked_page_is_never_ok():
    for name in ("captcha.html", "signin.html", "access_denied.html"):
        assert hp.classify_page(fx(name), 200) != hp.OK
