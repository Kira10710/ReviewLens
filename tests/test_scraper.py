"""Scraper tests. No live network: a fake fetcher serves synthetic fixtures."""
from pathlib import Path

import pytest
import requests

from src import amazon_scraper as az
from src.amazon_scraper import (FetchError, FetchResult, InvalidAmazonURL, RobotsRules, SafeFetcher,
                                extract_asin, scrape_product_reviews, validate_amazon_url)

FIX = Path(__file__).parent / "fixtures"
ASIN = "B000000001"
URL = f"https://www.amazon.com/Example-Product/dp/{ASIN}/ref=sr_1_1?keywords=x"


def fx(name):
    return (FIX / name).read_text(encoding="utf-8")


# ------------------------------------------------------------------ URL validation / ASIN
@pytest.mark.parametrize("url,asin", [
    ("https://www.amazon.com/dp/B07XJ8C8F5", "B07XJ8C8F5"),
    ("https://www.amazon.com/Some-Title-Here/dp/B07XJ8C8F5/ref=sr_1_3?x=1", "B07XJ8C8F5"),
    ("http://amazon.in/gp/product/B07XJ8C8F5", "B07XJ8C8F5"),
    ("https://www.amazon.co.uk/product-reviews/b07xj8c8f5/?pageNumber=2", "B07XJ8C8F5"),
    ("https://smile.amazon.de/gp/aw/d/B07XJ8C8F5?th=1", "B07XJ8C8F5"),
    ("https://www.amazon.com/dp/0306406152", "0306406152"),
])
def test_valid_urls_and_asin(url, asin):
    v = validate_amazon_url(url)
    assert v.asin == asin and v.canonical_url == f"https://{v.host}/dp/{asin}"


@pytest.mark.parametrize("url,msg", [
    ("", "enter"), (None, "enter"), ("   ", "enter"),
    ("ftp://www.amazon.com/dp/B07XJ8C8F5", "http"),
    ("https://example.com/dp/B07XJ8C8F5", "not a supported"),
    ("https://amazon.com.evil.com/dp/B07XJ8C8F5", "not a supported"),
    ("https://evilamazon.com/dp/B07XJ8C8F5", "not a supported"),
    ("https://www.amazon.com@evil.com/dp/B07XJ8C8F5", "credentials|not a supported"),
    ("https://user:pw@www.amazon.com/dp/B07XJ8C8F5", "credentials"),
    ("https://www.amazon.com:8443/dp/B07XJ8C8F5", "ports"),
    ("http://127.0.0.1/dp/B07XJ8C8F5", "not a supported"),
    ("https://www.amazon.com/", "ASIN"),
    ("https://www.amazon.com/dp/TOOSHORT", "ASIN"),
    ("not a url", "http"),
])
def test_invalid_urls_rejected(url, msg):
    with pytest.raises(InvalidAmazonURL, match=f"(?i)({msg})"):
        validate_amazon_url(url)


def test_extract_asin_none_when_absent():
    assert extract_asin("https://www.amazon.com/s?k=laptop") is None


# ------------------------------------------------------------------ robots.txt rules
def test_robots_wildcards_and_allow_override():
    rules = RobotsRules(fx("robots_sample.txt"), agent="otherbot")     # falls back to the * group
    assert not rules.allowed("/gp/cart/view")
    assert not rules.allowed(f"/product-reviews/{ASIN}/?pageNumber=1")
    assert not rules.allowed(f"/Some-Title/product-reviews/{ASIN}")
    assert rules.allowed("/product-reviews/public")                     # longest match: Allow wins
    assert rules.allowed(f"/dp/{ASIN}")


def test_robots_specific_agent_group_wins():
    rules = RobotsRules(fx("robots_sample.txt"), agent="reviewlens")
    assert not rules.allowed("/secret/x") and rules.allowed("/product-reviews/anything")


# ------------------------------------------------------------------ fake fetcher
class FakeFetcher:
    def __init__(self, routes, default=None):
        self.routes, self.default, self.calls = routes, default, []

    def get(self, url):
        self.calls.append(url)
        for key, val in self.routes.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                status, body = val
                return FetchResult(url, status, body, url)
        if self.default is not None:
            status, body = self.default
            return FetchResult(url, status, body, url)
        raise AssertionError(f"unexpected URL {url}")


ROBOTS_OPEN = (200, "User-agent: *\nDisallow: /gp/cart\n")
ROBOTS_NO_REVIEWS = (200, "User-agent: *\nDisallow: /product-reviews/\n")
noop_sleep = lambda s: None


def run(fetcher, max_pages=3):
    return scrape_product_reviews(URL, max_pages, fetcher=fetcher, delay=0, sleep=noop_sleep)


def test_successful_scrape_with_pagination_and_dedup():
    f = FakeFetcher({
        "robots.txt": ROBOTS_OPEN,
        f"/dp/{ASIN}": (200, fx("product_page_synthetic.html")),
        "pageNumber=1": (200, fx("review_page_synthetic.html")),
        "pageNumber=2": (200, fx("review_page_synthetic.html")),     # same page again -> all duplicates
    })
    res = run(f, max_pages=3)
    assert res.status == "ok" and res.ok and res.asin == ASIN
    assert res.stats["pages_attempted"] == 3 and res.stats["pages_parsed"] == 3
    assert res.stats["reviews_extracted"] == 1 + 5 + 5
    # unique ids: product-page review + R1, R2, R3, R4 (title-only review kept); the rest are duplicates
    assert res.stats["retained"] == 5 and res.stats["dropped_empty"] == 0
    assert res.stats["duplicates_removed"] == 6
    assert set(res.reviews["source"]) == {"amazon_scrape"} and set(res.reviews["product_id"]) == {ASIN}
    assert res.reviews["scraped_at"].notna().all()


def test_max_pages_is_respected():
    f = FakeFetcher({"robots.txt": ROBOTS_OPEN, "/dp/": (200, fx("product_page_synthetic.html"))})
    res = run(f, max_pages=1)
    assert res.stats["pages_attempted"] == 1 and len(f.calls) == 2      # robots.txt + product page


@pytest.mark.parametrize("fixture,status,expected_status,needle", [
    ("captcha.html", 200, "blocked", "CAPTCHA"),
    ("captcha.html", 503, "blocked", "CAPTCHA"),
    ("signin.html", 200, "blocked", "sign-in"),
    ("access_denied.html", 403, "blocked", "denied"),
])
def test_blocked_pages_are_failures_not_empty_successes(fixture, status, expected_status, needle):
    f = FakeFetcher({"robots.txt": ROBOTS_OPEN, "/dp/": (status, fx(fixture))})
    res = run(f)
    assert res.status == expected_status and not res.ok and len(res.reviews) == 0
    assert needle.lower() in res.message.lower()
    assert res.stats["pages_attempted"] == 1          # stopped immediately; did not keep hitting the site


def test_throttled_response_stops():
    f = FakeFetcher({"robots.txt": ROBOTS_OPEN, "/dp/": (429, "")})
    res = run(f)
    assert res.status == "blocked" and "throttl" in res.message.lower()


def test_block_after_some_pages_gives_partial_result():
    f = FakeFetcher({"robots.txt": ROBOTS_OPEN, "/dp/": (200, fx("product_page_synthetic.html")),
                     "pageNumber=1": (200, fx("captcha.html"))})
    res = run(f)
    assert res.status == "partial" and res.ok and res.stats["retained"] == 1
    assert "CAPTCHA" in res.message


def test_timeout_is_reported_as_failure():
    f = FakeFetcher({"robots.txt": ROBOTS_OPEN, "/dp/": FetchError("timeout", "Request timed out")})
    res = run(f)
    assert res.status == "failed" and len(res.reviews) == 0 and "network" in res.message.lower()
    assert res.pages[-1].outcome == "network_error"


def test_robots_unreachable_means_nothing_is_scraped():
    f = FakeFetcher({"robots.txt": FetchError("timeout", "timed out")})
    res = run(f)
    assert res.status == "failed" and "robots" in res.message.lower() and len(f.calls) == 1


def test_robots_captcha_page_counts_as_blocked():
    f = FakeFetcher({"robots.txt": (200, fx("captcha.html"))})
    assert run(f).status == "blocked"


def test_robots_disallowed_review_pages_are_skipped():
    f = FakeFetcher({"robots.txt": ROBOTS_NO_REVIEWS, "/dp/": (200, fx("product_page_synthetic.html"))})
    res = run(f, max_pages=4)
    assert res.status == "partial" and "robots.txt" in res.message
    assert not any("product-reviews" in c for c in f.calls)           # never requested
    assert any(p.outcome == "skipped_robots" for p in res.pages)


def test_unrecognised_layout_is_failure():
    f = FakeFetcher({"robots.txt": ROBOTS_OPEN}, default=(200, fx("js_shell.html")))
    res = run(f, max_pages=2)
    assert res.status == "failed" and "not recognised" in res.message and len(res.reviews) == 0


def test_page_without_reviews_is_no_reviews_status():
    f = FakeFetcher({"robots.txt": ROBOTS_OPEN, "/dp/": (200, fx("no_reviews.html")),
                     "pageNumber=1": (200, fx("no_reviews.html"))})
    res = run(f)
    assert res.status == "no_reviews" and len(res.reviews) == 0


def test_invalid_url_returns_failed_result_without_network():
    f = FakeFetcher({})
    res = scrape_product_reviews("https://evil.example/dp/B000000001", 2, fetcher=f)
    assert res.status == "failed" and f.calls == []


# ------------------------------------------------------------------ SafeFetcher (redirects, SSRF, timeouts)
class FakeResponse:
    def __init__(self, status, headers=None, body=b"<html></html>"):
        self.status_code, self.headers, self._body, self.encoding = status, headers or {}, body, "utf-8"

    def iter_content(self, n):
        yield self._body

    def close(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses, self.headers, self.urls = list(responses), {}, []

    def get(self, url, **kw):
        self.urls.append(url)
        assert kw["allow_redirects"] is False and "cookies" not in kw
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_redirect_to_external_host_is_refused():
    sess = FakeSession([FakeResponse(302, {"Location": "https://evil.example/steal"})])
    fetcher = SafeFetcher(resolver=lambda h: True, session=sess)
    with pytest.raises(FetchError) as ei:
        fetcher.get("https://www.amazon.com/dp/B000000001")
    assert ei.value.kind == "blocked_redirect" and len(sess.urls) == 1


def test_redirect_to_internal_address_is_refused():
    sess = FakeSession([FakeResponse(302, {"Location": "http://169.254.169.254/latest/meta-data"})])
    with pytest.raises(FetchError, match="non-Amazon"):
        SafeFetcher(resolver=lambda h: True, session=sess).get("https://www.amazon.com/dp/B000000001")


def test_amazon_host_resolving_to_private_ip_is_refused():
    fetcher = SafeFetcher(resolver=lambda h: False, session=FakeSession([]))
    with pytest.raises(FetchError) as ei:
        fetcher.get("https://www.amazon.com/dp/B000000001")
    assert ei.value.kind == "private_address"


def test_redirect_to_signin_is_reported_without_fetching_login_page():
    sess = FakeSession([FakeResponse(302, {"Location": "https://www.amazon.com/ap/signin?openid=x"})])
    res = SafeFetcher(resolver=lambda h: True, session=sess).get("https://www.amazon.com/product-reviews/B000000001")
    assert res.redirected_to_signin and len(sess.urls) == 1


def test_valid_amazon_redirect_followed_then_limit(monkeypatch):
    monkeypatch.setattr(az.time, "sleep", lambda s: None)
    sess = FakeSession([FakeResponse(301, {"Location": "https://www.amazon.com/dp/B000000001"}),
                        FakeResponse(200, body=b"<html>ok</html>")])
    res = SafeFetcher(resolver=lambda h: True, session=sess).get("https://amazon.com/dp/B000000001")
    assert res.status_code == 200 and res.final_url.startswith("https://www.amazon.com")
    loop = FakeSession([FakeResponse(302, {"Location": "https://www.amazon.com/x"}) for _ in range(10)])
    with pytest.raises(FetchError, match="redirects"):
        SafeFetcher(resolver=lambda h: True, session=loop).get("https://www.amazon.com/x")


def test_timeout_retries_then_raises(monkeypatch):
    monkeypatch.setattr(az.time, "sleep", lambda s: None)
    sess = FakeSession([requests.Timeout("slow")] * (az.MAX_RETRIES + 1))
    with pytest.raises(FetchError) as ei:
        SafeFetcher(resolver=lambda h: True, session=sess).get("https://www.amazon.com/dp/B000000001")
    assert ei.value.kind == "timeout" and len(sess.urls) == az.MAX_RETRIES + 1


def test_user_agent_is_honest_and_no_cookies_sent():
    f = SafeFetcher(resolver=lambda h: True, session=FakeSession([]))
    assert "ReviewLens" in f.session.headers["User-Agent"]


def test_non_ok_pages_record_diagnostics_without_changing_outcome():
    class F(FakeFetcher):
        def get(self, url):
            r = super().get(url)
            r.content_type = "text/html;charset=UTF-8"
            return r
    f = F({"robots.txt": ROBOTS_OPEN, "/dp/": (200, fx("captcha.html"))})
    res = run(f)
    assert res.status == "blocked"
    note = res.pages[-1].note
    assert "content-type='text/html;charset=UTF-8'" in note and "chars" in note
