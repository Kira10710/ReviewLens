"""Pure HTML parsing for Amazon pages. No network access happens here.

The parser targets the ``data-hook`` attributes Amazon has used for customer
reviews (``review``, ``review-title``, ``review-body``, ``review-date``,
``review-star-rating``, ``avp-badge``). Amazon changes markup regionally and over
time, so a layout this parser does not recognise is reported as ``UNRECOGNIZED``
rather than as a successful scrape with zero reviews.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

# Page states returned by classify_page()
OK = "ok"
CAPTCHA = "captcha"
SIGNIN = "signin_required"
ACCESS_DENIED = "access_denied"
THROTTLED = "throttled"
NOT_FOUND = "not_found"
SERVER_ERROR = "server_error"
NO_REVIEWS = "no_reviews"
UNRECOGNIZED = "unrecognized_layout"
EMPTY = "empty_response"

STATE_MESSAGES = {
    OK: "Page parsed.",
    CAPTCHA: "Amazon returned a CAPTCHA / robot-check page. ReviewLens does not solve or bypass CAPTCHAs.",
    SIGNIN: "Amazon requires sign-in to view these reviews. ReviewLens does not log in or use cookies.",
    ACCESS_DENIED: "Amazon denied access (HTTP 403 or an access-denied page).",
    THROTTLED: "Amazon is throttling requests (HTTP 429/503). Scraping was stopped to avoid hammering the site.",
    NOT_FOUND: "The product page was not found (HTTP 404 or 'page not found').",
    SERVER_ERROR: "Amazon returned a server error.",
    NO_REVIEWS: "The page loaded but contains no customer reviews.",
    UNRECOGNIZED: "The page loaded but its layout was not recognised (reviews may be rendered by JavaScript "
                  "or the markup differs from what this parser supports).",
    EMPTY: "The response body was empty.",
    "network_error": "A network error occurred while contacting Amazon (see the page log).",
}
BLOCKING_STATES = {CAPTCHA, SIGNIN, ACCESS_DENIED, THROTTLED}

_CAPTCHA_MARKERS = (
    "enter the characters you see below", "type the characters you see in this image",
    "/errors/validatecaptcha", "opfcaptcha", "not a robot", "automated access to amazon data",
)
_RATING_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:out of|von|sur|su|de|/)\s*5|(\d+(?:[.,]\d+)?)\s*(?:stars?|étoiles|sterne)",
                        re.IGNORECASE)
_STAR_CLASS_RE = re.compile(r"a-star-(?:mini-|small-)?(\d)(?:-(\d))?")


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", "html.parser")


def _review_nodes(soup: BeautifulSoup):
    nodes = soup.select('[data-hook="review"]')
    return [n for n in nodes if n.select_one('[data-hook="review-body"]') is not None]


def classify_page(html: str | None, status_code: int | None = 200) -> str:
    """Return the state of a fetched page (see module constants)."""
    if not html or not html.strip():
        if status_code in (429, 503):
            return THROTTLED
        if status_code == 403:
            return ACCESS_DENIED
        if status_code == 404:
            return NOT_FOUND
        return EMPTY
    soup = _soup(html)
    if _review_nodes(soup):
        return OK  # reviews present beats any other signal
    low = html.lower()
    if any(m in low for m in _CAPTCHA_MARKERS):
        return CAPTCHA
    title = (soup.title.get_text(" ", strip=True).lower() if soup.title else "")
    if (soup.select_one('form[name="signIn"]') is not None or soup.select_one("#ap_email") is not None
            or "sign in" in title or "sign-in" in title or "anmelden" in title):
        return SIGNIN
    if status_code == 429:
        return THROTTLED
    if status_code == 403 or "access denied" in title or "access to this page has been denied" in low:
        return ACCESS_DENIED
    if status_code == 503:
        return THROTTLED
    if status_code == 404 or "page not found" in title or "looking for something?" in low:
        return NOT_FOUND
    if status_code and status_code >= 500:
        return SERVER_ERROR
    if soup.select_one("#cm-cr-dp-review-list, #cm_cr-review_list, [data-hook='top-customer-reviews-widget']") \
            is not None or "no customer reviews" in low or "customer reviews" in low and "0 global ratings" in low:
        return NO_REVIEWS
    return UNRECOGNIZED


def _parse_rating(node) -> float | None:
    el = node.select_one('[data-hook="review-star-rating"], [data-hook="cmps-review-star-rating"]')
    if el is not None:
        m = _RATING_RE.search(el.get_text(" ", strip=True))
        if m:
            return float((m.group(1) or m.group(2)).replace(",", "."))
        for cls in el.get("class", []):
            c = _STAR_CLASS_RE.search(cls)
            if c:
                return float(f"{c.group(1)}.{c.group(2) or 0}")
    return None


def _clean_node_text(el, drop=("i",)) -> str:
    if el is None:
        return ""
    el = BeautifulSoup(str(el), "html.parser")
    for tag in el.find_all(list(drop)):
        tag.decompose()
    for tag in el.select(".a-expander-prompt, [data-hook='expand-collapse-read-more-less'], a.a-link-normal.a-declarative"):
        tag.decompose()
    for br in el.find_all("br"):
        br.replace_with(" ")
    return el.get_text(" ", strip=True)


def parse_review_page(html: str | None, source_url: str | None, product_id: str | None,
                      product_url: str | None = None, scraped_at: str | None = None) -> list[dict]:
    """Extract review records from one Amazon page (product page or review-list page).

    Returns raw records (uncleaned text) using the internal field names. Missing
    fields are ``None``; empty or malformed HTML yields ``[]``. Reviewer names are
    deliberately not collected.
    """
    if not html or not isinstance(html, str):
        return []
    scraped_at = scraped_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for node in _review_nodes(_soup(html)):
        body = _clean_node_text(node.select_one('[data-hook="review-body"]'))
        title = _clean_node_text(node.select_one('[data-hook="review-title"]'))
        date_el = node.select_one('[data-hook="review-date"]')
        date_raw = date_el.get_text(" ", strip=True) if date_el is not None else None
        verified = True if node.select_one('[data-hook="avp-badge"], [data-hook="avp-badge-linkless"]') is not None else None
        records.append({
            "review_id": node.get("id") or None,
            "product_id": product_id,
            "product_url": product_url,
            "source_url": source_url,
            "review_title": title or None,
            "review_text": body or None,
            "rating": _parse_rating(node),
            "review_date": date_raw,
            "review_date_raw": date_raw,
            "verified_purchase": verified,
            "scraped_at": scraped_at,
        })
    return records
