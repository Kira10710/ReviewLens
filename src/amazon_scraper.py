"""Amazon-specific review collection.

Design rules (see README "Scraping policy"):
* Only Amazon storefront hosts are accepted, including after every redirect hop,
  and hosts must resolve to public IP addresses.
* robots.txt is fetched and obeyed for every path requested.
* No CAPTCHA solving, no login, no cookies, no proxy rotation, no browser spoofing.
  The scraper identifies itself honestly and stops at the first blocking response.
* Network access (``SafeFetcher``) is separate from parsing (``html_parser``) so
  the logic can be tested offline with fixtures and a fake fetcher.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

import pandas as pd
import requests

from src import html_parser as hp
from src.review_preprocessing import SOURCE_SCRAPE, clean_and_deduplicate_reviews

USER_AGENT = "ReviewLens/1.0 (college NLP project; honest-bot; contact: project owner)"
CONNECT_TIMEOUT, READ_TIMEOUT = 5, 20
MAX_RETRIES = 2                      # extra attempts for timeouts / connection errors only
MAX_REDIRECTS = 4
MAX_BYTES = 3_000_000
DEFAULT_DELAY_SECONDS = 2.5          # pause between requests to the same site
MAX_PAGES_HARD_LIMIT = 10

# Registrable Amazon storefront domains. A host must equal one of these or be a subdomain.
ALLOWED_AMAZON_DOMAINS = (
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.it", "amazon.es", "amazon.nl",
    "amazon.se", "amazon.pl", "amazon.ca", "amazon.com.mx", "amazon.com.br", "amazon.in",
    "amazon.co.jp", "amazon.com.au", "amazon.sg", "amazon.ae", "amazon.sa", "amazon.com.tr",
    "amazon.eg", "amazon.com.be",
)
_ASIN_PATTERNS = (
    re.compile(r"/(?:dp|gp/product|gp/aw/d|product-reviews|gp/offer-listing|exec/obidos/ASIN)/([A-Z0-9]{10})(?:[/?#]|$)",
               re.IGNORECASE),
    re.compile(r"/(?:[^/]+)/dp/([A-Z0-9]{10})(?:[/?#]|$)", re.IGNORECASE),
)


class InvalidAmazonURL(ValueError):
    """The URL is not an acceptable Amazon product URL."""


class FetchError(RuntimeError):
    """A network-level failure (timeout, DNS, TLS, redirect policy violation, ...)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class ValidatedURL:
    host: str
    asin: str
    canonical_url: str
    original_url: str


@dataclass
class FetchResult:
    url: str
    status_code: int | None
    text: str
    final_url: str
    redirected_to_signin: bool = False
    content_type: str = ""


@dataclass
class PageOutcome:
    url: str
    kind: str                    # "product_page" | "review_page" | "robots.txt"
    outcome: str                 # an html_parser state, "skipped_robots", "network_error" ...
    http_status: int | None = None
    reviews_extracted: int = 0
    note: str = ""


@dataclass
class ScrapeResult:
    status: str                  # ok | partial | blocked | failed | no_reviews
    reviews: pd.DataFrame
    stats: dict
    pages: list[PageOutcome] = field(default_factory=list)
    asin: str | None = None
    canonical_url: str | None = None
    message: str = ""
    started_at: str = ""
    finished_at: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "partial") and len(self.reviews) > 0


# ---------------------------------------------------------------------------- URL validation
def is_allowed_amazon_host(host: str | None) -> bool:
    if not host:
        return False
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in ALLOWED_AMAZON_DOMAINS)


def extract_asin(url: str) -> str | None:
    path = urlsplit(url).path
    for pat in _ASIN_PATTERNS:
        m = pat.search(path)
        if m:
            return m.group(1).upper()
    return None


def validate_amazon_url(url) -> ValidatedURL:
    """Validate a user-supplied URL and extract its ASIN. Raises InvalidAmazonURL."""
    if not isinstance(url, str) or not url.strip():
        raise InvalidAmazonURL("Please enter an Amazon product URL.")
    url = url.strip()
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise InvalidAmazonURL("The URL could not be parsed.")
    if parts.scheme not in ("http", "https"):
        raise InvalidAmazonURL("The URL must start with http:// or https://.")
    if parts.username or parts.password:
        raise InvalidAmazonURL("URLs containing credentials are not accepted.")
    if port not in (None, 80, 443):
        raise InvalidAmazonURL("Custom ports are not accepted.")
    host = (parts.hostname or "").lower().rstrip(".")
    if not is_allowed_amazon_host(host):
        raise InvalidAmazonURL(f"'{host or url}' is not a supported Amazon storefront. Accepted: "
                               + ", ".join("www." + d for d in ALLOWED_AMAZON_DOMAINS[:6]) + ", ...")
    asin = extract_asin(url)
    if not asin:
        raise InvalidAmazonURL("No product identifier (ASIN) found. Use a product page URL such as "
                               "https://www.amazon.com/dp/B0XXXXXXXX or .../product-reviews/B0XXXXXXXX.")
    return ValidatedURL(host=host, asin=asin, canonical_url=f"https://{host}/dp/{asin}", original_url=url)


def host_resolves_publicly(host: str) -> bool:
    """True only if every address the host resolves to is a public IP (SSRF guard)."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return False
    return bool(infos)


# ---------------------------------------------------------------------------- robots.txt
class RobotsRules:
    """Minimal wildcard-aware robots.txt matcher (Python's robotparser ignores '*' in paths)."""

    def __init__(self, text: str = "", agent: str = "reviewlens"):
        self.rules: list[tuple[bool, re.Pattern, int]] = []
        groups, current, in_rules = {}, [], False
        for line in (text or "").splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (s.strip() for s in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if in_rules:
                    current, in_rules = [], False
                current.append(value.lower())
                groups.setdefault(value.lower(), [])
            elif key in ("allow", "disallow"):
                in_rules = True
                for agent_name in current:
                    groups[agent_name].append((key == "allow", value))
        chosen = groups.get(agent.lower())
        if chosen is None:
            chosen = groups.get("*", [])
        for allow, pattern in chosen:
            if pattern == "":
                continue
            regex = re.escape(pattern).replace(r"\*", ".*")
            if regex.endswith(r"\$"):
                regex = regex[:-2] + "$"
            self.rules.append((allow, re.compile(regex), len(pattern)))

    def allowed(self, path_and_query: str) -> bool:
        best = None
        for allow, regex, length in self.rules:
            if regex.match(path_and_query):
                if best is None or length > best[1] or (length == best[1] and allow):
                    best = (allow, length)
        return True if best is None else best[0]


# ---------------------------------------------------------------------------- network
class SafeFetcher:
    """HTTP client that validates every redirect hop and never sends cookies or credentials."""

    def __init__(self, resolver=host_resolves_publicly, session=None):
        self.resolver = resolver
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml",
                                     "Accept-Language": "en"})

    def _check_host(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not is_allowed_amazon_host(parts.hostname):
            raise FetchError("blocked_redirect", f"Refusing to contact non-Amazon host: {parts.hostname}")
        if not self.resolver(parts.hostname):
            raise FetchError("private_address", f"{parts.hostname} does not resolve to a public address.")

    def get(self, url: str) -> FetchResult:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            self._check_host(current)
            last_error = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = self.session.get(current, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                                            allow_redirects=False, stream=True)
                    break
                except requests.Timeout as exc:
                    last_error = FetchError("timeout", f"Request timed out after {READ_TIMEOUT}s: {exc}")
                except requests.RequestException as exc:
                    last_error = FetchError("network", f"Network error: {exc.__class__.__name__}: {exc}")
                time.sleep(1.5 * (attempt + 1))
            else:
                raise last_error
            try:
                if 300 <= resp.status_code < 400 and resp.headers.get("Location"):
                    nxt = urljoin(current, resp.headers["Location"])
                    if re.search(r"/ap/signin|/ap/widget|/errors/validateCaptcha", nxt):
                        return FetchResult(url, resp.status_code, "", nxt, redirected_to_signin=True)
                    current = nxt
                    continue
                body = b""
                for chunk in resp.iter_content(65536):
                    body += chunk
                    if len(body) > MAX_BYTES:
                        break
                text = body.decode(resp.encoding or "utf-8", errors="replace")
                return FetchResult(url, resp.status_code, text, current,
                                   content_type=resp.headers.get("Content-Type", ""))
            finally:
                resp.close()
        raise FetchError("too_many_redirects", f"More than {MAX_REDIRECTS} redirects.")


# ---------------------------------------------------------------------------- orchestration
def _empty_stats() -> dict:
    return {"pages_attempted": 0, "pages_parsed": 0, "reviews_extracted": 0, "duplicates_removed": 0,
            "dropped_empty": 0, "retained": 0}


def plan_pages(asin: str, host: str, max_pages: int) -> list[tuple[str, str]]:
    """(kind, url) request plan: product page first, then review-list pages."""
    plan = [("product_page", f"https://{host}/dp/{asin}")]
    for n in range(1, max_pages):
        plan.append(("review_page", f"https://{host}/product-reviews/{asin}/?pageNumber={n}&reviewerType=all_reviews"))
    return plan[:max_pages]


def scrape_product_reviews(product_url: str, max_pages: int = 3, fetcher=None,
                           delay: float = DEFAULT_DELAY_SECONDS, sleep=time.sleep,
                           progress=None) -> ScrapeResult:
    """Collect reviews for one Amazon product.

    ``max_pages`` caps the total number of page requests (product page + review pages).
    ``progress`` is an optional ``callable(message, fraction)``. Never fabricates
    records: if nothing can be collected the result says why.
    """
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = progress or (lambda msg, frac: None)
    stats, pages = _empty_stats(), []

    def finish(status, message, df=None, asin=None, canonical=None):
        return ScrapeResult(status, df if df is not None else pd.DataFrame(), stats, pages, asin,
                            canonical, message, started, datetime.now(timezone.utc).isoformat(timespec="seconds"))

    try:
        target = validate_amazon_url(product_url)
    except InvalidAmazonURL as exc:
        return finish("failed", str(exc))
    max_pages = max(1, min(int(max_pages), MAX_PAGES_HARD_LIMIT))
    fetcher = fetcher or SafeFetcher()

    # robots.txt (compliance check, not a review page)
    robots_url = f"https://{target.host}/robots.txt"
    report("Checking robots.txt ...", 0.02)
    try:
        rres = fetcher.get(robots_url)
    except FetchError as exc:
        pages.append(PageOutcome(robots_url, "robots.txt", "network_error", note=str(exc)))
        return finish("failed", f"Could not retrieve robots.txt ({exc.kind}): {exc}. Nothing was scraped "
                      "because robots rules could not be verified.", asin=target.asin, canonical=target.canonical_url)
    if rres.status_code == 200 and rres.text.strip():
        state = hp.classify_page(rres.text, rres.status_code)
        if state in hp.BLOCKING_STATES:
            pages.append(PageOutcome(robots_url, "robots.txt", state, rres.status_code))
            return finish("blocked", hp.STATE_MESSAGES[state] + " Even robots.txt was not served, so nothing was scraped.",
                          asin=target.asin, canonical=target.canonical_url)
        rules = RobotsRules(rres.text)
        pages.append(PageOutcome(robots_url, "robots.txt", "ok", 200, note=f"{len(rules.rules)} rules loaded"))
    elif rres.status_code == 404:
        rules = RobotsRules("")
        pages.append(PageOutcome(robots_url, "robots.txt", "not_found", 404, note="no robots.txt: nothing disallowed"))
    else:
        pages.append(PageOutcome(robots_url, "robots.txt", "unavailable", rres.status_code))
        return finish("failed", f"robots.txt could not be retrieved (HTTP {rres.status_code}); nothing was scraped "
                      "because robots rules could not be verified.", asin=target.asin, canonical=target.canonical_url)

    plan = plan_pages(target.asin, target.host, max_pages)
    raw_records: list[dict] = []
    stop_state = None
    last_bad_state = None
    skipped_robots = 0
    for i, (kind, url) in enumerate(plan):
        report(f"Page {i + 1}/{len(plan)}: {kind.replace('_', ' ')}", 0.1 + 0.85 * i / len(plan))
        parts = urlsplit(url)
        if not rules.allowed(parts.path + ("?" + parts.query if parts.query else "")):
            skipped_robots += 1
            pages.append(PageOutcome(url, kind, "skipped_robots", note="disallowed by robots.txt"))
            if kind == "review_page":
                break  # all review-list pages share the same disallowed prefix
            continue
        if stats["pages_attempted"] > 0:
            sleep(delay)
        stats["pages_attempted"] += 1
        try:
            res = fetcher.get(url)
        except FetchError as exc:
            pages.append(PageOutcome(url, kind, "network_error", note=f"{exc.kind}: {exc}"))
            stop_state = "network_error"
            break
        if res.redirected_to_signin:
            state = hp.SIGNIN
        else:
            state = hp.classify_page(res.text, res.status_code)
        if state == hp.OK:
            recs = hp.parse_review_page(res.text, url, target.asin, target.canonical_url)
            stats["pages_parsed"] += 1
            stats["reviews_extracted"] += len(recs)
            raw_records.extend(recs)
            pages.append(PageOutcome(url, kind, state, res.status_code, len(recs)))
            continue
        note = hp.STATE_MESSAGES.get(state, "")
        m = re.search(r"<title[^>]*>(.*?)</title>", res.text or "", re.IGNORECASE | re.DOTALL)
        note += (f" [diagnostic: content-type={res.content_type!r}, {len(res.text or '')} chars, "
                 f"title={(m.group(1).strip()[:80] if m else None)!r}]")
        pages.append(PageOutcome(url, kind, state, res.status_code, note=note))
        if state in hp.BLOCKING_STATES or (state == hp.NOT_FOUND and kind == "product_page"):
            stop_state = state            # never keep hitting a site that is refusing us
            break
        if kind == "review_page":
            if state in (hp.NO_REVIEWS, hp.NOT_FOUND):
                break                     # normal end of pagination
            stop_state = state            # unrecognised layout / server error on a review page
            break
        last_bad_state = state            # product page without parsable reviews: try review pages
    report("Cleaning and de-duplicating ...", 0.97)
    for rec in raw_records:
        rec["source"] = SOURCE_SCRAPE
    df, cstats = clean_and_deduplicate_reviews(raw_records, source=SOURCE_SCRAPE)
    stats.update({k: cstats[k] for k in ("duplicates_removed", "dropped_empty", "retained")})
    report("Done.", 1.0)

    if len(df):
        notes = []
        if stop_state:
            notes.append(f"Stopped early: {hp.STATE_MESSAGES.get(stop_state, stop_state)}")
        if skipped_robots:
            notes.append(f"{skipped_robots} page(s) skipped because robots.txt disallows them.")
        status = "partial" if notes else "ok"
        msg = (f"Collected {len(df)} unique review(s) from {stats['pages_parsed']} page(s)."
               + (" " + " ".join(notes) if notes else ""))
        return finish(status, msg, df, target.asin, target.canonical_url)
    if stop_state in hp.BLOCKING_STATES:
        return finish("blocked", hp.STATE_MESSAGES[stop_state] + " No reviews were collected.", df,
                      target.asin, target.canonical_url)
    bad = stop_state or last_bad_state
    if bad == hp.NO_REVIEWS and not stop_state:
        return finish("no_reviews", hp.STATE_MESSAGES[hp.NO_REVIEWS], df, target.asin, target.canonical_url)
    if bad:
        detail = hp.STATE_MESSAGES.get(bad, bad)
        return finish("failed", f"{detail} No reviews were collected.", df, target.asin, target.canonical_url)
    if skipped_robots and stats["pages_attempted"] == 0:
        return finish("failed", "Every page needed was disallowed by robots.txt, so nothing was requested.",
                      df, target.asin, target.canonical_url)
    return finish("no_reviews", "Pages were fetched and parsed, but they contained no reviews.", df,
                  target.asin, target.canonical_url)
