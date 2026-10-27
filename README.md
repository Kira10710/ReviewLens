# ReviewLens: Amazon Review Collection and Aspect-Based Sentiment Analysis

ReviewLens takes an **Amazon product URL**, collects the customer reviews Amazon makes publicly
accessible, cleans them, predicts whole-review sentiment, extracts product aspects (battery, screen,
price, ...), predicts the sentiment of **each aspect**, and shows everything in a Streamlit dashboard
with CSV export. If Amazon blocks access, the same pipeline runs on a **CSV you upload**.

> **Honest status of live scraping:** one real attempt was made from the build machine (see
> [Live scraping status](#live-scraping-status)). Amazon returned a CAPTCHA, the scraper stopped as
> designed, and **0 real reviews were collected**. The parser and failure handling are verified only
> against synthetic HTML fixtures. Live scraping success is **not** demonstrated and should not be assumed.

## Data flow

```
Amazon URL --validate--> robots.txt check --> product page / review pages (stop at first block)
   |                                                        |
   |                                              html_parser (pure, offline-testable)
   v                                                        v
CSV upload (fallback) ------------------> review_preprocessing (clean, dedupe, schema)
                                                            |
                                  +-------------------------+--------------------------+
                                  v                                                    v
                    sentiment_model (TF-IDF + LR)                       aspect_extractor (rules)
                    whole-review pos/neu/neg                                     |
                                  |                                   aspect_sentiment (trained ABSA model
                                  |                                   applied to each aspect's own clause)
                                  +---------------------> pipeline ----------------+
                                                              |
                                   Streamlit dashboard  +  3 CSV exports
```

| Module | Responsibility |
|---|---|
| `src/amazon_scraper.py` | URL validation, ASIN extraction, SSRF-safe fetching, robots.txt, pagination, failure reporting |
| `src/html_parser.py` | Parse review HTML; classify pages (OK / CAPTCHA / sign-in / denied / throttled / unrecognised) |
| `src/review_preprocessing.py` | Internal schema, cleaning, de-duplication, CSV validation |
| `src/sentiment_model.py`, `src/evaluation.py` | Whole-review sentiment: train, load, held-out evaluation |
| `src/aspect_extractor.py`, `config/aspect_vocabulary.json` | Rule-based aspect extraction |
| `src/aspect_sentiment.py`, `src/predictor.py`, `src/train.py`, `src/evaluate.py` | Aspect sentiment model (SemEval-2014 laptops) |
| `src/pipeline.py` | Ties it together, exports, frequency themes |
| `app.py` | Streamlit dashboard |

## Setup (Windows PowerShell)

Requires Python 3.11+ (built and tested with **3.12.0**; 3.11 was not installed on the build machine).

```powershell
cd C:\Prakrut\NLP\ReviewLens
py -3.12 -m venv .venv            # or: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train and evaluate the models once (downloads the public benchmark data; about 2 minutes total):

```powershell
python -m src.data_loader        # SemEval-2014 laptop aspect data (aspect sentiment model)
python -m src.train              # aspect sentiment model (cross-validation on train split only)
python -m src.evaluate           # -> reports/metrics.json (SemEval test split)
python -m src.sentiment_model    # whole-review model: downloads Amazon Reviews Multi (EN), trains, saves
python -m src.evaluation         # -> reports/review_sentiment_metrics.json (official test split)
python -m pytest -v              # 328 tests, no network needed, Gemini is mocked
```

## Run the app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open http://localhost:8501.

### Giving it an Amazon URL
Open **Scrape Amazon Reviews**, paste a product URL such as `https://www.amazon.com/dp/B0XXXXXXXXX` (any
`/dp/`, `/gp/product/` or `/product-reviews/` URL with a 10-character ASIN works), pick a page limit and click
**Start scraping**. The page shows the ASIN, pages attempted, reviews extracted, duplicates removed and
reviews retained, plus a page-by-page log. Accepted hosts are Amazon storefronts only (`amazon.com`, `.co.uk`,
`.de`, `.in`, ... see `ALLOWED_AMAZON_DOMAINS`); short links such as `amzn.to` are not expanded.

### Uploading a CSV (fallback when scraping is blocked)
Use the **Upload CSV instead** tab. Required column: `review_text`. Optional: `rating` (1-5),
`review_title`, `review_date`, `review_id`, `product_id`. Up to 5000 rows are processed; empty rows are
skipped and counted; malformed or column-less files give a clear error. Uploaded rows are tagged
`source = csv_upload` everywhere, never "scraped". A sample is in `data/samples/sample_reviews.csv`
(hand-written demo text, **not** real customer reviews).

## Scraping policy and live scraping status

What the scraper does: validates the host (Amazon only, re-checked on every redirect, resolved IPs must be
public), fetches and **obeys `robots.txt`** (wildcard-aware), identifies itself as `ReviewLens/1.0`, waits 2.5 s
between requests, uses timeouts (5 s connect / 20 s read), retries only on network errors (2 retries), sends no
cookies or login, and **stops at the first CAPTCHA, sign-in wall, access-denied or 429/503 response**. It does
not solve CAPTCHAs, rotate proxies, spoof browsers or use headless browsers. It collects public review fields
only (title, text, rating, date, verified flag); reviewer names are deliberately not collected.
Amazon's terms of use restrict automated access; this is a coursework tool, so use it sparingly and at your
own responsibility.

### Live scraping status
* **Tested:** one attempt on 2026-10-20 with `https://www.amazon.com/dp/B0BDHWDR12`, 2-page limit
  (`reports/live_scrape_attempt.json`).
* **Outcome:** `robots.txt` was served (137 rules; the paths requested were allowed). The product page returned
  HTTP 200 with a layout the parser did not recognise (reviews are probably rendered by JavaScript or hidden
  behind a bot check). The review-list page returned a **CAPTCHA**. Status `blocked`; **0 reviews collected**.
  Nothing was substituted or fabricated.
* **Therefore:** live end-to-end scraping is **unverified**. Amazon access varies by region, page layout,
  time and your network. The parser targets `data-hook="review"` markup observed in Amazon HTML; it was built
  and tested only on synthetic fixtures that imitate that structure.

## Imported datasets (separate from live scraping)

The **Import Dataset** page analyzes review files you already have, such as the Datafiniti Amazon exports in
`Dataset/`. Records are tagged `source = dataset_import`, are never labelled "scraped", and have no source URL.
`src/dataset_loader.py` does the work:

* **Header verification:** a header row is accepted only if at least two first-row cells are known column names;
  a file whose first row looks like data (timestamps, URLs, ASINs) is refused unless explicit `column_names` are
  given. The three provided files all have header rows (comma-delimited UTF-8; two with a BOM).
* **Column mapping** (source -> internal): `reviews.text` -> `review_text`, `reviews.rating` -> `rating`,
  `reviews.title` -> `review_title`, `reviews.date` -> `review_date` (ISO timestamps parsed),
  `reviews.id` -> `review_id`, `reviews.didPurchase` -> `verified_purchase`, `id` -> product grouping key,
  `asins` -> `product_id` (first ASIN; full list kept in `asins_all`), `name` -> `product_title` (most common
  non-empty name per product), plus `brand`, `categories`, `num_helpful`, `do_recommend`. Reviewer usernames and
  locations are never loaded. Generic names (`review_text`, `rating`, `stars`, ...) also work.
* **Products:** the page lists every product with its review count and requires you to choose one (more can be
  added explicitly). Reviews of different products are not mixed by default, and de-duplication is per product.
* **Memory:** files are streamed with the `csv` module and only the needed columns are kept, so the 250 MB file
  is never held in memory. Rows with the wrong number of fields are skipped and counted. A cap (default 2000,
  max 5000) applies to the selection, using the first reviews in file order, not a random sample.
* **Analysis limit:** choose 20, 100, 500, 1,000, 2,000 or the maximum available (safety cap 5,000). The page
  shows the exact number of unique valid reviews available for the selection, the limit, and how many will be
  analyzed. Duplicates and empty-text rows never count towards the limit, and the number loaded is the number sent
  to the pipeline (tested). The Data Export table is a 20-row *preview*; the CSV downloads contain every row.
* A file with no review-text column (metadata only) is reported as such; nothing is invented. Missing ratings stay
  empty.

Contents of the provided files (`reports/dataset_import_summary.json`):

| File | Data rows | Products | Rows with text | Unique valid reviews after de-duplication |
|---|---|---|---|---|
| `1429_1.csv` | 34,660 | 42 | 34,658 | 34,655 |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv` | 5,000 | 24 | 5,000 | 4,904 |
| `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` | 28,332 | 65 | 28,331 | 26,252 |

All three have 0 malformed rows. The product-level dashboards are built from one product at a time.

## AI Insights with Gemini (optional)

The **Aspect Insights** page has an **AI Insights (Gemini)** section that writes a natural-language summary of one
product's reviews: overall consensus, aspect-wise findings, pros, cons, a buying recommendation, and a confidence note
that preserves conflicting opinions. It is **optional**: without an API key the section says Gemini is unavailable and
every other feature (sentiment, aspects, review table, exports, evaluation) works unchanged.

### Get and configure an API key
1. Create a key in [Google AI Studio](https://aistudio.google.com/apikey) (a free tier with rate limits exists; check
   your own quota and billing there).
2. Provide it **either** as an environment variable (PowerShell, current session):
   ```powershell
   $env:GEMINI_API_KEY = "your-key"
   streamlit run app.py
   ```
   **or** in Streamlit secrets: copy `.streamlit\secrets.toml.example` to `.streamlit\secrets.toml` (git-ignored) and set
   `GEMINI_API_KEY = "your-key"`. The environment variable wins if both exist.
3. Restart the app after changing the key. The sidebar shows "configured (key hidden)"; the key is never printed,
   logged or shown in the UI, and is scrubbed from error messages. Requests are made server-side by the Python process.

Optional settings (environment variables): `GEMINI_MODEL` (default `gemini-2.5-flash`), `GEMINI_TIMEOUT_SECONDS`
(45), `AI_MAX_EXCERPTS` (24), `AI_MAX_INPUT_CHARS` (24000), `AI_EXCERPT_CHARS` (600). The excerpt limit can also be
changed on the page.

### Troubleshooting: "Gemini is unavailable"
Key lookup order (read fresh on every page run, never cached): 1) the `GEMINI_API_KEY` **environment variable of the app
process**, 2) Streamlit secrets, 3) the project's `.streamlit\secrets.toml` (read directly, so it works even if you launch
the app from another folder). Placeholders such as `your-gemini-api-key-here` are ignored, and surrounding quotes or
whitespace are stripped.
* The sidebar shows `Gemini: configured (<source>)` or `off`. Under **Aspect Insights > Gemini configuration check** the
  page shows which source was found, the files it looked for, the process ID, the port and the folder this instance
  was started from. The key itself is never shown or logged.
* **A variable set in one PowerShell window is invisible to an app started in another window or by another tool, and
  to an app that was already running.** Open the browser at the port printed by the terminal where you ran
  `streamlit run` (if 8501 is busy, Streamlit silently uses 8502, 8503, ...). Stop old instances with Ctrl+C.
* Check from a terminal (prints presence, source and length only): `python -m src.ai_insights --check`.

### How it stays grounded
* Gemini receives **ReviewLens's own results** (statistics computed from the real analysis) plus a bounded set of real
  review excerpts, chosen deterministically: one review per sentiment for each of the most-mentioned aspects, then
  filled by whole-review sentiment. The sentiment and aspect models are **not re-run**, and duplicates are removed with
  the same logic as the rest of the pipeline. The page shows reviews analyzed, excerpts sent, and any shortening or
  size-limit dropping, so nothing is truncated silently.
* The prompt tells the model to use only the supplied data, never invent reviews, numbers, specifications or aspects,
  to say "limited evidence" when thin, to keep disagreement visible, and to treat review text as **untrusted data, not
  instructions** (review text is JSON-encoded inside a delimited block and `<`/`>` are neutralised).
* The application **validates the answer**: every pro, con and aspect finding must cite supplied excerpt ids or it is
  dropped; findings about aspects that were not detected are dropped; numbers that are not in the supplied statistics
  and claims of "scraping" for non-scraped data are flagged. All statistics shown are computed by ReviewLens, not
  written by Gemini. Output is labelled "AI-generated by Gemini" and can still contain mistakes.
* One product at a time: mixed-product selections are refused. Imported-dataset reviews are never called live-scraped.

### What is sent to Google
When you click **Generate AI Insights** (and only then), the selected excerpts are sent to Google's Gemini API: review
text (title + body), star rating, ReviewLens's whole-review sentiment and aspect labels, plus aggregate statistics, the
product title and a one-line description of the data source. **Not sent:** usernames, locations, URLs, review or
product IDs, dates, the rest of the dataset. Do not use the feature with reviews you are not allowed to share with
Google.

### Quota, errors and caching
One request per click, no automatic retries. Quota/rate limits, rejected keys, unavailable models, timeouts, network
errors, blocked or malformed answers each show a specific message; the rest of the app is unaffected. Successful
summaries are cached for the same evidence (same reviews, analysis, limits, model and prompt version), so reruns and page
switches make no new calls; changing any of those clears the displayed summary, and **Regenerate** forces a new call.
Failures are not cached.

## Review schema

`review_id, product_id, product_url, source_url, review_title, review_text (original), cleaned_text,
rating, review_date (ISO or empty), review_date_raw, verified_purchase (true or empty), source
(amazon_scrape | csv_upload), scraped_at`. Unavailable fields stay empty; nothing is invented. Cleaning decodes
HTML entities, normalises Unicode/whitespace/quotes and strips control characters, and **keeps negations**
("not", "never", "no") and case. Duplicates are removed by `review_id`, or for id-less rows by a hash of
title + text + date + rating (so two different reviewers posting identical text with no id would be merged).

## Sentiment model (whole review)

* **Model:** TF-IDF (1-2 grams, no stop-word removal) + Logistic Regression with balanced class weights,
  selected over a Linear SVM by macro-F1 on the official **validation** split. Fitted pipeline saved to
  `models/review_sentiment.joblib`, so the dashboard never retrains.
* **Training data:** [Amazon Reviews Multi (English), SetFit mirror](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en):
  200,000 train / 5,000 validation / 5,000 test. **Labels are derived from star ratings** (1-2 negative,
  3 neutral, 4-5 positive). This is a proxy for sentiment and the main weakness of the benchmark: 3-star reviews
  are often not neutral in text. Scraped or uploaded reviews are never used for training.
* **Star ratings in the dashboard are shown separately** and are never used as labels or predictions.

### Held-out test results (`reports/review_sentiment_metrics.json`, 5000 test reviews)

| Accuracy | Macro-F1 | Weighted-F1 | Macro-P | Macro-R | Majority baseline acc. |
|---|---|---|---|---|---|
| **0.7174** | **0.6813** | **0.7263** | 0.6841 | 0.6840 | 0.400 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.800 | 0.744 | 0.771 | 2000 |
| neutral | 0.409 | 0.517 | 0.457 | 1000 |
| positive | 0.844 | 0.792 | 0.817 | 2000 |

Validation macro-F1 was 0.6849. Neutral is clearly the weakest class. 15 texts are identical in train and test.
These scores measure agreement with the star-derived label on Amazon benchmark reviews. They do not
measure agreement with human-judged sentiment, and they say nothing about scraped data.

## Aspect extraction and aspect sentiment

* **Extraction (rule-based, not a trained model):** a configurable vocabulary
  (`config/aspect_vocabulary.json`, 17 aspects: Battery, Camera, Display, Performance, Price, Design, Build
  Quality, Keyboard, Touchpad, Software, Audio, Storage, Connectivity, Heat & Noise, Comfort & Usability,
  Delivery & Packaging, Customer Service) matched with word boundaries, multiword phrases, plurals and
  hyphen/space variants; longest match wins; each aspect once per sentence. Only aspects actually found are
  shown. There is no "other/unrecognised" aspect: unmatched topics are simply not reported.
* **Context:** for each aspect, the clause around it is extracted (split at *but, however, although, while,
  yet, ;* ...) with negation marking, so "The display is excellent, but the battery drains too quickly" gives
  Display: positive and Battery: negative from different contexts, never the whole-review label copied to
  every aspect.
* **Aspect sentiment (trained):** TF-IDF + Logistic Regression trained on SemEval-2014 Task 4 **laptop**
  aspect terms (2,313 train / 638 test). Test-split results with *gold* aspect terms: accuracy 0.7320,
  macro-F1 0.6831, weighted-F1 0.7286 (`reports/metrics.json`). These do not evaluate the rule-based extractor,
  and the model was trained on laptops only, so accuracy on other product types is **unmeasured**.
* **Confidence:** shown only where the model outputs class probabilities (Logistic Regression), labelled as a raw
  model probability, not calibrated.
* **Aggregation:** `aspect_overall` = Mixed if any positive and any negative aspect; otherwise the single
  polarity present; Neutral if all neutral. `final_label` = `aspect_overall` when aspects were detected,
  otherwise the whole-review prediction (`final_label_basis` says which).
* **Themes** on the Insights page are plain word/phrase frequencies in predicted-negative vs predicted-positive
  reviews. There is no topic modelling.

## Dashboard pages
Overview · Scrape Amazon Reviews (+ CSV upload) · Import Dataset · Review Analysis (search/filter by sentiment, rating, aspect;
per-review detail) · Aspect Insights (distributions, top aspects, most negative aspects, drill-down to the
reviews behind a chart) · Data Export (cleaned reviews, review-level predictions, aspect-level predictions) ·
Model Evaluation (both benchmarks above). Empty states are shown before anything is loaded; no demo metrics
are displayed as live results.

## Four kinds of data (kept separate)
1. **Live scraped reviews:** only what Amazon serves to the scraper at run time. None were obtained so far.
2. **Uploaded reviews:** your CSV, tagged `csv_upload`; **imported datasets** from `Dataset/`, tagged `dataset_import`.
3. **Benchmark data:** SemEval-2014 laptops + Amazon Reviews Multi, used only for training/evaluating models.
4. **Test fixtures:** `tests/fixtures/*.html` and review strings inside `tests/`, invented for automated tests.
   They are never shown in the app.

## Tests

```powershell
python -m pytest -v
```
328 tests (last run: **328 passed**): URL validation and ASIN extraction, SSRF/redirect handling, robots rules,
fixture-based HTML parsing, missing fields, duplicates, empty/malformed HTML, timeouts, CAPTCHA / sign-in /
access-denied / throttling, CSV validation, preprocessing, model loading (missing and corrupt), prediction
structure, multiple and mixed aspects, unknown aspects, star-rating independence, CSV export, an end-to-end
fixture test (records -> preprocessing -> sentiment -> aspects -> exports) and headless Streamlit rendering of
every page. No test uses the network. Model-dependent tests are skipped if models are not trained.

## Known limitations
* Live Amazon scraping is unverified and frequently blocked; the product page itself shows only a few reviews.
  Layouts, regions and languages differ; dates in non-English storefronts may stay unparsed (raw text kept).
* DNS is checked before each request, but a determined DNS-rebinding attacker could still race the check.
* Star-derived training labels; weak neutral class (F1 0.457).
* Aspect extraction is vocabulary-bound: implicit aspects, misspellings and unlisted terms are missed, and some
  words are ambiguous ("value", "support", "fit", "quality"). Clause splitting uses a fixed connective list;
  opinions joined with "and", sarcasm and comparisons are handled poorly.
* The aspect sentiment model was trained on laptop reviews; other categories are out of domain and unmeasured.
  There is no labelled evaluation of the full extract-then-classify pipeline on Amazon reviews.
* Single-threaded, up to 10 pages per scrape and 5000 uploaded rows.
