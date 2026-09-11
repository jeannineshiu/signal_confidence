# Phase 1: data audit (2026-09-11, AAPL post-cutoff build)

Source: Kaggle `frankossai/apple-stock-aapl-historical-financial-news-data`, file `apple_news_data.csv`
(29,752 rows, 2016-02-19 → 2024-11-27). About 93% of links are finance.yahoo.com; the rest are fool.com,
investorplace and others. Listed as CC0 on Kaggle; the headlines belong to their publishers. See
`dataset_survey.md` for why this dataset replaced the Benzinga 2009–2020 build.

## Contamination boundary
- gpt-4o-mini's documented knowledge cutoff is **2023-10-01**. Sampling starts **2023-11-01** to leave a
  margin, and a data test asserts every sampled headline is after every allowed model's cutoff.

## Timestamps: accuracy checked against real events
- Every timestamp is `+00:00` UTC.
- **Apple's five earnings releases in the window** (16:30 ET on 2023-11-02, 2024-02-01, 05-02, 08-01,
  10-31): every results headline ("beats", "reports results", "revenue hits record") is stamped **at or
  after 16:30 ET**. "Apple reports fourth quarter results" is stamped exactly 16:30. Anything earlier on
  those days is a preview ("ahead of", "to report"). The timestamps are genuine publication times.
- **Date-only rows** are stored as exactly 00:00:00 UTC, which is 19:00/20:00 ET the *previous* evening.
  Read at face value they would anchor a day early, which is look-ahead. `labels.py` treats midnight UTC
  and midnight ET as "time unknown" (tested), and eligibility excludes these rows anyway.
- **Live blogs** ("Stock Market Today: S&P 500 closes at record high (Live Coverage)", stamped 11:45 ET)
  keep their first-publication timestamp while the title is rewritten through the day, so the title can
  describe events after its own timestamp. They are excluded (`LIVE_BLOG_PATTERN`). There are only 10
  post-cutoff, 4 of which name Apple.

## Eligibility funnel (pre-registered rules, `sigconf/config.py`)
raw 29,752 → since 2023-11-01: 6,648 → names Apple: 2,799 → exact time: 2,573 → not a live blog: 2,569 →
syndicated duplicates removed: **2,501**, spanning about 357 distinct days.

A regex filter for multi-stock list headlines was tried and dropped: it also removed Apple-specific
headlines ("Apple Is a Big Risk as Mag 7 Stocks Look Vulnerable"). Headlines that aren't really about
Apple should come back from the LLM as neutral and show up in the coverage figure, not be hand-filtered.

## Sample (seed 42)
- 150 headlines, at most one per anchor day. **dev 45** (2023-11-08 → 2024-03-01), **test 105**
  (2024-03-05 → 2024-11-22).
- Labels: dev down 29 · flat 2 · up 14; test down 38 · flat 8 · up 59.
- **Regime shift between the splits:** 33% of dev directional labels are up (AAPL fell in early 2024),
  versus **61% of test directional labels (n = 97)**. Two consequences for the README:
  - Always-up is a tough baseline on test (61%).
  - The prompt is iterated in a bearish period and scored in a bullish one.
- Many headlines are low-information ("Steve Jobs Didn't Become A Billionaire Because Of Apple…",
  Motley Fool listicles), so expect a high neutral rate.

## Label checks
- 10-row manual audit; every boundary case behaves as specified:
  - Sunday 17:41 anchors Monday.
  - Monday 17:00 anchors Tuesday.
  - Thursday 16:16 anchors Friday, with t1 skipping the Monday MLK holiday to Tuesday.
  - Tuesday 03:25 after Memorial Day anchors that Tuesday.
- Labels recomputed from **unadjusted** yfinance closes agree **150/150**.
