# Phase 1: data audit (2026-09-11)

Source: Kaggle `miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests`, file
`analyst_ratings_processed.csv` (157.6 MB unzipped). The Kaggle listing says CC0, and the uploader
notes the headlines are Benzinga's property.

## Raw file
- 1,400,469 rows. **2,578 (0.18%) are malformed**: a title with an embedded newline splits across
  two lines, which shifts the date into the title column. Those rows are dropped and counted, not repaired.
- 2009–2020 (the data ends June 2020). Every timestamp has minute precision, and the seconds are always 00.

## Timezone: checked, no problem
- The uploader describes the times as "UTC-4", but the offsets are **`-04:00` (907,506 rows) and
  `-05:00` (490,385 rows)**.
- **100% of offsets match America/New_York DST rules**, and the wall-clock part equals NY local time.
  These are correct US/Eastern timestamps, so the ±1 h anchor-day risk in PLAN §5 does not apply.
- Exactly 00:00:00 (time unknown): 0.05% of rows. Exactly 16:00:00: 0.20% (these roll to the next day
  under the "strictly after" rule).
- The hour histogram has a large spike at 16:xx (139k rows), which is after-close publishing. That is
  why the timestamp-aware anchor rule matters.

## Ticker choice → NVDA + JNJ
- **AAPL has only 469 rows over 86 distinct days in this dataset**, which is unusable. JPM and INTC have
  about 10 rows each.
- Company-specific, non-list headlines since 2011: NVDA 1,598 (718 days), JNJ 1,385 (782 days).
- This gives a contrast between high-volatility semis and a low-volatility defensive name.
- About half of each ticker's rows do not name the company at all ("5 Biggest Price Target Changes For
  Friday", sector ETF stories). The eligibility filter removes those and multi-stock list headlines.

## Sample (seed 42)
- 150 headlines, 75 per ticker, at most one per (ticker, anchor day).
- dev = 45 (2011-03-16 → 2014-08-25), test = 105 (2014-09-04 → 2020-05-10).
- Labels, all 150: down 67 · flat 7 · up 76.
- Test: down 49 · flat 5 · up 51, so the **always-up baseline ≈ 51/100 = 51%** of directional labels.
  Dev up rate is 58%.

## Label checks
- 10-row manual audit. Boundary cases behave as specified:
  - h006 was published Mon 16:00 and anchors Tue.
  - h065 was published Thu 15:59 and anchors Thu.
  - h103 NVDA 2018-03-27 at −7.8% matches the real drop that day.
- Independent check: labels recomputed from **unadjusted** yfinance closes agree **150/150**, so no
  dividend or split adjustment flips a direction in the sample.
- NVDA adjusted closes are small (≈0.3 in 2011, after the 2021 4:1 and 2024 10:1 splits). Ratios are
  unaffected.
