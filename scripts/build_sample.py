"""One-time: raw Kaggle CSV → data/sample/headlines.csv (+ price cache).

Needs the raw file in data/raw/ (apple_news_data.csv, or the .zip Kaggle
serves) and network access for yfinance. Everything it writes is committed, so
reproducing the results never requires running this again.

    python scripts/build_sample.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sigconf import config  # noqa: E402
from sigconf.data import headlines, prices  # noqa: E402


def main() -> int:
    raw_path = config.RAW_DIR / config.RAW_HEADLINES_FILE
    if not raw_path.exists():
        raw_path = raw_path.with_name(raw_path.name + ".zip")
    if not raw_path.exists():
        print(f"missing {raw_path.with_suffix('')} (or .zip) — download it from Kaggle first",
              file=sys.stderr)
        return 1

    raw, dropped = headlines.load_raw(raw_path, config.TICKER)
    print(f"raw rows: {len(raw):,} usable, {dropped:,} dropped (missing title/date)")

    candidates, funnel = headlines.eligible(
        raw, config.COMPANY_PATTERN, config.LIVE_BLOG_PATTERN, config.SAMPLE_START
    )
    print("eligibility funnel:", " → ".join(f"{k} {v:,}" for k, v in funnel.items()))

    calendar = prices.load_closes(config.TICKER, config.PRICE_START, config.PRICE_END).index
    sample = headlines.sample(candidates, calendar, config.N_HEADLINES, config.SEED)
    headlines.write_sample(sample)

    dev = sample[sample["split"] == "dev"]
    test = sample[sample["split"] == "test"]
    print(f"wrote {config.SAMPLE_PATH.relative_to(config.ROOT)}: {len(sample)} headlines")
    print(f"  dev : {len(dev):3d}  {dev['published_at'].min()} → {dev['published_at'].max()}")
    print(f"  test: {len(test):3d}  {test['published_at'].min()} → {test['published_at'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
