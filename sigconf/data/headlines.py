"""Headline ingestion, eligibility filtering, sampling and the dev/test split.

Source: Kaggle "Apple Stock (AAPL): Historical Financial News Data"
(frankossai/apple-stock-aapl-historical-financial-news-data, CC0 as listed),
file apple_news_data.csv — columns used: date (ISO 8601, UTC), title.
Timestamp accuracy was checked against Apple's five 16:30 ET earnings releases
in the sample window (notes/phase1_data_audit.md).

Sampling is label-blind: which headlines are drawn depends on the trading
calendar (to find each headline's anchor day) but never on price moves.
"""

import html
from pathlib import Path

import numpy as np
import pandas as pd

from sigconf.config import DEV_FRACTION, SAMPLE_PATH
from sigconf.data.labels import anchor_positions, is_time_unknown

SAMPLE_COLUMNS = ["id", "ticker", "published_at", "headline", "split"]


def load_raw(path: Path, ticker: str) -> tuple[pd.DataFrame, int]:
    """Read the raw CSV (plain or .zip) → (usable rows, number dropped).

    Titles are HTML-unescaped ("&amp;" → "&") and whitespace-collapsed so each
    prompt is one clean line. Rows with a missing title or unparseable date
    are dropped and counted.
    """
    raw = pd.read_csv(path, usecols=["date", "title"])
    ts = pd.to_datetime(raw["date"], utc=True, errors="coerce", format="ISO8601")
    ok = ts.notna() & raw["title"].notna()
    titles = raw.loc[ok, "title"].astype(str).map(html.unescape).str.split().str.join(" ")
    df = pd.DataFrame({"ticker": ticker, "published_at": ts[ok], "headline": titles})
    return df.reset_index(drop=True), int((~ok).sum())


def eligible(
    df: pd.DataFrame, company_pattern: str, live_blog_pattern: str, start: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the pre-registered eligibility rules; return rows and a count funnel.

    In order: published on/after `start` → names the company → exact
    publication time known → not a live blog → first copy of a syndicated title.
    """
    funnel = {"raw": len(df)}
    df = df[df["published_at"] >= pd.Timestamp(start, tz="UTC")]
    funnel["after_start"] = len(df)
    df = df[df["headline"].str.contains(company_pattern, case=False, regex=True)]
    funnel["names_company"] = len(df)
    df = df[~is_time_unknown(df["published_at"])]
    funnel["exact_time"] = len(df)
    df = df[~df["headline"].str.contains(live_blog_pattern, case=False, regex=True)]
    funnel["not_live_blog"] = len(df)
    df = df.sort_values(["published_at", "headline"], kind="stable")
    df = df[~df["headline"].str.lower().duplicated(keep="first")]
    funnel["deduplicated"] = len(df)
    return df.reset_index(drop=True), funnel


def sample(
    candidates: pd.DataFrame, trading_days: pd.DatetimeIndex, n: int, seed: int
) -> pd.DataFrame:
    """Draw n headlines, at most one per anchor day, then split chronologically.

    Two headlines sharing an anchor day share the same label, so keeping both
    would count one market outcome twice. The key is the anchor day t0, not the
    calendar date: a Saturday headline and a Monday-morning one both anchor on
    Monday. Only headlines with a complete label window (t0 and t1 both in the
    calendar) are eligible.
    """
    rng = np.random.default_rng(seed)
    rows = candidates.sort_values(["published_at", "headline"], kind="stable")
    pos = anchor_positions(rows["published_at"], trading_days)
    rows = rows.assign(_t0=pos)[pos + 1 < len(trading_days)]

    # One headline per anchor day, chosen at random; then n days at random.
    shuffled = rows.iloc[rng.permutation(len(rows))]
    one_per_day = shuffled.drop_duplicates("_t0", keep="first")
    if len(one_per_day) < n:
        raise ValueError(f"only {len(one_per_day)} eligible anchor days, need {n}")
    chosen = one_per_day.iloc[rng.choice(len(one_per_day), size=n, replace=False)]

    out = chosen.drop(columns="_t0").sort_values(["published_at", "headline"], kind="stable")
    out = out.reset_index(drop=True)
    out.insert(0, "id", [f"h{i:03d}" for i in range(len(out))])
    out["split"] = chronological_split(out["published_at"])
    return out[SAMPLE_COLUMNS]


def chronological_split(published_at: pd.Series, dev_fraction: float = DEV_FRACTION) -> pd.Series:
    """Earliest dev_fraction of rows → "dev", the rest → "test" (input sorted by time)."""
    if not published_at.is_monotonic_increasing:
        raise ValueError("published_at must be sorted ascending")
    n_dev = int(round(len(published_at) * dev_fraction))
    return pd.Series(["dev"] * n_dev + ["test"] * (len(published_at) - n_dev),
                     index=published_at.index)


def write_sample(df: pd.DataFrame, path: Path = SAMPLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    # Store publication time in US/Eastern with its offset: readable, unambiguous.
    out["published_at"] = (
        out["published_at"].dt.tz_convert("America/New_York").map(lambda t: t.isoformat())
    )
    out.to_csv(path, index=False)


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str, "ticker": str, "headline": str, "split": str})
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, format="ISO8601")
    if list(df.columns) != SAMPLE_COLUMNS:
        raise ValueError(f"unexpected columns in {path}: {list(df.columns)}")
    if df["id"].duplicated().any():
        raise ValueError(f"duplicate ids in {path}")
    return df
