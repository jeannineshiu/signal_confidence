"""Headline ingestion, eligibility filtering, sampling and the dev/test split.

Source: Kaggle "Daily Financial News for 6000+ Stocks" (miguelaenlle),
file analyst_ratings_processed.csv — columns: index, title, date, stock.
Timestamps carry explicit US/Eastern offsets (-04:00 / -05:00); every offset
was checked to match America/New_York DST rules on the full file.

Sampling is label-blind: which headlines are drawn depends on the trading
calendar (to find each headline's anchor day) but never on price moves.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sigconf.config import DEV_FRACTION, SAMPLE_PATH
from sigconf.data.labels import anchor_positions

SAMPLE_COLUMNS = ["id", "ticker", "published_at", "headline", "split"]


def load_raw(path: Path) -> tuple[pd.DataFrame, int]:
    """Read the raw CSV (plain or .zip) → (well-formed rows, number dropped).

    About 0.2% of raw rows are broken across two lines (a title with an
    embedded newline shifts the date into the title column). Those rows are
    dropped and counted rather than repaired.
    """
    raw = pd.read_csv(path, usecols=["title", "date", "stock"])
    ts = pd.to_datetime(raw["date"], utc=True, errors="coerce", format="ISO8601")
    ok = ts.notna() & raw["stock"].notna() & raw["title"].notna()
    df = pd.DataFrame(
        {
            "ticker": raw.loc[ok, "stock"].astype(str),
            "published_at": ts[ok],
            # Collapse embedded newlines / runs of spaces so each prompt is one line.
            "headline": raw.loc[ok, "title"].astype(str).str.split().str.join(" "),
        }
    )
    return df.reset_index(drop=True), int((~ok).sum())


def eligible(
    df: pd.DataFrame, ticker_patterns: dict[str, str], list_pattern: str, start: str
) -> pd.DataFrame:
    """Headlines for the chosen tickers that name the company and are not list items."""
    parts = []
    for ticker, pattern in ticker_patterns.items():
        rows = df[(df["ticker"] == ticker) & (df["published_at"] >= pd.Timestamp(start, tz="UTC"))]
        names_company = rows["headline"].str.contains(pattern, case=False, regex=True)
        is_list = rows["headline"].str.contains(list_pattern, case=False, regex=True)
        parts.append(rows[names_company & ~is_list])
    return pd.concat(parts, ignore_index=True)


def sample(
    candidates: pd.DataFrame,
    trading_days: dict[str, pd.DatetimeIndex],
    n_per_ticker: int,
    seed: int,
) -> pd.DataFrame:
    """Draw n_per_ticker headlines per ticker, at most one per anchor day.

    Two headlines sharing an anchor day share the same label, so keeping both
    would count one market outcome twice. The key is the anchor day t0, not the
    calendar date: a Saturday headline and a Monday-morning one both anchor on
    Monday. Only headlines with a complete label window (t0 and t1 both in the
    calendar) are eligible.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for ticker in sorted(trading_days):
        rows = candidates[candidates["ticker"] == ticker].sort_values(
            ["published_at", "headline"], kind="stable"
        )
        days = trading_days[ticker]
        pos = anchor_positions(rows["published_at"], days)
        rows = rows.assign(_t0=pos)[pos + 1 < len(days)]

        # One headline per anchor day, chosen at random; then n days at random.
        shuffled = rows.iloc[rng.permutation(len(rows))]
        one_per_day = shuffled.drop_duplicates("_t0", keep="first")
        if len(one_per_day) < n_per_ticker:
            raise ValueError(
                f"{ticker}: only {len(one_per_day)} eligible anchor days, need {n_per_ticker}"
            )
        chosen = rng.choice(len(one_per_day), size=n_per_ticker, replace=False)
        parts.append(one_per_day.iloc[np.sort(chosen)].drop(columns="_t0"))

    out = pd.concat(parts).sort_values(["published_at", "ticker"], kind="stable")
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
