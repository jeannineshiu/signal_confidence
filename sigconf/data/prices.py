"""Daily closes from yfinance, cached to CSV so reruns never touch the network.

Closes are split- and dividend-adjusted (`auto_adjust=True`, passed explicitly
because the yfinance default has changed across versions), so a close-to-close
ratio is a true total return. The cache files are committed: yfinance revises
adjusted history whenever a new dividend is paid, and the committed copy is what
makes every reported number reproducible.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from sigconf.config import CACHE_DIR

Fetcher = Callable[[str, str, str], pd.Series]


def fetch_closes(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted daily closes for [start, end) — network call."""
    import yfinance as yf  # imported lazily: tests and offline runs never need it

    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker} {start}..{end}")
    closes = hist["Close"].copy()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    closes.index.name = "date"
    closes.name = "close"
    return closes


def cache_path(ticker: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"prices_{ticker}.csv"


def load_closes(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path = CACHE_DIR,
    offline: bool = False,
    fetcher: Fetcher = fetch_closes,
) -> pd.Series:
    """Adjusted closes covering [start, end), from cache when it covers the range.

    A cache that does not span the requested range is refetched as a whole
    (never stitched), unless `offline`, in which case it is an error.
    """
    path = cache_path(ticker, cache_dir)
    if path.exists():
        cached = read_cache(path)
        if _covers(cached, start, end):
            return cached
        if offline:
            raise RuntimeError(f"{path} does not cover {start}..{end} and offline=True")
    elif offline:
        raise RuntimeError(f"no price cache at {path} and offline=True")

    closes = fetcher(ticker, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    closes.to_csv(path, header=True)
    return read_cache(path)  # return exactly what a later cached run will read


def read_cache(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    closes = df["close"].sort_index()
    if closes.index.has_duplicates or closes.isna().any():
        raise ValueError(f"corrupt price cache {path}: duplicate dates or missing closes")
    return closes


def _covers(closes: pd.Series, start: str, end: str) -> bool:
    # Allow a few days of slack at each end for weekends/holidays at the bounds.
    slack = pd.Timedelta(days=5)
    return (
        closes.index.min() <= pd.Timestamp(start) + slack
        and closes.index.max() >= pd.Timestamp(end) - slack
    )
