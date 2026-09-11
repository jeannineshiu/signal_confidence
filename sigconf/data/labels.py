"""Next-day direction labels with a strict no-look-ahead boundary.

The rule (pre-registered, PLAN.md §1):

  anchor day t0 = the first trading day whose 16:00 ET close is *strictly after*
                  the headline's publication time
  label window  = close[t0] → close[t1], where t1 is the trading day after t0

Consequences:
  * published 11:00 ET on trading day D   → t0 = D      (close D is after the news)
  * published 16:00:00 or later on D      → t0 = next trading day
  * weekend / holiday                     → t0 = next trading day
  * time-of-day unknown                   → treated as "sometime on that date",
    i.e. possibly after the close, so t0 = next trading day after that date.
    Sources mark a date-only item as midnight, in UTC or in ET; both count.
    Midnight UTC is 19:00/20:00 ET on the *previous* evening, so taking it at
    face value would anchor a day too early — exactly the look-ahead to avoid.

The entry price close[t0] is therefore always a price that existed *after* the
headline was public. A window starting at a close that predates the news would
let the label absorb the market's reaction to the headline itself, flattering
any signal read from it. This is stricter than a literal "date D → D+1" reading
for after-close headlines, which is the point.

Early-close days (13:00 ET) are treated as 16:00 closes; this is a documented
approximation affecting only headlines published between 13:00 and 16:00 on
those few days.
"""

from datetime import time

import numpy as np
import pandas as pd

from sigconf.config import DEAD_BAND, MARKET_CLOSE, MARKET_TZ

UP, DOWN, FLAT = "up", "down", "flat"


def close_times(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC timestamps of the 16:00 ET close on each trading day (DST-aware)."""
    days = pd.DatetimeIndex(trading_days).normalize()
    if days.tz is not None:
        days = days.tz_localize(None)
    local = (days + pd.Timedelta(hours=MARKET_CLOSE.hour, minutes=MARKET_CLOSE.minute))
    return local.tz_localize(MARKET_TZ).tz_convert("UTC")


def is_time_unknown(published_at: pd.Series) -> pd.Series:
    """True where the timestamp is exactly midnight in UTC or in ET (date-only)."""
    ts = pd.to_datetime(published_at, utc=True)
    local = ts.dt.tz_convert(MARKET_TZ)
    return (ts.dt.time == time(0, 0)) | (local.dt.time == time(0, 0))


def effective_publication(published_at: pd.Series) -> pd.Series:
    """Publication instant used for anchoring, in UTC.

    Date-only items are pushed to the last instant of their calendar date in
    ET. The date is read in UTC, which for both midnight conventions is the
    later candidate: midnight ET on D is 04:00/05:00 UTC on D, and midnight UTC
    on D is the evening of D-1 in ET but a date-D item in the source.
    """
    ts = pd.to_datetime(published_at, utc=True)
    unknown = is_time_unknown(ts)
    utc_date = ts.dt.tz_localize(None).dt.normalize()
    end_of_day = (
        (utc_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
        .dt.tz_localize(MARKET_TZ)
        .dt.tz_convert("UTC")
    )
    return ts.where(~unknown, end_of_day)


def anchor_positions(published_at: pd.Series, trading_days: pd.DatetimeIndex) -> np.ndarray:
    """Index into `trading_days` of each headline's anchor day t0.

    Returns len(trading_days) where no close after publication exists.
    """
    closes = _utc_ns(close_times(trading_days))
    if not np.all(np.diff(closes) > 0):
        raise ValueError("trading_days must be strictly increasing")
    pub = _utc_ns(effective_publication(published_at))
    # side="right": first close strictly greater than the publication instant.
    return np.searchsorted(closes, pub, side="right")


def _utc_ns(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Tz-aware timestamps → int64 nanoseconds since the epoch (UTC)."""
    naive_utc = pd.DatetimeIndex(ts).tz_convert("UTC").tz_localize(None)
    return naive_utc.to_numpy().astype("datetime64[ns]").astype(np.int64)


def direction_from_return(r: float, dead_band: float = DEAD_BAND) -> str:
    """up if r >= +dead_band, down if r <= -dead_band, otherwise flat."""
    if np.isnan(r):
        raise ValueError("return is NaN")
    if abs(r) < dead_band:
        return FLAT
    return UP if r > 0 else DOWN


def label_headlines(
    headlines: pd.DataFrame, prices: pd.Series, dead_band: float = DEAD_BAND
) -> pd.DataFrame:
    """Attach t0, t1, the next-day return and its direction to each headline.

    `headlines` needs a `published_at` column; `prices` is the close series for
    the headlines' ticker, indexed by trading date. Only close[t0] and close[t1]
    are read. Rows with no t1 in the price history get a missing label (NA) —
    never a filled-in value.
    """
    prices = prices.sort_index()
    days = pd.DatetimeIndex(prices.index)
    pos = anchor_positions(headlines["published_at"], days)
    n = len(days)

    out = headlines.copy()
    has_window = pos + 1 < n
    t0_pos = np.where(has_window, pos, 0)
    t1_pos = np.where(has_window, pos + 1, 0)
    values = prices.to_numpy(dtype=float)

    out["t0"] = pd.Series(days[t0_pos].date, index=out.index).where(has_window, None)
    out["t1"] = pd.Series(days[t1_pos].date, index=out.index).where(has_window, None)
    close_t0 = np.where(has_window, values[t0_pos], np.nan)
    close_t1 = np.where(has_window, values[t1_pos], np.nan)
    out["close_t0"] = close_t0
    out["close_t1"] = close_t1
    out["ret"] = close_t1 / close_t0 - 1.0
    out["label"] = [
        direction_from_return(r, dead_band) if ok else None
        for r, ok in zip(out["ret"], has_window, strict=True)
    ]
    return out
