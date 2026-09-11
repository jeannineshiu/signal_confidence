from datetime import date

import numpy as np
import pandas as pd
import pytest

from sigconf.data.labels import (
    DOWN,
    FLAT,
    UP,
    anchor_positions,
    close_times,
    direction_from_return,
    label_headlines,
)

# NYSE-like 2019 calendar: weekdays minus New Year's Day and Independence Day.
DAYS = pd.bdate_range("2019-01-01", "2019-12-31").drop(
    pd.DatetimeIndex(["2019-01-01", "2019-07-04"])
)


def t0_of(published: str) -> date:
    pos = anchor_positions(pd.Series(pd.to_datetime([published], utc=True)), DAYS)[0]
    return DAYS[pos].date()


# ── Anchor-day rule ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("published", "expected_t0"),
    [
        # intraday, before the close → same day
        ("2019-07-02T11:00:00-04:00", date(2019, 7, 2)),
        ("2019-07-02T15:59:00-04:00", date(2019, 7, 2)),
        # at or after the close → next trading day ("strictly after")
        ("2019-07-02T16:00:00-04:00", date(2019, 7, 3)),
        ("2019-07-02T18:30:00-04:00", date(2019, 7, 3)),
        # Wednesday after close, Thursday is a holiday → Friday
        ("2019-07-03T17:00:00-04:00", date(2019, 7, 5)),
        # on the holiday itself → next trading day
        ("2019-07-04T10:00:00-04:00", date(2019, 7, 5)),
        # Friday after close and weekend → Monday
        ("2019-07-05T16:30:00-04:00", date(2019, 7, 8)),
        ("2019-07-06T10:00:00-04:00", date(2019, 7, 8)),
        ("2019-07-07T23:59:00-04:00", date(2019, 7, 8)),
        # pre-market on a trading day → that day's close is after it
        ("2019-07-08T04:30:00-04:00", date(2019, 7, 8)),
    ],
)
def test_anchor_day_rule(published, expected_t0):
    assert t0_of(published) == expected_t0


def test_unknown_time_of_day_is_treated_as_after_the_close():
    # 00:00:00 local means "time unknown": the headline may have come after the close.
    assert t0_of("2019-07-02T00:00:00-04:00") == date(2019, 7, 3)


def test_a_real_time_just_after_midnight_is_not_treated_as_unknown():
    assert t0_of("2019-07-02T00:00:30-04:00") == date(2019, 7, 2)


@pytest.mark.parametrize(
    ("published_utc", "expected_t0"),
    [
        # Summer (EDT, UTC-4): 16:00 ET = 20:00 UTC
        ("2019-07-02T19:59:00Z", date(2019, 7, 2)),
        ("2019-07-02T20:00:00Z", date(2019, 7, 3)),
        # Winter (EST, UTC-5): 16:00 ET = 21:00 UTC
        ("2019-01-15T20:30:00Z", date(2019, 1, 15)),
        ("2019-01-15T21:00:00Z", date(2019, 1, 16)),
        # First trading day after DST starts (Sun 2019-03-10)
        ("2019-03-11T19:59:00Z", date(2019, 3, 11)),
        ("2019-03-11T20:00:00Z", date(2019, 3, 12)),
        # First trading day after DST ends (Sun 2019-11-03)
        ("2019-11-04T20:59:00Z", date(2019, 11, 4)),
        ("2019-11-04T21:00:00Z", date(2019, 11, 5)),
    ],
)
def test_close_is_16_00_eastern_across_dst(published_utc, expected_t0):
    assert t0_of(published_utc) == expected_t0


def test_no_look_ahead_and_no_skipping_on_random_publication_times():
    """The core invariant: close(t0) is after publication, close(t0 - 1) is not.

    The first half proves no look-ahead; the second proves t0 is the *first*
    such close, so no tradable day is skipped.
    """
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2019-01-02", tz="UTC").value
    end = pd.Timestamp("2019-12-20", tz="UTC").value
    # Nonzero seconds so no draw lands on the 00:00:00 "unknown time" convention.
    ns = rng.integers(start, end, size=5_000) // 60_000_000_000 * 60_000_000_000 + 7_000_000_000
    published = pd.Series(pd.to_datetime(ns, utc=True))
    pos = anchor_positions(published, DAYS)
    closes = close_times(DAYS)

    assert np.all(pos < len(DAYS))
    assert np.all(closes[pos] > published.to_numpy())
    has_prev = pos > 0
    assert np.all(closes[pos[has_prev] - 1] <= published[has_prev].to_numpy())


def test_close_times_are_4pm_new_york():
    ct = close_times(pd.DatetimeIndex(["2019-01-15", "2019-07-02"]))
    local = ct.tz_convert("America/New_York")
    assert list(local.hour) == [16, 16]
    assert list(ct.hour) == [21, 20]  # EST, EDT


def test_unsorted_trading_days_are_rejected():
    days = pd.DatetimeIndex(["2019-07-03", "2019-07-02"])
    with pytest.raises(ValueError):
        anchor_positions(pd.Series(pd.to_datetime(["2019-07-01T12:00Z"], utc=True)), days)


# ── Dead-band ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("r", "expected"),
    [
        (0.0010, UP),
        (0.00099, FLAT),
        (0.0, FLAT),
        (-0.00099, FLAT),
        (-0.0010, DOWN),
        (0.05, UP),
        (-0.05, DOWN),
    ],
)
def test_dead_band_boundaries(r, expected):
    assert direction_from_return(r, dead_band=0.0010) == expected


def test_nan_return_is_an_error_not_a_label():
    with pytest.raises(ValueError):
        direction_from_return(float("nan"))


# ── label_headlines ───────────────────────────────────────────────────────────


def _prices(values, days):
    return pd.Series(values, index=pd.DatetimeIndex(days), name="close", dtype=float)


def test_label_uses_close_t0_to_close_t1():
    prices = _prices([100.0, 102.0, 101.0], ["2019-07-01", "2019-07-02", "2019-07-03"])
    headlines = pd.DataFrame({"published_at": pd.to_datetime(
        ["2019-07-01T10:00:00-04:00", "2019-07-01T17:00:00-04:00"], utc=True)})
    out = label_headlines(headlines, prices)

    assert list(out["t0"]) == [date(2019, 7, 1), date(2019, 7, 2)]
    assert list(out["t1"]) == [date(2019, 7, 2), date(2019, 7, 3)]
    assert out["ret"].tolist() == pytest.approx([0.02, 101.0 / 102.0 - 1])
    assert list(out["label"]) == [UP, DOWN]


def test_label_depends_only_on_the_two_window_closes():
    days = pd.bdate_range("2019-07-01", "2019-07-12")
    base = _prices(np.linspace(100, 110, len(days)), days)
    headlines = pd.DataFrame({"published_at": pd.to_datetime(["2019-07-03T12:00-04:00"], utc=True)})
    before = label_headlines(headlines, base)

    t0 = pd.Timestamp(before["t0"].iloc[0])
    t1 = pd.Timestamp(before["t1"].iloc[0])
    scrambled = base * np.random.default_rng(1).uniform(0.5, 1.5, len(base))
    scrambled[t0] = base[t0]
    scrambled[t1] = base[t1]
    after = label_headlines(headlines, scrambled)

    assert after["ret"].iloc[0] == before["ret"].iloc[0]
    assert after["label"].iloc[0] == before["label"].iloc[0]


def test_headline_without_a_next_day_is_unlabelled_not_filled():
    prices = _prices([100.0, 101.0], ["2019-07-01", "2019-07-02"])
    headlines = pd.DataFrame({"published_at": pd.to_datetime(
        ["2019-07-01T12:00-04:00", "2019-07-01T17:00-04:00", "2019-07-10T12:00-04:00"], utc=True)})
    out = label_headlines(headlines, prices)

    assert out["label"].iloc[0] == UP
    assert out["label"].isna().tolist() == [False, True, True]
    assert out["t0"].isna().tolist() == [False, True, True]
    assert out["ret"].isna().tolist() == [False, True, True]


def test_label_headlines_accepts_unsorted_prices():
    prices = _prices([101.0, 100.0], ["2019-07-02", "2019-07-01"])
    headlines = pd.DataFrame({"published_at": pd.to_datetime(["2019-07-01T12:00-04:00"], utc=True)})
    assert label_headlines(headlines, prices)["label"].iloc[0] == UP
