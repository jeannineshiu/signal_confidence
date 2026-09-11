import pandas as pd
import pytest

from sigconf.data import headlines
from sigconf.data.labels import anchor_positions

RAW_CSV = """\
date,title,content,link,symbols
2024-05-02T21:40:07+00:00,Apple (AAPL) Beats Q2 Earnings,body,https://x,AAPL.US
2024-05-02T21:45:00+00:00,apple (aapl) beats q2 earnings,syndicated copy,https://y,AAPL.US
2024-05-03T00:00:00+00:00,Apple Date-Only Story,body,https://x,AAPL.US
2024-05-03T04:00:00+00:00,Apple Midnight ET Story,body,https://x,AAPL.US
2024-05-03T13:00:00+00:00,Stock Market Today: Apple leads Nasdaq higher (Live Coverage),b,u,s
2024-05-03T14:00:00+00:00,"Dow Rises;  Apple
Slides On Downgrade",body,https://x,AAPL.US
2024-05-03T15:00:00+00:00,S&amp;P 500 Rallies; Apple Up,body,https://x,AAPL.US
2024-05-03T16:00:00+00:00,Pineapple Prices Soar,body,https://x,AAPL.US
2023-10-15T12:00:00+00:00,Apple Before The Cutoff,body,https://x,AAPL.US
not-a-date,Apple Broken Row,body,https://x,AAPL.US
2024-05-06T12:00:00+00:00,,body,https://x,AAPL.US
"""

COMPANY = r"\bapple\b|\baapl\b"
LIVE = r"live coverage|stock market today"
DAYS = pd.bdate_range("2024-01-01", "2024-12-31").drop(
    pd.DatetimeIndex(["2024-01-01", "2024-07-04"])
)


@pytest.fixture
def raw(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(RAW_CSV)
    return headlines.load_raw(path, "AAPL")


def test_load_raw_drops_and_counts_unusable_rows(raw):
    df, dropped = raw
    assert dropped == 2  # unparseable date, missing title
    assert len(df) == 9
    assert (df["ticker"] == "AAPL").all()
    assert df["published_at"].dt.tz is not None


def test_load_raw_cleans_titles(raw):
    df, _ = raw
    titles = set(df["headline"])
    assert "Dow Rises; Apple Slides On Downgrade" in titles  # newline + double space
    assert "S&P 500 Rallies; Apple Up" in titles  # HTML entity


def test_eligibility_rules_and_funnel(raw):
    df, _ = raw
    got, funnel = headlines.eligible(df, COMPANY, LIVE, start="2023-11-01")

    assert got["headline"].tolist() == [
        "Apple (AAPL) Beats Q2 Earnings",       # syndicated lowercase copy dropped
        "Dow Rises; Apple Slides On Downgrade",
        "S&P 500 Rallies; Apple Up",
    ]
    assert funnel == {
        "raw": 9,
        "after_start": 8,     # pre-cutoff row out
        "names_company": 7,   # "Pineapple" is not Apple
        "exact_time": 5,      # midnight UTC and midnight ET out
        "not_live_blog": 4,
        "deduplicated": 3,
    }


def _candidates(stamps, titles=None):
    titles = titles or [f"Apple {i}" for i in range(len(stamps))]
    return pd.DataFrame(
        {"ticker": "AAPL", "published_at": pd.to_datetime(stamps, utc=True), "headline": titles}
    )


def test_sample_keeps_at_most_one_headline_per_anchor_day():
    # Saturday, Sunday and Monday pre-market all anchor on Monday 2024-07-08.
    cands = _candidates(["2024-07-06T10:00-04:00", "2024-07-07T10:00-04:00",
                         "2024-07-08T08:00-04:00", "2024-07-09T08:00-04:00"])
    out = headlines.sample(cands, DAYS, n=2, seed=0)
    assert len(out) == 2
    assert len(set(anchor_positions(out["published_at"], DAYS))) == 2
    with pytest.raises(ValueError, match="only 2 eligible anchor days"):
        headlines.sample(cands, DAYS, n=3, seed=0)


def test_sample_excludes_headlines_without_a_full_label_window():
    days = pd.DatetimeIndex(["2024-07-01", "2024-07-02", "2024-07-03"])
    cands = _candidates(["2024-07-01T10:00-04:00", "2024-07-03T10:00-04:00",
                         "2024-07-03T17:00-04:00"], ["ok", "t0 is last day", "no t0"])
    out = headlines.sample(cands, days, n=1, seed=0)
    assert out["headline"].tolist() == ["ok"]


def _many(n, start="2024-02-01"):
    stamps = (pd.bdate_range(start, periods=n) + pd.Timedelta(hours=14)).tz_localize(
        "America/New_York")
    return _candidates(stamps)


def test_sample_is_deterministic_per_seed():
    cands = _many(80)
    a = headlines.sample(cands, DAYS, n=40, seed=42)
    b = headlines.sample(cands, DAYS, n=40, seed=42)
    c = headlines.sample(cands, DAYS, n=40, seed=7)

    pd.testing.assert_frame_equal(a, b)
    assert set(a["headline"]) != set(c["headline"])
    assert list(a.columns) == headlines.SAMPLE_COLUMNS
    assert a["id"].is_unique


def test_split_is_chronological_with_dev_first():
    out = headlines.sample(_many(80), DAYS, n=40, seed=42)
    dev = out[out["split"] == "dev"]
    test = out[out["split"] == "test"]

    assert len(dev) == 12 and len(test) == 28  # 30% of 40
    assert dev["published_at"].max() <= test["published_at"].min()


def test_chronological_split_requires_sorted_input():
    unsorted = pd.Series(pd.to_datetime(["2024-07-02", "2024-07-01"], utc=True))
    with pytest.raises(ValueError):
        headlines.chronological_split(unsorted)


def test_sample_file_round_trip_preserves_instants(tmp_path):
    out = headlines.sample(_many(10), DAYS, n=5, seed=0)
    path = tmp_path / "s.csv"
    headlines.write_sample(out, path)
    back = headlines.load_sample(path)

    assert "-05:00" in path.read_text()  # stored in US/Eastern with offset
    pd.testing.assert_series_equal(
        back["published_at"].dt.tz_convert("UTC"),
        out["published_at"].dt.tz_convert("UTC"),
        check_dtype=False,
    )
    assert back.drop(columns="published_at").equals(out.drop(columns="published_at"))
