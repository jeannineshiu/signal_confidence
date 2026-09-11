import pandas as pd
import pytest

from sigconf.data import headlines
from sigconf.data.labels import anchor_positions

RAW_CSV = """\
,title,date,stock
0,Nvidia Beats Q2 Estimates,2019-07-02 11:00:00-04:00,NVDA
1,Sterne Agee Provides Color on Aaron's,,
,2011-07-20 06:43:00-04:00,AAN,
2,Stocks That Hit 52-Week Highs On Friday,2019-07-05 10:30:00-04:00,NVDA
3,NVDA Upgraded to Buy,2019-01-15 09:00:00-05:00,NVDA
4,Johnson & Johnson Settles Talc Suit,2019-07-03 08:00:00-04:00,JNJ
5,Pfizer Raises Guidance,2019-07-03 08:00:00-04:00,PFE
6,Nvidia Unveils New GPU,2010-06-01 08:00:00-04:00,NVDA
7,A Look at AMD's New Gaming APUs,2019-07-08 08:00:00-04:00,NVDA
8,"NVIDIA Shares Up  2%;
BMO Upbeat",2019-07-09 08:00:00-04:00,NVDA
"""

PATTERNS = {"NVDA": r"nvidia|\bnvda\b", "JNJ": r"johnson\s*&\s*johnson"}
LIST = r"stocks (?:that|moving)"
DAYS = pd.bdate_range("2019-01-01", "2019-12-31").drop(
    pd.DatetimeIndex(["2019-01-01", "2019-07-04"])
)


@pytest.fixture
def raw(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(RAW_CSV)
    return headlines.load_raw(path)


def test_load_raw_drops_and_counts_broken_rows(raw):
    df, dropped = raw
    assert dropped == 2  # the title and its orphaned continuation line
    assert len(df) == 8


def test_load_raw_collapses_whitespace_inside_quoted_titles(raw):
    df, _ = raw
    assert "NVIDIA Shares Up 2%; BMO Upbeat" in set(df["headline"])
    assert df["published_at"].dt.tz is not None


def test_load_raw_respects_both_eastern_offsets(raw):
    df, _ = raw
    by_title = df.set_index("headline")["published_at"]
    assert by_title["NVDA Upgraded to Buy"] == pd.Timestamp("2019-01-15T14:00Z")  # EST
    assert by_title["Nvidia Beats Q2 Estimates"] == pd.Timestamp("2019-07-02T15:00Z")  # EDT


def test_eligible_keeps_only_company_specific_non_list_headlines_in_window(raw):
    df, _ = raw
    got = headlines.eligible(df, PATTERNS, LIST, start="2011-01-01")
    assert sorted(got["headline"]) == [
        "Johnson & Johnson Settles Talc Suit",
        "NVDA Upgraded to Buy",
        "NVIDIA Shares Up 2%; BMO Upbeat",
        "Nvidia Beats Q2 Estimates",
    ]


def _candidates(rows):
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "published_at": pd.to_datetime([r[1] for r in rows], utc=True),
            "headline": [r[2] for r in rows],
        }
    )


def test_sample_keeps_at_most_one_headline_per_anchor_day():
    # Saturday, Sunday and Monday pre-market all anchor on Monday 2019-07-08.
    cands = _candidates(
        [
            ("NVDA", "2019-07-06T10:00-04:00", "sat"),
            ("NVDA", "2019-07-07T10:00-04:00", "sun"),
            ("NVDA", "2019-07-08T08:00-04:00", "mon"),
            ("NVDA", "2019-07-09T08:00-04:00", "tue"),
        ]
    )
    out = headlines.sample(cands, {"NVDA": DAYS}, n_per_ticker=2, seed=0)
    t0 = anchor_positions(out["published_at"], DAYS)
    assert len(out) == 2
    assert len(set(t0)) == 2
    with pytest.raises(ValueError, match="only 2 eligible anchor days"):
        headlines.sample(cands, {"NVDA": DAYS}, n_per_ticker=3, seed=0)


def test_sample_excludes_headlines_without_a_full_label_window():
    days = pd.DatetimeIndex(["2019-07-01", "2019-07-02", "2019-07-03"])
    cands = _candidates(
        [
            ("NVDA", "2019-07-01T10:00-04:00", "ok"),
            ("NVDA", "2019-07-03T10:00-04:00", "t0 is last day, no t1"),
            ("NVDA", "2019-07-03T17:00-04:00", "no t0 at all"),
        ]
    )
    out = headlines.sample(cands, {"NVDA": days}, n_per_ticker=1, seed=0)
    assert out["headline"].tolist() == ["ok"]


def _many(ticker, n, start="2019-02-01"):
    stamps = pd.bdate_range(start, periods=n) + pd.Timedelta(hours=14)
    return _candidates([(ticker, s.tz_localize("America/New_York"), f"{ticker}{i}")
                        for i, s in enumerate(stamps)])


def test_sample_is_deterministic_per_seed_and_balanced_per_ticker():
    cands = pd.concat([_many("NVDA", 60), _many("JNJ", 60)], ignore_index=True)
    cal = {"NVDA": DAYS, "JNJ": DAYS}
    a = headlines.sample(cands, cal, n_per_ticker=20, seed=42)
    b = headlines.sample(cands, cal, n_per_ticker=20, seed=42)
    c = headlines.sample(cands, cal, n_per_ticker=20, seed=7)

    pd.testing.assert_frame_equal(a, b)
    assert set(a["headline"]) != set(c["headline"])
    assert a["ticker"].value_counts().to_dict() == {"NVDA": 20, "JNJ": 20}
    assert list(a.columns) == headlines.SAMPLE_COLUMNS
    assert a["id"].is_unique


def test_split_is_chronological_with_dev_first():
    cands = pd.concat([_many("NVDA", 60), _many("JNJ", 60)], ignore_index=True)
    out = headlines.sample(cands, {"NVDA": DAYS, "JNJ": DAYS}, n_per_ticker=20, seed=42)
    dev = out[out["split"] == "dev"]
    test = out[out["split"] == "test"]

    assert len(dev) == 12 and len(test) == 28  # 30% of 40
    assert dev["published_at"].max() <= test["published_at"].min()


def test_chronological_split_requires_sorted_input():
    unsorted = pd.Series(pd.to_datetime(["2019-07-02", "2019-07-01"], utc=True))
    with pytest.raises(ValueError):
        headlines.chronological_split(unsorted)


def test_sample_file_round_trip_preserves_instants(tmp_path):
    cands = _many("NVDA", 10)
    out = headlines.sample(cands, {"NVDA": DAYS}, n_per_ticker=5, seed=0)
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
