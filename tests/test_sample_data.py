"""Checks on the committed data itself, not just the code that built it."""

import pandas as pd
import pytest

from sigconf import config
from sigconf.data.labels import close_times
from sigconf.pipeline import load_labeled


@pytest.fixture(scope="module")
def labeled():
    return load_labeled(offline=True)


def test_sample_size_balance_and_split(labeled):
    n_per_ticker = config.N_HEADLINES // len(config.TICKER_PATTERNS)
    assert len(labeled) == config.N_HEADLINES
    assert labeled["ticker"].value_counts().to_dict() == dict.fromkeys(
        config.TICKER_PATTERNS, n_per_ticker
    )
    assert (labeled["split"] == "dev").sum() == round(config.N_HEADLINES * config.DEV_FRACTION)


def test_dev_strictly_precedes_test(labeled):
    dev = labeled[labeled["split"] == "dev"]
    test = labeled[labeled["split"] == "test"]
    assert dev["published_at"].max() < test["published_at"].min()


def test_no_look_ahead_on_committed_sample(labeled):
    """Every entry close is strictly after the headline was published."""
    t0_close = close_times(pd.DatetimeIndex(pd.to_datetime(labeled["t0"])))
    assert (t0_close > pd.DatetimeIndex(labeled["published_at"])).all()
    assert (pd.to_datetime(labeled["t1"]) > pd.to_datetime(labeled["t0"])).all()


def test_every_headline_has_a_label_window_and_one_per_anchor_day(labeled):
    assert labeled["label"].notna().all()
    assert not labeled.duplicated(["ticker", "t0"]).any()


def test_headlines_name_the_company(labeled):
    for ticker, pattern in config.TICKER_PATTERNS.items():
        rows = labeled[labeled["ticker"] == ticker]
        assert rows["headline"].str.contains(pattern, case=False, regex=True).all()
