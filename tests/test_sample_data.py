"""Checks on the committed data itself, not just the code that built it."""

import pandas as pd
import pytest

from sigconf import config
from sigconf.data.labels import close_times, is_time_unknown
from sigconf.pipeline import load_labeled


@pytest.fixture(scope="module")
def labeled():
    return load_labeled(offline=True)


def test_sample_size_and_split(labeled):
    assert len(labeled) == config.N_HEADLINES
    assert (labeled["ticker"] == config.TICKER).all()
    assert (labeled["split"] == "dev").sum() == round(config.N_HEADLINES * config.DEV_FRACTION)


def test_every_headline_postdates_every_allowed_models_training_cutoff(labeled):
    latest_cutoff = max(pd.Timestamp(c, tz="UTC") for c in config.LLM_TRAINING_CUTOFFS.values())
    assert pd.Timestamp(config.SAMPLE_START, tz="UTC") > latest_cutoff
    assert (labeled["published_at"] >= pd.Timestamp(config.SAMPLE_START, tz="UTC")).all()


def test_dev_strictly_precedes_test(labeled):
    dev = labeled[labeled["split"] == "dev"]
    test = labeled[labeled["split"] == "test"]
    assert dev["published_at"].max() < test["published_at"].min()


def test_no_look_ahead_on_committed_sample(labeled):
    """Every entry close is strictly after the headline was published."""
    t0_close = close_times(pd.DatetimeIndex(pd.to_datetime(labeled["t0"])))
    assert (t0_close > pd.DatetimeIndex(labeled["published_at"])).all()
    assert (pd.to_datetime(labeled["t1"]) > pd.to_datetime(labeled["t0"])).all()


def test_every_headline_has_exact_time_and_a_label_window(labeled):
    assert not is_time_unknown(labeled["published_at"]).any()
    assert labeled["label"].notna().all()
    assert not labeled["t0"].duplicated().any()  # one headline per anchor day


def test_headlines_name_the_company_and_are_not_live_blogs(labeled):
    h = labeled["headline"]
    assert h.str.contains(config.COMPANY_PATTERN, case=False, regex=True).all()
    assert not h.str.contains(config.LIVE_BLOG_PATTERN, case=False, regex=True).any()
