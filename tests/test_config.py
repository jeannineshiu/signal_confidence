import math
import os

import pandas as pd
import pytest

from sigconf import config


def test_bin_edges_cover_unit_interval_in_increasing_order():
    edges = config.BIN_EDGES
    assert edges[0] == 0.0
    assert edges[-1] == 1.0
    assert all(a < b for a, b in zip(edges, edges[1:], strict=False))


def test_spec_buckets_start_at_half_with_a_separate_incoherent_bucket():
    # Spec §6 buckets are 0.5-0.6 … 0.9-1.0; below 0.5 is its own bucket.
    assert config.BIN_EDGES[1] == 0.5
    assert config.BIN_EDGES[2:] == (0.6, 0.7, 0.8, 0.9, 1.0)


def test_sampling_starts_after_every_allowed_models_training_cutoff():
    start = pd.Timestamp(config.SAMPLE_START)
    assert config.DEFAULT_LLM_MODEL in config.LLM_TRAINING_CUTOFFS
    assert all(pd.Timestamp(c) < start for c in config.LLM_TRAINING_CUTOFFS.values())


def test_split_and_dead_band_are_sane():
    assert 0 < config.DEV_FRACTION < 0.5  # test must be the larger split
    assert 0 < config.DEAD_BAND < 0.01


def test_no_api_key_during_tests():
    assert "OPENAI_API_KEY" not in os.environ


def test_llm_settings_defaults(monkeypatch):
    for var in ("LLM_MODEL", "LLM_MAX_CALLS", "LLM_MAX_COST_USD"):
        monkeypatch.delenv(var, raising=False)
    settings = config.llm_settings()
    assert settings.model == config.DEFAULT_LLM_MODEL
    assert settings.max_calls == config.DEFAULT_MAX_CALLS
    assert settings.max_cost_usd == config.DEFAULT_MAX_COST_USD


def test_llm_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_MAX_CALLS", "10")
    monkeypatch.setenv("LLM_MAX_COST_USD", "0.25")
    settings = config.llm_settings()
    assert settings == config.LLMSettings("some-model", 10, 0.25)


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("LLM_MAX_CALLS", "-1"),
        ("LLM_MAX_COST_USD", "-0.01"),
        ("LLM_MAX_COST_USD", str(math.nan)),
        ("LLM_MODEL", "  "),
    ],
)
def test_llm_settings_rejects_invalid_caps(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError):
        config.llm_settings()
