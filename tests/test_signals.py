"""Signal generation, driven by a scripted fake client — no network, no spend.

The headlines are real rows from the committed AAPL sample; only the model's
replies are scripted, because these tests check our handling of each kind of
reply, not the model.
"""

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from sigconf import config
from sigconf.config import LLMSettings
from sigconf.data.headlines import load_sample
from sigconf.signals import generate
from sigconf.signals.budget import BudgetExceeded, CostGuard, estimate_input_tokens
from sigconf.signals.client import Completion, LLMCallError
from sigconf.signals.generate import SignalCache, check_model_allowed, generate_signal, run_signals
from sigconf.signals.prompt import PROMPT_SHA, SYSTEM_PROMPT, USER_TEMPLATE, render_user
from sigconf.signals.schema import Signal

MODEL = config.DEFAULT_LLM_MODEL
GOOD = '{"direction": "bullish", "confidence": 0.7, "reasoning": "Strong demand."}'


class FakeClient:
    """Replays scripted replies; an Exception in the script is raised instead."""

    def __init__(self, script, tokens=(300, 30), model=MODEL):
        self.script = list(script)
        self.tokens = tokens
        self.model = model
        self.calls: list[str] = []

    def complete(self, system, user, logprobs=False):
        self.calls.append(user)
        reply = self.script.pop(0) if self.script else GOOD
        if isinstance(reply, Exception):
            raise reply
        return Completion(reply, self.tokens[0], self.tokens[1], self.model)


def refuse(_model):
    raise AssertionError("no client may be built on this path")


@pytest.fixture(scope="module")
def sample():
    return load_sample()


def guard(max_calls=100, max_cost=1.0):
    return CostGuard(MODEL, max_calls, max_cost)


def settings(max_calls=400, max_cost=1.0, model=MODEL):
    return LLMSettings(model=model, max_calls=max_calls, max_cost_usd=max_cost)


# ── Schema ────────────────────────────────────────────────────────────────────


def test_schema_accepts_a_valid_reply():
    s = Signal.model_validate_json(GOOD)
    assert (s.direction, s.confidence) == ("bullish", 0.7)


@pytest.mark.parametrize(
    "reply",
    [
        '{"direction": "bullish", "confidence": 1.5, "reasoning": "x"}',
        '{"direction": "BUY", "confidence": 0.7, "reasoning": "x"}',
        '{"direction": "Bullish", "confidence": 0.7, "reasoning": "x"}',
        '{"direction": "bullish", "reasoning": "x"}',
        '{"direction": "bullish", "confidence": 0.7, "reasoning": ""}',
        '{"direction": "bullish", "confidence": 0.7, "reasoning": "x", "extra": 1}',
        "I think the stock will go up.",
        "",
    ],
)
def test_schema_rejects_malformed_replies(reply):
    with pytest.raises(ValidationError):
        Signal.model_validate_json(reply)


# ── Retry-once path ───────────────────────────────────────────────────────────


def _one(sample, client, g=None):
    row = sample.iloc[0]
    return generate_signal(client, g or guard(), row["id"], row["ticker"], row["headline"])


def test_valid_first_reply_is_ok_in_one_attempt(sample):
    rec = _one(sample, FakeClient([GOOD]))
    assert (rec["status"], rec["attempts"], rec["direction"], rec["confidence"]) == (
        "ok", 1, "bullish", 0.7)


def test_malformed_then_valid_is_ok_after_one_retry_with_a_correction(sample):
    client = FakeClient(['{"direction": "BUY", "confidence": 0.7, "reasoning": "x"}', GOOD])
    rec = _one(sample, client)
    assert (rec["status"], rec["attempts"]) == ("ok", 2)
    assert "not valid" in client.calls[1] and "direction" in client.calls[1]
    assert client.calls[1].startswith(client.calls[0])  # same prompt + correction
    assert len(rec["raw_texts"]) == 2


def test_malformed_twice_is_a_counted_parse_failure_with_raw_text_kept(sample):
    rec = _one(sample, FakeClient(["not json", "still not json"]))
    assert (rec["status"], rec["attempts"]) == ("parse_failure", 2)
    assert rec["raw_texts"] == ["not json", "still not json"]
    assert rec["direction"] is None and rec["error"]


def test_api_error_is_not_a_parse_failure(sample):
    rec = _one(sample, FakeClient([LLMCallError("RateLimitError: slow down")]))
    assert (rec["status"], rec["attempts"]) == ("api_error", 1)
    assert "RateLimitError" in rec["error"]


def test_api_error_on_the_retry_is_an_api_error(sample):
    rec = _one(sample, FakeClient(["not json", LLMCallError("timeout")]))
    assert (rec["status"], rec["attempts"]) == ("api_error", 2)


def test_a_different_served_model_aborts(sample):
    with pytest.raises(RuntimeError, match="answered"):
        _one(sample, FakeClient([GOOD], model="gpt-4o-mini-2099-01-01"))


def test_usage_and_cost_are_recorded(sample):
    g = guard()
    rec = _one(sample, FakeClient(["bad", GOOD], tokens=(1000, 100)), g)
    assert (rec["input_tokens"], rec["output_tokens"]) == (2000, 200)
    assert rec["cost_usd"] == pytest.approx((2000 * 0.15 + 200 * 0.60) / 1e6)
    assert g.calls == 2


# ── Prompt content ────────────────────────────────────────────────────────────


def test_prompt_is_ticker_and_headline_only_never_the_date(sample):
    assert "{" not in USER_TEMPLATE.replace("{ticker}", "").replace("{headline}", "")
    for row in sample.itertuples():
        user = render_user(row.ticker, row.headline)
        assert user == f"Ticker: {row.ticker}\nHeadline: {row.headline}"
        published = row.published_at.tz_convert("America/New_York")
        assert published.strftime("%Y-%m-%d") not in user + SYSTEM_PROMPT
        assert published.strftime("%B %d") not in user + SYSTEM_PROMPT


# ── Model guard ───────────────────────────────────────────────────────────────


def test_dated_snapshot_with_early_cutoff_is_allowed():
    check_model_allowed(MODEL)


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5"])
def test_aliases_and_undocumented_models_are_refused(model):
    with pytest.raises(ValueError, match="training cutoff"):
        check_model_allowed(model)


def test_model_with_cutoff_after_sample_start_is_refused(monkeypatch):
    monkeypatch.setitem(generate.LLM_TRAINING_CUTOFFS, "late-model", "2024-01-01")
    with pytest.raises(ValueError, match="not before SAMPLE_START"):
        check_model_allowed("late-model")


def test_run_refuses_a_bad_model_before_building_a_client(sample, tmp_path):
    with pytest.raises(ValueError):
        run_signals(sample.head(2), settings(model="gpt-4o-mini"), refuse,
                    SignalCache(tmp_path / "s.jsonl"), log=lambda _: None)


# ── Cost guard ────────────────────────────────────────────────────────────────


def test_estimate_is_conservative_for_real_prompts(sample):
    # ~4 chars/token is typical English; the estimate assumes 3 plus overhead.
    for row in sample.head(20).itertuples():
        text = SYSTEM_PROMPT + render_user(row.ticker, row.headline)
        assert estimate_input_tokens(SYSTEM_PROMPT, render_user(row.ticker, row.headline)) \
            > len(text) / 4


@pytest.mark.parametrize(("max_calls", "max_cost"), [(5, 1.0), (400, 0.0001)])
def test_preflight_over_a_cap_aborts_with_zero_calls(sample, tmp_path, max_calls, max_cost):
    client = FakeClient([])
    with pytest.raises(BudgetExceeded):
        run_signals(sample.head(10), settings(max_calls, max_cost), lambda _: client,
                    SignalCache(tmp_path / "s.jsonl"), log=lambda _: None)
    assert client.calls == []
    assert not (tmp_path / "s.jsonl").exists()


def test_in_flight_stop_when_real_usage_exceeds_the_estimate(sample, tmp_path):
    # Each call bills 1M input tokens ($0.15): far above the estimate the
    # pre-flight approved, so the in-flight check must stop the batch.
    client = FakeClient([], tokens=(1_000_000, 0))
    cache = SignalCache(tmp_path / "s.jsonl")
    with pytest.raises(BudgetExceeded):
        run_signals(sample.head(10), settings(max_cost=0.2), lambda _: client, cache,
                    log=lambda _: None)
    # Call 1 bills $0.15; the guard now prices the next call at $0.15 too, and
    # $0.30 would exceed $0.20, so it stops. Spend never passes the cap here.
    assert len(client.calls) == 1
    lines = (tmp_path / "s.jsonl").read_text().splitlines()
    assert len(lines) == 1  # the finished item is safely cached before the stop
    assert json.loads(lines[0])["cost_usd"] == pytest.approx(0.15)


def test_cached_items_cost_nothing_and_skip_the_client(sample, tmp_path):
    items = sample.head(6)
    cache = SignalCache(tmp_path / "s.jsonl")
    first = FakeClient([])
    run_signals(items.head(4), settings(), lambda _: first, cache, log=lambda _: None)
    assert len(first.calls) == 4

    # Cap allows exactly the worst case for the 2 uncached items, not for all 6.
    g = CostGuard(MODEL, 4, 1.0)
    users = [render_user(t, h) for t, h in zip(items["ticker"], items["headline"], strict=True)]
    exact_two = g.preflight(SYSTEM_PROMPT, users[4:])
    second = FakeClient([])
    out = run_signals(items, settings(max_calls=4, max_cost=exact_two), lambda _: second,
                      SignalCache(tmp_path / "s.jsonl"), log=lambda _: None)
    assert len(second.calls) == 2
    assert out["id"].tolist() == items["id"].tolist()
    assert (out["status"] == "ok").all()


def test_offline_cache_hit_needs_no_client_and_miss_is_an_error(sample, tmp_path):
    items = sample.head(3)
    cache = SignalCache(tmp_path / "s.jsonl")
    run_signals(items, settings(), lambda _: FakeClient([]), cache, log=lambda _: None)

    reloaded = SignalCache(tmp_path / "s.jsonl")
    out = run_signals(items, settings(), refuse, reloaded, offline=True, log=lambda _: None)
    assert len(out) == 3
    with pytest.raises(RuntimeError, match="offline"):
        run_signals(sample.head(4), settings(), refuse, reloaded, offline=True,
                    log=lambda _: None)


def test_a_changed_prompt_misses_the_cache(sample, tmp_path):
    path = tmp_path / "s.jsonl"
    row = sample.iloc[0]
    stale = {"id": row["id"], "model": MODEL, "prompt_sha": "an-older-prompt", "status": "ok",
             "direction": "bullish", "confidence": 0.9, "attempts": 1, "cost_usd": 0.0}
    path.write_text(json.dumps(stale) + "\n")
    cache = SignalCache(path)
    assert cache.get(row["id"], MODEL) is None
    assert cache.get(row["id"], MODEL, "an-older-prompt") is not None
    assert PROMPT_SHA != "an-older-prompt"


def test_api_errors_are_cached_and_retried_only_on_request(sample, tmp_path):
    items = sample.head(2)
    cache = SignalCache(tmp_path / "s.jsonl")
    run_signals(items, settings(), lambda _: FakeClient([LLMCallError("down"), GOOD]), cache,
                log=lambda _: None)
    assert cache.get(items["id"].iloc[0], MODEL)["status"] == "api_error"

    run_signals(items, settings(), refuse, cache, log=lambda _: None)  # no retry by default
    retry = FakeClient([GOOD])
    out = run_signals(items, settings(), lambda _: retry, cache, retry_api_errors=True,
                      log=lambda _: None)
    assert len(retry.calls) == 1
    assert out["status"].tolist() == ["ok", "ok"]


def test_cache_file_is_one_json_record_per_line(sample, tmp_path):
    run_signals(sample.head(3), settings(), lambda _: FakeClient([]),
                SignalCache(tmp_path / "s.jsonl"), log=lambda _: None)
    records = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == sample.head(3)["id"].tolist()
    assert all(r["prompt_sha"] == PROMPT_SHA and r["model"] == MODEL for r in records)
    assert all("generated_at" in r for r in records)
    assert pd.DataFrame(records)["cost_usd"].gt(0).all()
