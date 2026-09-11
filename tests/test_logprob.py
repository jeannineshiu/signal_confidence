"""Token-probability confidence and the paired comparison (PLAN.md §8).

Extraction is tested on two real API replies captured with full logprobs
(tests/fixtures/openai_logprob_replies.json); the comparison statistics on
real FiveThirtyEight forecasts, against scikit-learn.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from conftest import favourite_form
from sklearn.metrics import roc_auc_score

from sigconf.eval.compare import auroc, compare_confidences
from sigconf.signals.budget import CostGuard
from sigconf.signals.client import Completion
from sigconf.signals.generate import generate_signal
from sigconf.signals.logprob import direction_distribution, logprob_confidence

REPLIES = json.loads((Path(__file__).parent / "fixtures" / "openai_logprob_replies.json")
                     .read_text())
MODEL = "gpt-4o-mini-2024-07-18"


def _probs(reply, token):
    """Probabilities of the alternatives listed at the direction token, by hand."""
    tok = next(t for t in reply["token_logprobs"] if t["token"] == token)
    return {alt: math.exp(lp) for alt, lp in tok["top"]}


# ── Extraction on real replies ────────────────────────────────────────────────


def test_directional_reply_confidence_matches_hand_computation():
    reply = REPLIES["directional"]
    assert json.loads(reply["text"])["direction"] == "bullish"
    alts = _probs(reply, "bull")
    # Alternatives that can only begin "bullish": bull, bul, bu, … ; "bearish": bear, …
    p_bull = sum(p for a, p in alts.items() if "bullish".startswith(a) and a and a[0] == "b"
                 and not "bearish".startswith(a))
    p_bear = sum(p for a, p in alts.items() if "bearish".startswith(a)
                 and not "bullish".startswith(a))

    dist = direction_distribution(reply["token_logprobs"])
    conf, opposite_listed = logprob_confidence(dist)
    assert dist["chosen"] == "bullish"
    assert dist["p"]["bullish"] == pytest.approx(p_bull, rel=1e-12)
    assert dist["p"]["bearish"] == pytest.approx(p_bear, rel=1e-12)
    assert opposite_listed
    assert conf == pytest.approx(p_bull / (p_bull + p_bear), rel=1e-12)
    assert 0.5 <= conf <= 1.0


def test_neutral_reply_has_no_directional_confidence():
    dist = direction_distribution(REPLIES["neutral"]["token_logprobs"])
    assert dist["chosen"] == "neutral"
    assert logprob_confidence(dist) == (None, True)


def test_real_replies_put_almost_all_mass_on_listed_labels():
    for key in ("directional", "neutral"):
        dist = direction_distribution(REPLIES[key]["token_logprobs"])
        assert dist["captured_mass"] == pytest.approx(1.0, abs=1e-3)


# ── Extraction edge cases (token layouts the API could produce) ──────────────


def _tokens(*pairs):
    """[(token, [(alt, prob), …]), …] → the API's token_logprobs shape."""
    return [{"token": t, "logprob": math.log(dict(top)[t]),
             "top": [[a, math.log(p)] for a, p in top]} for t, top in pairs]


def test_quote_attached_to_the_value_token_is_handled():
    toks = _tokens(('{"', [('{"', 1.0)]), ("direction", [("direction", 1.0)]),
                   ('":', [('":', 1.0)]), ('"bear', [('"bear', 0.7), ('"bull', 0.2), ('"n', 0.1)]),
                   ("ish", [("ish", 1.0)]))
    dist = direction_distribution(toks)
    assert dist["chosen"] == "bearish"
    assert logprob_confidence(dist)[0] == pytest.approx(0.7 / 0.9)


def test_ambiguous_prefix_is_not_counted_for_either_label():
    toks = _tokens(("{\"direction\":\"", [("{\"direction\":\"", 1.0)]),
                   ("bull", [("bull", 0.6), ("b", 0.3), ("bear", 0.1)]))
    dist = direction_distribution(toks)
    assert dist["p"] == pytest.approx({"bullish": 0.6, "bearish": 0.1, "neutral": 0.0})
    assert dist["captured_mass"] == pytest.approx(0.7)


def test_opposite_missing_from_top_k_is_flagged_as_an_upper_bound():
    toks = _tokens(("{\"direction\":\"", [("{\"direction\":\"", 1.0)]),
                   ("bull", [("bull", 0.98), ("neutral", 0.02)]))
    conf, opposite_listed = logprob_confidence(direction_distribution(toks))
    assert conf == 1.0 and not opposite_listed


def test_reply_without_a_direction_value_raises():
    with pytest.raises(ValueError):
        direction_distribution(_tokens(("{}", [("{}", 1.0)])))


# ── generate_signal with logprobs, replaying a real reply ────────────────────


class ReplayClient:
    def __init__(self, reply, text=None):
        self.reply, self.text, self.calls = reply, text, []

    def complete(self, system, user, logprobs=False):
        self.calls.append(logprobs)
        return Completion(self.text or self.reply["text"], 250, 30, MODEL,
                          self.reply["token_logprobs"] if logprobs else None)


def test_generate_signal_records_both_confidences_from_one_reply():
    client = ReplayClient(REPLIES["directional"])
    rec = generate_signal(client, CostGuard(MODEL, 10, 1.0), "h000", "AAPL", "x",
                          with_logprobs=True)
    assert client.calls == [True]
    assert rec["status"] == "ok" and rec["direction"] == "bullish"
    assert rec["confidence"] == json.loads(REPLIES["directional"]["text"])["confidence"]
    assert 0.5 <= rec["logprob_confidence"] <= 1.0
    assert rec["logprob_error"] is None and rec["direction_top_logprobs"]


def test_token_disagreeing_with_parsed_reply_is_an_error_not_a_guess():
    # Text says bearish but the token stream says bullish: refuse to report a number.
    text = REPLIES["directional"]["text"].replace("bullish", "bearish")
    rec = generate_signal(ReplayClient(REPLIES["directional"], text), CostGuard(MODEL, 10, 1.0),
                          "h000", "AAPL", "x", with_logprobs=True)
    assert rec["status"] == "ok"
    assert rec["logprob_confidence"] is None and "parsed reply" in rec["logprob_error"]


def test_logprobs_are_not_requested_unless_asked():
    client = ReplayClient(REPLIES["directional"])
    rec = generate_signal(client, CostGuard(MODEL, 10, 1.0), "h000", "AAPL", "x")
    assert client.calls == [False] and "logprob_confidence" not in rec


# ── Comparison statistics on real 538 forecasts ──────────────────────────────


def test_auroc_equals_sklearn_on_real_forecasts(nba_538):
    conf, hit = favourite_form(nba_538)
    assert auroc(conf, hit) == pytest.approx(roc_auc_score(hit, conf), rel=1e-12)


def test_auroc_handles_ties_like_sklearn(nba_538):
    conf, hit = favourite_form(nba_538)
    coarse = np.round(conf * 20) / 20  # discrete, like verbalised confidence
    assert auroc(coarse, hit) == pytest.approx(roc_auc_score(hit, coarse), rel=1e-12)


def test_auroc_undefined_with_one_class():
    assert auroc([0.6, 0.7], [1, 1]) is None


def test_paired_comparison_of_a_readout_with_itself_is_exactly_zero(nfl_538):
    conf, hit = favourite_form(nfl_538)
    m = compare_confidences(conf, conf, hit, names=("a", "b"), n_boot=300, n_sims=300)
    d = m["b_minus_a"]
    for stat in ("brier", "ece", "auroc"):
        assert d[stat]["value"] == 0.0 and d[stat]["ci95"] == [0.0, 0.0]


def test_paired_comparison_detects_a_real_coarsening(nfl_538):
    # 538's probabilities vs the same probabilities rounded to 0.05: the rounded
    # readout has fewer distinct values and cannot discriminate better.
    conf, hit = favourite_form(nfl_538)
    coarse = np.round(conf * 20) / 20
    m = compare_confidences(conf, coarse, hit, names=("exact", "coarse"), n_boot=500,
                            n_sims=500)
    assert m["coarse"]["n_distinct_values"] < m["exact"]["n_distinct_values"]
    assert m["coarse_minus_exact"]["auroc"]["value"] <= 1e-9
    assert m["accuracy"] == pytest.approx(hit.mean())
    assert json.loads(json.dumps(m)) == m
