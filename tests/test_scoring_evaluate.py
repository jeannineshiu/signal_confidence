import json

import numpy as np
import pandas as pd
import pytest
from conftest import favourite_form
from sklearn.metrics import brier_score_loss

from sigconf.eval.evaluate import evaluate
from sigconf.eval.scoring import STATES, funnel, score
from sigconf.pipeline import load_labeled


@pytest.fixture(scope="module")
def real_labeled():
    """The committed AAPL sample with its real next-day labels."""
    return load_labeled(offline=True)


def _signals(ids, rows):
    """Hand-written signal records for exercising the state logic (no LLM yet)."""
    return pd.DataFrame(
        [{"id": i, "status": s, "direction": d, "confidence": c}
         for i, (s, d, c) in zip(ids, rows, strict=True)]
    )


def test_every_state_is_assigned_in_funnel_order(real_labeled):
    flat = real_labeled[real_labeled["label"] == "flat"].head(2)
    up = real_labeled[real_labeled["label"] == "up"].head(4)
    down = real_labeled[real_labeled["label"] == "down"].head(1)
    items = pd.concat([flat, up, down])
    sig = _signals(items["id"], [
        ("ok", "bullish", 0.8),          # flat label → flat_label
        ("api_error", None, None),       # flat label, but the call failed first → api_error
        ("parse_failure", None, None),   # → parse_failure
        ("ok", "neutral", 0.6),          # → neutral
        ("ok", "bullish", 0.7),          # up, called up → scored, hit 1
        ("ok", "bearish", 0.9),          # up, called down → scored, hit 0
        ("ok", "bearish", 0.55),         # down, called down → scored, hit 1
    ])
    out = score(items, sig)

    assert out["state"].tolist() == ["flat_label", "api_error", "parse_failure", "neutral",
                                     "scored", "scored", "scored"]
    assert out["hit"].tolist()[4:] == [1, 0, 1]
    assert out["hit"].isna().tolist()[:4] == [True] * 4
    assert funnel(out) == {"api_error": 1, "parse_failure": 1, "neutral": 1, "flat_label": 1,
                           "scored": 3, "total": 7}


def test_a_headline_without_a_signal_is_an_error_not_a_skip(real_labeled):
    items = real_labeled.head(3)
    sig = _signals(items["id"][:2], [("ok", "bullish", 0.7)] * 2)
    with pytest.raises(ValueError, match="no signal"):
        score(items, sig)


def test_signals_for_other_splits_are_ignored(real_labeled):
    dev = real_labeled[real_labeled["split"] == "dev"]
    sig = _signals(real_labeled["id"], [("ok", "neutral", 0.5)] * len(real_labeled))
    assert len(score(dev, sig)) == len(dev)


@pytest.mark.parametrize(
    "bad",
    [("ok", "BUY", 0.7), ("ok", "bullish", 1.5), ("ok", "bullish", None), ("weird", None, None)],
)
def test_invalid_signal_values_are_rejected(real_labeled, bad):
    items = real_labeled.head(1)
    with pytest.raises(ValueError):
        score(items, _signals(items["id"], [bad]))


def test_duplicate_signal_ids_are_rejected(real_labeled):
    items = real_labeled.head(1)
    sig = _signals([items["id"].iloc[0]] * 2, [("ok", "bullish", 0.7)] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        score(items, sig)


# ── evaluate(): the full metrics dict, run on real 538 forecasts ─────────────


def _538_as_scored(games: pd.DataFrame) -> pd.DataFrame:
    """Map real 538 games onto this project's shapes: 'up' = team 1 won;
    the forecaster calls the favourite ('bullish' = team 1) with its probability."""
    labeled = pd.DataFrame({
        "id": [f"g{i}" for i in range(len(games))],
        "label": np.where(games["prob1_outcome"] == 1, "up", "down"),
    })
    conf, _ = favourite_form(games)
    signals = pd.DataFrame({
        "id": labeled["id"],
        "status": "ok",
        "direction": np.where(games["prob1"] >= games["prob2"], "bullish", "bearish"),
        "confidence": conf,
    })
    return score(labeled, signals)


@pytest.fixture(scope="module")
def nfl_metrics(nfl_538):
    return evaluate(_538_as_scored(nfl_538), n_boot=2_000, n_sims=2_000)


def test_evaluate_numbers_agree_with_independent_computation(nfl_538, nfl_metrics):
    conf, hit = favourite_form(nfl_538)
    m = nfl_metrics
    assert m["funnel"]["scored"] == m["n_headlines"] == len(nfl_538)
    assert m["coverage"] == 1.0
    assert m["accuracy"]["value"] == pytest.approx(hit.mean())
    assert m["brier"]["value"] == pytest.approx(brier_score_loss(hit, conf), rel=1e-12)
    # always-up here = "always pick team 1"; its accuracy is team 1's real win rate.
    assert m["baselines"]["always_up_same_items"]["value"] == pytest.approx(
        (nfl_538["prob1_outcome"] == 1).mean())


def test_evaluate_reports_every_number_with_uncertainty_and_baseline(nfl_metrics):
    m = nfl_metrics
    for key in ("accuracy", "brier", "ece"):
        lo, hi = m[key]["ci95"]
        assert lo <= m[key]["value"] <= hi
    assert m["brier"]["coin_flip_reference"] == 0.25
    assert "perfect_calibration_null" in m["ece"]
    assert m["ece"]["perfect_calibration_null"]["method"].startswith("simulation")
    assert set(m["baselines"]) == {"coin_flip", "always_up_same_items",
                                   "always_up_all_directional_labels", "llm_vs_always_up_mcnemar"}
    assert sum(r["n"] for r in m["reliability_table"]) == m["accuracy"]["n"]


def test_evaluate_output_is_json_serialisable_and_deterministic(nfl_538, nfl_metrics):
    again = evaluate(_538_as_scored(nfl_538), n_boot=2_000, n_sims=2_000)
    assert json.dumps(nfl_metrics, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_evaluate_refuses_an_empty_scored_set(real_labeled):
    items = real_labeled.head(3)
    sig = _signals(items["id"], [("ok", "neutral", 0.5)] * 3)
    with pytest.raises(ValueError, match="no scored"):
        evaluate(score(items, sig))


def test_funnel_keys_follow_state_order(nfl_metrics):
    assert list(nfl_metrics["funnel"])[:-1] == list(STATES)
