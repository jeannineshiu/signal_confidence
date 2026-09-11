"""CORP recalibration and Brier decomposition, validated on real forecasts.

References: exact hand arithmetic on five real NFL games, scikit-learn's
IsotonicRegression on every real 538 game (continuous confidences) and on the
committed AAPL signals (a few discrete confidences, so ties matter), and the
algebraic identities the decomposition must satisfy on real data.
"""

import numpy as np
import pytest
from conftest import favourite_form
from sklearn.isotonic import IsotonicRegression
from test_calibration_brier import FIVE_GAMES_BRIER, FIVE_GAMES_CONF, FIVE_GAMES_HIT

from sigconf.eval.calibration import expected_calibration_error
from sigconf.eval.corp import corp_analysis, decompose, pav, recalibrate
from sigconf.pipeline import run_split


def sklearn_isotonic(conf, hit) -> np.ndarray:
    return IsotonicRegression(out_of_bounds="clip").fit(conf, hit).predict(conf)


def refuse(_model):
    raise AssertionError("these tests must not call the LLM")


@pytest.fixture(scope="module")
def aapl_rows():
    """The committed AAPL signals for both splits, scored offline."""
    out = {}
    for split in ("dev", "test"):
        _, scored = run_split(split, offline=True, client_factory=refuse, log=lambda _: None)
        rows = scored[scored["state"] == "scored"]
        out[split] = (rows["confidence"].to_numpy(dtype=float), rows["hit"].to_numpy(dtype=float))
    return out


# ── Hand-computed fixture: the same five NFL playoff games as the Brier/ECE tests ──
# Sorted by confidence the hits are 1, 1, 0, 1, 0 (0.636, 0.653, 0.681, 0.685, 0.810).
# PAV pools every violation back to the start, so the fit is the constant 3/5:
# the confidences' ordering is worth nothing here, DSC = 0, and
# MCB = Brier − UNC = 0.294686304623487 − 0.6·0.4.


def test_pav_on_five_games_matches_hand_computation():
    np.testing.assert_allclose(recalibrate(FIVE_GAMES_CONF, FIVE_GAMES_HIT), [0.6] * 5,
                               rtol=1e-12)


def test_decomposition_on_five_games_matches_hand_computation():
    d = decompose(FIVE_GAMES_CONF, FIVE_GAMES_HIT)
    assert d["unc"] == pytest.approx(0.24, rel=1e-12)
    assert d["dsc"] == pytest.approx(0.0, abs=1e-15)
    assert d["mcb"] == pytest.approx(FIVE_GAMES_BRIER - 0.24, rel=1e-12)


def test_weighted_pooling_arithmetic():
    # 1 (weight 1) then 0 (weight 3) violate the order → one block at 1/4.
    np.testing.assert_allclose(pav(np.array([1.0, 0.0]), np.array([1.0, 3.0])), [0.25, 0.25])
    np.testing.assert_allclose(pav(np.array([0.2, 0.5, 0.4, 0.9]), np.ones(4)),
                               [0.2, 0.45, 0.45, 0.9])


# ── Independent implementation: scikit-learn ──────────────────────────────────


@pytest.mark.parametrize("league", ["nba", "nfl"])
@pytest.mark.parametrize("form", ["favourite", "two_sided"])
def test_pav_equals_sklearn_on_real_538_forecasts(request, league, form):
    df = request.getfixturevalue(f"{league}_538")
    if form == "favourite":
        conf, hit = favourite_form(df)
    else:
        conf, hit = df["prob1"].to_numpy(), df["prob1_outcome"].to_numpy(dtype=float)
    np.testing.assert_allclose(recalibrate(conf, hit), sklearn_isotonic(conf, hit),
                               rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_pav_equals_sklearn_with_tied_confidences(aapl_rows, split):
    conf, hit = aapl_rows[split]
    assert np.unique(conf).size < 10  # the LLM states only a handful of values
    np.testing.assert_allclose(recalibrate(conf, hit), sklearn_isotonic(conf, hit),
                               rtol=1e-12, atol=1e-15)


# ── Identities that must hold on real data ────────────────────────────────────


@pytest.mark.parametrize("league", ["nba", "nfl"])
def test_brier_equals_mcb_minus_dsc_plus_unc(request, league):
    conf, hit = favourite_form(request.getfixturevalue(f"{league}_538"))
    d = decompose(conf, hit)
    assert d["mcb"] - d["dsc"] + d["unc"] == pytest.approx(d["brier"], rel=1e-12)
    assert d["mcb"] >= 0 and d["dsc"] >= 0
    assert d["unc"] == pytest.approx(hit.mean() * (1 - hit.mean()), rel=1e-12)


def test_recalibrated_forecast_has_no_miscalibration(nba_538):
    conf, hit = favourite_form(nba_538)
    recal = recalibrate(conf, hit)
    d = decompose(recal, hit)
    assert d["mcb"] == pytest.approx(0.0, abs=1e-15)
    assert d["dsc"] == pytest.approx(decompose(conf, hit)["dsc"], rel=1e-12)


def test_constant_forecast_has_no_discrimination(nba_538):
    _, hit = favourite_form(nba_538)
    d = decompose(np.full(hit.size, 0.5), hit)
    assert d["dsc"] == pytest.approx(0.0, abs=1e-15)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        decompose([], [])
    with pytest.raises(ValueError):
        decompose([0.7], [0.5])


# ── The analysis: references, band, reproducibility ───────────────────────────


@pytest.fixture(scope="module")
def nfl_analysis(nfl_538):
    conf, hit = favourite_form(nfl_538)
    return conf, hit, corp_analysis(conf, hit, n_sims=200, seed=7)


def test_analysis_is_reproducible_for_a_seed(nfl_analysis):
    conf, hit, first = nfl_analysis
    assert corp_analysis(conf, hit, n_sims=200, seed=7) == first


def test_analysis_agrees_with_decompose_and_orders_its_band(nfl_analysis):
    conf, hit, a = nfl_analysis
    d = decompose(conf, hit)
    assert a["mcb"]["value"] == d["mcb"] and a["dsc"]["value"] == d["dsc"]
    curve = a["curve"]
    assert sum(p["n"] for p in curve) == len(conf)
    assert all(p["band_low"] <= p["band_high"] for p in curve)
    assert [p["confidence"] for p in curve] == sorted(p["confidence"] for p in curve)
    for key, null in (("mcb", "perfect_calibration_null"), ("dsc", "no_discrimination_null")):
        assert 0 < a[key][null]["p_value"] <= 1


def test_references_on_the_committed_aapl_test_calls(aapl_rows):
    """The two readings the README makes, recomputed from the committed rows."""
    conf, hit = aapl_rows["test"]
    a = corp_analysis(conf, hit, n_sims=2000)
    assert a["mcb"]["perfect_calibration_null"]["p_value"] < 0.01
    assert a["dsc"]["no_discrimination_null"]["p_value"] > 0.05
    assert [p["n"] for p in a["curve"]] == [11, 34, 27, 5]


def test_readme_claims_about_bins_on_the_committed_rows(aapl_rows):
    # Every stated value is over-confident, so pooling 0.70 and 0.75 into one
    # bin cancels nothing: one bin per stated value gives the same ECE.
    conf, hit = aapl_rows["test"]
    levels = np.unique(conf)
    per_value = np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2, [1.0]])
    assert expected_calibration_error(conf, hit, per_value) == pytest.approx(
        expected_calibration_error(conf, hit), rel=1e-12)
    # 0.70 vs 0.75 on test (15/34 vs 17/27) reverses on dev (10/16 vs 1/5).
    for split, want in (("test", {0.7: (15, 34), 0.75: (17, 27)}),
                        ("dev", {0.7: (10, 16), 0.75: (1, 5)})):
        c, h = aapl_rows[split]
        got = {v: (int(h[c == v].sum()), int((c == v).sum())) for v in want}
        assert got == want
