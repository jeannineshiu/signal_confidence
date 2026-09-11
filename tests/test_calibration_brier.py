"""Brier, reliability table and ECE, validated on real FiveThirtyEight forecasts.

Two independent references: exact hand arithmetic on five real games, and
scikit-learn on all 8.9k real NBA games.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import favourite_form
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from sigconf.eval.brier import COIN_FLIP_BRIER, brier_score, brier_skill_score
from sigconf.eval.calibration import (
    assign_bins,
    ece_batch,
    expected_calibration_error,
    reliability_table,
)

# ── Hand-computed fixture: last five games of FiveThirtyEight's 2021 NFL season ──
# Each row: favourite's published win probability, 1 if the favourite won.
#   Super Bowl LVI   Rams (0.6849395710053819) beat Bengals        → hit 1, (1−c)² = 0.0992630739
#   NFC Championship Rams (0.6360927506267418) beat 49ers          → hit 1, (1−c)² = 0.1324284861
#   AFC Championship Chiefs (0.810417382104968) lost to Bengals    → hit 0,     c² = 0.6567763332
#   AFC Divisional   Chiefs (0.6527196431565626) beat Bills        → hit 1, (1−c)² = 0.1206036462
#   NFC Divisional   Buccaneers (0.6814396404565678) lost to Rams  → hit 0,     c² = 0.4643599836
FIVE_GAMES_CONF = [0.6849395710053819, 0.6360927506267418, 0.810417382104968,
                   0.6527196431565626, 0.6814396404565678]
FIVE_GAMES_HIT = [1, 1, 0, 1, 0]
# Brier = (0.0992630739 + 0.1324284861 + 0.6567763332 + 0.1206036462 + 0.4643599836) / 5
FIVE_GAMES_BRIER = 0.294686304623487
# Bins: [0.6,0.7) holds four games, 3 won, confidences sum 2.6551916052; [0.8,0.9) holds one, lost.
# ECE = (|3 − 2.6551916052| + |0 − 0.8104173821|) / 5 = (0.3448083948 + 0.8104173821) / 5
FIVE_GAMES_ECE = 0.23104515537194278


def test_fixture_rows_match_the_committed_538_file(nfl_538):
    conf, hit = favourite_form(nfl_538.head(5))
    assert conf.tolist() == FIVE_GAMES_CONF
    assert hit.tolist() == FIVE_GAMES_HIT


def test_brier_matches_hand_computation():
    got = brier_score(FIVE_GAMES_CONF, FIVE_GAMES_HIT)
    assert got == pytest.approx(FIVE_GAMES_BRIER, rel=1e-12)


def test_ece_matches_hand_computation():
    ece = expected_calibration_error(FIVE_GAMES_CONF, FIVE_GAMES_HIT)
    assert ece == pytest.approx(FIVE_GAMES_ECE, rel=1e-12)


def test_reliability_table_matches_hand_computation():
    t = reliability_table(FIVE_GAMES_CONF, FIVE_GAMES_HIT)
    assert t[["bin_lo", "bin_hi", "n"]].values.tolist() == [[0.6, 0.7, 4], [0.8, 0.9, 1]]
    assert t["accuracy"].tolist() == [0.75, 0.0]
    expected_means = [2.6551916052452544 / 4, 0.810417382104968]
    assert t["mean_confidence"].tolist() == pytest.approx(expected_means)
    assert t["gap"].iloc[1] == pytest.approx(-0.810417382104968)  # over-confident bin


# ── Independent implementation: scikit-learn on every real NBA game ─────────────

TENTHS = np.linspace(0, 1, 11)


def test_no_nba_probability_sits_on_a_bin_edge(nba_538):
    # Precondition for the sklearn comparison: sklearn's bins are right-closed,
    # ours left-closed; they agree whenever no value equals an interior edge.
    probs = np.concatenate([nba_538["prob1"], nba_538["prob2"]])
    assert not np.isin(probs, TENTHS[1:-1]).any()


@pytest.mark.parametrize("form", ["favourite", "two_sided"])
def test_reliability_table_equals_sklearn_calibration_curve(nba_538, form):
    if form == "favourite":
        conf, hit = favourite_form(nba_538)
    else:
        conf, hit = nba_538["prob1"].to_numpy(), nba_538["prob1_outcome"].to_numpy()
    prob_true, prob_pred = calibration_curve(hit, conf, n_bins=10, strategy="uniform")
    ours = reliability_table(conf, hit, edges=TENTHS)

    np.testing.assert_allclose(ours["accuracy"], prob_true, rtol=1e-12)
    np.testing.assert_allclose(ours["mean_confidence"], prob_pred, rtol=1e-12)


def test_ece_equals_ece_built_from_sklearn_and_numpy_histogram(nba_538):
    conf, hit = favourite_form(nba_538)
    prob_true, prob_pred = calibration_curve(hit, conf, n_bins=10, strategy="uniform")
    counts, _ = np.histogram(conf, bins=TENTHS)  # numpy: [a, b) bins, last closed
    counts = counts[counts > 0]
    reference = np.sum(counts / counts.sum() * np.abs(prob_true - prob_pred))
    ours = expected_calibration_error(conf, hit, edges=TENTHS)
    assert ours == pytest.approx(reference, rel=1e-12)


def test_brier_equals_sklearn(nba_538):
    conf, hit = favourite_form(nba_538)
    assert brier_score(conf, hit) == pytest.approx(brier_score_loss(hit, conf), rel=1e-12)


def test_brier_identity_favourite_vs_two_sided_form(nba_538):
    """(c − hit)² scored on the favourite = (P(team1) − team1_won)² on real games."""
    conf, hit = favourite_form(nba_538)
    two_sided = brier_score(nba_538["prob1"], nba_538["prob1_outcome"])
    assert brier_score(conf, hit) == pytest.approx(two_sided, rel=1e-12)


def test_fast_ece_equals_table_definition_on_real_data(nba_538):
    conf, hit = favourite_form(nba_538)
    t = reliability_table(conf, hit)
    definition = float((t["n"] / t["n"].sum() * t["gap"].abs()).sum())
    assert expected_calibration_error(conf, hit) == pytest.approx(definition, rel=1e-12)


def test_ece_batch_row_by_row_matches_single_computation(nfl_538):
    conf, hit = favourite_form(nfl_538)
    rows = [slice(0, 300), slice(300, 600), slice(600, 900)]
    c = np.stack([conf[s] for s in rows])
    h = np.stack([hit[s] for s in rows])
    batch = ece_batch(np.stack([assign_bins(r) for r in c]), c, h, 6)
    single = [expected_calibration_error(ci, hi) for ci, hi in zip(c, h, strict=True)]
    np.testing.assert_allclose(batch, single, rtol=1e-12)


# ── Bin edges and input validation ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("c", "expected_lo"),
    [(0.0, 0.0), (0.4999, 0.0), (0.5, 0.5), (0.6, 0.6), (0.6999, 0.6), (0.9, 0.9), (1.0, 0.9)],
)
def test_bin_edges_are_left_closed_and_last_bin_includes_one(c, expected_lo):
    t = reliability_table([c], [1])
    assert t["bin_lo"].iloc[0] == expected_lo


def test_empty_bins_are_omitted_and_weights_sum_to_one(nba_538):
    conf, hit = favourite_form(nba_538)
    t = reliability_table(conf, hit)
    assert (t["n"] > 0).all()
    assert t["n"].sum() == len(conf)
    assert t["bin_lo"].min() == 0.5  # favourites never state < 0.5


@pytest.mark.parametrize(
    ("conf", "hit"),
    [([], []), ([1.2], [1]), ([-0.1], [0]), ([np.nan], [1]), ([0.7], [0.5]), ([0.7, 0.8], [1])],
)
def test_invalid_inputs_raise(conf, hit):
    with pytest.raises(ValueError):
        expected_calibration_error(conf, hit)
    with pytest.raises(ValueError):
        brier_score(conf, hit)


def test_tied_nfl_games_are_rejected_as_non_binary():
    raw = pd.read_csv("tests/fixtures/538/nfl_games.csv")
    ties = raw[raw["prob1_outcome"] == 0.5]
    assert len(ties) > 0
    with pytest.raises(ValueError, match="binary"):
        brier_score(ties["prob1"], ties["prob1_outcome"])


def test_brier_reference_points():
    assert brier_score([0.5, 0.5], [0, 1]) == COIN_FLIP_BRIER
    assert brier_score([1.0, 1.0], [1, 1]) == 0.0
    assert brier_skill_score(COIN_FLIP_BRIER) == 0.0
    assert brier_skill_score(0.2) == pytest.approx(0.2)
