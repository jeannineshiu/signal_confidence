import math

import numpy as np
import pytest
from conftest import favourite_form

from sigconf.eval.baselines import (
    accuracy_summary,
    binomial_central_interval,
    binomial_two_sided_p,
    mcnemar_exact,
    wilson_interval,
)
from sigconf.eval.brier import brier_score
from sigconf.eval.calibration import expected_calibration_error
from sigconf.eval.uncertainty import (
    bootstrap_brier,
    bootstrap_ece,
    bootstrap_indices,
    ece_null_distribution,
    percentile_interval,
    upper_tail_p_value,
)

# ── Closed-form statistics against hand-computed values ───────────────────────


def test_wilson_interval_known_values():
    # p = 0.5, n = 100: centre 0.5, half-width 1.96/(1+0.0384) · sqrt(0.0025 + 0.000096)
    assert wilson_interval(50, 100) == pytest.approx((0.40383153, 0.59616847), abs=1e-8)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.27753279, abs=1e-8)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_exact_binomial_two_sided_p():
    # P(X ≥ 8) for Bin(10, ½) = (45 + 10 + 1) / 1024; doubled by symmetry.
    # Integer arithmetic: these are exact, not approximate.
    assert binomial_two_sided_p(8, 10) == 112 / 1024
    assert binomial_two_sided_p(10, 10) == 2 / 1024
    assert binomial_two_sided_p(5, 10) == 1.0


@pytest.mark.parametrize(("k", "n"), [(16, 31), (8, 31), (10, 12), (14, 43), (1212, 1882),
                                      (941, 1882)])
def test_binomial_p_matches_scipy_including_real_sample_sizes(k, n):
    # Independent reference (scipy ships with scikit-learn, test-only).
    # (1212, 1882) is the real NFL favourites count that overflowed a float C(n, k).
    from scipy.stats import binomtest

    assert binomial_two_sided_p(k, n) == pytest.approx(binomtest(k, n, 0.5).pvalue, rel=1e-9)


def test_coin_flip_range_is_the_central_95_percent_of_binomial():
    # Bin(10, ½): P(X ≤ 1) = 11/1024 < 2.5% ≤ P(X ≤ 2); P(X ≤ 7) < 97.5% ≤ P(X ≤ 8).
    assert binomial_central_interval(10) == (0.2, 0.8)
    lo, hi = binomial_central_interval(100)
    assert lo < 0.5 < hi
    assert sum(math.comb(100, k) for k in range(int(lo * 100), int(hi * 100) + 1)) / 2**100 >= 0.95


def test_mcnemar_uses_only_discordant_items():
    a = [1] * 7 + [1, 0]
    b = [0] * 7 + [1, 0]
    got = mcnemar_exact(a, b)
    assert (got["a_only"], got["b_only"]) == (7, 0)
    assert got["p_value"] == pytest.approx(2 / 128)
    assert mcnemar_exact([1, 0], [1, 0])["p_value"] == 1.0


def test_accuracy_summary_on_real_nfl_favourites(nfl_538):
    _, hit = favourite_form(nfl_538)
    s = accuracy_summary(hit)
    assert s["k"] == int(hit.sum()) and s["n"] == len(hit)
    assert s["ci95"][0] < s["value"] < s["ci95"][1]
    assert s["p_vs_coin_flip"] < 1e-6  # 538 favourites win far more than half the time


# ── Bootstrap: resamples the real rows ────────────────────────────────────────


def test_bootstrap_is_deterministic_and_brackets_the_estimate(nfl_538):
    conf, hit = favourite_form(nfl_538)
    idx = bootstrap_indices(len(conf), 2_000, seed=42)
    assert np.array_equal(idx, bootstrap_indices(len(conf), 2_000, seed=42))

    lo, hi = percentile_interval(bootstrap_brier(conf, hit, idx))
    assert lo < brier_score(conf, hit) < hi
    lo, hi = percentile_interval(bootstrap_ece(conf, hit, idx))
    assert lo < expected_calibration_error(conf, hit) < hi


def test_bootstrap_brier_row_equals_direct_brier_on_that_resample(nfl_538):
    conf, hit = favourite_form(nfl_538)
    idx = bootstrap_indices(len(conf), 3, seed=1)
    for row, value in zip(idx, bootstrap_brier(conf, hit, idx), strict=True):
        assert value == pytest.approx(brier_score(conf[row], hit[row]), rel=1e-12)


# ── ECE null distribution: a simulation, checked against exact math ──────────


def test_null_simulation_matches_exact_expectation_for_one_bin():
    # All confidences 0.7, n = 20 → one bin; ECE = |k/20 − 0.7| with k ~ Bin(20, 0.7).
    # Exact E[ECE] = Σ_k C(20,k) 0.7^k 0.3^(20−k) · |k/20 − 0.7|.
    exact = sum(math.comb(20, k) * 0.7**k * 0.3 ** (20 - k) * abs(k / 20 - 0.7) for k in range(21))
    null = ece_null_distribution(np.full(20, 0.7), n_sims=200_000, seed=0)
    assert null.mean() == pytest.approx(exact, abs=1e-3)


def test_null_is_deterministic_per_seed_and_shrinks_with_n(nba_538):
    conf, _ = favourite_form(nba_538)
    a = ece_null_distribution(conf[:100], n_sims=2_000, seed=3)
    assert np.array_equal(a, ece_null_distribution(conf[:100], n_sims=2_000, seed=3))
    big = ece_null_distribution(conf[:2000], n_sims=2_000, seed=3)
    assert big.mean() < a.mean()  # small-sample ECE is biased upward


def test_upper_tail_p_value():
    null = np.array([0.1, 0.2, 0.3, 0.4])
    assert upper_tail_p_value(0.0, null) == 1.0
    assert upper_tail_p_value(0.35, null) == pytest.approx(2 / 5)
    assert upper_tail_p_value(9.9, null) == pytest.approx(1 / 5)
