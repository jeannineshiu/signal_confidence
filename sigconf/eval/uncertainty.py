"""Sampling uncertainty for Brier and ECE.

Two different tools, deliberately kept apart:

* **Bootstrap** resamples the *observed* scored rows with replacement. No
  value is invented; it answers "how much would this number move with a
  different draw of headlines?"

* **ECE null distribution** is a simulation-based significance test, not data.
  It keeps the model's own confidences and draws outcomes hit ~ Bernoulli(c),
  i.e. it asks what ECE a *perfectly calibrated* forecaster with exactly these
  confidences would show at this sample size. ECE is biased upward in small
  samples, so an observed ECE is only evidence of miscalibration if it sits
  in the upper tail of this distribution.
"""

import numpy as np

from sigconf.config import BIN_EDGES
from sigconf.eval.calibration import assign_bins, check_inputs, ece_batch


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_brier(confidence, hit, idx: np.ndarray) -> np.ndarray:
    c, h = check_inputs(confidence, hit)
    return np.mean((c[idx] - h[idx]) ** 2, axis=1)


def bootstrap_ece(confidence, hit, idx: np.ndarray, edges=BIN_EDGES) -> np.ndarray:
    c, h = check_inputs(confidence, hit)
    bins = assign_bins(c, edges)
    return ece_batch(bins[idx], c[idx], h[idx], len(edges) - 1)


def percentile_interval(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def ece_null_distribution(confidence, n_sims: int, seed: int, edges=BIN_EDGES) -> np.ndarray:
    """ECE of a perfectly calibrated forecaster with these exact confidences."""
    c = np.asarray(confidence, dtype=float)
    check_inputs(c, np.zeros_like(c))
    rng = np.random.default_rng(seed)
    hits = (rng.random((n_sims, c.size)) < c).astype(float)
    bins = np.broadcast_to(assign_bins(c, edges), hits.shape)
    return ece_batch(bins, np.broadcast_to(c, hits.shape), hits, len(edges) - 1)


def upper_tail_p_value(observed: float, null: np.ndarray) -> float:
    """P(null ≥ observed) with the +1 correction, so it is never exactly 0."""
    return float((1 + np.sum(null >= observed)) / (1 + null.size))
