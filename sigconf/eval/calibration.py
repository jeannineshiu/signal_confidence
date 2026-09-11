"""Reliability table and Expected Calibration Error (ECE).

Inputs throughout: `confidence`, the forecaster's stated probability that its
directional call is right, and `hit`, 1 if the call was right and 0 if not.
Only directional predictions reach this module; neutral calls and flat labels
are removed upstream by sigconf/eval/scoring.py and reported as coverage.

Bins are left-closed [lo, hi) with the last bin closed on both ends, so a
confidence of exactly 0.6 lands in [0.6, 0.7) and 1.0 lands in [0.9, 1.0].
"""

import numpy as np
import pandas as pd

from sigconf.config import BIN_EDGES
from sigconf.eval.baselines import wilson_interval


def check_inputs(confidence, hit) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(confidence, dtype=float)
    h = np.asarray(hit, dtype=float)
    if c.ndim != 1 or c.shape != h.shape:
        raise ValueError("confidence and hit must be 1-D arrays of equal length")
    if c.size == 0:
        raise ValueError("no predictions to evaluate")
    if not np.all(np.isfinite(c)) or np.any((c < 0) | (c > 1)):
        raise ValueError("confidence must lie in [0, 1]")
    if not np.all((h == 0) | (h == 1)):
        raise ValueError("hit must be 0 or 1 (a tie is not a binary outcome)")
    return c, h


def assign_bins(confidence, edges=BIN_EDGES) -> np.ndarray:
    """Index of the bin each confidence falls in."""
    e = np.asarray(edges, dtype=float)
    if e.ndim != 1 or e.size < 2 or not np.all(np.diff(e) > 0):
        raise ValueError("edges must be strictly increasing")
    c = np.asarray(confidence, dtype=float)
    idx = np.searchsorted(e, c, side="right") - 1
    idx = np.where(c == e[-1], e.size - 2, idx)
    if np.any((idx < 0) | (idx > e.size - 2)):
        raise ValueError("confidence outside the bin edges")
    return idx


def reliability_table(confidence, hit, edges=BIN_EDGES) -> pd.DataFrame:
    """One row per non-empty bin: n, mean confidence, empirical accuracy,
    Wilson 95% CI on that accuracy, and the gap accuracy − mean confidence
    (negative = over-confident)."""
    c, h = check_inputs(confidence, hit)
    e = np.asarray(edges, dtype=float)
    bins = assign_bins(c, e)
    rows = []
    for i in np.unique(bins):
        in_bin = bins == i
        n, k = int(in_bin.sum()), int(h[in_bin].sum())
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "bin_lo": float(e[i]),
                "bin_hi": float(e[i + 1]),
                "n": n,
                "mean_confidence": float(c[in_bin].mean()),
                "accuracy": k / n,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = table["accuracy"] - table["mean_confidence"]
    return table


def expected_calibration_error(confidence, hit, edges=BIN_EDGES) -> float:
    """ECE = Σ_b (n_b / N) · |accuracy_b − mean_confidence_b| over non-empty bins."""
    c, h = check_inputs(confidence, hit)
    bins = assign_bins(c, edges)
    return float(ece_batch(bins[None, :], c[None, :], h[None, :], len(edges) - 1)[0])


def ece_batch(bins: np.ndarray, confidence: np.ndarray, hit: np.ndarray, n_bins: int) -> np.ndarray:
    """ECE for R forecast sets at once; each argument has shape (R, N).

    Uses (n_b / N)·|k_b/n_b − Σc_b/n_b| = |Σ_b (hit − c)| / N, so each bin
    needs only the sum of (hit − confidence) over its members. Empty bins
    contribute zero. This is what makes bootstrap and null resampling cheap.
    """
    r, n = bins.shape
    flat = (bins + np.arange(r)[:, None] * n_bins).ravel()
    per_bin = np.bincount(flat, weights=(hit - confidence).ravel(), minlength=r * n_bins)
    return np.abs(per_bin.reshape(r, n_bins)).sum(axis=1) / n
