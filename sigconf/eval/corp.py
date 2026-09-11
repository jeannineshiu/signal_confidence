"""CORP reliability diagram and Brier decomposition (Dimitriadis, Gneiting &
Jordan, PNAS 2021).

CORP replaces hand-set bins with the pool-adjacent-violators (PAV) fit: the
non-decreasing function of confidence that best predicts the outcome. Its
groups come from the data, so there is no binning choice to make. The same fit
splits the mean Brier score exactly into three parts:

    Brier = MCB − DSC + UNC

* MCB (miscalibration) = Brier(original) − Brier(recalibrated): what is lost
  because stated confidence differs from the PAV-recalibrated confidence.
* DSC (discrimination) = Brier(constant ȳ) − Brier(recalibrated): what the
  confidence's *ordering* is worth, i.e. how well it separates hits from misses.
* UNC (uncertainty) = ȳ(1 − ȳ): the Brier of always stating the hit rate.

MCB ≥ 0 and DSC ≥ 0 by construction (both the original confidence and a
constant are non-decreasing, and PAV is the best such fit). Both are also
biased upward at small n, because PAV fits noise. Each is therefore judged
against what it would be at this n if its effect were absent:

* MCB against a perfectly calibrated forecaster with the same confidences
  (hit ~ Bernoulli(confidence)), the same simulation as the ECE null;
* DSC against confidence unrelated to correctness: the observed hits
  permuted across the observed confidences (a resampling of real rows).

No bootstrap interval is reported for MCB or DSC. A percentile interval for a
quantity that cannot go below zero and is biased upward excludes zero even
when there is no effect, so it would misread as evidence.

Inputs are `confidence` and `hit`, as in sigconf/eval/calibration.py.
"""

import numpy as np
import pandas as pd

from sigconf.config import N_BOOTSTRAP, SEED
from sigconf.eval import brier, uncertainty
from sigconf.eval.calibration import check_inputs


def pav(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-decreasing fit to `values` (already in x order)."""
    means, totals, sizes = [], [], []
    for v, w in zip(values, weights, strict=True):
        means.append(float(v))
        totals.append(float(w))
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:  # pool adjacent violators
            w_new = totals[-2] + totals[-1]
            means[-2] = (means[-2] * totals[-2] + means[-1] * totals[-1]) / w_new
            totals[-2] = w_new
            sizes[-2] += sizes[-1]
            del means[-1], totals[-1], sizes[-1]
    return np.repeat(means, sizes)


def _levels(confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct confidence values (ascending) and each item's index into them.
    Tied confidences are one level: PAV must give them one recalibrated value."""
    return np.unique(confidence, return_inverse=True)


def _components(levels, counts, hits_per_level, n) -> tuple[float, float, float, np.ndarray]:
    """(MCB, DSC, UNC, recalibrated value per level) from per-level tallies."""
    fit = pav(hits_per_level / counts, counts)
    misses = counts - hits_per_level
    score = (hits_per_level * (1 - levels) ** 2 + misses * levels**2).sum() / n
    recal = (hits_per_level * (1 - fit) ** 2 + misses * fit**2).sum() / n
    rate = hits_per_level.sum() / n
    unc = rate * (1 - rate)
    return score - recal, unc - recal, unc, fit


def recalibrate(confidence, hit) -> np.ndarray:
    """PAV-recalibrated confidence for each item (the CORP reliability curve)."""
    c, h = check_inputs(confidence, hit)
    levels, inverse = _levels(c)
    counts = np.bincount(inverse).astype(float)
    return pav(np.bincount(inverse, weights=h) / counts, counts)[inverse]


def decompose(confidence, hit) -> dict:
    """Brier = MCB − DSC + UNC (the identity holds to float precision)."""
    c, h = check_inputs(confidence, hit)
    levels, inverse = _levels(c)
    counts = np.bincount(inverse).astype(float)
    mcb, dsc, unc, _ = _components(levels, counts, np.bincount(inverse, weights=h), c.size)
    return {"brier": brier.brier_score(c, h), "mcb": float(mcb), "dsc": float(dsc),
            "unc": float(unc)}


def _per_row_hits(inverse: np.ndarray, hits: np.ndarray, n_levels: int) -> np.ndarray:
    """Hit tallies per level for each row of an (R, N) hit matrix → (R, n_levels)."""
    r = hits.shape[0]
    flat = (inverse[None, :] + np.arange(r)[:, None] * n_levels).ravel()
    return np.bincount(flat, weights=hits.ravel(), minlength=r * n_levels).reshape(r, n_levels)


def corp_analysis(
    confidence, hit, n_sims: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = 0.05
) -> dict:
    """Decomposition, its two small-sample references, and the reliability curve
    with a consistency band (where a perfectly calibrated forecaster's
    recalibrated curve falls, pointwise, in 1 − alpha of simulations)."""
    c, h = check_inputs(confidence, hit)
    levels, inverse = _levels(c)
    counts = np.bincount(inverse).astype(float)
    k = levels.size
    mcb, dsc, unc, fit = _components(levels, counts, np.bincount(inverse, weights=h), c.size)

    rng = np.random.default_rng(seed + 1)
    calibrated = (rng.random((n_sims, c.size)) < c).astype(float)
    sim_hits = _per_row_hits(inverse, calibrated, k)
    sims = [_components(levels, counts, row, c.size) for row in sim_hits]
    mcb_null = np.array([s[0] for s in sims])
    band = np.quantile(np.stack([s[3] for s in sims]), [alpha / 2, 1 - alpha / 2], axis=0)

    rng = np.random.default_rng(seed + 2)
    shuffled = np.stack([rng.permutation(h) for _ in range(n_sims)])
    dsc_null = np.array([_components(levels, counts, row, c.size)[1]
                         for row in _per_row_hits(inverse, shuffled, k)])

    hits_per_level = np.bincount(inverse, weights=h)
    curve = pd.DataFrame({
        "confidence": levels,
        "n": counts.astype(int),
        "hit_rate": hits_per_level / counts,
        "recalibrated": fit,
        "band_low": band[0],
        "band_high": band[1],
    })

    def reference(observed: float, null: np.ndarray, method: str) -> dict:
        return {"method": method, "n_sims": n_sims, "mean": float(null.mean()),
                "p95": float(np.quantile(null, 0.95)),
                "p_value": uncertainty.upper_tail_p_value(observed, null)}

    return {
        "n": int(c.size),
        "brier": brier.brier_score(c, h),
        "mcb": {"value": float(mcb), "perfect_calibration_null": reference(
            mcb, mcb_null, "simulation: hit ~ Bernoulli(confidence), same confidences")},
        "dsc": {"value": float(dsc), "no_discrimination_null": reference(
            dsc, dsc_null, "permutation: observed hits shuffled across observed confidences")},
        "unc": float(unc),
        "n_pav_groups": int(np.unique(fit).size),
        "curve": curve.to_dict(orient="records"),
        "settings": {"n_sims": n_sims, "seed": seed, "band": 1 - alpha},
    }
