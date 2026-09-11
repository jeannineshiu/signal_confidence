"""Two confidence readouts for the same calls, compared on the same items.

Both confidences belong to the same directional call, so accuracy is
identical by construction and every difference is a difference in the
*confidence*. Two questions, kept apart:

* calibration — does the stated probability match the hit rate? (Brier, ECE)
* discrimination — does higher confidence pick out the calls that were right?
  (AUROC: the chance a random right call carries more confidence than a
  random wrong one; 0.5 = no better than chance, ties count one half)

Differences use a *paired* bootstrap: each resample draws the same items for
both readouts, so item-to-item noise cancels instead of widening the interval.
"""

import numpy as np

from sigconf.config import BIN_EDGES, N_BOOTSTRAP, SEED
from sigconf.eval import brier, calibration, uncertainty


def auroc(confidence, hit) -> float | None:
    """Mann–Whitney AUROC with average ranks for ties; None if one class is empty."""
    c, h = calibration.check_inputs(confidence, hit)
    pos = h == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    _, inverse, counts = np.unique(c, return_inverse=True, return_counts=True)
    first_rank = np.cumsum(counts) - counts + 1
    ranks = (first_rank + (counts - 1) / 2)[inverse]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _bootstrap_auroc(c: np.ndarray, h: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = [auroc(c[row], h[row]) for row in idx]
    return np.array([np.nan if v is None else v for v in out])


def _ci(samples: np.ndarray) -> list[float]:
    samples = samples[~np.isnan(samples)]
    return list(uncertainty.percentile_interval(samples))


def _summary(c, h, boot: dict, null_seed, edges, n_sims) -> dict:
    ece = calibration.expected_calibration_error(c, h, edges)
    null = uncertainty.ece_null_distribution(c, n_sims, null_seed, edges)
    values, counts = np.unique(c, return_counts=True)
    return {
        "mean_confidence": float(c.mean()),
        "mean_minus_accuracy": float(c.mean() - h.mean()),
        "brier": {"value": brier.brier_score(c, h), "ci95": _ci(boot["brier"])},
        "ece": {"value": ece,
                "ci95": _ci(boot["ece"]),
                "perfect_calibration_null": {
                    "method": "simulation: hit ~ Bernoulli(confidence), same confidences",
                    "mean": float(null.mean()),
                    "p_value": uncertainty.upper_tail_p_value(ece, null)}},
        "auroc": {"value": auroc(c, h), "ci95": _ci(boot["auroc"])},
        "n_distinct_values": int(values.size),
        "share_at_or_above_0.99": float((c >= 0.99).mean()),
        "reliability_table": calibration.reliability_table(c, h, edges).to_dict(orient="records"),
        "most_common_values": {f"{v:.6g}": int(k) for v, k in
                               sorted(zip(values, counts, strict=True), key=lambda x: -x[1])[:5]},
    }


def compare_confidences(
    conf_a, conf_b, hit, names=("a", "b"), edges=BIN_EDGES,
    n_boot: int = N_BOOTSTRAP, n_sims: int = N_BOOTSTRAP, seed: int = SEED,
) -> dict:
    """Calibration and discrimination of two confidence readouts on the same items."""
    a, h = calibration.check_inputs(conf_a, hit)
    b, _ = calibration.check_inputs(conf_b, hit)
    idx = uncertainty.bootstrap_indices(len(h), n_boot, seed)

    boot = {
        name: {"brier": uncertainty.bootstrap_brier(x, h, idx),
               "ece": uncertainty.bootstrap_ece(x, h, idx, edges),
               "auroc": _bootstrap_auroc(x, h, idx)}
        for name, x in ((names[0], a), (names[1], b))
    }

    def paired(stat: str, point: float | None) -> dict:
        return {"value": point, "ci95": _ci(boot[names[1]][stat] - boot[names[0]][stat])}

    sa = _summary(a, h, boot[names[0]], seed + 1, edges, n_sims)
    sb = _summary(b, h, boot[names[1]], seed + 2, edges, n_sims)
    auc_diff = (None if sa["auroc"]["value"] is None or sb["auroc"]["value"] is None
                else sb["auroc"]["value"] - sa["auroc"]["value"])
    return {
        "n": int(len(h)),
        "accuracy": float(h.mean()),
        names[0]: sa,
        names[1]: sb,
        f"{names[1]}_minus_{names[0]}": {
            "brier": paired("brier", sb["brier"]["value"] - sa["brier"]["value"]),
            "ece": paired("ece", sb["ece"]["value"] - sa["ece"]["value"]),
            "auroc": paired("auroc", auc_diff),
            "method": "paired bootstrap: same resampled items for both readouts",
        },
        "settings": {"bin_edges": list(edges), "n_boot": n_boot, "seed": seed},
    }
