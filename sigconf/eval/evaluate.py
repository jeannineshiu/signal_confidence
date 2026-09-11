"""Every headline metric, each next to its baseline, as one JSON-ready dict.

The rule this module enforces is structural: no number is produced without
its sample size, its uncertainty, and the baseline it has to beat.
"""

import numpy as np
import pandas as pd

from sigconf.config import BIN_EDGES, N_BOOTSTRAP, SEED
from sigconf.data.labels import FLAT, UP
from sigconf.eval import baselines, brier, calibration, scoring, uncertainty


def evaluate(
    scored: pd.DataFrame,
    edges=BIN_EDGES,
    n_boot: int = N_BOOTSTRAP,
    n_sims: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Metrics for one split. `scored` is the output of scoring.score()."""
    counts = scoring.funnel(scored)
    rows = scored[scored["state"] == "scored"]
    if rows.empty:
        raise ValueError("no scored predictions: nothing to evaluate")

    conf = rows["confidence"].to_numpy(dtype=float)
    hit = rows["hit"].to_numpy(dtype=float)
    went_up = (rows["label"] == UP).to_numpy()
    n = len(rows)

    llm = baselines.accuracy_summary(hit)
    always_up_same = baselines.accuracy_summary(went_up.astype(int))
    directional_labels = scored[scored["label"] != FLAT]
    always_up_all = baselines.accuracy_summary((directional_labels["label"] == UP).astype(int))

    b = brier.brier_score(conf, hit)
    ece = calibration.expected_calibration_error(conf, hit, edges)
    idx = uncertainty.bootstrap_indices(n, n_boot, seed)
    brier_ci = uncertainty.percentile_interval(uncertainty.bootstrap_brier(conf, hit, idx))
    ece_ci = uncertainty.percentile_interval(uncertainty.bootstrap_ece(conf, hit, idx, edges))
    null = uncertainty.ece_null_distribution(conf, n_sims, seed + 1, edges)
    table = calibration.reliability_table(conf, hit, edges)

    values, value_counts = np.unique(conf, return_counts=True)
    distinct = {f"{v:g}": int(k) for v, k in zip(values, value_counts, strict=True)}
    return {
        "n_headlines": counts["total"],
        "funnel": counts,
        "coverage": n / counts["total"],
        "accuracy": llm,
        "baselines": {
            "coin_flip": {
                "value": 0.5,
                "range95_at_n": list(baselines.binomial_central_interval(n)),
            },
            "always_up_same_items": always_up_same,
            "always_up_all_directional_labels": always_up_all,
            "llm_vs_always_up_mcnemar": baselines.mcnemar_exact(hit, went_up.astype(int)),
        },
        "brier": {
            "value": b,
            "ci95": list(brier_ci),
            "coin_flip_reference": brier.COIN_FLIP_BRIER,
            "skill_vs_coin_flip": brier.brier_skill_score(b),
        },
        "ece": {
            "value": ece,
            "ci95": list(ece_ci),
            "perfect_calibration_null": {
                "method": "simulation: hit ~ Bernoulli(confidence), same confidences",
                "n_sims": n_sims,
                "mean": float(null.mean()),
                "p95": float(np.quantile(null, 0.95)),
                "p_value": uncertainty.upper_tail_p_value(ece, null),
            },
        },
        "confidence": {
            "mean": float(conf.mean()),
            "mean_minus_accuracy": float(conf.mean() - llm["value"]),
            "distinct_values": distinct,
            "below_half": int((conf < 0.5).sum()),
        },
        "reliability_table": table.to_dict(orient="records"),
        "settings": {"bin_edges": list(edges), "n_boot": n_boot, "seed": seed},
    }
