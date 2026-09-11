"""Brier score over directional predictions.

Scored as (confidence − hit)². For a binary outcome this is identical to the
usual (P(up) − y)² with P(up) = confidence for a bullish call and
1 − confidence for a bearish one, because both terms measure the same
distance from the truth; tests check the identity on real forecasts.
"""

import numpy as np

from sigconf.eval.calibration import check_inputs

# A forecaster that always says 0.5 scores exactly (0.5 − hit)² = 0.25.
COIN_FLIP_BRIER = 0.25


def brier_score(confidence, hit) -> float:
    c, h = check_inputs(confidence, hit)
    return float(np.mean((c - h) ** 2))


def brier_skill_score(brier: float, reference: float = COIN_FLIP_BRIER) -> float:
    """1 − B / B_ref: positive beats the reference, 0 ties it, negative is worse."""
    return 1.0 - brier / reference
