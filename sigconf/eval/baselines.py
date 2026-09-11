"""Accuracy, the baselines it is judged against, and the exact tests behind them.

All inference here is closed-form (Wilson interval, exact binomial), written
out rather than imported so the whole method is inspectable and tested
against hand-computed values. No simulation happens in this module.
"""

import math
from fractions import Fraction

import numpy as np

Z95 = 1.959963984540054  # standard-normal 97.5th percentile


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, n={n}]")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, centre - half), min(1.0, centre + half)


# Every binomial question here is about a fair coin (p = ½): accuracy vs
# guessing, and McNemar's discordant pairs. Then P(X = k) = C(n, k) / 2ⁿ, which
# Python evaluates *exactly* with integers — no overflow at any n, no libm
# rounding, so p-values are bit-identical on every platform (a log-space
# version differed between macOS and Linux in the last digit).


def _fair_coin_counts(n: int) -> list[int]:
    if n <= 0:
        raise ValueError("n must be positive")
    return [math.comb(n, i) for i in range(n + 1)]


def binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided test of k successes in n fair-coin trials: total
    probability of outcomes no more likely than the observed one."""
    counts = _fair_coin_counts(n)
    if not 0 <= k <= n:
        raise ValueError(f"invalid k={k}, n={n}")
    observed = counts[k]
    return min(1.0, sum(c for c in counts if c <= observed) / 2**n)


def binomial_central_interval(n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Central (1 - alpha) range of k/n for k ~ Binomial(n, ½): where a
    coin-flipping forecaster's accuracy falls at this sample size."""
    cumulative = np.cumsum(np.array(_fair_coin_counts(n), dtype=object))
    total = 2**n
    tail = Fraction(str(alpha)) / 2
    lo = next(i for i, c in enumerate(cumulative) if c >= tail * total)
    hi = next(i for i, c in enumerate(cumulative) if c >= (1 - tail) * total)
    return lo / n, hi / n


def accuracy_summary(hits) -> dict:
    """k, n, accuracy, Wilson 95% CI and exact p-value against 0.5."""
    h = np.asarray(hits)
    if h.size == 0:
        raise ValueError("no predictions")
    if not np.all((h == 0) | (h == 1)):
        raise ValueError("hits must be 0 or 1")
    k, n = int(h.sum()), int(h.size)
    lo, hi = wilson_interval(k, n)
    return {
        "k": k,
        "n": n,
        "value": k / n,
        "ci95": [lo, hi],
        "p_vs_coin_flip": binomial_two_sided_p(k, n),
    }


def mcnemar_exact(hits_a, hits_b) -> dict:
    """Exact McNemar test for two forecasters scored on the same items.

    Only discordant items carry information: `a_only` where A is right and B
    wrong, `b_only` the reverse. Under "equally accurate", a_only ~ Bin(a_only
    + b_only, 0.5).
    """
    a = np.asarray(hits_a).astype(bool)
    b = np.asarray(hits_b).astype(bool)
    if a.shape != b.shape:
        raise ValueError("forecasters must be scored on the same items")
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else binomial_two_sided_p(a_only, n)
    return {"a_only": a_only, "b_only": b_only, "p_value": p}
