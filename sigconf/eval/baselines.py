"""Accuracy, the baselines it is judged against, and the exact tests behind them.

All inference here is closed-form (Wilson interval, exact binomial), written
out rather than imported so the whole method is inspectable and tested
against hand-computed values. No simulation happens in this module.
"""

import math

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


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Computed in log space: C(n, k) overflows a float beyond n ≈ 1,000."""
    if not 0 < p < 1:
        raise ValueError("p must lie strictly between 0 and 1")
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(log_comb + k * math.log(p) + (n - k) * math.log1p(-p))


def binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test: total probability of outcomes no more
    likely than the observed one (the "minlike" convention)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"invalid k={k}, n={n}")
    observed = binomial_pmf(k, n, p)
    pmfs = [binomial_pmf(i, n, p) for i in range(n + 1)]
    return min(1.0, sum(q for q in pmfs if q <= observed * (1 + 1e-7)))


def binomial_central_interval(n: int, p: float = 0.5, alpha: float = 0.05) -> tuple[float, float]:
    """Central (1 - alpha) range of k/n for k ~ Binomial(n, p).

    For p = 0.5 this is the range a coin-flipping forecaster's accuracy falls
    in at this sample size — the honest "random guess" band.
    """
    cdf = np.cumsum([binomial_pmf(i, n, p) for i in range(n + 1)])
    lo = int(np.searchsorted(cdf, alpha / 2 - 1e-12))
    hi = int(np.searchsorted(cdf, 1 - alpha / 2 - 1e-12))
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
        "p_vs_coin_flip": binomial_two_sided_p(k, n, 0.5),
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
    p = 1.0 if n == 0 else binomial_two_sided_p(a_only, n, 0.5)
    return {"a_only": a_only, "b_only": b_only, "p_value": p}
