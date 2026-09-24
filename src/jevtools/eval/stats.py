"""Small exact statistics in pure Python (no numpy/scipy, spec §9 dependency policy).

- :func:`clopper_pearson_upper`: the one-sided Clopper–Pearson upper confidence bound of a binomial rate, used by
  threshold tuning (§11.3) and critical-tier certification (§11.4);
- :func:`binom_cdf`: ``P(X ≤ k)`` for ``X ~ Bin(n, p)`` computed in log space;
- :func:`quantile`: linear-interpolation quantile (numpy's default), used for the hysteresis width (§11.2 item 5).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def binom_cdf(k: int, n: int, p: float) -> float:
    """``P(X ≤ k)`` for ``X ~ Binomial(n, p)`` (summed in log space; exact enough for n up to ~10⁶)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        for i in range(k + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper–Pearson upper bound on a rate after ``k`` events in ``n`` trials.

    The bound ``u`` solves ``P(X ≤ k | n, u) = 1 − conf`` (bisection to 1e-12). ``n = 0`` gives 1.0 (no evidence);
    ``k = n`` gives 1.0; ``k = 0`` has the closed form ``1 − (1 − conf)^(1/n)`` (≈ 3/n: the rule of three).
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    if n == 0 or k == n:
        return 1.0
    alpha = 1.0 - conf
    if k == 0:
        return 1.0 - float(alpha ** (1.0 / n))
    lo, hi = k / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return hi


def rule_of_three_cases(alpha: float, conf: float = 0.95) -> int:
    """Smallest ``n`` whose zero-error Clopper–Pearson bound is ≤ ``alpha`` (≈ 3/alpha; 2,995 for 0.1%)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil(math.log(1.0 - conf) / math.log1p(-alpha))


def quantile(values: Sequence[float], q: float) -> float:
    """The ``q``-quantile with linear interpolation between order statistics (``values`` non-empty)."""
    if not values:
        raise ValueError("quantile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


__all__ = ["binom_cdf", "clopper_pearson_upper", "quantile", "rule_of_three_cases"]
