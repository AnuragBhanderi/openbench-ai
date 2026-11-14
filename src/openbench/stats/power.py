"""Sample-size and power planning for paired model comparisons (v0.5.0, issue #34).

Binary metrics (exact McNemar test, as used by ``openbench analyze`` for single runs):

* ``p01`` = P(baseline wrong, challenger right), ``p10`` = P(baseline right, challenger wrong).
* The difference in accuracy is ``delta = p01 - p10``; the disagreement (discordance) rate is
  ``psi = p01 + p10``. Power depends on both: more disagreement means more noise.
* Power at ``n`` items is computed **exactly**: the number of discordant items
  ``D ~ Binomial(n, psi)`` and, given ``D = d``, the challenger's wins ``X ~ Binomial(d, p01/psi)``;
  the test rejects when the two-sided exact binomial p-value of ``X`` is ``<= alpha``.
* If ``psi`` is unknown, it is set from the two accuracies **assuming the models' errors are
  independent**. Real models usually agree more than that, so this usually overstates ``psi``
  and the sample size, which is the cautious direction.

Continuous metrics (sign-flip test): a normal approximation using the standard deviation of the
per-item differences, ``n = ((z_{1-alpha/2} + z_{power}) * sd / delta)^2``.

Exact power is not monotone in ``n`` (the test is discrete). The planner reports the smallest
``n`` with power at or above the target and checks that the next items keep it there.
"""

from __future__ import annotations

import math
from functools import lru_cache
from statistics import NormalDist
from typing import Any

EXACT_MAX_N = 3_000
_STD_NORMAL = NormalDist()


def _check_prob(name: str, value: float, *, open_interval: bool = False) -> float:
    v = float(value)
    if not math.isfinite(v) or (v <= 0.0 if open_interval else v < 0.0) or (v >= 1.0 if open_interval else v > 1.0):
        raise ValueError(f"{name} must be in {'(0, 1)' if open_interval else '[0, 1]'}")
    return v


def discordance_under_independence(baseline_accuracy: float, delta: float) -> tuple[float, float]:
    """Return ``(p01, p10)`` when the two models' errors are independent."""
    p_a = _check_prob("baseline_accuracy", baseline_accuracy)
    p_b = p_a + float(delta)
    if not 0.0 <= p_b <= 1.0:
        raise ValueError("baseline_accuracy + difference must be within [0, 1]")
    return (1.0 - p_a) * p_b, p_a * (1.0 - p_b)


def _log_binom_pmf_row(d: int, q: float) -> list[float]:
    """log P(X = x) for X ~ Binomial(d, q), x = 0..d."""
    if q <= 0.0:
        return [0.0 if x == 0 else -math.inf for x in range(d + 1)]
    if q >= 1.0:
        return [0.0 if x == d else -math.inf for x in range(d + 1)]
    lq, l1q = math.log(q), math.log1p(-q)
    lg_d = math.lgamma(d + 1)
    return [lg_d - math.lgamma(x + 1) - math.lgamma(d - x + 1) + x * lq + (d - x) * l1q for x in range(d + 1)]


@lru_cache(maxsize=65_536)
def _mcnemar_critical_low(d: int, alpha: float) -> int:
    """Largest ``c`` such that 2 * P(Bin(d, 1/2) <= c) <= alpha, or -1 if none."""
    if d == 0:
        return -1
    log_half_d = -d * math.log(2.0)
    cum = 0.0
    crit = -1
    lg_d = math.lgamma(d + 1)
    for x in range(d // 2 + 1):
        cum += math.exp(lg_d - math.lgamma(x + 1) - math.lgamma(d - x + 1) + log_half_d)
        if 2.0 * cum <= alpha * (1.0 + 1e-12):
            crit = x
        else:
            break
    return crit


def mcnemar_power(n: int, p01: float, p10: float, alpha: float = 0.05) -> float:
    """Exact power of the two-sided exact McNemar test at ``n`` paired items."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    p01 = _check_prob("p01", p01)
    p10 = _check_prob("p10", p10)
    psi = p01 + p10
    if psi > 1.0 + 1e-12:
        raise ValueError("p01 + p10 must not exceed 1")
    _check_prob("alpha", alpha, open_interval=True)
    if psi <= 0.0:
        return 0.0
    if n > EXACT_MAX_N:
        return _mcnemar_power_normal(n, p01, p10, alpha)
    q = p01 / psi
    log_d = _log_binom_pmf_row(n, min(psi, 1.0))
    # Only discordant counts within ~10 SD of the mean carry weight; skip the rest.
    centre = n * psi
    spread = 10.0 * math.sqrt(max(n * psi * (1.0 - psi), 1.0)) + 10.0
    d_lo, d_hi = max(1, int(centre - spread)), min(n, int(centre + spread) + 1)
    power = 0.0
    for d in range(d_lo, d_hi + 1):
        w = math.exp(log_d[d])
        if w < 1e-16:
            continue
        crit = _mcnemar_critical_low(d, alpha)
        if crit < 0:
            continue
        row = _log_binom_pmf_row(d, q)
        reject = math.fsum(math.exp(row[x]) for x in range(crit + 1))
        reject += math.fsum(math.exp(row[d - x]) for x in range(crit + 1) if d - x > crit)
        power += w * reject
    return min(1.0, power)


def _mcnemar_power_normal(n: int, p01: float, p10: float, alpha: float) -> float:
    """Normal approximation (Connor 1987) for large ``n``."""
    psi = p01 + p10
    delta = p01 - p10
    z_a = _STD_NORMAL.inv_cdf(1 - alpha / 2)
    var_alt = psi - delta * delta
    if var_alt <= 0:
        return 1.0
    num = abs(delta) * math.sqrt(n) - z_a * math.sqrt(psi)
    return _STD_NORMAL.cdf(num / math.sqrt(var_alt))


def plan_binary(
    *,
    delta: float,
    baseline_accuracy: float | None = None,
    p01: float | None = None,
    p10: float | None = None,
    discordance: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    max_n: int = 100_000,
) -> dict[str, Any]:
    """Smallest number of paired items for the exact McNemar test to reach ``power``."""
    _check_prob("alpha", alpha, open_interval=True)
    _check_prob("power", power, open_interval=True)
    if delta == 0:
        raise ValueError("the difference to detect must be non-zero")
    assumption: str
    if p01 is not None and p10 is not None:
        q01, q10 = float(p01), float(p10)
        assumption = "p01 and p10 given (e.g. estimated from pilot runs)"
    elif discordance is not None:
        psi = _check_prob("discordance", discordance)
        if psi < abs(delta):
            raise ValueError("discordance must be at least the absolute difference")
        q01, q10 = (psi + delta) / 2.0, (psi - delta) / 2.0
        assumption = "discordance rate given"
    elif baseline_accuracy is not None:
        q01, q10 = discordance_under_independence(baseline_accuracy, delta)
        assumption = "independent errors (usually overstates disagreement, so the sample size is cautious)"
    else:
        raise ValueError("give baseline_accuracy, discordance, or p01 and p10")
    n_needed = _search_n(lambda n: mcnemar_power(n, q01, q10, alpha), power, max_n)
    return {
        "metric_kind": "binary",
        "test": "mcnemar_exact",
        "difference": q01 - q10,
        "p01": q01,
        "p10": q10,
        "discordance": q01 + q10,
        "assumption": assumption,
        "alpha": alpha,
        "target_power": power,
        "items_needed": n_needed,
        "power_at_items_needed": mcnemar_power(n_needed, q01, q10, alpha) if n_needed else None,
        "method": "exact" if (n_needed or 0) <= EXACT_MAX_N else "normal_approximation",
    }


def _search_n(power_at, target: float, max_n: int) -> int | None:
    """Smallest n with power >= target that stays >= target for the next 2% (5 to 25) items."""
    raw = power_at
    memo: dict[int, float] = {}

    def power_at(m: int) -> float:  # noqa: F811 - memoised wrapper
        if m not in memo:
            memo[m] = raw(m)
        return memo[m]

    if power_at(max_n) < target:
        return None
    lo, hi = 1, max_n
    while lo < hi:
        mid = (lo + hi) // 2
        if power_at(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    n = lo
    # Discrete tests: walk back to the first n that holds the target over a short window.
    while n > 1 and power_at(n - 1) >= target:
        n -= 1
    window = min(25, max(5, n // 50))
    while any(power_at(m) < target for m in range(n + 1, min(n + window, max_n) + 1)):
        n += 1
    return n


def minimum_detectable_difference_binary(
    n: int, *, baseline_accuracy: float, alpha: float = 0.05, power: float = 0.8, direction: int = 1
) -> float | None:
    """Smallest accuracy gain detectable at ``n`` items (independent-errors assumption)."""
    _check_prob("baseline_accuracy", baseline_accuracy)
    top = (1.0 - baseline_accuracy) if direction > 0 else baseline_accuracy
    if top <= 0:
        return None

    def pw(delta: float) -> float:
        q01, q10 = discordance_under_independence(baseline_accuracy, delta * direction)
        return mcnemar_power(n, q01, q10, alpha)

    if pw(top) < power:
        return None
    lo, hi = 0.0, top
    for _ in range(40):
        mid = (lo + hi) / 2
        if pw(mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def plan_continuous(*, delta: float, sd_difference: float, alpha: float = 0.05, power: float = 0.8) -> dict[str, Any]:
    """Normal-approximation sample size for a paired mean difference (sign-flip / t)."""
    _check_prob("alpha", alpha, open_interval=True)
    _check_prob("power", power, open_interval=True)
    if delta == 0 or sd_difference <= 0:
        raise ValueError("difference must be non-zero and sd_difference positive")
    z = _STD_NORMAL.inv_cdf(1 - alpha / 2) + _STD_NORMAL.inv_cdf(power)
    n = math.ceil((z * sd_difference / abs(delta)) ** 2)
    return {
        "metric_kind": "continuous",
        "test": "sign_flip (planned with a normal approximation)",
        "difference": delta,
        "sd_difference": sd_difference,
        "alpha": alpha,
        "target_power": power,
        "items_needed": max(n, 2),
        "method": "normal_approximation",
        "note": "Accurate when items are many and differences are not extremely skewed; add a margin for small n.",
    }


def minimum_detectable_difference_continuous(n: int, *, sd_difference: float, alpha: float = 0.05, power: float = 0.8) -> float:
    z = _STD_NORMAL.inv_cdf(1 - alpha / 2) + _STD_NORMAL.inv_cdf(power)
    return z * sd_difference / math.sqrt(n)
