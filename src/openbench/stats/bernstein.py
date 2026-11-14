"""Equivalence (TOST) for bounded continuous metrics with empirical Bernstein bounds (v0.5.0, #20).

The v0.4.0 bootstrap-t TOST failed its calibration on skewed and zero-inflated differences, so
it stays descriptive. This procedure replaces it as an **opt-in** method with a finite-sample
guarantee instead of an asymptotic one:

* Per-item differences ``d_i = y_B - y_A`` of a metric bounded in ``[0, 1]`` lie in
  ``[-1, 1]``. Rescaled to ``x_i = (d_i + 1) / 2`` in ``[0, 1]``, the empirical Bernstein
  inequality (Maurer and Pontil 2009, Theorem 4) gives, for ``n >= 2`` independent items and
  any ``delta`` in ``(0, 1)``, with probability at least ``1 - delta``:

  ``E[x] >= mean(x) - sqrt(2 V ln(2 / delta) / n) - 7 ln(2 / delta) / (3 (n - 1))``

  where ``V`` is the unbiased sample variance of ``x`` (and symmetrically for the upper bound).
* Each one-sided test of the TOST rejects its boundary when the corresponding bound clears
  the margin. Its p-value is the smallest ``delta`` at which it does (closed form, from a
  quadratic in ``sqrt(ln(2 / delta))``). ``p_TOST = max(p_lower, p_upper)``.
* The size is at most ``alpha`` for **every** distribution of bounded differences (no normality,
  symmetry or large-sample assumption). The price is power: it needs more items than an
  asymptotic test. The ``(1 - 2 alpha)`` interval is the pair of bounds at ``delta = alpha``.

Assumption: items are independent draws from the population of items, as in every other
OpenBench procedure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from openbench.stats.results import EquivalenceInterval, EquivalenceResult, StatWarning

PROCEDURE = "empirical_bernstein_tost"
LOWER_BOUND, UPPER_BOUND = -1.0, 1.0
_RANGE = UPPER_BOUND - LOWER_BOUND


def _half_width(variance01: float, n: int, delta: float) -> float:
    """Bernstein deviation on the [0, 1] scale."""
    t = math.log(2.0 / delta)
    return math.sqrt(2.0 * variance01 * t / n) + 7.0 * t / (3.0 * (n - 1))


def _p_one_sided(gap01: float, variance01: float, n: int) -> float:
    """Smallest delta with deviation(delta) < gap (on the [0, 1] scale); 1.0 if gap <= 0."""
    if gap01 <= 0:
        return 1.0
    # gap = sqrt(2 V t / n) + 7 t / (3 (n - 1)), t = ln(2 / delta), s = sqrt(t):
    # a s^2 + b s - gap = 0 with a = 7 / (3 (n - 1)), b = sqrt(2 V / n).
    a = 7.0 / (3.0 * (n - 1))
    b = math.sqrt(2.0 * variance01 / n)
    s = (-b + math.sqrt(b * b + 4.0 * a * gap01)) / (2.0 * a)
    delta = 2.0 * math.exp(-(s * s))
    return min(1.0, delta)


def tost_empirical_bernstein(differences: Sequence[float], margin: float, alpha: float = 0.05) -> EquivalenceResult:
    """Finite-sample TOST of ``|E[d]| < margin`` for differences in ``[-1, 1]``."""
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if isinstance(margin, bool) or not (isinstance(margin, (int, float)) and margin > 0 and math.isfinite(margin)):
        raise ValueError("margin must be a positive finite number")
    data = [float(d) for d in differences]
    n = len(data)
    assumptions = (
        "items are independent draws from the item population",
        "per-item differences lie in [-1, 1] (metric bounded in [0, 1])",
    )

    def unavailable(reason: str) -> EquivalenceResult:
        return EquivalenceResult(
            procedure=PROCEDURE, margin=float(margin), alpha=alpha, sample_count=n,
            estimate=(math.fsum(data) / n) if n else None, p_lower=None, p_upper=None, p_tost=None,
            p_value_method="closed_form_bound_inversion", result_unadjusted="not_established",
            interval=None, reason=reason, assumptions=assumptions,
        )

    if n < 2:
        return unavailable("insufficient_items")
    if any(not math.isfinite(d) or d < LOWER_BOUND - 1e-12 or d > UPPER_BOUND + 1e-12 for d in data):
        return unavailable("differences_outside_bounds")
    scaled = [(d - LOWER_BOUND) / _RANGE for d in data]
    mean01 = math.fsum(scaled) / n
    # Two-pass variance with correctly rounded sums (deterministic across Python versions).
    var01 = math.fsum((x - mean01) ** 2 for x in scaled) / (n - 1)
    estimate = math.fsum(data) / n
    # On the [0, 1] scale the boundaries -margin and +margin sit at (±margin + 1) / 2.
    lo01 = (-margin - LOWER_BOUND) / _RANGE
    hi01 = (margin - LOWER_BOUND) / _RANGE
    p_lower = _p_one_sided(mean01 - lo01, var01, n)   # H0: E[d] <= -margin
    p_upper = _p_one_sided(hi01 - mean01, var01, n)   # H0: E[d] >= +margin
    p_tost = max(p_lower, p_upper)
    hw = _half_width(var01, n, alpha) * _RANGE
    interval = EquivalenceInterval(
        lower=max(LOWER_BOUND, estimate - hw), upper=min(UPPER_BOUND, estimate + hw), level=1 - 2 * alpha,
        method="empirical_bernstein_bounds", label=f"{100 * (1 - 2 * alpha):g}% equivalence interval (not multiplicity-adjusted)",
    )
    warnings: list[StatWarning] = []
    if n < 100:
        warnings.append(StatWarning("bernstein_low_power", f"Only {n} items; finite-sample bounds are wide, so equivalence is rarely established below a few hundred items."))
    return EquivalenceResult(
        procedure=PROCEDURE,
        margin=float(margin),
        alpha=alpha,
        sample_count=n,
        estimate=estimate,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        p_value_method="closed_form_bound_inversion",
        result_unadjusted="established" if p_tost <= alpha else "not_established",
        interval=interval,
        statistic={"mean_scaled": mean01, "variance_scaled": var01, "half_width_at_alpha": hw},
        assumptions=assumptions,
        diagnostics={"guarantee": "size <= alpha for any distribution of bounded differences (Maurer and Pontil 2009, Thm 4)"},
        warnings=tuple(warnings),
    )
