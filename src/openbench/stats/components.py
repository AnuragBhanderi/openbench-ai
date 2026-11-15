"""Variance components and a two-level confidence interval for repeated runs (v0.5.0, issue #45).

For one model run ``R`` times on ``n`` items, with ``y[i][r]`` the score of item ``i`` in
repetition ``r`` (complete cases: items valid in every repetition), the one-way random-effects
model ``y = mu + item_i + rep_ir`` gives the usual ANOVA (method-of-moments) estimates:

* ``MSW = sum_i sum_r (y_ir - ybar_i)^2 / (n (R - 1))`` estimates the **repetition** (within-item)
  variance ``sigma2_rep``;
* ``MSB = R * sum_i (ybar_i - ybar)^2 / (n - 1)`` and ``sigma2_item = max(0, (MSB - MSW) / R)``
  estimates the **item** variance.

The share of variance from re-running is ``sigma2_rep / (sigma2_rep + sigma2_item)``.

The **two-level CI** for the mean resamples items with replacement, then repetitions with
replacement within each chosen item (percentile interval). It accounts for both sources of
variation; with few repetitions it is somewhat wider than necessary (the within-item variance
is counted twice), which errs on the safe side. The existing item-level CI (resampling item
means only) is reported beside it for comparison.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from openbench.stats.bootstrap import bootstrap_mean_ci
from openbench.stats.rng import derive_stream

TWO_LEVEL_MAX_RESAMPLES = 2_000


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def variance_components(matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    """ANOVA estimates of item and repetition variance from an ``n x R`` matrix."""
    rows = [[float(v) for v in r] for r in matrix]
    n = len(rows)
    reps = len(rows[0]) if rows else 0
    if n < 2 or reps < 2 or any(len(r) != reps for r in rows):
        return {
            "items": n,
            "repetitions": reps,
            "unavailable_reason": "need at least 2 items and 2 repetitions per item (complete cases)",
        }
    item_means = [math.fsum(r) / reps for r in rows]
    grand = math.fsum(item_means) / n
    ssw = math.fsum((v - item_means[i]) ** 2 for i, r in enumerate(rows) for v in r)
    msw = ssw / (n * (reps - 1))
    msb = reps * math.fsum((m - grand) ** 2 for m in item_means) / (n - 1)
    sigma2_rep = msw
    sigma2_item_raw = (msb - msw) / reps
    sigma2_item = max(0.0, sigma2_item_raw)
    total = sigma2_rep + sigma2_item
    return {
        "items": n,
        "repetitions": reps,
        "grand_mean": grand,
        "sigma2_item": sigma2_item,
        "sigma2_repetition": sigma2_rep,
        "sigma2_item_unclipped": sigma2_item_raw,
        "share_repetition": (sigma2_rep / total) if total > 0 else None,
        "share_item": (sigma2_item / total) if total > 0 else None,
        "mean_squares": {"between_items": msb, "within_items": msw},
        "standard_error_of_mean_anova": math.sqrt(msb / (n * reps)),
        "estimator": "one-way random effects, ANOVA (method of moments); negative item variance clipped at 0",
    }


def two_level_bootstrap_ci(
    matrix: Sequence[Sequence[float]],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    analysis_seed: int = 0,
    metric: str = "",
    arm: str = "",
) -> dict[str, Any]:
    """Percentile CI for the mean, resampling items and then repetitions within items."""
    rows = [[float(v) for v in r] for r in matrix]
    n = len(rows)
    if n < 2 or not 0 < confidence_level < 1:
        return {"estimate": None, "lower": None, "upper": None, "unavailable_reason": "need at least 2 items"}
    grand = math.fsum(v for r in rows for v in r) / sum(len(r) for r in rows)
    rng = derive_stream(analysis_seed, "hierarchical", metric, arm)
    stats: list[float] = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for _ in range(n):
            row = rows[rng.randrange(n)]
            m = len(row)
            for _ in range(m):
                total += row[rng.randrange(m)]
            count += m
        stats.append(total / count)
    stats.sort()
    a = (1 - confidence_level) / 2
    return {
        "estimate": grand,
        "lower": _percentile(stats, a),
        "upper": _percentile(stats, 1 - a),
        "confidence_level": confidence_level,
        "resamples": resamples,
        "method": "two_level_percentile_bootstrap (items, then repetitions within items)",
        "analysis_seed": analysis_seed,
    }


def repeated_run_components(
    matrix: Sequence[Sequence[float]],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    analysis_seed: int = 0,
    metric: str = "",
    arm: str = "",
) -> dict[str, Any]:
    """Variance components plus two-level and item-level CIs for one repeat group."""
    comps = variance_components(matrix)
    if comps.get("unavailable_reason"):
        return {"components": comps}
    # The two-level bootstrap is costlier per resample; 2,000 resamples are enough for a 95% percentile CI.
    two_level = two_level_bootstrap_ci(matrix, confidence_level=confidence_level, resamples=min(resamples, TWO_LEVEL_MAX_RESAMPLES),
                                       analysis_seed=analysis_seed, metric=metric, arm=arm)
    item_means = [math.fsum(r) / len(r) for r in matrix]
    item_level = bootstrap_mean_ci(item_means, confidence_level=confidence_level, resamples=resamples,
                                   analysis_seed=analysis_seed, metric=f"components|{metric}", arm_a=arm)
    share = comps.get("share_repetition")
    if share is None:
        sentence = "All scores are identical; there is no variation to split."
    else:
        sentence = (
            f"Re-running explains {100 * share:.0f}% of the variation in scores; "
            f"the choice of questions explains {100 * (1 - share):.0f}%."
        )
    return {
        "components": comps,
        "two_level_confidence_interval": two_level,
        "item_level_confidence_interval": item_level.to_dict(),
        "summary": sentence,
    }
