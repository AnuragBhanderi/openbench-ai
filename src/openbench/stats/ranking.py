"""Ranking several models with uncertainty (v0.5.0, issue #33).

Input: an ``n x k`` table of per-item scores (items aligned across all ``k`` models, complete
cases only). Output, per model:

* the mean score with a percentile bootstrap CI;
* a **rank interval** built from all ``k (k - 1) / 2`` pairwise paired tests with Holm
  correction: the best possible rank is ``1 + #models significantly better`` and the worst is
  ``k - #models significantly worse``. Because Holm controls the familywise error rate, all rank
  intervals hold together with probability of roughly ``1 - alpha`` (see the methodology note on
  directional errors);
* whether the model is **statistically tied with the top model** (not significantly different
  from the best observed model after Holm);
* a descriptive bootstrap share of resamples in which the model ranks first (``p_best``). This is
  not a probability that the model is truly best.

Ties in observed means share the average rank. Higher scores are better.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openbench.stats.bootstrap import bootstrap_mean_ci
from openbench.stats.effects import binary_paired_counts
from openbench.stats.rng import derive_stream
from openbench.stats.tests import (
    adjust_p_values,
    mcnemar_exact,
    select_difference_test,
    sign_flip_test,
)

RANK_METHOD = "holm_pairwise_rank_intervals"


def _average_ranks(values: Sequence[float], tol: float = 1e-12) -> list[float]:
    """Rank 1 = highest value; tied values (within ``tol``) share their average rank."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(values[order[j + 1]] - values[order[i]]) <= tol:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


@dataclass(frozen=True)
class RankingResult:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def rank_models(
    labels: Sequence[str],
    table: Sequence[Sequence[float]],
    *,
    score_kind: str = "bounded_continuous",
    repetitions: Sequence[int] | None = None,
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    permutations: int = 9_999,
    analysis_seed: int = 0,
    metric: str = "",
) -> RankingResult:
    """Rank ``k`` models from an ``n x k`` table of aligned per-item scores (rows = items)."""
    k = len(labels)
    if k < 2:
        raise ValueError("ranking needs at least two models")
    if len(set(labels)) != k:
        raise ValueError("model labels must be unique")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rows = [list(map(float, r)) for r in table]
    if any(len(r) != k for r in rows):
        raise ValueError("every table row must have one score per model")
    if any(not math.isfinite(v) for r in rows for v in r):
        raise ValueError("scores must be finite")
    n = len(rows)
    reps = list(repetitions) if repetitions is not None else [1] * k
    if len(reps) != k:
        raise ValueError("repetitions must have one entry per model")

    columns = [[r[j] for r in rows] for j in range(k)]
    means = [math.fsum(c) / n if n else float("nan") for c in columns]
    observed_ranks = _average_ranks(means) if n else [float("nan")] * k

    # Pairwise tests (i < j by label order), Holm across the whole family.
    pair_index: list[tuple[int, int]] = [(i, j) for i in range(k) for j in range(i + 1, k)]
    tests: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for i, j in pair_index:
        diffs = [r[j] - r[i] for r in rows]
        test_name = select_difference_test(score_kind if score_kind in ("binary", "bounded_continuous") else "bounded_continuous", reps[i], reps[j])
        if test_name == "mcnemar_exact":
            counts = binary_paired_counts(columns[i], columns[j])
            result = mcnemar_exact(counts.b, counts.c)
        else:
            result = sign_flip_test(
                diffs, permutations=permutations, analysis_seed=analysis_seed, metric=f"rank|{metric}",
                arm_a=labels[i], arm_b=labels[j],
            )
        p = result.p_value if result.p_value is not None else 1.0
        raw_p.append(p)
        tests.append(
            {
                "model_a": labels[i],
                "model_b": labels[j],
                "mean_difference_b_minus_a": (math.fsum(diffs) / n) if n else None,
                "test": result.test,
                "p_value": result.p_value,
                "unavailable_reason": result.unavailable_reason,
            }
        )
    adjusted = adjust_p_values(raw_p, "holm") if raw_p else None
    better = [[False] * k for _ in range(k)]  # better[a][b]: a significantly better than b
    for idx, (i, j) in enumerate(pair_index):
        p_adj = adjusted.adjusted[idx] if adjusted else 1.0
        tests[idx]["adjusted_p_value"] = p_adj
        diff = tests[idx]["mean_difference_b_minus_a"] or 0.0
        significant = p_adj <= alpha and diff != 0.0  # same rule as `openbench analyze`
        tests[idx]["significant"] = significant
        if significant:
            if diff > 0:
                better[j][i] = True
            else:
                better[i][j] = True

    top = min(range(k), key=lambda t: (observed_ranks[t], t)) if n else 0

    # Descriptive bootstrap: share of item resamples in which each model ranks first.
    rng = derive_stream(analysis_seed, "rank", metric)
    first_counts = [0.0] * k
    boot = min(resamples, 2_000) if n else 0
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bmeans = [math.fsum(columns[j][t] for t in idx) / n for j in range(k)]
        best = max(bmeans)
        winners = [j for j in range(k) if abs(bmeans[j] - best) <= 1e-12]
        for w in winners:
            first_counts[w] += 1.0 / len(winners)

    models: list[dict[str, Any]] = []
    for j in range(k):
        ci = bootstrap_mean_ci(columns[j], confidence_level=confidence_level, resamples=resamples,
                               analysis_seed=analysis_seed, metric=f"rank_mean|{metric}", arm_a=labels[j])
        n_better = sum(1 for t in range(k) if better[t][j])
        n_worse = sum(1 for t in range(k) if better[j][t])
        tied_with_top = j == top or not (better[top][j] or better[j][top])
        models.append(
            {
                "model": labels[j],
                "repetitions": reps[j],
                "mean": means[j] if n else None,
                "mean_confidence_interval": ci.to_dict(),
                "observed_rank": observed_ranks[j] if n else None,
                "rank_interval": [1 + n_better, k - n_worse],
                "significantly_better_than": sorted(labels[t] for t in range(k) if better[j][t]),
                "significantly_worse_than": sorted(labels[t] for t in range(k) if better[t][j]),
                "tied_with_top": tied_with_top,
                "p_best_bootstrap": (first_counts[j] / boot) if boot else None,
                "p_best_resamples": boot,
            }
        )
    models.sort(key=lambda m: (m["observed_rank"] if m["observed_rank"] is not None else math.inf, m["model"]))

    warnings: list[dict[str, str]] = []
    if n < 30:
        warnings.append({"code": "low_power_wide_intervals", "message": f"Only {n} aligned items; rank intervals will be wide."})
    tied = [m["model"] for m in models if m["tied_with_top"]]
    if len(tied) > 1:
        warnings.append({"code": "no_clear_winner", "message": f"{len(tied)} models are statistically tied with the top model: {', '.join(tied)}."})

    return RankingResult(
        {
            "method": RANK_METHOD,
            "metric": metric,
            "score_kind": score_kind,
            "items": n,
            "models_ranked": k,
            "alpha": alpha,
            "confidence_level": confidence_level,
            "resamples": resamples,
            "permutations": permutations,
            "analysis_seed": analysis_seed,
            "top_model": labels[top] if n else None,
            "models": models,
            "pairwise_tests": tests,
            "correction": "holm",
            "warnings": warnings,
            "interpretation": (
                "Rank intervals come from all pairwise paired tests with Holm correction: best rank = 1 + number of "
                "models significantly better, worst rank = number of models minus the number significantly worse. "
                "'Tied with top' means not significantly different from the best observed model. p_best_bootstrap "
                "is the share of item resamples in which the model scored highest; it describes stability, not the "
                "probability that the model is truly best."
            ),
        }
    )
