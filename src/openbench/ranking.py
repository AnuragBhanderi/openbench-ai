"""Rank stored runs, repeat groups or evaluation files (v0.5.0, issue #33).

Uses the same arms, item alignment and compatibility checks as ``openbench analyze``, then
hands an aligned ``items x models`` table to :func:`openbench.stats.ranking.rank_models`.
Never calls a model provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openbench.exceptions import ConfigurationError, IncompatibleComparisonError
from openbench.itemscores import collect_item_scores, metric_score_kind
from openbench.paired import ComparisonArm, check_compatibility
from openbench.provenance import provenance_block
from openbench.stats.ranking import rank_models


def rank_arms(
    arms: Sequence[ComparisonArm],
    metric: str,
    *,
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    permutations: int = 9_999,
    analysis_seed: int = 0,
    selected_ids: Sequence[str] | None = None,
    allow_incompatible: bool = False,
) -> dict[str, Any]:
    arms = list(arms)
    if len(arms) < 2:
        raise ConfigurationError("Ranking needs at least two runs, repeat groups or evaluation files")
    ids = [a.arm_id for a in arms]
    if len(set(ids)) != len(ids):
        raise ConfigurationError(f"Arm IDs must be unique: {ids}")
    for arm in arms:
        if metric not in arm.evaluations[0].aggregate_results:
            raise ConfigurationError(f"Metric '{metric}' was not computed for '{arm.arm_id}'")

    observations = {a.arm_id: collect_item_scores(a.evaluations, metric) for a in arms}
    shared = set.intersection(*(set(o) for o in observations.values()))
    if selected_ids is not None:
        shared &= set(selected_ids)
    universe = sorted(shared)

    compat: list[dict[str, Any]] = []
    hard = []
    base = arms[0]
    for other in arms[1:]:
        report = check_compatibility(base, other, [metric], universe)
        compat.append({"baseline": base.arm_id, "challenger": other.arm_id, **report.to_dict()})
        if not report.compatible:
            hard.append(f"{base.arm_id} vs {other.arm_id}:\n{report.describe()}")
    if hard and not allow_incompatible:
        raise IncompatibleComparisonError("Inputs are incompatible for ranking:\n" + "\n".join(hard), report=None)

    table: list[list[float]] = []
    excluded = 0
    for rid in universe:
        row = []
        for arm in arms:
            obs = observations[arm.arm_id].get(rid)
            row.append(obs.mean if obs is not None else None)
        if any(v is None for v in row):
            excluded += 1
            continue
        table.append([float(v) for v in row])

    kind = metric_score_kind(metric)
    result = rank_models(
        ids,
        table,
        score_kind=kind,
        repetitions=[a.repetitions for a in arms],
        alpha=alpha,
        confidence_level=confidence_level,
        resamples=resamples,
        permutations=permutations,
        analysis_seed=analysis_seed,
        metric=metric,
    ).to_dict()
    for entry in result["models"]:
        arm = next(a for a in arms if a.arm_id == entry["model"])
        entry["model_identity"] = arm.model_identity
        entry["source"] = arm.source
        entry["is_mock_provider"] = arm.is_mock
    parameters = {
        "metric": metric, "alpha": alpha, "confidence_level": confidence_level, "resamples": resamples,
        "permutations": permutations, "missing_policy": "complete_case",
    }
    return {
        "schema_version": "0.5.0",
        "analysis": "model_ranking",
        "analysis_mode": "exploratory" if hard else "inferential",
        "banner": ("EXPLORATORY: some inputs failed hard compatibility checks; treat this ranking as descriptive." if hard else None),
        "arms": [a.summary() for a in arms],
        "items_shared": len(universe),
        "items_excluded_incomplete": excluded,
        "compatibility": compat,
        **result,
        "provenance": provenance_block(
            parameters=parameters,
            inputs=[{"arm": a.arm_id, "input_sha256": list(a.input_sha256)} for a in arms],
            analysis_seed=analysis_seed,
        ),
    }
