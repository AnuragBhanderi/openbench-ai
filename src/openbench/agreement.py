"""Agreement between an automatic judgement and human preferences (v0.5.0, issues #46 and #19).

The automatic side is any per-task preference ``"A"``, ``"B"`` or ``"tie"``: derived from a
metric (the response with the higher score wins) or from an LLM judge. The human side is the
majority outcome of a task's annotations in model space (``A``, ``B`` or ``tie``; ``neither`` is
ignored). Tasks without a strict human majority are excluded and counted.

Reported: percent agreement with a bootstrap CI over tasks, Cohen's kappa between the automatic
label and the human majority, agreement on decisive tasks only, a confusion table and a
breakdown by category. Agreement with people measures how well the automatic method imitates
these annotators on these tasks; it is not the same as correctness.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from openbench.preferences import cohens_kappa
from openbench.stats.bootstrap import bootstrap_mean_ci

LABELS = ("A", "B", "tie")


def human_majority(labels: Sequence[str]) -> str | None:
    """Strict majority among A/B/tie labels ('neither' ignored); ``None`` if there is none."""
    counted = Counter(label for label in labels if label in LABELS)
    if not counted:
        return None
    (top, top_n), *rest = counted.most_common()
    if rest and rest[0][1] == top_n:
        return None
    return top


def metric_preference(score_a: float | None, score_b: float | None, tie_tolerance: float = 1e-9) -> str | None:
    """'A' if A scored higher, 'B' if B did, 'tie' within the tolerance; None if a score is missing."""
    if score_a is None or score_b is None:
        return None
    diff = float(score_b) - float(score_a)
    if abs(diff) <= tie_tolerance:
        return "tie"
    return "B" if diff > 0 else "A"


def agreement_report(
    human_labels: Mapping[str, Sequence[str]],
    automatic: Mapping[str, str | None],
    *,
    categories: Mapping[str, str] | None = None,
    automatic_name: str = "automatic",
    analysis_seed: int = 0,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Compare automatic per-task labels with the human majority per task."""
    excluded = Counter()
    tasks: list[str] = []
    for task_id in sorted(set(human_labels) | set(automatic)):
        majority = human_majority(human_labels.get(task_id, ()))
        auto = automatic.get(task_id)
        if task_id not in human_labels:
            excluded["no_human_annotations"] += 1
        elif majority is None:
            excluded["no_human_majority"] += 1
        elif auto not in LABELS:
            excluded["no_automatic_label"] += 1
        else:
            tasks.append(task_id)
    pairs = [(human_majority(human_labels[t]), automatic[t]) for t in tasks]
    matches = [1.0 if h == a else 0.0 for h, a in pairs]
    ci = bootstrap_mean_ci(matches, confidence_level=confidence_level, resamples=resamples,
                           analysis_seed=analysis_seed, metric=f"agreement|{automatic_name}")
    kappa = cohens_kappa([h for h, _ in pairs], [a for _, a in pairs]) if len(pairs) >= 2 else None
    decisive = [(h, a) for h, a in pairs if h != "tie" and a != "tie"]
    confusion = {h: {a: sum(1 for x, y in pairs if x == h and y == a) for a in LABELS} for h in LABELS}
    by_category: dict[str, Any] = {}
    if categories:
        for cat in sorted({categories.get(t, "unknown") for t in tasks}):
            sub = [m for t, m in zip(tasks, matches, strict=True) if categories.get(t, "unknown") == cat]
            by_category[cat] = {"tasks": len(sub), "percent_agreement": math.fsum(sub) / len(sub) if sub else None}
    warnings = []
    if len(pairs) < 30:
        warnings.append({"code": "few_tasks", "message": f"Only {len(pairs)} tasks compared; agreement estimates are imprecise."})
    return {
        "automatic": automatic_name,
        "tasks_compared": len(pairs),
        "tasks_excluded": dict(sorted(excluded.items())),
        "percent_agreement": (math.fsum(matches) / len(matches)) if matches else None,
        "percent_agreement_confidence_interval": {**ci.to_dict(), "method": "percentile_bootstrap_over_tasks"},
        "cohens_kappa_vs_human_majority": kappa,
        "decisive_tasks": len(decisive),
        "decisive_agreement": (sum(1 for h, a in decisive if h == a) / len(decisive)) if decisive else None,
        "confusion_human_by_automatic": confusion,
        "by_category": by_category,
        "warnings": warnings,
        "note": (
            "Human label = strict majority of A/B/tie annotations per task ('neither' ignored). Agreement measures how "
            "well the automatic method imitates these annotators on these tasks; it is not correctness."
        ),
    }
