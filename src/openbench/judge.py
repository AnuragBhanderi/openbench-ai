"""LLM-as-judge with bias controls (v0.5.0, issue #19).

Pairwise judging of two models' responses by a third model, measured rather than trusted:

* **Both orders.** Every pair is judged twice, with the responses shown as A-then-B and
  B-then-A. A verdict counts only when both orders agree; otherwise the pair is
  *inconsistent* and scored as a tie. The position-consistency rate is reported.
* **Position bias.** The share of decisive single-order verdicts that pick the response shown
  first, with a bootstrap CI.
* **Length bias.** Among consistent decisive verdicts with responses of different lengths, the
  share won by the longer response, with a bootstrap CI.
* **Self-preference.** Flagged when the judge model has the same name as one of the judged models.
* **Agreement with people.** When blinded human annotations exist for the same pair, Cohen's
  kappa between the judge and the human majority (see :mod:`openbench.agreement`).
* **Provenance and caching.** Judge provider, model, prompt version and prompt hash are recorded;
  judge calls go through the execution engine, so they are cached, retried and rate-limited like
  any other run.

Judge verdicts are opinions of another model. Reports never present them as ground truth.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openbench.adapters.base import BaseModelAdapter
from openbench.cache import ResponseCache
from openbench.dataset import Dataset, DatasetRecord
from openbench.exceptions import ConfigurationError
from openbench.execution import ExecutionConfig, ExecutionEngine
from openbench.preferences import PreferenceSource
from openbench.stats.bootstrap import bootstrap_mean_ci

JUDGE_PROMPT_VERSION = "pairwise-v1"

DEFAULT_CRITERIA = (
    "Judge which response answers the user's question better: correctness first, then "
    "completeness and clarity. Do not prefer a response because it is longer."
)

PAIRWISE_TEMPLATE = """You are an impartial judge comparing two AI responses to the same question.
{criteria}

[Question]
{question}
{reference_block}
[Response 1]
{response_1}

[Response 2]
{response_2}

Reply with exactly one word on the first line: 1 if Response 1 is better, 2 if Response 2 is better, or tie."""

_VERDICT = re.compile(r"\b(1|2|tie)\b", re.IGNORECASE)
JUDGE_LIMITATION = (
    "Judge verdicts are another model's opinion. They can share the judged models' blind spots and "
    "are not ground truth; check their agreement with people before relying on them."
)


def parse_verdict(text: str) -> str | None:
    """Map judge output to '1', '2' or 'tie' (first match on the first non-empty line)."""
    for line in (text or "").splitlines():
        line = line.strip().strip("*").strip()
        if not line:
            continue
        m = _VERDICT.search(line)
        return m.group(1).lower() if m else None
    return None


def build_prompt(question: str, first: str, second: str, *, criteria: str = DEFAULT_CRITERIA, reference: str | None = None) -> str:
    reference_block = f"\n[Reference answer]\n{reference}\n" if reference else ""
    return PAIRWISE_TEMPLATE.format(
        criteria=criteria.strip(), question=question, reference_block=reference_block,
        response_1=first, response_2=second,
    )


def _to_model_space(verdict: str | None, order: str) -> str | None:
    if verdict is None:
        return None
    if verdict == "tie":
        return "tie"
    shown_first = order[0]  # "A" for AB, "B" for BA
    shown_second = order[1]
    return shown_first if verdict == "1" else shown_second


def _same_model(name_a: str, name_b: str) -> bool:
    def norm(s: str) -> str:
        return s.strip().lower().split(":")[0]

    return norm(name_a) == norm(name_b)


@dataclass
class PairwiseJudgeResult:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    @property
    def labels(self) -> dict[str, str | None]:
        return {t["record_id"]: t["verdict"] for t in self.payload["tasks"]}


class PairwiseJudge:
    """Judge two preference sources with a judge model, in both orders."""

    def __init__(
        self,
        judge: BaseModelAdapter,
        *,
        criteria: str = DEFAULT_CRITERIA,
        show_reference: bool = False,
        execution_config: ExecutionConfig | None = None,
        cache: ResponseCache | None = None,
        analysis_seed: int = 0,
        confidence_level: float = 0.95,
        resamples: int = 10_000,
    ) -> None:
        self.judge = judge
        self.criteria = criteria
        self.show_reference = show_reference
        self.execution_config = execution_config or ExecutionConfig(timeout_seconds=judge.timeout_seconds)
        self.cache = cache if cache is not None else ResponseCache(enabled=False)
        self.seed = analysis_seed
        self.confidence_level = confidence_level
        self.resamples = resamples

    def prompt_sha256(self) -> str:
        template = PAIRWISE_TEMPLATE + "\n" + self.criteria + f"\nreference={self.show_reference}"
        return hashlib.sha256(template.encode("utf-8")).hexdigest()

    def judge_sources(self, source_a: PreferenceSource, source_b: PreferenceSource, record_ids: Sequence[str] | None = None) -> PairwiseJudgeResult:
        shared = sorted(set(source_a.items) & set(source_b.items))
        if record_ids is not None:
            wanted = set(record_ids)
            shared = [r for r in shared if r in wanted]
        eligible = [
            r for r in shared
            if source_a.items[r]["valid"] and source_b.items[r]["valid"] and source_a.items[r]["prompt"] == source_b.items[r]["prompt"]
        ]
        if not eligible:
            raise ConfigurationError("No record has a valid response from both models; nothing to judge")
        records: list[DatasetRecord] = []
        for rid in eligible:
            ia, ib = source_a.items[rid], source_b.items[rid]
            ref = ia["reference"] if self.show_reference else None
            for order, (first, second) in (("AB", (ia, ib)), ("BA", (ib, ia))):
                records.append(
                    DatasetRecord(
                        id=f"{rid}|{order}",
                        category=ia["category"],
                        prompt=build_prompt(ia["prompt"], first["response"], second["response"], criteria=self.criteria, reference=ref),
                        expected_answer="",
                        metadata={"difficulty": ia.get("difficulty") or "unspecified"},
                    )
                )
        dataset = Dataset(
            name=f"openbench-judge/{source_a.source_id}-vs-{source_b.source_id}",
            version=f"{JUDGE_PROMPT_VERSION}+{self.prompt_sha256()[:12]}",
            records=records,
            source_path="judge",
        )
        engine = ExecutionEngine(self.judge, self.execution_config, self.cache)
        outcome = engine.execute_dataset(dataset)
        raw = {item.id: item.response for item in outcome.response_set.responses}

        tasks: list[dict[str, Any]] = []
        first_shown_wins: list[float] = []
        longer_wins: list[float] = []
        for rid in eligible:
            ia, ib = source_a.items[rid], source_b.items[rid]
            v_ab = parse_verdict(raw.get(f"{rid}|AB", "")) if f"{rid}|AB" in raw else None
            v_ba = parse_verdict(raw.get(f"{rid}|BA", "")) if f"{rid}|BA" in raw else None
            m_ab, m_ba = _to_model_space(v_ab, "AB"), _to_model_space(v_ba, "BA")
            for v in (v_ab, v_ba):
                if v in ("1", "2"):
                    first_shown_wins.append(1.0 if v == "1" else 0.0)
            if m_ab is None or m_ba is None:
                status, verdict = "unparsed_or_failed", None
            elif m_ab == m_ba:
                status, verdict = "consistent", m_ab
            else:
                status, verdict = "inconsistent", "tie"
            len_a, len_b = len(ia["response"] or ""), len(ib["response"] or "")
            if status == "consistent" and verdict in ("A", "B") and len_a != len_b:
                longer = "A" if len_a > len_b else "B"
                longer_wins.append(1.0 if verdict == longer else 0.0)
            tasks.append(
                {
                    "record_id": rid,
                    "category": ia["category"],
                    "verdict_order_ab": m_ab,
                    "verdict_order_ba": m_ba,
                    "status": status,
                    "verdict": verdict,
                    "response_lengths": {"A": len_a, "B": len_b},
                }
            )
        judged = [t for t in tasks if t["verdict"] is not None]
        scores = [{"A": 1.0, "tie": 0.5, "B": 0.0}[t["verdict"]] for t in judged]
        win_ci = bootstrap_mean_ci(scores, confidence_level=self.confidence_level, resamples=self.resamples,
                                   analysis_seed=self.seed, metric="judge_win_rate_a")
        pos_ci = bootstrap_mean_ci(first_shown_wins, confidence_level=self.confidence_level, resamples=self.resamples,
                                   analysis_seed=self.seed, metric="judge_position_bias")
        len_ci = bootstrap_mean_ci(longer_wins, confidence_level=self.confidence_level, resamples=self.resamples,
                                   analysis_seed=self.seed, metric="judge_length_bias")
        consistent = sum(1 for t in tasks if t["status"] == "consistent")
        inconsistent = sum(1 for t in tasks if t["status"] == "inconsistent")
        warnings: list[dict[str, str]] = []
        judge_name = self.judge.model_name
        self_pref = [s.source_id for s in (source_a, source_b) if _same_model(judge_name, s.model_name)]
        if self_pref:
            warnings.append({"code": "self_preference_risk", "message": f"The judge model '{judge_name}' also produced responses ({', '.join(self_pref)}); judges tend to prefer their own outputs."})
        if pos_ci.lower is not None and (pos_ci.lower > 0.5 or pos_ci.upper < 0.5):
            warnings.append({"code": "position_bias", "message": "The judge picks the first-shown response more or less often than half the time, beyond its CI."})
        if len_ci.lower is not None and len_ci.lower > 0.5:
            warnings.append({"code": "length_bias", "message": "The longer response wins more often than half the time, beyond its CI; the judge may favour length."})
        if tasks and inconsistent / len(tasks) > 0.2:
            warnings.append({"code": "low_position_consistency", "message": f"{inconsistent} of {len(tasks)} pairs changed verdict when the order was swapped."})
        failed = sum(1 for t in tasks if t["status"] == "unparsed_or_failed")
        if failed:
            warnings.append({"code": "unparsed_verdicts", "message": f"{failed} pair(s) had a failed request or an answer that was not 1, 2 or tie."})
        payload = {
            "analysis": "llm_judge_pairwise",
            "judge": {
                "provider": self.judge.provider_name,
                "model_name": judge_name,
                "model_version": self.judge.model_version,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "prompt_sha256": self.prompt_sha256(),
                "criteria": self.criteria,
                "show_reference": self.show_reference,
                "generation": self.judge.reproducibility_metadata(),
            },
            "model_a": source_a.identity(),
            "model_b": source_b.identity(),
            "pairs_judged": len(tasks),
            "judge_calls": len(records),
            "cached_judge_calls": sum(1 for log in outcome.execution_logs if log.get("event") == "cache_hit"),
            "position_consistency_rate": consistent / len(tasks) if tasks else None,
            "outcome_counts": {k: sum(1 for t in judged if t["verdict"] == k) for k in ("A", "B", "tie")},
            "inconsistent_pairs_scored_as_tie": inconsistent,
            "win_rate_a": (math.fsum(scores) / len(scores)) if scores else None,
            "win_rate_a_confidence_interval": win_ci.to_dict(),
            "position_bias": {"share_first_shown_wins": (math.fsum(first_shown_wins) / len(first_shown_wins)) if first_shown_wins else None,
                              "decisive_single_order_verdicts": len(first_shown_wins), "confidence_interval": pos_ci.to_dict()},
            "length_bias": {"share_longer_wins": (math.fsum(longer_wins) / len(longer_wins)) if longer_wins else None,
                            "pairs": len(longer_wins), "confidence_interval": len_ci.to_dict()},
            "tasks": tasks,
            "warnings": warnings,
            "limitations": JUDGE_LIMITATION,
            "methodology": (
                "Each pair is judged in both display orders. A verdict counts only when both orders agree; otherwise the "
                "pair is scored as a tie. Win rate of A: A = 1, tie = 0.5, B = 0, with a percentile bootstrap CI over pairs."
            ),
        }
        return PairwiseJudgeResult(payload)
