"""Calibration studies for the v0.5.0 statistical methods (standard library only).

Run from the repository root (takes several minutes):

    PYTHONPATH=src python tests/reference/_ref_v050_calibration.py

It writes ``tests/reference/data/v050_calibration_results.json``. The test
``tests/reference/test_reference_v050_calibration.py`` checks the committed results against the
pre-specified pass rules below (it does not re-run the simulation).

Pass rules (fixed before running):

* **Rank intervals** (``openbench rank``): simultaneous coverage, i.e. every model's rank
  interval contains one of its true ranks, at least ``0.95 - 2 * SE`` in every configuration,
  where ``SE = sqrt(0.95 * 0.05 / sims)``.
* **Two-level bootstrap CI** (repeated runs): coverage of the true mean at least ``0.95 - 2 * SE``
  in every configuration with at least 30 items; configurations with fewer items are reported
  but only need ``0.90`` (small-sample percentile bootstrap intervals are known to be narrow).
* **Power planner** (``openbench plan``): exact McNemar power within ``3 * SE`` of a Monte Carlo
  estimate of the real test's rejection rate in every configuration.
* **Bounded TOST** (``--continuous-equivalence empirical_bernstein``): on the same scenario grid
  as the v0.4.0 bootstrap-t calibration (``_ref_continuous_calibration.py``), the empirical size of
  each one-sided test at its boundary is at most ``alpha + 2 * SE`` and at most ``0.06`` in every
  cell (the unchanged v0.4.0 acceptance rule).
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

from openbench.stats.bernstein import tost_empirical_bernstein
from openbench.stats.components import two_level_bootstrap_ci
from openbench.stats.power import mcnemar_power
from openbench.stats.ranking import rank_models
from openbench.stats.tests import mcnemar_exact

OUT = Path(__file__).resolve().parent / "data" / "v050_calibration_results.json"
MASTER_SEED = 20261115


def _true_rank_sets(means: list[float]) -> list[set[int]]:
    """Ranks each model may correctly take (tied true means may take any rank in their block)."""
    order = sorted(range(len(means)), key=lambda i: -means[i])
    sets: list[set[int]] = [set() for _ in means]
    pos = 0
    while pos < len(order):
        end = pos
        while end + 1 < len(order) and means[order[end + 1]] == means[order[pos]]:
            end += 1
        for t in range(pos, end + 1):
            sets[order[t]] = set(range(pos + 1, end + 2))
        pos = end + 1
    return sets


def rank_study(sims: int) -> list[dict]:
    rng = random.Random(MASTER_SEED)
    configs = [
        {"kind": "continuous", "means": [0.0, 0.0, 0.0], "sd": 1.0, "n": 30},
        {"kind": "continuous", "means": [0.0, 0.2, 0.4, 0.6, 0.8], "sd": 1.0, "n": 30},
        {"kind": "continuous", "means": [0.0, 0.0, 0.3, 0.3, 1.0], "sd": 1.0, "n": 60},
        {"kind": "binary", "means": [0.6, 0.6, 0.6, 0.6], "n": 100},
        {"kind": "binary", "means": [0.5, 0.6, 0.7, 0.8], "n": 100},
        {"kind": "binary", "means": [0.55, 0.6, 0.6, 0.75, 0.9], "n": 80},
    ]
    results = []
    for cfg in configs:
        k, n = len(cfg["means"]), cfg["n"]
        allowed = _true_rank_sets(cfg["means"])
        covered = 0
        for _ in range(sims):
            if cfg["kind"] == "binary":
                rows = [[1.0 if rng.random() < p else 0.0 for p in cfg["means"]] for _ in range(n)]
            else:
                rows = [[rng.gauss(m, cfg["sd"]) for m in cfg["means"]] for _ in range(n)]
            out = rank_models([f"m{i}" for i in range(k)], rows, score_kind=cfg["kind"] if cfg["kind"] == "binary" else "bounded_continuous",
                              resamples=50, permutations=299, analysis_seed=rng.randrange(2**31)).to_dict()
            ok = all(any(m["rank_interval"][0] <= r <= m["rank_interval"][1] for r in allowed[int(m["model"][1:])]) for m in out["models"])
            covered += ok
        cov = covered / sims
        se = math.sqrt(0.95 * 0.05 / sims)
        results.append({**cfg, "sims": sims, "simultaneous_coverage": cov, "threshold": 0.95 - 2 * se, "passed": cov >= 0.95 - 2 * se})
        print(f"rank {cfg} -> {cov:.3f}", flush=True)
    return results


def two_level_study(sims: int) -> list[dict]:
    rng = random.Random(MASTER_SEED + 1)
    configs = [
        {"kind": "gaussian", "n": 20, "reps": 3, "sd_item": 1.0, "sd_rep": 1.0},
        {"kind": "gaussian", "n": 50, "reps": 2, "sd_item": 1.0, "sd_rep": 1.0},
        {"kind": "gaussian", "n": 50, "reps": 5, "sd_item": 0.3, "sd_rep": 1.0},
        {"kind": "gaussian", "n": 100, "reps": 3, "sd_item": 1.0, "sd_rep": 0.2},
        {"kind": "binary", "n": 50, "reps": 3},
        {"kind": "binary", "n": 100, "reps": 5},
    ]
    results = []
    for cfg in configs:
        hits = 0
        for _ in range(sims):
            rows = []
            if cfg["kind"] == "gaussian":
                truth = 0.0
                for _ in range(cfg["n"]):
                    item = rng.gauss(0.0, cfg["sd_item"])
                    rows.append([item + rng.gauss(0.0, cfg["sd_rep"]) for _ in range(cfg["reps"])])
            else:
                # Item difficulty p_i ~ Beta(2, 2) (mean 0.5); responses Bernoulli(p_i).
                truth = 0.5
                for _ in range(cfg["n"]):
                    p = rng.betavariate(2.0, 2.0)
                    rows.append([1.0 if rng.random() < p else 0.0 for _ in range(cfg["reps"])])
            ci = two_level_bootstrap_ci(rows, resamples=500, analysis_seed=rng.randrange(2**31))
            hits += ci["lower"] <= truth <= ci["upper"]
        cov = hits / sims
        se = math.sqrt(0.95 * 0.05 / sims)
        threshold = 0.95 - 2 * se if cfg["n"] >= 30 else 0.90
        results.append({**cfg, "sims": sims, "coverage": cov, "threshold": threshold, "passed": cov >= threshold})
        print(f"two-level {cfg} -> {cov:.3f}", flush=True)
    return results


def power_study(sims: int) -> list[dict]:
    rng = random.Random(MASTER_SEED + 2)
    configs = [(40, 0.2, 0.05), (100, 0.15, 0.08), (150, 0.12, 0.08), (200, 0.3, 0.25), (60, 0.1, 0.1), (300, 0.05, 0.02)]
    results = []
    for n, p01, p10 in configs:
        hits = 0
        for _ in range(sims):
            b = c = 0
            for _ in range(n):
                u = rng.random()
                if u < p01:
                    b += 1
                elif u < p01 + p10:
                    c += 1
            hits += mcnemar_exact(b, c).p_value <= 0.05
        mc = hits / sims
        exact = mcnemar_power(n, p01, p10)
        se = math.sqrt(max(exact * (1 - exact), 1e-4) / sims)
        results.append({"n": n, "p01": p01, "p10": p10, "sims": sims, "monte_carlo": mc, "exact": exact,
                        "abs_difference": abs(mc - exact), "tolerance": 3 * se, "passed": abs(mc - exact) <= 3 * se})
        print(f"power n={n} p01={p01} p10={p10}: exact {exact:.4f} mc {mc:.4f}", flush=True)
    return results


# ---- Bounded TOST: the v0.4.0 scenario grid, re-implemented with the standard library.
def _two_point(lo, plo, hi):
    def gen(rng, n, delta, sign):
        return [sign * delta + (lo if rng.random() < plo else hi) for _ in range(n)]
    return gen


def _mirror(g):
    def gen(rng, n, delta, sign):
        return [2 * sign * delta - x for x in g(rng, n, delta, sign)]
    return gen


def _sym_trunc_normal(rng, n, delta, sign):
    return [sign * delta + max(-0.6, min(0.6, rng.gauss(0.0, 0.15))) for _ in range(n)]


def _three_point_zero(rng, n, delta, sign):
    out = []
    for _ in range(n):
        u = rng.random()
        out.append(sign * delta + (-0.1 if u < 0.3 else (0.0 if u < 0.9 else 0.3)))
    return out


def _per_item_mean(reps):
    def gen(rng, n, delta, sign):
        out = []
        for _ in range(n):
            if sign > 0:
                pa = rng.uniform(0.1, 0.8)
                pb = pa + delta
            else:
                pa = rng.uniform(0.2, 0.9)
                pb = pa - delta
            ya = sum(rng.random() < pa for _ in range(reps)) / reps
            yb = sum(rng.random() < pb for _ in range(reps)) / reps
            out.append(yb - ya)
        return out
    return gen


def _zero_inflated(rng, n, delta, sign):
    return [sign * 0.5 if rng.random() < 2 * delta else 0.0 for _ in range(n)]


def bernstein_study(reps: int) -> list[dict]:
    shape_only = {
        "symmetric_trunc_normal": _sym_trunc_normal,
        "two_point_70_30_right_skew": _two_point(-0.06, 0.7, 0.14),
        "two_point_70_30_left_skew": _mirror(_two_point(-0.06, 0.7, 0.14)),
        "two_point_85_15_right_skew": _two_point(-0.03, 0.85, 0.17),
        "two_point_85_15_left_skew": _mirror(_two_point(-0.03, 0.85, 0.17)),
        "three_point_60pct_tied_zero": _three_point_zero,
    }
    margin_dependent = {
        "per_item_mean_R3": _per_item_mean(3),
        "per_item_mean_R5": _per_item_mean(5),
        "zero_inflated_near_bounds": _zero_inflated,
    }
    jobs = [(name, g, 0.10, "upper") for name, g in shape_only.items()]
    for name, g in margin_dependent.items():
        for delta in (0.02, 0.05, 0.10, 0.20):
            for bd in ("lower", "upper"):
                jobs.append((name, g, delta, bd))
    rows = []
    alpha = 0.05
    mcse = math.sqrt(alpha * (1 - alpha) / reps)
    for name, g, delta, bd in jobs:
        for n in (30, 50, 100, 200, 500):
            rng = random.Random(f"{MASTER_SEED}|bernstein|{name}|{bd}|{delta}|{n}")
            sign = 1 if bd == "upper" else -1
            rejected = 0
            for _ in range(reps):
                res = tost_empirical_bernstein(g(rng, n, delta, sign), delta, alpha)
                p = res.p_upper if bd == "upper" else res.p_lower
                rejected += p is not None and p <= alpha
            size = rejected / reps
            rows.append({"scenario": name, "boundary": bd, "delta": delta, "n": n, "replicates": reps, "size": size,
                         "passed": size <= alpha + 2 * mcse and size <= 0.06})
        print(f"bernstein {name} {bd} {delta}: max size {max(r['size'] for r in rows[-5:]):.4f}", flush=True)
    return rows


STUDIES = {
    "rank_intervals": lambda quick: rank_study(60 if quick else 300),
    "two_level_bootstrap": lambda quick: two_level_study(60 if quick else 400),
    "power_planner": lambda quick: power_study(500 if quick else 4000),
    "bounded_tost": lambda quick: bernstein_study(200 if quick else 2000),
}


def main() -> int:
    quick = "--quick" in sys.argv
    start = time.time()
    if "--only" in sys.argv:
        # Re-run one study and merge it into the committed results.
        name = sys.argv[sys.argv.index("--only") + 1]
        payload = json.loads(OUT.read_text(encoding="utf-8"))
        payload[name] = STUDIES[name](quick)
        payload["all_passed"] = all(r["passed"] for key in STUDIES for r in payload.get(key, []))
        OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"updated {name} in {OUT}; all passed: {payload['all_passed']}")
        return 0
    payload = {
        "description": "v0.5.0 calibration studies; see the module docstring for the pre-specified pass rules.",
        "master_seed": MASTER_SEED,
        "python": sys.version.split()[0],
        "rank_intervals": rank_study(60 if quick else 300),
        "two_level_bootstrap": two_level_study(60 if quick else 400),
        "power_planner": power_study(500 if quick else 4000),
        "bounded_tost": bernstein_study(200 if quick else 2000),
    }
    payload["all_passed"] = all(r["passed"] for key in ("rank_intervals", "two_level_bootstrap", "power_planner", "bounded_tost") for r in payload[key])
    payload["seconds"] = round(time.time() - start, 1)
    if not quick:
        OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")
    print("ALL PASSED" if payload["all_passed"] else "SOME CONFIGURATIONS FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
