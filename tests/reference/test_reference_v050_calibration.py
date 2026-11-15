"""Checks the committed v0.5.0 calibration results against their pre-specified rules.

The simulation itself is ``_ref_v050_calibration.py`` (several minutes); this test only reads
its committed output, so it runs in the normal suite.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "data" / "v050_calibration_results.json"


def _load():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_rank_intervals_meet_simultaneous_coverage():
    rows = _load()["rank_intervals"]
    assert len(rows) == 6 and all(r["sims"] == 300 for r in rows)
    assert all(r["passed"] and r["simultaneous_coverage"] >= r["threshold"] for r in rows)


def test_power_planner_matches_simulation():
    rows = _load()["power_planner"]
    assert len(rows) == 6 and all(r["passed"] and r["abs_difference"] <= r["tolerance"] for r in rows)


def test_bounded_tost_controls_size_on_the_v040_grid():
    rows = _load()["bounded_tost"]
    assert len(rows) == 150 and all(r["replicates"] == 2000 for r in rows)
    assert all(r["passed"] and r["size"] <= 0.06 for r in rows)


def test_two_level_ci_results_and_documented_follow_up():
    rows = _load()["two_level_bootstrap"]
    assert len(rows) == 6
    failed = [r for r in rows if not r["passed"]]
    # One cell missed its threshold in the pre-specified run; it is kept, not overwritten,
    # and carries a documented post-hoc check (docs/statistical_methodology.md, section 20).
    assert len(failed) <= 1
    for r in failed:
        follow = r["follow_up"]
        assert follow["replicates"] >= 1000
        assert follow["two_level_coverage"] >= 0.94
        assert abs(follow["two_level_coverage"] - follow["item_level_coverage_same_data"]) <= 0.02
