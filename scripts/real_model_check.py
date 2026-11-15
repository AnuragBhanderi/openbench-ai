"""End-to-end workflow check against a real model provider (v0.5.0, issue #16).

Runs the full v0.4.0+ workflow through the CLI and asserts on *structure*, not scores:

1. a seeded repeat group (2 repetitions) and a second single run;
2. ``openbench analyze`` on the two arms (valid JSON report);
3. ``openbench preferences create`` (tasks and key written);
4. ``openbench runs verify <group> --quality-checks`` (no integrity errors).

Used by ``.github/workflows/real-model.yml`` with Ollama; it also runs with ``--provider mock``
so the script itself is tested in the normal suite. Exit code 0 means every check passed.

    python scripts/real_model_check.py --provider ollama --model qwen2.5:0.5b --output real_model_out
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _cli(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "openbench.cli", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"$ openbench {' '.join(args)}\n{proc.stdout}{proc.stderr}", flush=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"FAILED (exit {proc.returncode}): openbench {' '.join(args)}")
    return proc


def run_checks(provider: str, model: str, model_b: str | None, endpoint: str | None, dataset: str, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    exp = out / "experiments"
    common = ["--dataset", dataset, "--provider", provider, "--experiments-dir", str(exp), "--cache-dir", str(out / "cache")]
    if endpoint:
        common += ["--endpoint", endpoint]
    problems: list[str] = []

    _cli(["run", *common, "--model", model, "--repetitions", "2", "--seed", "1"])
    _cli(["run", *common, "--model", model_b or model, "--seed", "2"])

    listing = json.loads(_cli(["runs", "list", "--experiments-dir", str(exp), "--format", "json"]).stdout)
    if len(listing) != 3:
        problems.append(f"expected 3 runs, found {len(listing)}")
    for run in listing:
        if run.get("status") != "completed":
            problems.append(f"{run.get('run_id')} status is {run.get('status')}")
        lineage = run.get("lineage") or {}
        supported = lineage.get("seed_supported_by_adapter")
        if supported and lineage.get("seed_status") != "passed_to_provider":
            problems.append(f"{run.get('run_id')} seed_status is {lineage.get('seed_status')}")
    group_ids = sorted({r["lineage"]["repeat_group_id"] for r in listing if (r.get("lineage") or {}).get("repeat_group_id")})
    singles = [r["run_id"] for r in listing if not (r.get("lineage") or {}).get("repeat_group_id")]
    if len(group_ids) != 1 or len(singles) != 1:
        problems.append(f"expected one repeat group and one single run, got {group_ids} and {singles}")
        return problems
    group_id, single_id = group_ids[0], singles[0]

    report = out / "comparison.json"
    proc = _cli(
        ["analyze", "--runs", single_id, group_id, "--experiments-dir", str(exp), "--format", "json", "--output", str(report)],
        check=False,
    )
    if proc.returncode != 0:
        problems.append(f"analyze exited with {proc.returncode}")
    else:
        try:
            json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"comparison report is not valid JSON: {exc}")

    prefs = out / "preferences"
    # Preference tasks pair two runs: the single run and the group's first repetition.
    first_rep = sorted(r["run_id"] for r in listing if (r.get("lineage") or {}).get("repeat_group_id") == group_id)[0]
    proc = _cli(
        ["preferences", "create", "--runs", single_id, first_rep, "--experiments-dir", str(exp), "--output", str(prefs), "--overwrite"],
        check=False,
    )
    if proc.returncode != 0 or not (prefs / "tasks.jsonl").exists():
        problems.append("preferences create did not write tasks.jsonl")

    audit = out / "verify_audit.json"
    proc = _cli(
        ["runs", "verify", group_id, "--experiments-dir", str(exp), "--quality-checks", "--audit-output", str(audit)],
        check=False,
    )
    if proc.returncode != 0:
        problems.append(f"runs verify exited with {proc.returncode}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-b", default=None, help="Second model (default: the same model with another seed).")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--dataset", default="datasets/reasoning/sample.jsonl")
    parser.add_argument("--output", default="real_model_out")
    args = parser.parse_args(argv)
    problems = run_checks(args.provider, args.model, args.model_b, args.endpoint, args.dataset, Path(args.output))
    if problems:
        print("\nREAL-MODEL CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nREAL-MODEL CHECK PASSED: runs completed, reports valid, integrity verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
