# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — Quality + cost benchmark runner.

Runs the task set in benchmarks/tasks/ through Keystone Inference and (when
API keys are configured) frontier models, scores each completion by
actually executing it in a real self-hosted sandbox, and reports pass rate
+ token cost comparison side by side.

Usage:
  python -m benchmarks.run_benchmark                    # Keystone Inference only
  python -m benchmarks.run_benchmark --with-frontier     # + any configured frontier models
  python -m benchmarks.run_benchmark --gpu-count 4 --gpu-hourly-cost 1.89 --gpu-tokens-per-second 4000

Requires: SANDBOX_DAEMON_URL reachable (see src/sandbox/daemon.py), and
either VLLM_CODING_URL reachable or --with-frontier with an API key set —
running with neither just reports "no models available" rather than
fabricating results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from benchmarks.cost_model import CostComparison, GPUCostProfile
from benchmarks.model_clients import (
    CompletionClient,
    KeystoneInferenceClient,
    available_frontier_clients,
)
from benchmarks.scoring import BenchmarkTask, load_all_tasks, score_completion
from src.sandbox.manager import SandboxManager

TASKS_DIR = Path(__file__).parent / "tasks"


async def _run_one(task: BenchmarkTask, label: str, client: CompletionClient, manager: SandboxManager) -> dict:
    try:
        completion = await client.complete(task.prompt)
    except Exception as exc:
        return {"task_id": task.id, "model": label, "passed": False, "error": f"completion_failed: {exc}"}

    result = await score_completion(
        task,
        completion.text,
        label,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        manager=manager,
    )
    return result.to_dict()


async def run_benchmark(
    *,
    with_frontier: bool,
    keystone_role: str,
    gpu_profile: GPUCostProfile,
) -> dict:
    tasks = load_all_tasks(TASKS_DIR)
    if not tasks:
        raise RuntimeError(f"No benchmark tasks found in {TASKS_DIR}")

    clients: dict[str, CompletionClient] = {"keystone-inference": KeystoneInferenceClient(keystone_role)}
    if with_frontier:
        clients.update(available_frontier_clients())
        if len(clients) == 1:
            print(
                "--with-frontier set but no ANTHROPIC_API_KEY/OPENAI_API_KEY found — comparing "
                "Keystone Inference against itself only.",
                file=sys.stderr,
            )

    manager = SandboxManager()
    results: list[dict] = []
    try:
        for task in tasks:
            for label, client in clients.items():
                outcome = await _run_one(task, label, client, manager)
                results.append(outcome)
                status = "PASS" if outcome.get("passed") else "FAIL"
                print(f"[{status}] {task.id} / {label}")
    finally:
        await manager.aclose()

    # Cost comparison: use each result's actual measured token counts against
    # every known frontier price, regardless of whether that frontier model
    # was actually queried — this answers "what WOULD it have cost", which is
    # the point of the comparison even when only Keystone Inference ran.
    cost_rows = []
    for r in results:
        if r.get("model") != "keystone-inference" and not r.get("model", "").startswith("keystone-inference"):
            continue
        prompt_tokens, completion_tokens = r.get("prompt_tokens", 0), r.get("completion_tokens", 0)
        if prompt_tokens == 0 and completion_tokens == 0:
            continue
        for frontier_key in ("claude-opus", "claude-sonnet", "gpt-4o", "gpt-4o-mini"):
            comparison = CostComparison(prompt_tokens, completion_tokens, gpu_profile, frontier_key)
            cost_rows.append({"task_id": r["task_id"], **comparison.to_dict()})

    pass_rate_by_model: dict[str, dict] = {}
    for r in results:
        m = r["model"]
        bucket = pass_rate_by_model.setdefault(m, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(bool(r.get("passed")))

    return {
        "results": results,
        "pass_rate_by_model": {
            m: {"passed": b["passed"], "total": b["total"], "pass_rate": round(b["passed"] / b["total"], 3)}
            for m, b in pass_rate_by_model.items()
        },
        "cost_comparison": cost_rows,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-frontier", action="store_true", help="Also query configured frontier models")
    parser.add_argument("--keystone-role", default="coding", help="Model role to use for Keystone Inference")
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--gpu-hourly-cost", type=float, default=1.89, help="USD/hour per GPU")
    parser.add_argument("--gpu-tokens-per-second", type=float, default=4000.0, help="Measured aggregate throughput")
    parser.add_argument("--gpu-utilization", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None, help="Write full JSON report to this path")
    return parser


async def _main() -> int:
    args = _build_arg_parser().parse_args()
    gpu_profile = GPUCostProfile(
        gpu_count=args.gpu_count,
        hourly_cost_usd_per_gpu=args.gpu_hourly_cost,
        measured_tokens_per_second=args.gpu_tokens_per_second,
        utilization=args.gpu_utilization,
    )

    report = await run_benchmark(
        with_frontier=args.with_frontier,
        keystone_role=args.keystone_role,
        gpu_profile=gpu_profile,
    )

    print("\n=== Pass rate by model ===")
    for model, stats in report["pass_rate_by_model"].items():
        print(f"  {model}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.0%})")

    if report["cost_comparison"]:
        print("\n=== Cost comparison (Keystone Inference vs. frontier, per task) ===")
        for row in report["cost_comparison"]:
            print(
                f"  {row['task_id']} vs {row['frontier_model']}: "
                f"self-hosted ${row['self_hosted_cost_usd']:.6f} vs frontier ${row['frontier_cost_usd']:.6f} "
                f"({row['savings_multiple']}x)"
            )

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
