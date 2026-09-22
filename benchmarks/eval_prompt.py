# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — ad hoc single-input eval.

benchmarks/run_benchmark.py compares models across the fixed task suite in
benchmarks/tasks/. This module answers a narrower, more common question:
"how does Keystone Inference do on THIS exact prompt, right now, versus a
frontier model" — without writing a benchmark task file first.

Two modes, both real, no LLM-as-judge:

  Unscored (no test provided): each configured model gets the same prompt,
  and the raw completions + real token counts + real cost comparison are
  reported side by side. There is no pass/fail here — that's the honest
  behavior for an arbitrary prompt with no way to verify correctness.

  Scored (--test-code-file + --test-command given): the exact same
  extraction + real-sandbox-execution path benchmarks/scoring.py uses for
  the fixed task suite, built on the fly from CLI args instead of a
  benchmarks/tasks/*.json file. Real pass/fail, from a real exit code.

Usage:
  # Unscored — compare raw completions and cost for one prompt
  python -m benchmarks.eval_prompt --prompt "Write a Python LRU cache with O(1) get/put" --with-frontier

  # Scored — real pass/fail via a real sandboxed test
  python -m benchmarks.eval_prompt \
    --prompt-file task_prompt.txt \
    --language python --test-code-file test_solution.py \
    --test-command "pytest test_solution.py" \
    --with-frontier

Requires: either VLLM_CODING_URL reachable or --with-frontier with a
frontier API key set — with neither, Keystone Inference's own completion
call fails and is reported as an error, not fabricated. Scored mode also
needs SANDBOX_DAEMON_URL reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from benchmarks.cost_model import CostComparison, GPUCostProfile
from benchmarks.model_clients import CompletionClient, KeystoneInferenceClient, available_frontier_clients
from benchmarks.scoring import BenchmarkTask, score_completion
from src.sandbox.manager import SandboxManager

_DEFAULT_SOLUTION_FILENAMES = {
    "python": "solution.py",
    "javascript": "solution.js",
    "typescript": "solution.ts",
    "go": "solution.go",
}


def build_adhoc_task(
    *,
    prompt: str,
    language: str,
    test_code: str,
    test_command: str,
    solution_filename: str | None,
    test_filename: str | None,
) -> BenchmarkTask:
    return BenchmarkTask(
        id="adhoc",
        language=language,
        prompt=prompt,
        solution_filename=solution_filename or _DEFAULT_SOLUTION_FILENAMES.get(language, "solution.txt"),
        test_filename=test_filename or f"test_{_DEFAULT_SOLUTION_FILENAMES.get(language, 'solution.txt')}",
        test_code=test_code,
        test_command=test_command,
    )


async def _eval_one_unscored(label: str, client: CompletionClient, prompt: str) -> dict:
    try:
        completion = await client.complete(prompt)
    except Exception as exc:
        return {"model": label, "error": str(exc)}
    return {
        "model": label,
        "completion": completion.text,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
    }


async def _eval_one_scored(label: str, client: CompletionClient, task: BenchmarkTask, manager: SandboxManager) -> dict:
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


def _build_clients(*, with_frontier: bool, keystone_role: str, models: list[str] | None) -> dict[str, CompletionClient]:
    clients: dict[str, CompletionClient] = {"keystone-inference": KeystoneInferenceClient(keystone_role)}
    if with_frontier:
        clients.update(available_frontier_clients())
        if len(clients) == 1:
            print(
                "--with-frontier set but no ANTHROPIC_API_KEY/OPENAI_API_KEY found — comparing "
                "Keystone Inference against itself only.",
                file=sys.stderr,
            )
    if models:
        unknown = set(models) - set(clients)
        if unknown:
            raise ValueError(f"--models named {sorted(unknown)}, not in {sorted(clients)} — check spelling/API keys")
        clients = {k: v for k, v in clients.items() if k in models}
    return clients


async def run_eval(
    *,
    prompt: str,
    task: BenchmarkTask | None,
    with_frontier: bool,
    keystone_role: str,
    models: list[str] | None,
    gpu_profile: GPUCostProfile,
) -> dict:
    clients = _build_clients(with_frontier=with_frontier, keystone_role=keystone_role, models=models)

    results: list[dict] = []
    if task is None:
        for label, client in clients.items():
            results.append(await _eval_one_unscored(label, client, prompt))
    else:
        manager = SandboxManager()
        try:
            for label, client in clients.items():
                results.append(await _eval_one_scored(label, client, task, manager))
        finally:
            await manager.aclose()

    # Only Keystone Inference results get a cost comparison row — the
    # question this answers is "what would frontier pricing have cost for
    # the tokens Keystone actually used," which is meaningless applied to a
    # frontier result's own tokens (see run_benchmark.py's identical guard).
    cost_rows = []
    for r in results:
        if not r.get("model", "").startswith("keystone-inference"):
            continue
        prompt_tokens, completion_tokens = r.get("prompt_tokens", 0), r.get("completion_tokens", 0)
        if prompt_tokens == 0 and completion_tokens == 0:
            continue
        for frontier_key in ("claude-opus", "claude-sonnet", "gpt-4o", "gpt-4o-mini"):
            comparison = CostComparison(prompt_tokens, completion_tokens, gpu_profile, frontier_key)
            cost_rows.append({"model": r["model"], **comparison.to_dict()})

    return {"scored": task is not None, "results": results, "cost_comparison": cost_rows}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="The prompt text, inline")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a file containing the prompt text")

    parser.add_argument("--with-frontier", action="store_true", help="Also query configured frontier models")
    parser.add_argument("--keystone-role", default="coding", help="Model role to use for Keystone Inference")
    parser.add_argument(
        "--models",
        help="Comma-separated subset of model labels to run (default: all configured) — "
        "e.g. keystone-inference,claude-opus",
    )

    parser.add_argument("--language", default="python", help="Only used with --test-code-file (scored mode)")
    parser.add_argument("--test-code-file", type=Path, help="Real test file's contents — enables scored mode")
    parser.add_argument("--test-command", help="Real command to run the test (e.g. 'pytest test_solution.py')")
    parser.add_argument("--solution-filename", default=None)
    parser.add_argument("--test-filename", default=None)

    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--gpu-hourly-cost", type=float, default=1.89, help="USD/hour per GPU")
    parser.add_argument("--gpu-tokens-per-second", type=float, default=4000.0, help="Measured aggregate throughput")
    parser.add_argument("--gpu-utilization", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None, help="Write full JSON report to this path")
    return parser


async def _main() -> int:
    args = _build_arg_parser().parse_args()
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text()

    if bool(args.test_code_file) != bool(args.test_command):
        print("--test-code-file and --test-command must be given together (or neither, for unscored mode)")
        return 2

    task = None
    if args.test_code_file:
        task = build_adhoc_task(
            prompt=prompt,
            language=args.language,
            test_code=args.test_code_file.read_text(),
            test_command=args.test_command,
            solution_filename=args.solution_filename,
            test_filename=args.test_filename,
        )

    gpu_profile = GPUCostProfile(
        gpu_count=args.gpu_count,
        hourly_cost_usd_per_gpu=args.gpu_hourly_cost,
        measured_tokens_per_second=args.gpu_tokens_per_second,
        utilization=args.gpu_utilization,
    )

    report = await run_eval(
        prompt=prompt,
        task=task,
        with_frontier=args.with_frontier,
        keystone_role=args.keystone_role,
        models=args.models.split(",") if args.models else None,
        gpu_profile=gpu_profile,
    )

    print(f"\n=== {'Scored' if report['scored'] else 'Unscored'} eval, {len(report['results'])} model(s) ===")
    for r in report["results"]:
        if "error" in r:
            print(f"\n[{r['model']}] ERROR: {r['error']}")
            continue
        if report["scored"]:
            status = "PASS" if r.get("passed") else "FAIL"
            print(f"\n[{status}] {r['model']}  ({r['prompt_tokens']}+{r['completion_tokens']} tokens)")
            if not r.get("passed") and r.get("stderr_tail"):
                print(f"  stderr: {r['stderr_tail'][:300]}")
        else:
            print(f"\n--- {r['model']}  ({r['prompt_tokens']}+{r['completion_tokens']} tokens) ---")
            print(r["completion"])

    if report["cost_comparison"]:
        print("\n=== Cost comparison (per model, what the same tokens would cost each frontier vendor) ===")
        for row in report["cost_comparison"]:
            print(
                f"  {row['model']} vs {row['frontier_model']}: "
                f"self-hosted ${row['self_hosted_cost_usd']:.6f} vs frontier ${row['frontier_cost_usd']:.6f} "
                f"({row['savings_multiple']}x)"
            )

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
