# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Regenerate docs/benchmarks/latest.md from the persisted benchmark runs.

    python -m benchmarks.report                 # writes docs/benchmarks/latest.md
    python -m benchmarks.report --days 30       # only runs from the last 30 days
    python -m benchmarks.report --stdout        # print instead of writing

For each backend label the report shows the solve rate over the LATEST run
of each task (an older attempt at the same task is superseded, not
averaged in), the tokens and wall time those latest runs took, and the
per-task grid. Cost per backend is shown only where MODEL_PRICES_PER_MILLION
prices that label's role; nothing is invented. The rendering is pure over
`BenchmarkRun` rows so it is unit-testable without a database.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.runs import load_runs
from src.billing.ledger import price_per_million
from src.db.models import BenchmarkRun

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "latest.md"


def latest_per_task(runs: list[BenchmarkRun]) -> dict[str, dict[str, BenchmarkRun]]:
    """{backend_label: {task_id: latest run}} — `runs` must be oldest-first."""
    latest: dict[str, dict[str, BenchmarkRun]] = defaultdict(dict)
    for run in runs:
        latest[run.backend_label][run.task_id] = run
    return latest


def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


def _fmt_ms(ms: int) -> str:
    return f"{ms / 1000:.0f}s" if ms else "—"


def render(runs: list[BenchmarkRun], *, generated_at: datetime | None = None) -> str:
    """Markdown for docs/benchmarks/latest.md from BenchmarkRun rows (oldest-first)."""
    when = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Benchmark results",
        "",
        f"_Generated {when} by `python -m benchmarks.report` from the `benchmark_runs` table — "
        "every number below is a persisted row, reproducible with `benchmarks/agent_runner.py --persist`._",
        "",
    ]
    if not runs:
        lines += [
            "No benchmark runs have been persisted yet. Run:",
            "",
            "```bash",
            "python -m benchmarks.agent_runner --persist --backend-label coding",
            "```",
            "",
        ]
        return "\n".join(lines)

    by_backend = latest_per_task(runs)
    task_ids = sorted({r.task_id for r in runs})

    lines += [
        "## Solve rate per backend (latest run of each task)",
        "",
        "| Backend | Model | Solved | Tasks | Prompt tokens | Completion tokens | Wall time | Cost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in sorted(by_backend):
        latest = by_backend[label]
        solved = sum(1 for r in latest.values() if r.solved)
        prompt = sum(r.prompt_tokens for r in latest.values())
        completion = sum(r.completion_tokens for r in latest.values())
        wall = sum(r.duration_ms for r in latest.values())
        model = next((r.model_id for r in latest.values() if r.model_id), None) or "—"
        price = price_per_million(label)
        cost = f"${(prompt + completion) / 1_000_000 * price:.4f}" if price is not None else "not priced"
        lines.append(
            f"| `{label}` | `{model}` | **{solved}/{len(latest)}** ({solved / len(latest):.0%}) | {len(latest)} | "
            f"{_fmt_tokens(prompt)} | {_fmt_tokens(completion)} | {_fmt_ms(wall)} | {cost} |"
        )

    lines += [
        "",
        "## Per task",
        "",
        "| Task | " + " | ".join(f"`{b}`" for b in sorted(by_backend)) + " |",
        "|---|" + "---|" * len(by_backend),
    ]
    for task_id in task_ids:
        cells = []
        for label in sorted(by_backend):
            run = by_backend[label].get(task_id)
            if run is None:
                cells.append("—")
            elif run.solved:
                cells.append(
                    f"✅ [{_fmt_ms(run.duration_ms)}]({run.pr_url})" if run.pr_url else f"✅ {_fmt_ms(run.duration_ms)}"
                )
            else:
                reason = run.status if run.status != "completed" else "tests failed"
                cells.append(f"❌ {reason}")
        lines.append(f"| `{task_id}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## How to read this",
        "",
        "- **Solved** means every `fail_to_pass` test now passes AND every `pass_to_pass` test still passes, "
        "re-run on a fresh clone of the branch the agent pushed — not the agent's own claim.",
        "- Tokens and wall time are for the latest run of each task, summed per backend.",
        "- Cost uses `MODEL_PRICES_PER_MILLION` for the backend label's role; a frontier backend is billed by "
        "its vendor and is shown as not priced unless you set a price for that label.",
        "- The full runner result for every row (branch, PR, per-test outcomes, errors) is in "
        "`benchmark_runs.details`.",
        "",
    ]
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=None, help="Only runs from the last N days")
    parser.add_argument("--stdout", action="store_true", help="Print the report instead of writing it")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    return parser


async def _main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    runs = await load_runs(days=args.days)
    text = render(runs)
    if args.stdout:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output} ({len(runs)} runs)")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    sys.exit(main())
