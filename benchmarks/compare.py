# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Sweep the repo-task suite across several model backends and persist every
run, then regenerate docs/benchmarks/latest.md — the "base vs base+RAG vs
fine-tuned vs frontier" comparison, produced by the same harness for each.

    python -m benchmarks.compare \\
        --backend coding=http://vllm-coding:8000/v1:zai-org/GLM-5.3-Flash \\
        --backend frontier=http://localhost:8091/v1:frontier \\
        --backend my-adapter=http://vllm-coding:8000/v1:my-adapter

Each `--backend LABEL=URL[:MODEL_ID]` points the coding role at that
OpenAI-compatible endpoint for one full pass of the suite (the same real
agent loop, Gitea, sandbox and held-out verification every time), tagged
with LABEL in `benchmark_runs`. Settings and the inference client cache
are reset between backends so nothing leaks from one pass to the next.
Requires the same live services as benchmarks/agent_runner.py.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass

from benchmarks import report
from benchmarks.agent_runner import run_suite
from benchmarks.runs import persist_results


@dataclass(frozen=True)
class Backend:
    label: str
    url: str
    model_id: str | None

    @classmethod
    def parse(cls, spec: str) -> Backend:
        """LABEL=URL[:MODEL_ID] — the model id is whatever comes after the URL's `/v1`."""
        if "=" not in spec:
            raise argparse.ArgumentTypeError(f"--backend must be LABEL=URL[:MODEL_ID], got {spec!r}")
        label, rest = spec.split("=", 1)
        url, _, model_id = rest.partition("/v1:")
        url = url + "/v1" if model_id else rest
        return cls(label=label.strip(), url=url.strip(), model_id=model_id.strip() or None)


def _point_coding_role_at(backend: Backend) -> None:
    os.environ["VLLM_CODING_URL"] = backend.url
    if backend.model_id:
        os.environ["CODING_MODEL_ID"] = backend.model_id
    from src.config import get_settings
    from src.inference import client as inference_client

    get_settings.cache_clear()
    inference_client._clients.clear()


async def sweep(backends: list[Backend], *, only: str | None, max_iterations: int, timeout_seconds: int) -> None:
    for backend in backends:
        print(f"\n##### Backend {backend.label!r}: {backend.url} model={backend.model_id or '(configured)'} #####")
        _point_coding_role_at(backend)
        results = await run_suite(only, max_iterations, timeout_seconds)
        from src.config import get_settings

        ids = await persist_results(
            results, backend_label=backend.label, model_id=backend.model_id or get_settings().coding_model_id
        )
        solved = sum(1 for r in results if r.get("solved"))
        print(f"##### {backend.label}: {solved}/{len(results)} solved, {len(ids)} runs persisted #####")
    await report._main([])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", action="append", type=Backend.parse, required=True, metavar="LABEL=URL[:MODEL]")
    parser.add_argument("--task", default=None, help="Run only this task id")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    asyncio.run(
        sweep(args.backend, only=args.task, max_iterations=args.max_iterations, timeout_seconds=args.timeout_seconds)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
