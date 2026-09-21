# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — Benchmark scoring.

Scores a model's completion for one benchmark task by actually running it:
write the extracted code + the task's real test file into a self-hosted
sandbox (the same SandboxManager the production agent uses — see
src/sandbox/manager.py) and execute the real test command. Pass/fail is
whatever the test process's exit code says, not an LLM's opinion of its
own code.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

from src.sandbox.manager import SandboxManager

_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


@dataclasses.dataclass(frozen=True)
class BenchmarkTask:
    id: str
    language: str
    prompt: str
    solution_filename: str
    test_filename: str
    test_code: str
    test_command: str

    @classmethod
    def load(cls, path: Path) -> BenchmarkTask:
        data = json.loads(path.read_text())
        return cls(**data)


@dataclasses.dataclass
class ScoreResult:
    task_id: str
    model_label: str
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    extracted_code: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model_label,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "stderr_tail": self.stderr[-500:] if not self.passed else "",
        }


def load_all_tasks(tasks_dir: Path) -> list[BenchmarkTask]:
    return sorted(
        (BenchmarkTask.load(p) for p in tasks_dir.glob("*.json")),
        key=lambda t: t.id,
    )


def extract_code(model_response: str) -> str:
    """
    Pull the first fenced code block out of a model's response. If the model
    ignored the "only a code fence" instruction and returned bare code with
    no fence, fall back to the raw response — real models do this often
    enough that failing the task outright on formatting would understate
    actual coding quality.
    """
    match = _CODE_FENCE_RE.search(model_response)
    return match.group(1).strip() if match else model_response.strip()


async def score_completion(
    task: BenchmarkTask,
    model_response: str,
    model_label: str,
    *,
    tenant_id: str = "benchmark",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    manager: SandboxManager | None = None,
) -> ScoreResult:
    """
    Real scoring: write the extracted solution + the task's test file into a
    fresh self-hosted sandbox, run the real test command, report the actual
    exit code. No LLM-as-judge, no static heuristic — the tests either pass
    or they don't.
    """
    code = extract_code(model_response)
    owns_manager = manager is None
    manager = manager or SandboxManager()

    handle = await manager.create(template=task.language, tenant_id=tenant_id)
    try:
        await manager.write_file(handle, task.solution_filename, code)
        await manager.write_file(handle, task.test_filename, task.test_code)
        result = await manager.execute(handle, task.test_command, timeout=60)
    finally:
        await manager.destroy(handle)
        if owns_manager:
            await manager.aclose()

    return ScoreResult(
        task_id=task.id,
        model_label=model_label,
        passed=result["exit_code"] == 0,
        exit_code=result["exit_code"],
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        duration_ms=result.get("duration_ms", 0),
        extracted_code=code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
