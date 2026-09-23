#!/usr/bin/env python3
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

"""
Record the VS Code extension's test fixtures from the real routes.

Writes vscode/extension/src/__tests__/fixtures/task-stream.sse: the exact bytes
`GET /v1/keystone/tasks/{id}/stream` (src/api/routes/agents.py) sends for one
task, captured through the real FastAPI app over ASGITransport, fed by the real
publishers in src/orchestrator/events.py over a real Redis Stream — so the frame
format (`id:` lines, `data:` JSON, blank-line separators, the closing `final`
event) is what the extension's parser meets in production, not a hand-written
imitation of it. And vscode/extension/src/__tests__/fixtures/model-library.json:
the real `GET /v1/keystone/models` response (src/inference/library.py) with the
`coding` role served by tests/fixtures/fake_vllm_server.py (probed READY) and
the other roles pointed at a dead port (probed UNHEALTHY) — the mix the
extension's model picker orders.

Every step payload is shaped exactly as the publishing call site emits it:
  route_decision  src/orchestrator/engine.py::submit_task
  deps_install    src/orchestrator/nodes/_shared.py::install_dependencies
  repo_map        src/orchestrator/nodes/_shared.py::ensure_repo_map
  tool_call / tool_result / model_text
                  src/orchestrator/nodes/coding.py::_run_agentic_loop
  quality_findings src/orchestrator/nodes/quality.py
  test_output     src/orchestrator/nodes/testing.py
  diff / pr       src/orchestrator/engine.py::_run_git_workflow
  node            src/orchestrator/events.py::_summarize / publish_task_final
The tool_call / tool_result / model_text / repo_map frames are copied from a
real coding-loop run (tests/test_coding_agentic_loop.py against the fake vLLM
fixture), including the failed first apply_patch. What this is NOT: a recording
of a live model writing code — that needs a GPU or the frontier proxy.

Needs the integration environment (DATABASE_URL, REDIS_URL — tests/conftest.py).
Usage:  PYTHONPATH=. python scripts/record_task_stream_fixture.py [--output PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api.middleware.auth import generate_api_key  # noqa: E402
from src.api.middleware.rate_limiter import get_redis  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.connection import get_db_context  # noqa: E402
from src.db.models import APIKey, Tenant, TenantTier  # noqa: E402
from src.inference.health import endpoint_health  # noqa: E402
from src.main import create_app  # noqa: E402
from src.orchestrator.events import publish_task_event, publish_task_final, publish_task_step  # noqa: E402

FIXTURES = ROOT / "vscode" / "extension" / "src" / "__tests__" / "fixtures"
DEFAULT_OUTPUT = FIXTURES / "task-stream.sse"
DEFAULT_LIBRARY_OUTPUT = FIXTURES / "model-library.json"

REPO_URL = "https://git.example.internal/acme/greeter"
BRANCH = "keystone/ada/3f2a9c1e"
COMMIT = "9b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c"
PR_URL = "https://git.example.internal/acme/greeter/pulls/42"

DIFF = """diff --git a/app.py b/app.py
index 3b18e51..a1f0c2d 100644
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def greet(name):
-    return f"hello {name}"
+    return f"HI {name}"
"""

TEST_STDOUT = """============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.2, pluggy-1.5.0
rootdir: /workspace
collected 2 items

tests/test_app.py ..                                                     [100%]

============================== 2 passed in 0.03s ===============================
"""


def _node_state(phase: str, **overrides: object) -> dict:
    """The fields src/orchestrator/graph.py::_state_to_dict hands to events.py::_summarize."""
    state: dict = {
        "phase": phase,
        "iteration": 1,
        "max_iterations": 15,
        "plan": "Change greet() to return an upper-case greeting and keep the existing test passing.",
        "plan_steps": ["Read app.py", "Patch greet()", "Run the tests"],
        "current_plan_step": 0,
        "file_changes": [],
        "review_passed": None,
        "review_comments": [],
        "tests_passed": None,
        "test_results": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "result_summary": "",
        "error_message": None,
    }
    state.update(overrides)
    return state


async def _publish_one_task(task_id: uuid.UUID) -> None:
    # engine.submit_task — model="auto" resolved before the task was persisted
    await publish_task_step(
        task_id,
        "route_decision",
        "submit",
        {"requested": "auto", "resolved_role": "coding", "complexity_routing_enabled": False},
        phase="pending",
    )
    # planning: clone, install, repo map, plan
    await publish_task_step(
        task_id,
        "deps_install",
        "planning",
        {
            "command": "pip install -r requirements.txt",
            "ok": True,
            "exit_code": 0,
            "output": "Requirement already satisfied: pytest in /usr/lib/python3/dist-packages (8.3.2)\n",
            "duration_ms": 1412,
        },
    )
    await publish_task_step(
        task_id,
        "repo_map",
        "planning",
        {
            "map": "Repository map (1 files, 1 symbols; (refs N) = references across the repo):\n"
            "app.py:\n  1: function greet(name)"
        },
    )
    await publish_task_event(task_id, "planning", _node_state("coding"))
    # coding: the real agentic loop's frames (a failed apply_patch, then the fix, then the final message)
    await publish_task_step(
        task_id,
        "tool_call",
        "coding",
        {"turn": 0, "name": "apply_patch", "arguments": {"path": "app.py", "search": "NOPE NOT THERE", "replace": "x"}},
        phase="coding",
    )
    await publish_task_step(
        task_id,
        "tool_result",
        "coding",
        {
            "turn": 0,
            "name": "apply_patch",
            "ok": False,
            "output": "",
            "error": "The `search` text was not found in 'app.py' (not even approximately). Re-read the file — it may "
            "have changed since you last saw it — and copy the exact text (including indentation) you want replaced.",
        },
        phase="coding",
    )
    await publish_task_step(
        task_id,
        "tool_call",
        "coding",
        {
            "turn": 1,
            "name": "apply_patch",
            "arguments": {"path": "app.py", "search": "hello {name}", "replace": "HI {name}"},
        },
        phase="coding",
    )
    await publish_task_step(
        task_id,
        "tool_result",
        "coding",
        {"turn": 1, "name": "apply_patch", "ok": True, "output": "Updated app.py (41 bytes)", "error": ""},
        phase="coding",
    )
    await publish_task_step(
        task_id,
        "model_text",
        "coding",
        {"turn": 2, "content": "Fixed after the first search didn't match.", "final": True},
        phase="coding",
    )
    coding_state = _node_state(
        "quality",
        current_plan_step=3,
        file_changes=[{"path": "app.py"}],
        total_prompt_tokens=2210,
        total_completion_tokens=188,
        result_summary="Fixed after the first search didn't match.",
    )
    await publish_task_event(task_id, "coding", coding_state)
    # quality gates: one informational ruff finding, nothing blocking
    await publish_task_step(
        task_id,
        "quality_findings",
        "quality",
        {
            "commands": ["ruff check --output-format json .", "bandit -q -r -f json ."],
            "blocking_tools": ["bandit", "mypy"],
            "findings": [
                {
                    "tool": "ruff",
                    "path": "app.py",
                    "line": 1,
                    "severity": "warning",
                    "code": "D103",
                    "message": "Missing docstring in public function",
                }
            ],
            "blocking": [],
        },
    )
    await publish_task_event(task_id, "quality", {**coding_state, "phase": "review"})
    await publish_task_event(
        task_id, "review", {**coding_state, "phase": "testing", "review_passed": True, "total_completion_tokens": 402}
    )
    # testing: related tests first, then the full suite
    for name, command in (("related", "python -m pytest tests/test_app.py -q"), ("all", "python -m pytest -q")):
        await publish_task_step(
            task_id,
            "test_output",
            "testing",
            {
                "name": name,
                "command": command,
                "exit_code": 0,
                "passed": True,
                "stdout": TEST_STDOUT,
                "stderr": "",
                "duration_ms": 640,
            },
        )
    await publish_task_event(
        task_id,
        "testing",
        {
            **coding_state,
            "phase": "complete",
            "review_passed": True,
            "tests_passed": True,
            "test_results": [{"passed": True}, {"passed": True}],
            "total_completion_tokens": 402,
        },
    )
    # engine._run_git_workflow: commit, push, PR — then the final event closes the stream
    await publish_task_step(
        task_id, "diff", "finalize", {"diff": DIFF, "branch_name": BRANCH, "commit_sha": COMMIT}, phase="complete"
    )
    await publish_task_step(task_id, "pr", "finalize", {"pr_url": PR_URL, "pr_number": 42}, phase="complete")
    await publish_task_final(
        task_id,
        "complete",
        result_summary="Fixed after the first search didn't match.",
        git_result={"branch_name": BRANCH, "commit_sha": COMMIT, "pr_url": PR_URL, "pr_number": 42},
    )


@asynccontextmanager
async def _fake_vllm_server() -> AsyncIterator[str]:
    """tests/fixtures/fake_vllm_server.py as a real subprocess (the same fixture tests/conftest.py
    starts): a real OpenAI-compatible HTTP server for the `coding` role, so the model library's
    health probe records a genuine READY next to the dead ports' UNHEALTHY."""
    fixture = ROOT / "tests" / "fixtures" / "fake_vllm_server.py"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "uvicorn",
        "fake_vllm_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        cwd=str(fixture.parent),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient() as probe:
            for _ in range(60):
                try:
                    await probe.get(f"{base_url}/last_payload", timeout=1.0)
                    break
                except httpx.TransportError:
                    await asyncio.sleep(0.1)
            else:
                raise RuntimeError("fake_vllm_server did not start in time")
        yield base_url
    finally:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5)


def _point_roles_at(coding_url: str) -> None:
    """coding → the fake server (READY); the other roles → a port nothing listens on (UNHEALTHY)."""
    os.environ["VLLM_CODING_URL"] = coding_url
    os.environ["CODING_MODEL_ID"] = "Qwen/Qwen2.5-Coder-7B-Instruct"
    os.environ["VLLM_CODING_FALLBACK_URL"] = "http://127.0.0.1:1/v1"
    os.environ["CODING_FALLBACK_MODEL_ID"] = "Qwen/Qwen2.5-Coder-32B-Instruct"
    os.environ["VLLM_REASONING_URL"] = "http://127.0.0.1:1/v1"
    os.environ["INFERENCE_HEALTH_CACHE_SECONDS"] = "0"
    get_settings.cache_clear()
    endpoint_health.reset()


async def _record(stream_output: Path, library_output: Path) -> int:
    task_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    full_key, prefix, key_hash = generate_api_key()
    async with get_db_context() as db:
        db.add(Tenant(id=tenant_id, name="sse-fixture-recorder", email=f"{tenant_id}@test.dev", tier=TenantTier.FREE))
        await db.flush()
        db.add(
            APIKey(
                tenant_id=tenant_id,
                name="recorder",
                key_prefix=prefix,
                key_hash=key_hash,
                scopes=["agent", "inference"],
            )
        )
        await db.flush()
    headers = {"Authorization": f"Bearer {full_key}"}
    try:
        await _publish_one_task(task_id)
        async with _fake_vllm_server() as coding_url:
            _point_roles_at(coding_url)
            app = create_app()
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18080") as client:
                    async with client.stream("GET", f"/v1/keystone/tasks/{task_id}/stream", headers=headers) as resp:
                        if resp.status_code != 200:
                            print(f"stream route answered {resp.status_code}", file=sys.stderr)
                            return 1
                        raw = b"".join([chunk async for chunk in resp.aiter_bytes()])
                    library = await client.get("/v1/keystone/models", headers=headers)
                    if library.status_code != 200:
                        print(f"model library route answered {library.status_code}: {library.text}", file=sys.stderr)
                        return 1
        await asyncio.to_thread(_write, stream_output, raw)
        await asyncio.to_thread(_write, library_output, json.dumps(library.json(), indent=2).encode() + b"\n")
        frames = raw.count(b"\n\n")
        states = {m["id"]: m["state"] for m in library.json()["models"] if m["kind"] == "role"}
        print(f"wrote {stream_output} ({len(raw)} bytes, {frames} frames, task {task_id})")
        print(f"wrote {library_output} (role states: {states})")
        return 0
    finally:
        redis = await get_redis()
        await redis.delete(f"keystone:task:{task_id}:events")
        async with get_db_context() as db:
            tenant = await db.get(Tenant, tenant_id)
            if tenant is not None:
                await db.delete(tenant)


def _write(output: Path, raw: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="the SSE fixture path")
    parser.add_argument("--library-output", type=Path, default=DEFAULT_LIBRARY_OUTPUT, help="the model-library path")
    args = parser.parse_args()
    return asyncio.run(_record(args.output, args.library_output))


if __name__ == "__main__":
    sys.exit(main())
