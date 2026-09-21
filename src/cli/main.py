# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — command-line client.

Most commands are a real HTTP client against a running Keystone
deployment's API — never talk to Postgres directly, so they work from any
machine (a developer's laptop, an OpenCode session via
`cli/opencode/commands/memory.md`, CI) that can reach the API, the same
way the OpenAI-compatible endpoints do. Configured the same way
`cli/opencode.config.json` already is:

  KEYSTONE_INFERENCE_URL   base URL, e.g. http://localhost:8080
  KEYSTONE_API_KEY         a real ks-... API key with the "agent" scope

`doctor` is the one exception — it diagnoses a deployment that may not be
up *yet*, so it connects to configured infrastructure directly (real
Postgres/Redis/Qdrant/sandbox-daemon/model-endpoint/git-host checks —
see src/cli/doctor.py) using this same machine's real environment/.env,
not the API.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from src.cli.doctor import CheckStatus, run_all_checks
from src.config import get_settings

app = typer.Typer(add_completion=False, help="Keystone — self-hosted LLM inference & autonomous coding agents.")
memory_app = typer.Typer(add_completion=False, help="Inspect and manage what the agent has learned.")
finetune_app = typer.Typer(add_completion=False, help="Start and manage fine-tuning jobs.")
app.add_typer(memory_app, name="memory")
app.add_typer(finetune_app, name="finetune")

console = Console()
error_console = Console(stderr=True, style="bold red")


def _base_url() -> str:
    url = os.environ.get("KEYSTONE_INFERENCE_URL")
    if not url:
        error_console.print("KEYSTONE_INFERENCE_URL is not set — point it at your Keystone deployment.")
        raise typer.Exit(code=1)
    return url.rstrip("/")


def _api_key() -> str:
    key = os.environ.get("KEYSTONE_API_KEY")
    if not key:
        error_console.print("KEYSTONE_API_KEY is not set — see `make api-key` or the admin UI.")
        raise typer.Exit(code=1)
    return key


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=_base_url(),
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=30.0,
    )


def _request(method: str, path: str, **kwargs) -> httpx.Response:
    try:
        with _client() as client:
            resp = client.request(method, path, **kwargs)
    except httpx.ConnectError as exc:
        error_console.print(f"Could not reach {_base_url()}: {exc}")
        raise typer.Exit(code=1) from exc

    if resp.status_code == 401:
        error_console.print("Unauthorized — check KEYSTONE_API_KEY.")
        raise typer.Exit(code=1)
    if resp.status_code == 404:
        error_console.print("Not found.")
        raise typer.Exit(code=1)
    if resp.status_code >= 400:
        error_console.print(f"Request failed ({resp.status_code}): {resp.text}")
        raise typer.Exit(code=1)
    return resp


def _request_one(method: str, path: str, **kwargs) -> dict:
    """For endpoints that return a single memory object."""
    return _request(method, path, **kwargs).json()


def _request_many(method: str, path: str, **kwargs) -> list[dict]:
    """For endpoints that return a list of memory objects."""
    return _request(method, path, **kwargs).json()


def _memory_table(records: list[dict]) -> Table:
    table = Table(show_lines=False)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Scope")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Pinned")
    table.add_column("Content", overflow="fold")
    for r in records:
        table.add_row(
            str(r["id"])[:8],
            r["scope"],
            r["kind"],
            r["status"],
            "📌" if r["pinned"] else "",
            r["content"],
        )
    return table


@memory_app.command("list")
def memory_list(
    repository: Annotated[str | None, typer.Option(help="Filter to a specific repo's memories")] = None,
    scope: Annotated[str | None, typer.Option(help="repo | tenant")] = None,
    status: Annotated[str | None, typer.Option(help="proposed | approved | forgotten")] = None,
):
    """List memories visible to your tenant."""
    params = {k: v for k, v in {"repository": repository, "scope": scope, "status": status}.items() if v}
    records = _request_many("GET", "/v1/keystone/memory", params=params)
    if not records:
        console.print("No memories found.")
        return
    console.print(_memory_table(records))


@memory_app.command("add")
def memory_add(
    content: Annotated[str, typer.Argument(help="The memory text")],
    kind: Annotated[str, typer.Option(help="convention | preference | fact | avoid")] = "fact",
    repository: Annotated[str | None, typer.Option(help="Repo clone URL — omit for a tenant-wide memory")] = None,
    pinned: Annotated[bool, typer.Option(help="Always include this memory regardless of query relevance")] = False,
):
    """Add a new memory."""
    body = {"content": content, "kind": kind, "repository": repository, "pinned": pinned}
    record = _request_one("POST", "/v1/keystone/memory", json=body)
    console.print(f"[green]Created[/green] {record['id']} ({record['scope']}/{record['kind']})")


@memory_app.command("recall")
def memory_recall(
    query: Annotated[str, typer.Argument(help="The task/question to preview recall for")],
    repository: Annotated[str | None, typer.Option(help="Repo clone URL for repo-scope recall")] = None,
    budget_tokens: Annotated[int, typer.Option(help="Token budget, matching the agent's own")] = 2000,
):
    """Preview exactly what the agent would recall for a given query — the real ranking, not an approximation."""
    body = {"query": query, "repository": repository, "budget_tokens": budget_tokens}
    records = _request_many("POST", "/v1/keystone/memory/recall", json=body)
    if not records:
        console.print("Nothing would be recalled for this query.")
        return
    console.print(_memory_table(records))


@memory_app.command("approve")
def memory_approve(memory_id: Annotated[str, typer.Argument()]):
    """Approve a proposed memory so it becomes recallable."""
    record = _request_one("POST", f"/v1/keystone/memory/{memory_id}/approve")
    console.print(f"[green]Approved[/green] {record['id']}")


@memory_app.command("forget")
def memory_forget(memory_id: Annotated[str, typer.Argument()]):
    """Forget a memory (soft delete — it stops being recalled, but the record and its attribution stay)."""
    record = _request_one("POST", f"/v1/keystone/memory/{memory_id}/forget")
    console.print(f"[yellow]Forgotten[/yellow] {record['id']}")


@memory_app.command("pin")
def memory_pin(memory_id: Annotated[str, typer.Argument()]):
    """Pin a memory — it's always recalled regardless of query relevance."""
    record = _request_one("POST", f"/v1/keystone/memory/{memory_id}/pin")
    console.print(f"[green]Pinned[/green] {record['id']}")


@memory_app.command("unpin")
def memory_unpin(memory_id: Annotated[str, typer.Argument()]):
    """Unpin a memory."""
    record = _request_one("POST", f"/v1/keystone/memory/{memory_id}/unpin")
    console.print(f"[green]Unpinned[/green] {record['id']}")


def _parse_config_value(raw: str):
    """A --set VALUE is JSON first (so `--set lora_r=16` and `--set eval_steps=25`
    become real ints, `--set use_qlora=true` a real bool), falling back to the
    literal string when it isn't valid JSON (e.g. a bare model path)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_config_sets(sets: list[str]) -> dict:
    config: dict = {}
    for item in sets:
        if "=" not in item:
            error_console.print(f"--set expects KEY=VALUE, got: {item!r}")
            raise typer.Exit(code=1)
        key, _, raw_value = item.partition("=")
        config[key] = _parse_config_value(raw_value)
    return config


def _finetune_job_table(jobs: list[dict]) -> Table:
    table = Table(show_lines=False)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Type")
    table.add_column("Base model")
    table.add_column("Status")
    table.add_column("Created")
    for j in jobs:
        table.add_row(str(j["id"])[:8], j["job_type"], j["base_model"], j["status"], j["created_at"])
    return table


def _print_finetune_job(job: dict) -> None:
    console.print(
        f"[bold]{job['id']}[/bold]  {job['job_type']}  {job['base_model']}  status=[cyan]{job['status']}[/cyan]"
    )
    if job.get("metrics"):
        console.print(f"  metrics: {job['metrics']}")
    if job.get("output_model_path"):
        console.print(f"  output: {job['output_model_path']}")
    if job.get("error_message"):
        error_console.print(f"  error: {job['error_message']}")


@finetune_app.command("start")
def finetune_start(
    job_type: Annotated[str, typer.Argument(help="lora | sft | dpo")],
    base_model: Annotated[str, typer.Argument(help="Base model id, e.g. Qwen/Qwen2.5-Coder-32B-Instruct")],
    training_data_path: Annotated[str, typer.Argument(help="Path to the training_data JSONL, reachable by the server")],
    set_: Annotated[
        list[str], typer.Option("--set", help="Trainer config override KEY=VALUE, repeatable (e.g. --set num_epochs=3)")
    ] = [],  # noqa: B006 — typer reads the default list, never mutated
):
    """Start a fine-tuning job."""
    config = _parse_config_sets(set_)
    body = {"job_type": job_type, "base_model": base_model, "training_data_path": training_data_path, "config": config}
    job = _request_one("POST", "/v1/finetune/jobs", json=body)
    console.print(f"[green]Started[/green] job {job['id']}")
    _print_finetune_job(job)


@finetune_app.command("list")
def finetune_list(status: Annotated[str | None, typer.Option(help="Filter by status")] = None):
    """List fine-tuning jobs for your tenant."""
    params = {"status": status} if status else {}
    jobs = _request_many("GET", "/v1/finetune/jobs", params=params)
    if not jobs:
        console.print("No fine-tuning jobs found.")
        return
    console.print(_finetune_job_table(jobs))


@finetune_app.command("status")
def finetune_status(job_id: Annotated[str, typer.Argument()]):
    """Show one job's current status and metrics."""
    job = _request_one("GET", f"/v1/finetune/jobs/{job_id}")
    _print_finetune_job(job)


@finetune_app.command("preview")
def finetune_preview(
    job_id: Annotated[str, typer.Argument()],
    lines: Annotated[int, typer.Option(help="Number of records to preview")] = 5,
):
    """Preview the first records of a job's real training data."""
    result = _request_one("GET", f"/v1/finetune/jobs/{job_id}/dataset-preview", params={"lines": lines})
    console.print(f"{result['path']}  ({result['total_records']} records)")
    for record in result["preview"]:
        console.print(record)


@finetune_app.command("promote")
def finetune_promote(job_id: Annotated[str, typer.Argument()]):
    """Promote a completed job's adapter to the tenant's default for its base model."""
    job = _request_one("POST", f"/v1/finetune/jobs/{job_id}/promote")
    console.print(f"[green]Promoted[/green] {job['id']} — new requests for {job['base_model']} now route to it.")


@finetune_app.command("rollback")
def finetune_rollback(job_id: Annotated[str, typer.Argument()]):
    """Retire a promoted job's adapter — routing falls back to the base model."""
    job = _request_one("POST", f"/v1/finetune/jobs/{job_id}/rollback")
    console.print(f"[yellow]Rolled back[/yellow] {job['id']} — routing now falls back to the base model.")


@finetune_app.command("watch")
def finetune_watch(job_id: Annotated[str, typer.Argument()]):
    """Stream a job's live progress until it reaches a terminal status."""
    url = f"{_base_url()}/v1/finetune/jobs/{job_id}/stream"
    headers = {"Authorization": f"Bearer {_api_key()}"}
    try:
        with (
            httpx.Client(timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)) as client,
            client.stream("GET", url, headers=headers) as resp,
        ):
            if resp.status_code != 200:
                error_console.print(f"Request failed ({resp.status_code}): {resp.read().decode()}")
                raise typer.Exit(code=1)
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[len("data: ") :])
                console.print(f"[cyan]{payload['status']}[/cyan]  {payload.get('message', '')}")
                if payload.get("metrics"):
                    console.print(f"  metrics: {payload['metrics']}")
    except httpx.ConnectError as exc:
        error_console.print(f"Could not reach {_base_url()}: {exc}")
        raise typer.Exit(code=1) from exc


_STATUS_STYLE = {
    CheckStatus.OK: ("green", "OK"),
    CheckStatus.WARN: ("yellow", "WARN"),
    CheckStatus.FAIL: ("bold red", "FAIL"),
    CheckStatus.SKIP: ("dim", "SKIP"),
}


@app.command("doctor")
def doctor():
    """Diagnose your Keystone deployment — real checks against Postgres, Redis, Qdrant, the
    sandbox daemon, model endpoints, the git host, package mirrors, and secret strength, each
    reporting exactly what's wrong (and often how to fix it), not just pass/fail."""
    settings = get_settings()
    results = asyncio.run(run_all_checks(settings))

    table = Table(show_lines=False)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for r in results:
        style, label = _STATUS_STYLE[r.status]
        table.add_row(r.name, f"[{style}]{label}[/{style}]", r.message)
    console.print(table)

    fail_count = sum(1 for r in results if r.status == CheckStatus.FAIL)
    warn_count = sum(1 for r in results if r.status == CheckStatus.WARN)
    if fail_count:
        error_console.print(f"\n{fail_count} check(s) FAILED, {warn_count} warning(s).")
        raise typer.Exit(code=1)
    if warn_count:
        console.print(f"\n[yellow]{warn_count} warning(s)[/yellow] — nothing failing outright.")
    else:
        console.print("\n[green]All checks passed.[/green]")


if __name__ == "__main__":
    app()
