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
import shutil
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from src.cli.doctor import CheckStatus, run_all_checks
from src.cli.ide import ContinueConfigInputs, render_continue_config
from src.cli.init import (
    BACKEND_CHOICES,
    BACKEND_NOTES,
    backend_overrides,
    default_overrides,
    find_env_template,
    render_env_file,
)
from src.cli.ops import (
    CLOUDS,
    build_bundle_build_images_command,
    build_bundle_download_models_command,
    build_bundle_import_command,
    build_down_command,
    build_helm_install_command,
    build_migrate_command,
    build_sandbox_images_command,
    build_tofu_command,
    build_tofu_output_command,
    build_up_command,
    cloud_env_dir,
    default_cloud_values_files,
    find_iac_binary,
    find_repo_root,
)
from src.cli.runpod_serverless import GPU_TIER_24GB, GPU_TIER_80GB, RunPodServerless, explain_api_error
from src.config import get_settings

app = typer.Typer(add_completion=False, help="Keystone — self-hosted LLM inference & autonomous coding agents.")
memory_app = typer.Typer(add_completion=False, help="Inspect and manage what the agent has learned.")
finetune_app = typer.Typer(add_completion=False, help="Start and manage fine-tuning jobs.")
tenants_app = typer.Typer(add_completion=False, help="Bootstrap tenants (KEYSTONE_ROOT_ADMIN_TOKEN required).")
users_app = typer.Typer(add_completion=False, help="Manage tenant users (KEYSTONE_ROOT_ADMIN_TOKEN required).")
bundle_app = typer.Typer(add_completion=False, help="Build/import the air-gapped offline bundle (wraps airgap/*.sh).")
deploy_app = typer.Typer(
    add_completion=False,
    help="Deploy to a cloud: a RunPod Serverless model backend, or a GPU Kubernetes pilot on AWS/Azure/GCP.",
)
ide_app = typer.Typer(add_completion=False, help="Generate IDE configuration (VS Code via Continue).")
app.add_typer(memory_app, name="memory")
app.add_typer(finetune_app, name="finetune")
app.add_typer(tenants_app, name="tenants")
app.add_typer(users_app, name="users")
app.add_typer(bundle_app, name="bundle")
app.add_typer(deploy_app, name="deploy")
app.add_typer(ide_app, name="ide")

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


def _admin_token() -> str:
    token = os.environ.get("KEYSTONE_ROOT_ADMIN_TOKEN")
    if not token:
        error_console.print(
            "KEYSTONE_ROOT_ADMIN_TOKEN is not set — this is the bootstrap token from .env, not a tenant API key."
        )
        raise typer.Exit(code=1)
    return token


def _client(*, admin: bool = False) -> httpx.Client:
    token = _admin_token() if admin else _api_key()
    return httpx.Client(
        base_url=_base_url(),
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )


def _request(method: str, path: str, *, admin: bool = False, **kwargs) -> httpx.Response:
    try:
        with _client(admin=admin) as client:
            resp = client.request(method, path, **kwargs)
    except httpx.ConnectError as exc:
        error_console.print(f"Could not reach {_base_url()}: {exc}")
        raise typer.Exit(code=1) from exc

    if resp.status_code == 401:
        which = "KEYSTONE_ROOT_ADMIN_TOKEN" if admin else "KEYSTONE_API_KEY"
        error_console.print(f"Unauthorized — check {which}.")
        raise typer.Exit(code=1)
    if resp.status_code == 409:
        error_console.print(f"Conflict: {resp.text}")
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


@finetune_app.command("guided")
def finetune_guided(
    repo: Annotated[list[str], typer.Option("--repo", help="Allow-listed git URL to learn from (repeatable)")],
    goal: Annotated[str, typer.Option(help="What the adapter should get better at, in plain words")],
    base_model: Annotated[str, typer.Option(help="Catalog model id, or 'auto' for the default SLM")] = "auto",
    epochs: Annotated[int, typer.Option(min=1)] = 1,
    holdout_ratio: Annotated[float, typer.Option(min=0.05, max=0.5)] = 0.2,
    gpu: Annotated[
        list[str] | None,
        typer.Option("--gpu", help="Target GPU as NAME:VRAM_GB (repeatable); omitted = detect on the server"),
    ] = None,
    gpu_hourly_cost: Annotated[float | None, typer.Option(help="Your GPU price in USD/hour, for the cost line")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Approve the plan without asking")] = False,
    force: Annotated[bool, typer.Option(help="Approve even if the planner says it does not fit")] = False,
):
    """Describe → plan & cost → approve → train. Builds a dataset from the repositories' real history and
    this tenant's accepted agent tasks, shows the plan, and starts the job once you approve it."""
    gpus = None
    if gpu:
        gpus = []
        for spec in gpu:
            name, _, vram = spec.rpartition(":")
            if not name or not vram:
                error_console.print(f"--gpu must be NAME:VRAM_GB, got {spec!r}")
                raise typer.Exit(code=1)
            gpus.append({"name": name, "vram_gb": float(vram)})
    body = {
        "repositories": repo,
        "goal": goal,
        "base_model": base_model,
        "epochs": epochs,
        "holdout_ratio": holdout_ratio,
        "gpus": gpus,
        "gpu_hourly_cost_usd": gpu_hourly_cost,
    }
    plan = _request_one("POST", "/v1/finetune/plans", json=body)
    p = plan["plan"]
    ds = plan["dataset"]
    table = Table(title=f"Plan {plan['job_id']}", show_header=False)
    table.add_row("Base model", plan["base_model"])
    table.add_row("Method", f"{p['method']}  ({'fits' if p['fits'] else 'DOES NOT FIT'})")
    table.add_row(
        "Hardware",
        f"{p['gpu_count']} x {p['gpu_name'] or '?'} ({p['vram_available_gb']} GB; needs ~{p['vram_required_gb']} GB)",
    )
    table.add_row(
        "Examples",
        f"{p['train_examples']} train / {p['holdout_examples']} held out  (from {ds['total_examples']} collected)",
    )
    table.add_row(
        "Sources",
        ", ".join(f"{k}: {v}" for k, v in ds["per_repository"].items()) + f"; accepted tasks: {ds['trajectories']}",
    )
    table.add_row(
        "Safety",
        f"{ds['dropped_by_secret_scan']} record(s) dropped by the secret scan, "
        f"{ds['records_with_pii_redacted']} with PII redacted",
    )
    table.add_row("Steps", f"{p['total_steps']} optimizer steps ({p['epochs']} epoch(s), LoRA r={p['lora_r']})")
    table.add_row("Time", f"~{p['estimated_hours']} h" if p["estimated_hours"] is not None else "—")
    table.add_row(
        "Cost", f"~${p['estimated_cost_usd']}" if p["estimated_cost_usd"] is not None else "— (no GPU price given)"
    )
    table.add_row("Basis", p["estimate_basis"])
    for reason in p["reasons"]:
        table.add_row("Note", reason)
    for warning in ds["warnings"]:
        table.add_row("[yellow]Warning[/yellow]", warning)
    console.print(table)
    if not yes and not typer.confirm("Approve and start training?", default=False):
        console.print(f"Not started. Approve later with: keystone finetune approve {plan['job_id']}")
        return
    job = _request_one("POST", f"/v1/finetune/plans/{plan['job_id']}/approve", json={"force": force})
    console.print(
        f"[green]Started[/green] job {job['id']} ({job['status']}) — follow it with: "
        f"keystone finetune watch {job['id']}"
    )


@finetune_app.command("approve")
def finetune_approve(
    job_id: Annotated[str, typer.Argument()],
    force: Annotated[bool, typer.Option(help="Approve even if the planner says it does not fit")] = False,
):
    """Approve a planned guided fine-tune and start it."""
    job = _request_one("POST", f"/v1/finetune/plans/{job_id}/approve", json={"force": force})
    console.print(f"[green]Started[/green] job {job['id']} ({job['status']})")


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
def finetune_promote(
    job_id: Annotated[str, typer.Argument()],
    force: Annotated[
        bool, typer.Option("--force", help="Admin only: promote despite a fail/unknown verdict (audit-logged)")
    ] = False,
):
    """Promote a completed job's adapter to the tenant's default for its base model.

    Gated by the verdict: the adapter must have beaten the base model on the
    held-out split. The promoted adapter is written to the lora_modules
    manifest and live-loaded into the serving vLLM when it allows it.
    """
    job = _request_one("POST", f"/v1/finetune/jobs/{job_id}/promote", params={"force": "true"} if force else None)
    verdict = (job.get("metrics") or {}).get("verdict") or {}
    serving = (job.get("metrics") or {}).get("serving") or {}
    console.print(f"[green]Promoted[/green] {job['id']} — new requests for {job['base_model']} now route to it.")
    if verdict:
        console.print(f"Verdict: {verdict.get('status')} — {verdict.get('reason')}")
    if serving:
        loaded = "[green]loaded live[/green]" if serving.get("loaded") else "[yellow]not loaded live[/yellow]"
        console.print(f"Serving: {loaded} — {serving.get('detail')}")
        if serving.get("manifest"):
            console.print(f"Manifest: {serving['manifest']}")
        if serving.get("manifest_error"):
            console.print(f"[red]{serving['manifest_error']}[/red]")


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


@tenants_app.command("create")
def tenant_create(
    name: Annotated[str, typer.Argument()],
    email: Annotated[str, typer.Argument()],
    tier: Annotated[str, typer.Option(help="free | pro | enterprise")] = "free",
):
    """Create a new tenant — the real bootstrap step every API key/user belongs to."""
    body = {"name": name, "email": email, "tier": tier}
    tenant = _request_one("POST", "/v1/admin/tenants", admin=True, json=body)
    console.print(f"[green]Created tenant[/green] {tenant['id']}  ({tenant['name']}, {tenant['tier']})")


@tenants_app.command("list")
def tenant_list():
    """List every tenant on this deployment."""
    # No list-all-tenants route exists yet (only per-tenant sub-resources) —
    # this is a real, honest gap, not a bug: say so instead of pretending.
    error_console.print(
        "No GET /v1/admin/tenants (list-all) route exists yet — only per-tenant "
        "sub-resources (keys, users) are listable today. See ROADMAP.md's Phase 7."
    )
    raise typer.Exit(code=1)


@app.command("keys-create")
def keys_create(
    tenant_id: Annotated[str, typer.Argument()],
    name: Annotated[str, typer.Option()] = "default",
    scopes: Annotated[str, typer.Option(help="Comma-separated, e.g. inference,agent,finetune")] = "inference,agent",
    expires_in_days: Annotated[int | None, typer.Option()] = None,
):
    """Create a real API key for a tenant — prints the key exactly once, like every other real API-key flow."""
    body = {
        "name": name,
        "scopes": [s.strip() for s in scopes.split(",") if s.strip()],
        "expires_in_days": expires_in_days,
    }
    key = _request_one("POST", f"/v1/admin/tenants/{tenant_id}/keys", admin=True, json=body)
    console.print(f"[green]Created key[/green] {key['name']} ({key['key_prefix']}...)")
    console.print(f"[bold yellow]{key['key']}[/bold yellow]  [dim](save this — it will not be shown again)[/dim]")


@app.command("keys-rotate")
def keys_rotate(
    tenant_id: Annotated[str, typer.Argument()],
    key_prefix: Annotated[str, typer.Argument(help="The ks-xxxxxxxx prefix of the key to replace")],
    grace_hours: Annotated[int, typer.Option(help="How long the old key keeps working; 0 = revoke now")] = 24,
):
    """Replace an API key without an outage: mints a new key with the same scopes and limits and expires the
    old one after --grace-hours (ADR 0004). Prints the new key exactly once."""
    key = _request_one(
        "POST", f"/v1/admin/tenants/{tenant_id}/keys/{key_prefix}/rotate", admin=True, json={"grace_hours": grace_hours}
    )
    console.print(f"[green]Rotated[/green] {key_prefix} -> {key['key_prefix']}... (old key valid {grace_hours}h)")
    console.print(f"[bold yellow]{key['key']}[/bold yellow]  [dim](save this — it will not be shown again)[/dim]")


@tenants_app.command("set-limits")
def tenant_set_limits(
    tenant_id: Annotated[str, typer.Argument()],
    monthly_budget_usd: Annotated[float | None, typer.Option(help="USD per calendar month; 0 = no budget")] = None,
    daily_token_limit: Annotated[int | None, typer.Option(help="tokens/day; 0 = unlimited")] = None,
    monthly_token_limit: Annotated[int | None, typer.Option(help="tokens/month; 0 = unlimited")] = None,
    max_concurrent_agents: Annotated[int | None, typer.Option(help="running tasks; 0 = no cap")] = None,
):
    """Set a tenant's caps — a monthly dollar budget, token budgets, concurrency. Only the options given change."""
    body = {
        k: v
        for k, v in {
            "monthly_budget_usd": monthly_budget_usd,
            "daily_token_limit": daily_token_limit,
            "monthly_token_limit": monthly_token_limit,
            "max_concurrent_agents": max_concurrent_agents,
        }.items()
        if v is not None
    }
    if not body:
        error_console.print("Give at least one limit to change.")
        raise typer.Exit(code=2)
    tenant = _request_one("POST", f"/v1/admin/tenants/{tenant_id}/limits", admin=True, json=body)
    console.print(
        f"[green]Updated[/green] {tenant['name']}: budget ${tenant['monthly_budget_usd']}/month, "
        f"tokens {tenant['daily_token_limit']}/day {tenant['monthly_token_limit']}/month, "
        f"concurrency {tenant['max_concurrent_agents']} (0 = unlimited)"
    )


@users_app.command("add")
def user_add(
    tenant_id: Annotated[str, typer.Argument()],
    name: Annotated[str, typer.Argument()],
    email: Annotated[str, typer.Argument()],
    role: Annotated[str, typer.Option(help="admin | lead | developer")] = "developer",
):
    """Add a user to a tenant — real identity attribution on every task/memory action they take."""
    body = {"name": name, "email": email, "role": role}
    user = _request_one("POST", f"/v1/admin/tenants/{tenant_id}/users", admin=True, json=body)
    console.print(f"[green]Added user[/green] {user['id']}  ({user['name']} <{user['email']}>, {user['role']})")


def _users_table(users: list[dict]) -> Table:
    table = Table(show_lines=False)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name")
    table.add_column("Email")
    table.add_column("Role")
    table.add_column("Active")
    for u in users:
        table.add_row(str(u["id"])[:8], u["name"], u["email"], u["role"], "yes" if u["is_active"] else "no")
    return table


@users_app.command("list")
def user_list(tenant_id: Annotated[str, typer.Argument()]):
    """List every user on a tenant."""
    users = _request_many("GET", f"/v1/admin/tenants/{tenant_id}/users", admin=True)
    if not users:
        console.print("No users found.")
        return
    console.print(_users_table(users))


def _require_repo_root():
    root = find_repo_root()
    if root is None:
        error_console.print(
            "Could not find docker-compose.yml — run this from inside a Keystone repo checkout "
            "(or a subdirectory of one)."
        )
        raise typer.Exit(code=1)
    return root


def _run_streamed(cmd: list[str], *, cwd=None) -> None:
    """Runs `cmd` with stdout/stderr inherited (not captured) so the user
    sees the same real, live output the wrapped tool (docker compose, an
    airgap/*.sh script) already produces — this is a thin wrapper, not a
    reimplementation that would have to re-derive or summarize that
    output itself."""
    import subprocess

    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, cwd=cwd)  # noqa: S603 — argv built entirely from this module's own fixed strings
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@app.command("up")
def up(
    dev: Annotated[
        bool, typer.Option(help="Only infra (postgres/redis/qdrant) — no app/worker/sandbox-daemon")
    ] = False,
    migrate: Annotated[bool, typer.Option(help="Run alembic migrations once the stack is up")] = True,
    sandbox_images: Annotated[bool, typer.Option(help="Build the sandbox runtime image too")] = True,
    with_demo_model: Annotated[
        bool,
        typer.Option(
            "--with-demo-model",
            help="Also start the no-GPU demo model (llama.cpp, Qwen2.5-Coder-0.5B) — pair with "
            "`keystone init --backend demo-cpu`",
        ),
    ] = False,
):
    """Bring up the Keystone stack — a real wrapper around `docker compose up -d`
    (+ optionally the sandbox image build and migrations), matching what the README's
    manual `make certs && make build && make up && make db-migrate` steps do in one command."""
    root = _require_repo_root()
    compose_file = root / "docker-compose.yml"

    _run_streamed(build_up_command(compose_file, dev_only=dev, with_demo_model=with_demo_model), cwd=root)

    if dev:
        console.print("[green]Dev infra is up.[/green] (--dev skips sandbox-images/migrate too)")
        return

    if sandbox_images:
        _run_streamed(build_sandbox_images_command(root), cwd=root)
    if migrate:
        _run_streamed(build_migrate_command(compose_file), cwd=root)

    console.print(
        "[green]Stack is up.[/green] Next: `keystone doctor`, then create an API key with `keystone keys-create`."
    )


@app.command("down")
def down():
    """Stop the Keystone stack — a real wrapper around `docker compose down`."""
    root = _require_repo_root()
    _run_streamed(build_down_command(root / "docker-compose.yml"), cwd=root)


@bundle_app.command("download-models")
def bundle_download_models(output_dir: Annotated[str, typer.Argument()] = "./airgap/models"):
    """Download every model weight onto this (internet-connected) build machine — wraps airgap/download_models.sh."""
    root = _require_repo_root()
    _run_streamed(build_bundle_download_models_command(root, output_dir), cwd=root)


@bundle_app.command("build-images")
def bundle_build_images(output_dir: Annotated[str, typer.Argument()] = "./airgap/output/images"):
    """Pull and tarball every container image onto this build machine — wraps airgap/build_image_bundle.sh."""
    root = _require_repo_root()
    _run_streamed(build_bundle_build_images_command(root, output_dir), cwd=root)


@bundle_app.command("import")
def bundle_import(
    bundle_dir: Annotated[str | None, typer.Argument()] = None,
    push: Annotated[bool, typer.Option(help="Also push each re-tagged image to the internal registry")] = False,
):
    """Load the transferred bundle on the air-gapped target — wraps airgap/import_bundle.sh."""
    root = _require_repo_root()
    _run_streamed(build_bundle_import_command(root, bundle_dir, push=push), cwd=root)


@deploy_app.command("runpod-serverless")
def deploy_runpod_serverless(
    name: Annotated[str, typer.Option(help="Endpoint/template name — re-running with the same name reuses them")] = (
        "keystone-coder"
    ),
    model: Annotated[str, typer.Option(help="Hugging Face model id to serve")] = "Qwen/Qwen2.5-Coder-7B-Instruct",
    served_name: Annotated[str, typer.Option(help="The `model` name clients send")] = "keystone-coder",
    gpu_tier: Annotated[str, typer.Option(help="24gb | 80gb — which serverless GPU pool to schedule on")] = "24gb",
    max_model_len: Annotated[int, typer.Option()] = 16384,
    workers_max: Annotated[int, typer.Option(help="Upper bound on concurrent GPU workers (cost ceiling)")] = 2,
    idle_timeout: Annotated[int, typer.Option(help="Seconds idle before a worker scales to zero")] = 60,
    destroy: Annotated[bool, typer.Option(help="Delete the endpoint and template of this name instead")] = False,
):
    """Create (or reuse) a RunPod Serverless vLLM endpoint serving an open-weight model,
    scaled to zero when idle — a real GPU backend with no hardware to buy. Needs
    RUNPOD_API_KEY in .env. Prints the exact .env lines that point a role at it."""
    settings = get_settings()
    if settings.runpod_api_key is None:
        error_console.print("RUNPOD_API_KEY is not set in .env — see docs/deployment/RUNPOD_SETUP.md.")
        raise typer.Exit(code=1)
    tiers = {"24gb": GPU_TIER_24GB, "80gb": GPU_TIER_80GB}
    if gpu_tier not in tiers:
        error_console.print(f"--gpu-tier must be one of {sorted(tiers)}")
        raise typer.Exit(code=1)

    with RunPodServerless(settings.runpod_api_key.get_secret_value()) as rp:
        try:
            if destroy:
                removed = rp.destroy(name)
                console.print(f"Removed endpoint={removed['endpoint']} template={removed['template']} for {name!r}")
                return
            deployment = rp.deploy(
                name=name,
                model=model,
                served_model_name=served_name,
                gpu_type_ids=tiers[gpu_tier],
                max_model_len=max_model_len,
                workers_max=workers_max,
                idle_timeout_seconds=idle_timeout,
            )
        except httpx.HTTPStatusError as exc:
            error_console.print(explain_api_error(exc.response.status_code, exc.response.text))
            raise typer.Exit(code=1) from exc

    verb_t = "created" if deployment.created_template else "reused"
    verb_e = "created" if deployment.created_endpoint else "reused"
    console.print(f"[green]Template[/green] {deployment.template_id} ({verb_t})  serving {deployment.model}")
    console.print(
        f"[green]Endpoint[/green] {deployment.endpoint_id} ({verb_e})  gpu tier {gpu_tier}, workers 0..{workers_max}"
    )
    console.print(f"[bold]{deployment.openai_base_url}[/bold]")
    console.print("\nPoint a role at it — add to .env (the first request after idle is a cold start):")
    console.print(f"  RUNPOD_ENDPOINT_URL={deployment.openai_base_url}")
    console.print(f"  VLLM_CODING_URL={deployment.openai_base_url}")
    console.print("  VLLM_CODING_API_KEY=${RUNPOD_API_KEY}")
    console.print(f"  CODING_MODEL_ID={deployment.served_model_name}")
    console.print("\nThen `keystone doctor` checks it, and `keystone deploy runpod-serverless --destroy` removes it.")


@deploy_app.command("cloud")
def deploy_cloud(
    cloud: Annotated[str, typer.Option(help="aws | azure | gcp")],
    env: Annotated[
        str, typer.Option(help="Environment name — runs in infra/opentofu/environments/<cloud>-<env>")
    ] = "pilot",
    plan: Annotated[bool, typer.Option("--plan", help="init + plan, change nothing (the default)")] = False,
    apply: Annotated[bool, typer.Option("--apply", help="init + apply — asks for the usual yes/no first")] = False,
    destroy: Annotated[
        bool, typer.Option("--destroy", help="init + destroy the whole environment — asks for yes/no first")
    ] = False,
    var_file: Annotated[
        Path | None,
        typer.Option(help="Extra -var-file (terraform.tfvars in the environment directory is loaded automatically)"),
    ] = None,
    helm_install: Annotated[
        bool,
        typer.Option(
            "--helm-install",
            help="Run `helm upgrade --install` with values-client-vpc.yaml + values-<cloud>.yaml against the "
            "current kubeconfig context — after --apply, or on its own once the cluster exists",
        ),
    ] = False,
):
    """Stand up (or tear down) a GPU Kubernetes pilot on AWS, Azure or GCP — a thin wrapper around
    `tofu -chdir=infra/opentofu/environments/<cloud>-<env> init|plan|apply|destroy` (Terraform is
    used when OpenTofu is not installed) and, with --helm-install, the matching Helm install.
    Every command is printed before it runs. See docs/guides/deploy-<cloud>.md for what you get,
    what it costs, and what has and has not been verified."""
    if cloud not in CLOUDS:
        error_console.print(f"--cloud must be one of {', '.join(CLOUDS)}, not {cloud!r}.")
        raise typer.Exit(code=1)
    if sum([plan, apply, destroy]) > 1:
        error_console.print("Pass at most one of --plan, --apply, --destroy.")
        raise typer.Exit(code=1)
    if destroy and helm_install:
        error_console.print("--helm-install makes no sense with --destroy.")
        raise typer.Exit(code=1)

    root = _require_repo_root()
    env_dir = cloud_env_dir(root, cloud, env)
    if not env_dir.is_dir():
        available = sorted(p.name for p in (root / "infra" / "opentofu" / "environments").iterdir() if p.is_dir())
        error_console.print(f"No environment at {env_dir} — available: {', '.join(available)}.")
        raise typer.Exit(code=1)

    run_iac = apply or destroy or plan or not helm_install
    if run_iac:
        binary = find_iac_binary()
        if binary is None:
            error_console.print(
                "Neither `tofu` nor `terraform` is on PATH. Install OpenTofu 1.6+ "
                "(https://opentofu.org/docs/intro/install/ — `brew install opentofu` on macOS) "
                "or Terraform 1.6+; the HCL under infra/opentofu works with either."
            )
            raise typer.Exit(code=1)
        resolved_var_file: Path | None = None
        if var_file is not None:
            resolved_var_file = var_file.resolve()
            if not resolved_var_file.is_file():
                error_console.print(f"--var-file {var_file} does not exist.")
                raise typer.Exit(code=1)
        action = "apply" if apply else "destroy" if destroy else "plan"
        _run_streamed(build_tofu_command(env_dir, "init", binary=binary), cwd=root)
        _run_streamed(build_tofu_command(env_dir, action, resolved_var_file, binary=binary), cwd=root)

        if action == "plan":
            console.print(
                f"\n[green]Plan only — nothing changed.[/green] Re-run with --apply to create the {cloud} {env} tier."
            )
            return
        if action == "destroy":
            console.print(
                f"\n[green]Destroyed.[/green] Cloud billing for the {cloud} {env} tier stops with the last resource."
            )
            return

        kubeconfig_cmd = _read_tofu_output(binary, env_dir, "kubeconfig_command", cwd=root)
        console.print("\n[green]Cluster is up.[/green] Point kubectl/helm at it:")
        console.print(f"  {kubeconfig_cmd}")
        if not helm_install:
            console.print("Then install the chart (or re-run this command with --helm-install):")
            console.print(f"  {' '.join(build_helm_install_command(cloud, 'keystone', 'keystone'))}")
            return

    if shutil.which("helm") is None:
        error_console.print("`helm` is not on PATH — install Helm 3 (https://helm.sh/docs/intro/install/).")
        raise typer.Exit(code=1)
    values_files = default_cloud_values_files(cloud, root / "helm" / "keystone")
    _run_streamed(
        build_helm_install_command(cloud, "keystone", "keystone", values_files, chart_dir=root / "helm" / "keystone"),
        cwd=root,
    )
    console.print(
        "\n[green]Chart installed.[/green] `kubectl -n keystone get pods` shows the rollout; the vLLM pod "
        "downloads the model into the shared cache on first start."
    )


def _read_tofu_output(binary: str, env_dir: Path, name: str, *, cwd: Path) -> str:
    """One captured `tofu output -raw <name>` — the only tofu call whose output the CLI reads
    rather than streams, because it needs the string to print an instruction with it."""
    import subprocess

    result = subprocess.run(  # noqa: S603 — argv from build_tofu_output_command's fixed strings
        build_tofu_output_command(env_dir, name, binary=binary), cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        return f"{binary} -chdir={env_dir} output -raw {name}   # ({result.stderr.strip()})"
    return result.stdout.strip()


@ide_app.command("continue-config")
def ide_continue_config(
    base_url: Annotated[
        str, typer.Option(help="Keystone gateway URL, no trailing /v1 (e.g. https://keystone.internal:8080)")
    ],
    api_key: Annotated[
        str, typer.Option(help="A Keystone API key with the inference and agent scopes", envvar="KEYSTONE_API_KEY")
    ],
    output: Annotated[
        Path | None, typer.Option(help="Write here (typically ~/.continue/config.yaml); prints to stdout when omitted")
    ] = None,
    ca_bundle: Annotated[
        Path | None, typer.Option(help="Internal CA bundle (pki/) for an air-gapped TLS deployment")
    ] = None,
    coding_model: Annotated[str, typer.Option(help="Model name for chat/edit/apply")] = "coding",
    autocomplete_model: Annotated[str, typer.Option(help="Model name for autocomplete")] = "coding_fallback",
):
    """Render Continue's config.yaml (VS Code) pointing chat/edit/apply, autocomplete and MCP memory at
    this Keystone deployment — the same gateway and key OpenCode uses. See docs/guides/use-from-vscode.md."""
    text = render_continue_config(
        ContinueConfigInputs(
            base_url=base_url,
            api_key=api_key,
            coding_model=coding_model,
            autocomplete_model=autocomplete_model,
            ca_bundle_path=str(ca_bundle) if ca_bundle else None,
        )
    )
    if output is None:
        console.print(text, markup=False, highlight=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    console.print(f"Wrote {output} — restart VS Code (or reload Continue) to pick it up.")


@app.command("ingest")
def ingest(
    repository_url: Annotated[str, typer.Argument(help="Git clone URL — must be on GIT_ALLOWED_HOSTS")],
    branch: Annotated[str, typer.Option()] = "main",
    max_file_size_kb: Annotated[int, typer.Option(help="Skip files larger than this")] = 500,
):
    """Ingest a repository into the memory system's RAG index — a real wrapper around the
    existing POST /v1/keystone/ingest route (src/memory/ingestion.py's CodeIngestionPipeline).
    Incremental: a second run only re-embeds files that actually changed."""
    body = {"repository_url": repository_url, "branch": branch, "max_file_size_kb": max_file_size_kb}
    result = _request_one("POST", "/v1/keystone/ingest", json=body)
    console.print(
        f"[green]Ingested[/green] {result['repository']} @ {result['commit_sha'][:8]} (branch={result['branch']})"
    )
    console.print(
        f"  {result['files_processed']} file(s) processed, {result['files_unchanged']} unchanged, "
        f"{result['files_skipped']} skipped, {result['files_deleted']} deleted"
    )
    console.print(
        f"  {result['chunks_created']} chunk(s) created, {result['chunks_upserted']} upserted, "
        f"{result['stale_chunks_deleted']} stale chunk(s) removed"
    )


@app.command("init")
def init(
    output: Annotated[str, typer.Option(help="Where to write the .env file")] = ".env",
    yes: Annotated[
        bool, typer.Option("--yes", help="Non-interactive: auto-generate all secrets, skip every other prompt")
    ] = False,
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Coding model backend: demo-cpu (llama.cpp 0.5B, no GPU), single-gpu (vLLM serving "
            "Qwen2.5-Coder-7B on one 24 GB GPU), frontier-proxy (benchmarks/frontier_proxy.py), or "
            "self-hosted-gpu (the compose/Helm defaults). Asked interactively when omitted.",
        ),
    ] = None,
):
    """Guided setup — writes a real .env file (from .env.example) for `make up` to use.
    Run `keystone doctor` afterward to verify what's actually reachable."""
    from pathlib import Path

    from rich.prompt import Confirm, Prompt

    output_path = Path(output)
    if (
        output_path.exists()
        and not yes
        and not Confirm.ask(f"{output_path} already exists — overwrite?", default=False)
    ):
        console.print("Aborted — nothing written.")
        raise typer.Exit(code=1)

    template = find_env_template()
    if template is None:
        error_console.print(
            "Could not find .env.example — run `keystone init` from inside a Keystone repo checkout "
            "(or a subdirectory of one)."
        )
        raise typer.Exit(code=1)

    overrides = default_overrides(auto_secrets=True)
    console.print(
        "[green]Generated strong random secrets[/green] for Postgres/Redis/Qdrant/admin-token/app secret key."
    )

    if not yes:
        configure_git = Confirm.ask(
            "\nConfigure a git host now? (needed for agentic coding tasks that open PRs — "
            "you can skip and do this later)",
            default=False,
        )
        if configure_git:
            host = Prompt.ask("Git host hostname (e.g. gitea.mycompany.com, or localhost for a local Gitea)")
            api_url = Prompt.ask("Git host API base URL", default=f"https://{host}/api/v1")
            token = Prompt.ask("Git host API token", password=True)
            overrides["GIT_ALLOWED_HOSTS"] = json.dumps([host])
            overrides["GIT_HOST_API_URL"] = api_url
            overrides["GIT_HOST_TOKEN"] = token

    if backend is None and not yes:
        backend = Prompt.ask(
            "\nCoding model backend",
            choices=list(BACKEND_CHOICES),
            default="demo-cpu",
        )
    if backend is not None:
        if backend not in BACKEND_CHOICES:
            error_console.print(f"Unknown --backend {backend!r}; choose one of: {', '.join(BACKEND_CHOICES)}")
            raise typer.Exit(code=2)
        overrides.update(backend_overrides(backend))
        console.print(f"\n[dim]{BACKEND_NOTES[backend]}[/dim]")

    content = render_env_file(template.read_text(), overrides)
    output_path.write_text(content)
    console.print(f"\n[green]Wrote {output_path}[/green]")
    console.print(
        "\nNext: [bold]make certs && make build && make up && make db-migrate && make sandbox-images[/bold]"
        "\nThen: [bold]keystone doctor[/bold] to confirm what's actually reachable."
    )


_STATUS_STYLE = {
    CheckStatus.OK: ("green", "OK"),
    CheckStatus.WARN: ("yellow", "WARN"),
    CheckStatus.FAIL: ("bold red", "FAIL"),
    CheckStatus.SKIP: ("dim", "SKIP"),
}


@app.command("doctor")
def doctor():
    """Diagnose your Keystone deployment — real checks against Postgres (and its migration revision), Redis,
    Qdrant, the sandbox daemon, model endpoints, the git host, package mirrors, secret strength, TLS
    certificate validity, free disk, and the running app's own readiness when KEYSTONE_INFERENCE_URL is set —
    each reporting exactly what's wrong (and often how to fix it), not just pass/fail."""
    settings = get_settings()
    results = asyncio.run(run_all_checks(settings, app_base_url=os.environ.get("KEYSTONE_INFERENCE_URL")))

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
