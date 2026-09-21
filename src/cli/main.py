# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — command-line client.

A real HTTP client against a running Keystone deployment's API — never
talks to Postgres directly, so it works from any machine (a developer's
laptop, an OpenCode session via `cli/opencode/commands/memory.md`, CI)
that can reach the API, the same way the OpenAI-compatible endpoints do.
Configured the same way `cli/opencode.config.json` already is:

  KEYSTONE_INFERENCE_URL   base URL, e.g. http://localhost:8080
  KEYSTONE_API_KEY         a real ks-... API key with the "agent" scope
"""

from __future__ import annotations

import os
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="Keystone — self-hosted LLM inference & autonomous coding agents.")
memory_app = typer.Typer(add_completion=False, help="Inspect and manage what the agent has learned.")
app.add_typer(memory_app, name="memory")

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


if __name__ == "__main__":
    app()
