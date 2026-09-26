# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — the real LSP stdio client script and language-server map behind the
`lsp_diagnostics` tool (implemented in `tools/impl.py`, which owns `ToolResult` — this module
holds no dependency on it, to keep the two free of a circular import).

`repo_map.py` gives the agent a ctags-based, textual outline of the repo — cheap and good enough
for orientation, but heuristic (its "references" are a `grep -F` count, not real semantic
resolution). `lsp_diagnostics` gives the agent something ctags can't: a real LSP server's own
type diagnostics on one file, mid-loop, before the model ever signals done — catching a broken
signature change immediately instead of only at the quality gate afterward.

Python only for now (`python-lsp-server`, pip-installed into the sandbox image — deliberately
NOT `pyright`, whose PyPI package fetches its actual JS implementation over the network on first
run, which this project's air-gap posture can't allow). Node/Go support is the same pattern —
install the language server binary in that runtime's Dockerfile, add its command/language id
below — once each has been verified the same way this one was: for real, inside a real sandbox,
against a file with a real, known error.

The client speaks the real LSP stdio wire protocol (`Content-Length: N\r\n\r\n<json>` framing) —
no python "lsp" library dependency inside the sandbox, since the sandbox images are deliberately
minimal (see docker/sandbox-runtimes/*.Dockerfile's own reasoning for baking in only what's
needed). It runs as a short-lived one-shot script per call: start the server, initialize, open
the one file, wait for its `textDocument/publishDiagnostics`, shut down — not a persistent
session held across tool calls, which would need connection lifecycle management this feature
doesn't need yet to be real and useful.
"""

from __future__ import annotations

LANGUAGE_SERVERS: dict[str, tuple[str, str]] = {
    ".py": ("pylsp", "python"),
}

DIAGNOSTIC_TIMEOUT_SECONDS = 20
CLIENT_SCRIPT_NAME = ".keystone-lsp-client.py"

CLIENT_SCRIPT = r"""
import asyncio, json, sys

async def read_message(reader):
    headers = {}
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode().partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    body = await reader.readexactly(length) if length else b""
    return json.loads(body) if body else {}

def encode(obj):
    body = json.dumps(obj).encode()
    return f"Content-Length: {len(body)}\r\n\r\n".encode() + body

async def main():
    command, root, path, language_id, timeout_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
    with open(path, encoding="utf-8") as f:
        content = f.read()
    proc = await asyncio.create_subprocess_exec(
        command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )

    async def send(obj):
        proc.stdin.write(encode(obj))
        await proc.stdin.drain()

    file_uri = "file://" + path
    diagnostics = None
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    try:
        await send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"processId": None, "rootUri": "file://" + root, "capabilities": {}},
        })
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(read_message(proc.stdout), timeout=remaining)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                break
            if msg.get("id") == 1 and "result" in msg:
                await send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
                doc = {"uri": file_uri, "languageId": language_id, "version": 1, "text": content}
                await send({"jsonrpc": "2.0", "method": "textDocument/didOpen", "params": {"textDocument": doc}})
            if msg.get("method") == "textDocument/publishDiagnostics" and msg.get("params", {}).get("uri") == file_uri:
                diagnostics = msg["params"]["diagnostics"]
                break
        try:
            await send({"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": None})
            await send({"jsonrpc": "2.0", "method": "exit", "params": None})
        except Exception:
            pass
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
    print(json.dumps({"diagnostics": diagnostics if diagnostics is not None else [], "timed_out": diagnostics is None}))

asyncio.run(main())
"""


def severity_name(severity: int | None) -> str:
    return {1: "error", 2: "warning", 3: "info", 4: "hint"}.get(severity or 0, "unknown")
