# Use Keystone from VS Code (Continue)

Keystone's gateway is OpenAI-compatible, so the [Continue](https://www.continue.dev/) extension for VS Code talks to it like any other model server — chat, inline edit, apply, and autocomplete — and Continue's MCP client gets Keystone's memory tools (`memory_search`, `memory_add`) the same way OpenCode does. No Keystone-specific extension is required for this path. Continue's agent mode works too: the gateway passes `tools`, `tool_choice` and `tool` messages through to the model unchanged (`tests/test_messages_api.py` proves the round trip against a real backend), and an extension that speaks the Anthropic Messages API instead can point at `/v1/messages` with the same key.

Verified on: 2026-09-22 — `tests/test_vscode_continue_config.py` parses the generated file and checks every field name against Continue's own config-yaml schema (`packages/config-yaml` in `continuedev/continue`).

## 1. Install Continue

In VS Code: Extensions → search **Continue** → Install. (Or `code --install-extension Continue.continue`.)

## 2. Generate the config

You need the gateway URL and an API key with the `inference` and `agent` scopes (`keystone keys-create`, or the admin API).

```bash
# Prints the YAML; add --output to write it where Continue reads it
keystone ide continue-config --base-url https://keystone.internal:8080 --api-key ks-XXXX-XXXXXXXX

# Write it for VS Code directly
keystone ide continue-config --base-url https://keystone.internal:8080 --api-key ks-XXXX-XXXXXXXX \
  --output ~/.continue/config.yaml

# Air-gapped deployment with the internal CA (pki/): tell Continue to trust it
keystone ide continue-config --base-url https://keystone.internal:8080 --api-key ks-XXXX-XXXXXXXX \
  --ca-bundle /etc/keystone/pki/ca.crt --output ~/.continue/config.yaml
```

`KEYSTONE_API_KEY` in the environment is picked up when `--api-key` is omitted. A checked-in rendering with placeholders is at [`vscode/continue/config.example.yaml`](https://github.com/GGChamp85/keystone/blob/main/vscode/continue/config.example.yaml).

## 3. What the config does

| Continue role | Keystone model | Why |
|---|---|---|
| `chat`, `edit`, `apply`, `summarize` | `coding` (the primary open-weight coder, or whatever you pointed the coding role at — including the frontier proxy) | Quality where it matters |
| `autocomplete` | `coding_fallback` | Cheaper, lower latency, short completions (`maxTokens: 256`, temperature 0) |
| MCP server `Keystone memory` | `GET/POST <gateway>/v1/keystone/mcp` (streamable-HTTP, bearer auth) | The agent's per-repo/per-tenant memory, readable and writable from the IDE |

Model names are the gateway's **roles**, not vendor model ids — swapping the model behind a role (`CODING_MODEL_ID`, a promoted fine-tuned adapter, a frontier proxy) needs no change in VS Code. Per-tenant adapter routing applies to these requests exactly as it does to every other gateway call.

## 4. Try it

- Open any file, select a function, press the Continue edit shortcut and ask for a change — the diff comes from your Keystone deployment.
- In Continue's chat, ask "what conventions does this repo have?" — the agent calls `memory_search` over MCP and answers from Keystone memory.
- Every request lands in the tenant's usage ledger (`GET /v1/keystone/usage`, the web UI's **Spend** view) with the model role it used.

## Background tasks from the IDE

Submitting a full background task (clone → branch → quality gates → tests → PR) is the web UI / `POST /v1/keystone/tasks` today; a first-party VS Code extension with a task panel and the live trace is on the roadmap. Until then, the OpenCode integration (`docs/OPENCODE_SETUP.md`) covers the terminal-agent workflow.
