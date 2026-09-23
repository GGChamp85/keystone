# Keystone Agents for VS Code

Submit a background coding task to your self-hosted [Keystone](https://github.com/GGChamp85/keystone) deployment, watch it work in a live trace panel — plan, every tool call and result, quality-gate findings, test output, the diff, the pull request — list the team's recent tasks, and search the team memory the agent learns from. One gateway URL, one `ks-...` API key, the same REST API the web UI uses.

Continue (chat, inline edit, autocomplete against the same gateway) is a separate extension: `keystone ide continue-config` generates its config. This extension covers what Continue does not — the background agent.

## Install

From the `.vsix` built by the repository's CI (`vscode-extension` job artifact) or locally:

```bash
cd vscode/extension && npm ci && npm run build && npm run package   # → keystone-agents.vsix
code --install-extension keystone-agents.vsix
```

## Settings and commands

| Setting | Default | Meaning |
|---|---|---|
| `keystone.baseUrl` | `http://localhost:8080` | The gateway origin (serves `/v1/*` and the web UI at `/app/`) |
| `keystone.defaultModel` | `auto` | The role pre-selected in the model picker |
| API key | — | Not a setting: **Keystone: Set API key** stores it in VS Code's secret storage (the OS keychain) |

| Command | What it does |
|---|---|
| **Keystone: Submit background task** | Description → repository URL (defaults to the workspace's `origin`) → model picker from `GET /v1/keystone/models` (READY roles first; an unhealthy role is shown with its state and cannot be picked) → `POST /v1/keystone/tasks`, then the trace opens |
| **Keystone: Open task trace** | The live trace panel for a task id — `GET /v1/keystone/tasks/{id}/stream` read as Server-Sent Events with a Bearer header; replays history, then streams. When the task finishes with a PR, a notification offers **Open PR** |
| **Keystone: List recent tasks** | The team's last 50 tasks with status, model role, repository and PR; pick one to open its trace |
| **Keystone: Search team memory** | `POST /v1/keystone/memory/recall` — exactly what the agent would recall for the query, scoped to the workspace's repository when it has one; pick a memory to copy it |
| **Keystone: Open the Playground (web UI)** | Opens `<baseUrl>/app/?view=playground` in the browser |
| **Keystone: Set API key** | Stores the key in secret storage |

The status bar shows how many of the tenant's tasks are pending or running (polled from `GET /v1/keystone/tasks` every 15 s while a key is stored); click it for the task list.

## What is verified, and how

- `npm test` — typecheck plus vitest unit tests for the SSE decoder, the step renderer and the model-picker ordering. They run on **recorded real frames**: `src/__tests__/fixtures/task-stream.sse` is the byte-exact output of the real stream route (captured through the FastAPI app, fed by the real event publishers over a real Redis Stream) and `model-library.json` the real `GET /v1/keystone/models` response with one role served by a real backend process — both produced by `scripts/record_task_stream_fixture.py` in the repository. The fixture is not a recording of a live model writing code; the payload shapes are those the publishing call sites emit.
- `npm run test:integration` — `@vscode/test-electron` downloads VS Code, loads the extension and checks activation and every command. With `KEYSTONE_URL` and `KEYSTONE_API_KEY` set it also submits a real task (`model: auto`) and asserts the trace panel receives real events (the routing decision first); without them that test prints why and skips.
- The event types in `src/api.ts` (`NodeEvent`, `StepEventType`, `StepEvent`, `TaskEvent`) are checked byte-for-byte against `web/src/api.ts` by `scripts/check_ts_types_in_sync.py` in the repository's CI lint job.

## Development

```bash
npm ci
npm run watch        # esbuild, rebuilds dist/extension.js on save — then F5 in VS Code (Run Extension)
npm test             # typecheck + unit tests
npm run test:integration
```

Apache-2.0 — see `LICENSE` (a pointer to the repository's license).
