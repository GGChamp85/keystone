# Changelog — Keystone Agents for VS Code

## 0.1.0

First release.

- Commands: submit background task (description → repository from the workspace `origin` → model picker with live READY/UNHEALTHY/NOT_DEPLOYED/TRAINING states from `GET /v1/keystone/models`), open task trace, list recent tasks, search team memory (`POST /v1/keystone/memory/recall`), open the Playground, set API key (VS Code secret storage).
- Live trace panel streaming `GET /v1/keystone/tasks/{id}/stream` over a fetch-based SSE reader with Bearer auth — phases, plan, per-node stats, every step (tool calls and results, model messages, dependency install, repository map, routing decision, quality findings, test output, diff, PR) with full bodies, and a notification with **Open PR** when the task completes.
- Status bar item with the tenant's pending/running task count.
- Unit tests on recorded real stream and model-library fixtures; a VS Code integration test that runs a real submission when `KEYSTONE_URL`/`KEYSTONE_API_KEY` are set and self-skips otherwise.
