# The coding agent

**What it is.** A background engineer for your repositories. Give it a task and a repository; it clones the repository into a sandbox, reads and searches the real code, edits it with real tools, runs your quality gates and your test suite, asks a second model to review, fixes what failed, and opens a pull request for you to read like any contributor's. You watch every step live.

**Who it is for.** Development teams that want to hand off well-defined work (a bug with a failing test, a pagination endpoint, a migration) and get back a reviewed pull request; leads who need to see what an agent did, not trust that it did it.

## What "done" means

- **Verified, not asserted.** Pass or fail comes from the real exit code of your linters, type checkers, security scanners and test suite, run inside a sandbox. No model grades its own work.
- **Fails closed.** A review the model could not complete, a gate that could not run, a test the sandbox could not execute: each blocks the task. Nothing is labelled reviewed or tested that was not.
- **A real git workflow.** Clone, branch (`keystone/<user>/<task_id>`), commit, push, pull request on your git server, then a fresh clone of the pushed branch is re-verified before the task is called complete.
- **Repository-aware.** It starts from a ranked map of your repository's symbols, reads line ranges rather than whole files, installs your dependencies once, runs the tests related to what it touched before the full suite, and can re-plan from the root cause after repeated failures rather than retrying blindly.

## Submit a task

From the web UI (**Tasks**), from VS Code, from the terminal agent, or over the API:

```bash
curl -X POST http://localhost:8080/v1/keystone/tasks \
  -H "Authorization: Bearer ks-XXXX-XXXXXXXX" -H "Content-Type: application/json" \
  -d '{
    "task": "Add pagination to the /users endpoint with cursor-based navigation",
    "repository_url": "https://<your-git-server>/myorg/myapi",
    "model": "coding",
    "max_iterations": 10
  }'
```

`"model": "auto"` lets the router choose the role from the task description, recorded on the task as the role actually used, never a hidden live decision. `max_iterations` is the one per-task safety bound, with no ceiling.

Then watch: `GET /v1/keystone/tasks/{id}/stream` is a live event stream (the web UI renders it) with every tool call and result, every diff, every test run and quality finding, the review, and the pull request link. Nothing is truncated in storage: full test output, full diffs and full install logs stay on the task record.

## What it will cost

Before you submit, `POST /v1/keystone/tasks/estimate` (the web UI shows it live as you type) answers "roughly how many tokens, and how much": it draws on this tenant's own completed tasks when there are any, falls back to every completed task on the deployment, and only uses a documented rough default when nothing has ever completed — `estimate_basis` always says which, so the number is never presented as more precise than it is. After a task finishes, `GET /v1/keystone/tasks/{id}` returns `cost_breakdown`: the same execution trace grouped by phase and model role and priced by your `MODEL_PRICES_PER_MILLION` — the real cost, not a second guess — shown as a table in the web UI once the task completes.

## Best of N

For work worth more than one attempt, set `best_of_n` on the task (or `AGENT_BEST_OF_N` for the deployment; default 1, no ceiling). The agent makes N independent attempts from the same clean base, each on its own branch in the sandbox, scores every attempt with the repository's own lint, type checks and the tests related to what it touched, and carries the best one forward as ordinary uncommitted edits into the normal quality, review and test gates. The trace shows every attempt's score and the ranking; losing branches are deleted. It multiplies model spend by N, which is why it is off by default.

## Where the work happens

Each task gets its own sandbox (gVisor by default; Firecracker microVMs where KVM is available) with the repository cloned inside it, its own branch and its own pull request. The sandbox runs the runtime image for the repository's ecosystem — Python, Node or Go toolchains baked in (`make sandbox-images` builds all three), chosen from the repository's own tooling once it is cloned. The sandbox's network egress is restricted to the git hosts you allow and the package mirrors you configure. Tasks never share a working copy, so many developers can run many tasks against many repositories at once; throughput is bounded by the GPU, worker and sandbox capacity you deploy, not by a policy number.

## Connect your git server

The agent only talks to hosts you list. Gitea is the reference implementation (`src/git/gitea.py`); a local one takes five minutes.

```bash
# .env
GIT_ALLOWED_HOSTS=["git.yourcompany.internal"]
GIT_HOST_API_URL=https://git.yourcompany.internal/api/v1
GIT_HOST_TOKEN=<a bot account token with push and pull-request permissions>
```

A repository URL outside the allowlist is rejected before any clone, with the host named in the error.

## Safety on untrusted content

Everything the agent reads from a repository is untrusted. Every tool result is wrapped in an explicit delimiter naming its source before it re-enters the model's context, and common prompt-injection phrasings are flagged with a visible notice inside that result. Content is never altered or dropped, so this is on by default. It is defence in depth, not a guarantee: a disguised injection can still evade the patterns.

When a role is pointed at a frontier model through the proxy, `FRONTIER_PROXY_REDACT_PII=1` redacts emails, phone numbers, national identifiers, card numbers (with a real checksum) and IP addresses from everything that leaves the network. It is deterministic pattern matching, auditable and dependency-free, and it will not catch free-text personal details.

## Other ways in

- **Web UI**: `http://localhost:8080/app/`: submit, watch the trace, browse the team's tasks, manage memory, see spend.
- **VS Code**: the Continue extension against the gateway, with the agent's memory over MCP ([guide](use-from-vscode.md)).
- **Terminal**: [OpenCode](https://github.com/sst/opencode) pre-configured against your roles, with permission gates that ask before every edit and every command ([setup](../deployment/OPENCODE_SETUP.md)).

## Memory

The agent records what it learned per repository and per tenant (conventions, gotchas, decisions) and retrieves it on later tasks. Humans can inspect, approve, pin or forget any memory from the web UI, the `keystone memory` commands, or the MCP tools an IDE uses.

## Steer a running task

A task does not have to run to completion unsupervised. `POST /v1/keystone/tasks/{id}/steer` queues a new instruction for a task that is still running or waiting to start; the agent picks it up at the start of its next iteration and treats it as a new turn in the conversation, never mid-tool-call, so an instruction can never split a tool call from its result. Nothing you send is lost — a message that arrives between two checks is simply picked up on the next one — and every steer is visible in the live trace, so the whole team sees what changed and when.

## Skills: standing instructions

A skill is a standing instruction an admin writes once, applied automatically to every matching task from then on — "run the PCI checklist when touching payment code" — distinct from memory, which the agent proposes for itself from what happened on one specific task. `POST /v1/keystone/skills` creates one, with an optional list of trigger keywords (leave it empty and the skill applies to every task); `POST /v1/keystone/skills/match` previews exactly which skills a task description would trigger before you ever submit it.

## Verify as you go

Right after an edit, the agent can ask a real language server to check the file it just changed — a broken signature or an undefined name is caught before the slower test suite runs, not after. Python is supported today, through `python-lsp-server`; `pyright` was deliberately left out, since its package fetches its real implementation over the network on first run, which does not fit an air-gapped deployment.

## Automate: schedules and webhooks

Save a task as a template and it can submit itself later, with no one watching: on a cron schedule, or the instant an external system calls its own unguessable webhook URL. `POST /v1/keystone/schedules` creates one; enable, disable or delete it any time. A CI pipeline, a nightly job, or any other internal tool can trigger a real task this way, through the exact same submission path a human uses from the web UI or the API.

## Replay and export a trace

Every step of a task — the plan, each edit, every tool call, every test run, the review, the final diff — stays on the task record, not just for as long as the task is running. `GET /v1/keystone/tasks/{id}/replay` reconstructs the exact live trace after the fact, for a task from last month as easily as one that finished five minutes ago. `GET /v1/keystone/tasks/{id}/trace` renders the same execution as a real OpenTelemetry trace, ready to load into Jaeger, Tempo, Honeycomb, or any other OTLP-compatible tool your team already runs.

## Verified how

`tests/test_git_workflow_integration.py` runs a task end to end against a live Gitea, a live sandbox daemon and a live sandbox; `tests/test_tool_impl.py`, `tests/test_repo_map.py` and `tests/test_fuzzy_patch.py` exercise the tools in a real sandbox (including the language-server check, against a live sandbox running `python-lsp-server`); `tests/test_review_node.py` and `tests/test_quality_*` prove the gates fail closed; `tests/test_steering.py`, `tests/test_steer_route.py`, `tests/test_skills_store.py` and `tests/test_skills_route.py` prove a steer or a skill actually reaches the prompt sent to the model; `tests/test_schedules_store.py` and `tests/test_schedules_route.py` prove a schedule submits a real task, on both the cron and webhook paths; `tests/test_replay.py` proves a replay reproduces the exact recorded event sequence; `tests/test_otel_export.py` proves the exported trace is valid against the real OpenTelemetry encoding library; `benchmarks/agent_runner.py` runs the whole loop on seeded repository tasks ([benchmarks](../benchmarks/README.md)).
