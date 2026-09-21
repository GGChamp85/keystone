# Keystone Agents — Interactive CLI (OpenCode) setup

Keystone Agents' interactive surface is [OpenCode](https://opencode.ai)
(`opencode-ai` on npm, MIT-licensed — verified against the published
package on 2026-09-20: `npm view opencode-ai`), configured to talk to
Keystone Inference instead of a third-party API. We don't build or
maintain a CLI ourselves — OpenCode already is a mature Claude-Code-style
terminal agent (tool-use loop, permission system, streaming TUI, plan
mode), and re-implementing that would be pure duplicated effort.

## Install

```bash
npm install -g opencode-ai
```

**Air-gapped:** `airgap/build_npm_mirror.sh` builds a self-contained
`node_modules/` bundle on the build machine (internet access) that the
air-gapped target can run directly, with no npm registry access of its own.

This one needed real debugging, not just packaging: `opencode-ai` ships its
CLI as a platform-specific native (Bun-compiled) binary, distributed via
per-platform `optionalDependencies` and staged into place by a
`postinstall` script. That script decides which binary to use from the
**build machine's own** `os.platform()/os.arch()` — it ignores `npm install
--os=... --cpu=...` — and falls back to its own nested `npm install`
straight against the live registry whenever the matching platform package
isn't present locally. A naive cross-platform build (building the
air-gapped bundle for a Linux server on a macOS laptop, say) would silently
stage the *build machine's* binary and quietly phone home while doing it —
exactly the kind of hidden network dependency air-gapping is supposed to
rule out. `build_npm_mirror.sh` installs with `--ignore-scripts` (so
postinstall never runs, never reaches the network) and manually stages the
correct target-platform binary itself, then verifies the staged file's
type (`ELF`/`Mach-O`/`PE32`) matches the requested target. Verified for
real: a same-platform round trip (`opencode --version` printed a real
version), and a cross-build from a macOS/ARM machine correctly staging a
genuine Linux/x86_64 ELF binary.

```bash
./airgap/build_npm_mirror.sh ./airgap/npm-mirror latest linux-x64
# -> ship ./airgap/npm-mirror into the air-gapped environment, then:
alias opencode='/opt/keystone/opencode/node_modules/.bin/opencode'  # run directly — NOT `node .../opencode`, it's a native binary
```

## Configure

Copy `cli/opencode.config.json` to your project root as `opencode.json` (or
to `~/.config/opencode/opencode.json` for a user-wide default), and set:

```bash
export KEYSTONE_INFERENCE_URL="https://keystone.<your-deployment>"   # bare host — no /v1 suffix
export KEYSTONE_API_KEY="ks-...."   # from `make api-key` or POST /v1/admin/tenants/{id}/keys
```

One env var, used consistently as the bare deployment host everywhere in
`cli/opencode.config.json` and `cli/opencode/`: the shipped config appends
`/v1` itself for the model provider (`options.baseURL`) and
`/v1/keystone/mcp` for the MCP server entry below; the `keystone` CLI, the
`/memory` command, and the `keystone-memory.ts` plugin all build their own
full paths (`/v1/keystone/...`) off the same bare value too.

Then from any repo:

```bash
opencode models keystone-inference   # confirms the three Keystone models are visible
opencode                             # starts the interactive TUI
```

## What the config actually does

Verified against OpenCode's real config schema (not assumed) — see
`packages/web/src/content/docs/providers.mdx` and `permissions.mdx` in
github.com/sst/opencode:

- **Provider registration**: `@ai-sdk/openai-compatible` pointed at
  `KEYSTONE_INFERENCE_URL`, branded "Keystone Inference" in OpenCode's
  model picker (not a generic "custom-openai" label) — confirmed working
  end-to-end against a local mock server during development: `opencode
  models keystone-inference` correctly listed `coding`, `coding_fallback`,
  and `reasoning`.
- **Auto mode is real but the default direction is the opposite of what
  you'd guess**: OpenCode's own out-of-the-box default is "allow all
  operations without approval." The shipped `cli/opencode.config.json`
  explicitly overrides this to `"permission": {"*": "ask", ...}` so a
  developer's first session is prompt-before-acting by default — the safer
  choice for a client deployment. From there:
  - `opencode --auto` (or `opencode run --auto "..."`) runs a whole session
    in auto-approve mode.
  - Inside the TUI, the command palette has **Enable/Disable auto-approve
    permissions** to toggle mid-session without restarting.
  - Explicit `"deny"` rules (we block `rm -rf*` and `git push*` by default)
    are enforced even in auto mode.
- **Model switching**: `opencode models` / the in-TUI model picker lists
  all three Keystone roles under the "Keystone Inference" provider; switch
  mid-session without editing config or restarting.

## Team memory (commands + plugin + MCP)

Keystone Agents' memory system (`src/memory/store.py` — pinned-first,
repo-over-tenant ranked recall; see `docs/` for the background-agent side)
also reaches an interactive OpenCode session, the same way `cli/opencode.config.json`
gets copied into a project. Three pieces ship under `cli/opencode/`
(the shipped `cli/opencode.config.json` already wires up the third one —
see [MCP server](#mcp-server) below):

- `cli/opencode/commands/memory.md` — a `/memory` command. Copy it to
  `.opencode/commands/memory.md` (project-level; OpenCode's real docs use
  the *plural* `commands/` directory — not the singular `command/` this
  repo's own internal dev tooling happens to use under `.opencode/command/`,
  which is a separate, unrelated convention). It shells out to the real
  `keystone` CLI (`keystone memory $ARGUMENTS`, using `$ARGUMENTS` exactly
  as OpenCode's command-file spec defines it), so it needs the same
  `KEYSTONE_INFERENCE_URL`/`KEYSTONE_API_KEY` env vars as everything else on
  this page, plus `pip install -e .` run once from the Keystone repo root
  (installs the `keystone` console script — see `pyproject.toml`'s
  `[project.scripts]`). Inside a session: `/memory list`, `/memory add "..."
  --kind avoid`, `/memory recall "topic"`, etc. — anything `keystone memory
  --help` lists.
- `cli/opencode/plugins/keystone-memory.ts` — a local plugin. Copy it to
  `.opencode/plugins/keystone-memory.ts` (project- or
  `~/.config/opencode/plugins/` for user-wide); OpenCode auto-discovers
  local plugins under that directory with no config registration needed
  (registration in `opencode.json`'s `plugin` array is only for
  npm-published plugins, which this is deliberately not, to keep it
  air-gap-safe with zero extra network dependency). It hooks
  `chat.message` to capture the latest user message text, then
  `experimental.chat.system.transform` to call `POST
  /v1/keystone/memory/recall` with that text as the query (scoped to the
  session's git repo, via `git remote get-url origin`) and push the
  result into the system prompt — so an interactive session sees the same
  learned conventions/avoid-list the background agent's own
  planning/coding nodes do, without the developer having to run `/memory
  recall` by hand every turn. Verified: real TypeScript syntax against
  OpenCode's actual `Plugin`/`Hooks` types
  (`packages/plugin/src/index.ts` in github.com/anomalyco/opencode, the
  project's current home after moving from `sst/opencode`) via `tsc
  --noEmit` — the only reported errors were the expected missing
  `@opencode-ai/plugin` module and Node types in this repo's own
  standalone check, not real bugs. It fails open: any Keystone
  unreachability (down endpoint, missing env vars, non-git directory)
  degrades to "no memory injected," never blocks or errors the session.

### MCP server

`src/api/routes/mcp.py` runs a real MCP server (the official `mcp` SDK,
MIT-licensed — its 2.x API renamed the old `FastMCP` to
`mcp.server.mcpserver.MCPServer`, confirmed by reading the installed
package's own source) over the Streamable HTTP transport, mounted into
the main FastAPI app at **`/v1/keystone/mcp`**. It exposes two tools —
`memory_search` (the same ranked recall the CLI/plugin/background agent
all use) and `memory_add` — to *any* MCP-speaking client, not just
OpenCode: Claude Desktop, Claude Code, or another IDE can all add it as a
remote MCP server the same way.

`cli/opencode.config.json` already registers it:

```json
"mcp": {
  "keystone-memory": {
    "type": "remote",
    "url": "{env:KEYSTONE_INFERENCE_URL}/v1/keystone/mcp",
    "enabled": true,
    "headers": { "Authorization": "Bearer {env:KEYSTONE_API_KEY}" }
  }
}
```

so once `opencode.json` is in place and the two env vars above are set,
prompts like "use keystone-memory to remember that ..." or "search
keystone-memory for our testing conventions" work with no extra setup.
This is deliberately a second, complementary path to the same data as the
plugin above — the plugin passively injects relevant memory into every
turn's system prompt; the MCP tools let the model actively search or add
memories on demand, and let non-OpenCode MCP clients reach the same data.

Auth: every tool call carries the same `ks-...` Bearer API key as the rest
of the Keystone API (`Authorization` header, read directly off the MCP
request — there's no FastAPI `Depends()` cycle inside an MCP tool call, so
this reuses the exact same key-resolution function `require_scope("agent")`
calls for every REST route, not a second implementation). Verified for
real: a genuine MCP JSON-RPC round trip (`initialize` ->
`notifications/initialized` -> `tools/list` -> `tools/call`) via curl
against a live app — `memory_add` creating a real row, `memory_search`
finding it with `hit_count` incremented by the real recall ranking — plus
both auth-failure paths (missing/invalid key) returning a clear,
client-visible error rather than a leaked stack trace. This exact flow is
also covered by `tests/test_mcp_server.py` (5 tests, real Postgres/Redis,
`requires_integration_env`), run as part of `python -m pytest`.

Two things worth knowing before relying on this at scale: (1) the
Streamable HTTP session manager runs in-process by default
(`stateless_http=False`), so an MCP session is pinned to whichever
uvicorn worker/replica handled its `initialize` call — fine for a
single-worker deployment, but a horizontally-scaled one needs either
sticky routing or `stateless_http=True` in `build_mcp_asgi_app()`. (2) DNS
rebinding protection is always on (`src/api/routes/mcp.py` passes an
explicit `TransportSecuritySettings`, not the SDK's `host="127.0.0.1"`
auto-default), and only allows `Host:`/`Origin:` headers matching
`settings.mcp_allowed_hosts`/`mcp_allowed_origins` (`src/config.py`) —
defaulted to `127.0.0.1`/`localhost`/`::1` (verified for real above). A
production deployment reachable at a real hostname must set
`MCP_ALLOWED_HOSTS`/`MCP_ALLOWED_ORIGINS` (comma-separated, or a JSON
array — standard pydantic-settings `list[str]` parsing) to that hostname,
or every MCP request will be correctly rejected the same way an arbitrary
Host header is today.

**Known limitation from testing**: a full real OpenCode+Bun interactive
TUI session driving `/memory`, the plugin's hook, and the `keystone-memory`
MCP entry end-to-end was not exercised in this pass — see the
air-gapped-binary and mock-server caveats above; the same constraints
apply here. What *was* verified for real: the `keystone` CLI itself
(`keystone memory list|add|recall|approve|forget|pin|unpin`) against a
live Keystone deployment (real Postgres + Redis + the FastAPI app, real
HTTP round trips, real error handling for unreachable/401/404), the
plugin's TypeScript against OpenCode's real published types, and the full
real MCP protocol round trip described above. Do a real interactive-session
pass once a real OpenCode+Bun environment and a
GPU-backed Keystone Inference endpoint are both available.

## Known limitation from testing

A live chat-completion round trip was verified against a minimal mock
server (real HTTP requests, real 200 responses, real auth header
delivery) — but the mock always returns a plain assistant-text response
with no tool-call structure, which caused OpenCode's agent loop to keep
re-prompting indefinitely rather than stopping. That's a property of the
*mock* not producing a response shape OpenCode recognizes as "done," not a
flaw in the provider config — a real vLLM-backed Keystone Inference
endpoint returns proper structured completions. This wasn't re-verified
against a real model in this pass (no GPU available); do that before
calling the interactive CLI fully production-verified.
