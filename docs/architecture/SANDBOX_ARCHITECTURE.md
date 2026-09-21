# Keystone — Self-Hosted Sandbox Architecture

Every piece of code Keystone Agents writes or is asked to test runs inside
this sandbox layer — never on the host running the FastAPI app, never in a
third-party cloud sandbox (this replaced an earlier E2B-hosted-SaaS
dependency entirely). The layer has three parts: a pluggable execution
**backend**, a privileged-isolation **daemon** the backend runs inside, and
a thin **client** the rest of the app talks to. Source: `src/sandbox/`.

## Why a separate daemon process at all

`keystoned` (`src/sandbox/daemon.py`) is the only process that touches
`/dev/kvm` (Firecracker) or the Docker socket (gVisor) — real privileged
access no other Keystone process should ever need. The main API/worker
processes talk to it over HTTP (`src/sandbox/manager.py`, an `httpx`
client), so a bug or compromise in the agent-orchestration code path can't
directly escalate to sandbox-host privileges; it would first have to
compromise `keystoned` itself through its narrow HTTP surface
(`POST /sandboxes`, `POST /sandboxes/{handle}/files`,
`POST /sandboxes/{handle}/exec`, `DELETE /sandboxes/{handle}`,
`GET /health`, `GET /metrics`).

## Backend interface (`src/sandbox/backends/base.py`)

A `SandboxBackend` ABC — `create` / `write_file` / `read_file` / `execute`
/ `destroy` / `apply_egress_policy` / `health` — with a "handle" (an opaque
backend-specific ID: a container ID for gVisor, a microVM ID for
Firecracker) as the only thing callers pass around. `keystoned` is
otherwise backend-agnostic; adding a third backend means implementing this
interface, nothing else changes.

## Backend selection (`daemon.py::_select_backend`)

```python
if preferred == "firecracker" and firecracker_available():
    return FirecrackerBackend()
# else: gVisor, whether preferred was "gvisor" or "firecracker" was
# requested but /dev/kvm isn't present on this host
return GVisorBackend()
```

`firecracker_available()` just checks for `/dev/kvm`. This makes the
production/client-VPC bare-metal tier automatically get real Firecracker
microVMs (`SANDBOX_BACKEND=firecracker` in config, KVM present), while the
RunPod dev/test tier — standard GPU pods, no nested virtualization —
automatically and silently falls back to gVisor instead of failing to
boot. Nothing in the calling code (orchestrator, RL rollout coordinator,
benchmark harness) needs to know which backend is actually running.

## gVisor backend (`backends/gvisor.py`) — the portable default

Each sandbox is a Docker container run under the `runsc` OCI runtime (a
userspace-kernel sandbox — isolates syscalls without requiring KVM). If
`runsc` isn't registered as a Docker runtime on the host (checked via
`docker info`'s `Runtimes` map, cached after the first check), falls back
further to plain `runc` with hardened flags — `no-new-privileges`, all
Linux capabilities dropped, read-only rootfs, PID/process limits. That
fallback is a real, if weaker, isolation boundary — never an unsandboxed
`subprocess.run` on the host (that path exists only as a
`VS_ALLOW_UNSANDBOXED_DEV`-gated, hard-blocked-in-production dev
convenience, see `src/sandbox/runtime.py`).

## Firecracker backend (`backends/firecracker.py`) — the client-VPC primary

Launches real Firecracker microVMs via the `jailer` wrapper
(`_spawn_jailer`, using `asyncio.create_subprocess_exec` — never a
blocking `subprocess.Popen` on the async event loop). Two boot paths:

- **`_cold_boot`**: full kernel boot for the first VM of a given runtime
  image.
- **`_restore_from_snapshot`**: Firecracker's own snapshot/restore
  mechanism — the same trick that gives hosted sandbox providers their
  sub-second cold starts — resumes a pre-booted, pre-warmed VM image
  instead of booting from scratch.

Guest communication happens over **vsock**, not a network socket (no IP
stack needed to reach the guest at all): `_VsockAgentClient` sends a small
JSON request/response protocol (`write_file` / `read_file` / `exec`) over
a Unix domain socket bridged to the guest's vsock device
(`_configure_vsock`). `_disable_networking` means the VM has no network
interface at all by default — egress is opt-in per sandbox, not
opt-out.

## Egress control — enforced from the host side, not inside the sandbox

The single most important security property of this layer, and the reason
it's built this way rather than "run iptables inside the container":
**a workload with shell access inside the sandbox must never be able to
alter its own firewall rules.** Both backends implement
`apply_egress_policy` entirely from the host/hypervisor side:

- **gVisor**: `iptables` rules in the host's `DOCKER-USER` chain, keyed on
  the container's assigned IP on the dedicated `keystone-sandbox-net`
  bridge network — default-deny, with explicit `-j ACCEPT` rules inserted
  for each allowed CIDR/port from `EgressPolicy`. These rules live in the
  host's network namespace; nothing running inside the container can see
  or modify them.
- **Firecracker**: the guest has no network device unless explicitly
  configured (`_disable_networking` is the default), so egress control is
  the strictly stronger "no network stack exists" rather than "network
  exists but is filtered."

The policy itself — which hosts/ports a sandbox may reach, secret-value
scrubbing from captured stdout/stderr, SSRF-pattern rejection — lives in
`src/sandbox/security.py` (`EgressPolicy`, `DEFAULT_EGRESS_RULES`), unchanged
from the original scaffold's design (it was already sound) and re-pointed
at internal mirror hostnames for the air-gapped tier
(`airgap/*.sh`) rather than public package registries.

## Zero-retention

`destroy(handle)` tears the sandbox down completely — the container is
removed (gVisor) or the microVM process is killed and its jail directory
deleted (Firecracker). Nothing about a task's sandboxed execution persists
across tasks; each `RolloutEnvironment.reset()` (RL) or task-execution
cycle (the agent orchestrator) gets a fresh sandbox from
`SandboxManager.get_or_create()`.

## Tenant concurrency limiting

`daemon.py`'s `_TenantConcurrency` caps how many sandboxes a single tenant
can have live at once (`SANDBOX_MAX_CONCURRENT`), independent of the
warm-pool/autoscaling story — this bounds blast radius (a runaway/malicious
tenant workload can't exhaust the daemon's capacity for everyone else) and
maps naturally onto the subscription/plan tiers Keystone Inference already
tracks.

## Idempotent creation for Temporal retries

`SandboxManager.get_or_create(task_id, ...)` — not just `create()` — exists
specifically because Temporal activities can be retried after a partial
failure (the activity's process crashed after the sandbox was created but
before the activity recorded success). Retrying a plain `create()` would
leak an orphaned sandbox on every retry; `get_or_create` is keyed so a
retry reuses the same sandbox instead of creating a new one each time.

## What this doesn't cover yet

- On-demand GPU-accelerated sandboxes (workstream 12's stated extension —
  attaching a GPU to a sandbox instance via the Kubernetes device plugin,
  for research/dataset-generation workloads that need GPU *inside* the
  sandbox, distinct from GPU-serving the LLMs themselves) is not
  implemented.
- Firecracker's snapshot/restore warm-pool sizing and the KEDA-based
  autoscaler referenced in the original project plan are not wired up as
  an actual autoscaling controller in this pass — `keystoned` creates
  sandboxes on demand per request; there's no pre-warmed pool manager
  sitting in front of it yet. Both backends have been exercised for real
  (gVisor via `tests/test_sandbox_gvisor_integration.py`'s live Docker
  lifecycle tests, RL rollouts, and the benchmark harness; Firecracker's
  code path has not been exercised against real KVM hardware in this dev
  environment — flag for a smoke test on client bare-metal before
  handoff, consistent with `docs/deployment/KUBERNETES_CLIENT_VPC.md`'s
  GPU/multi-node caveats).
