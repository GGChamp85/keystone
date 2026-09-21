# Keystone — Offline (Air-Gapped) Install Runbook

Two machines, two phases. The **build machine** has internet access and
never runs in the client's production environment. The **target** is the
air-gapped client VPC (or a network-isolated test host standing in for
it) and makes zero outbound connections once this runbook is done. Every
script referenced here has been run for real during development (not just
written) — see each step's "Verified" note. Run this on a real
network-isolated host, not just trust the notes, before an actual client
handoff — none of these steps have been chained together end-to-end on a
genuinely offline machine in this pass.

## Phase 1 — Build machine (has internet)

### 1. Container images

```bash
./airgap/build_image_bundle.sh ./airgap/output/images
```

Pulls and pins every image Keystone depends on (Postgres, Valkey, Qdrant,
vLLM, Temporal, nginx, OpenBao, Prometheus, Alertmanager, Grafana, Loki,
Promtail — see the script for the exact pinned tags, kept in sync with
`docker-compose.yml` /
`helm/keystone/values.yaml`) plus the two locally-built images
(`keystone-app`, `keystone-sandbox-daemon` — build those first with
`docker build`). Writes a `manifest.txt` with each image's sha256.

**Verified**: ran end-to-end with a real minimal manifest/tarball,
confirmed `docker load` + re-tag succeeds.

### 2. Model weights

```bash
./airgap/download_models.sh ./airgap/models
```

Downloads the coding/coding_fallback/reasoning LLMs and the BGE embedding model
via `huggingface_hub.snapshot_download`, skipping redundant non-safetensors
weight formats. Output is laid out exactly as `snapshot_download` leaves
it — mount these directories straight into the target's vLLM/embedding
services, no further transformation needed.

**Not run end-to-end in this pass** (the actual weights are tens of GB
each; not downloaded during development). The download call itself is the
standard, well-tested `huggingface_hub` API — the risk here is model-ID
drift (a model getting renamed/removed upstream), not the mechanism.
Re-run and confirm the manifest before a real client bundle.

### 3. Python dependency wheelhouse

```bash
./airgap/build_wheelhouse.sh ./airgap/wheelhouse
# add --with-finetuning to also include torch/transformers/trl/etc.
```

Downloads wheels for the **Dockerfile's actual target** (linux/x86_64,
Python 3.12 — `TARGET_PLATFORM`/`TARGET_PYTHON_VERSION` env vars override
this) regardless of what platform the build machine itself runs. This
matters: a naive same-platform `pip download` on a macOS/ARM build machine
silently produces wheels the air-gapped Linux target can't install at all.

**Verified**: ran for real from a macOS/ARM machine, produced 135 correctly
platform-tagged `manylinux2014_x86_64`/`cp312` wheels, including `torch`
when `--with-finetuning` was tested separately.

### 4. npm mirror (Keystone Agents' interactive CLI)

```bash
./airgap/build_npm_mirror.sh ./airgap/npm-mirror latest linux-x64
```

Stages a self-contained `node_modules/` for `opencode-ai`. **Real bug
found and fixed here**: `opencode-ai`'s own `postinstall` script always
stages the *build machine's* platform binary (ignoring `npm install
--os/--cpu`) and falls back to its own live npm-registry fetch when that
binary isn't already present — exactly the hidden-network-dependency this
whole runbook exists to prevent. `build_npm_mirror.sh` installs with
`--ignore-scripts` and manually stages the correct **target** platform's
binary instead, then verifies the staged file's type.

**Verified**: same-platform round trip actually ran (`opencode --version`
printed a real version); cross-build from macOS/ARM correctly staged a
genuine linux/x86_64 ELF binary (confirmed via `file`).

### 5. Internal CA / TLS

```bash
make certs-ca                              # keystone.local by default
KEYSTONE_HOST=keystone.client.example.com make certs-ca   # or a real hostname
```

Generates a real internal CA (`pki/generate_ca.sh`, 4096-bit RSA, 10-year
validity, proper `CA:TRUE` constraint) and a CA-signed leaf cert with a
correct SAN extension (`pki/issue_cert.sh`), writing the result to
`nginx/certs/` in the same `fullchain.pem`/`privkey.pem` layout
`nginx.conf` already expects — no nginx config changes needed. `make
certs` (the bare self-signed leaf, no CA) still exists for a single
developer's quick local boot; use `make certs-ca` for anything beyond a
laptop, since only the CA-backed chain is something a client's other
machines/services can actually be told to trust (`pki/ca/ca.crt`
distributed once, `pki/issue_cert.sh` mints as many leaf certs off it as
needed afterward without redistributing trust each time).

**Verified**: real end-to-end — generated a CA, issued a leaf cert,
confirmed `openssl verify` passes, then terminated actual TLS in an nginx
container with it and did a real (non-`-k`) `curl --cacert ca.crt`
handshake — `SSL certificate verify ok`, genuine HTTPS response.

### 6. Secrets bootstrap

```bash
docker compose --profile secrets up -d openbao
./scripts/openbao_bootstrap.sh
```

Drives OpenBao's real init/unseal/seed lifecycle (not `-dev` mode — the
same file-backed storage mode production uses) and renders
`.env.generated` from values read back out of OpenBao.

**Verified**: full lifecycle — init, unseal, KV write/read, wrong-token
rejection (403), delete, idempotent re-run, and restart-then-reunseal —
all confirmed against a real OpenBao 2.1 container.

## Transfer

Move `airgap/output/`, `airgap/models/`, `airgap/wheelhouse/`,
`airgap/npm-mirror/`, and this repo's source tree onto physical media (or
across a one-way data diode) into the target environment. Everything
under `airgap/` is `.gitignore`d — it's build output, not repo content.

## Phase 2 — Air-gapped target (zero internet access)

### 7. Import images

```bash
./airgap/import_bundle.sh ./airgap/output/images [--push]
```

Verifies each tarball's sha256 against the transferred manifest **before**
loading it — a bundle carried across an air gap on physical media is
exactly the scenario where silent transfer corruption is a real risk.

**Verified**: real load+retag round trip; both failure modes (corrupted
tarball, incomplete transfer/missing file) correctly abort with a clear
error and non-zero exit rather than loading a bad image silently.

### 8. Point config at local paths

There's no env-var indirection for vLLM's model choice — `--model` is a
literal CLI flag in `docker-compose.yml`'s `command:` for each
`vllm-{coding,coding-fallback,reasoning}` service (and `values.yaml`'s
`vllm.{coding,coding-fallback,reasoning}.model` for Helm). Edit those directly
to the local downloaded directory from step 2 instead of the HF repo ID:

```yaml
# docker-compose.yml, e.g. the vllm-coding service:
command: >
  --model /opt/keystone/models/qwen2.5-coder-32b-instruct   # was: Qwen/Qwen2.5-Coder-32B-Instruct
  --served-model-name qwen-coder-32b ...
```

```yaml
# helm/keystone/values-client-vpc.yaml:
vllm:
  coding:
    model: /opt/keystone/models/qwen2.5-coder-32b-instruct
```

The embedding model *does* have a proper settings field —
`EMBEDDING_MODEL_PATH` (`src/config.py`'s `embedding_model_path`) —
because it's loaded in-process by `src/memory/embeddings.py`, not passed
as another service's CLI arg:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
EMBEDDING_MODEL_PATH=/opt/keystone/models/bge-large-en-v1.5
```

### 9. Bring up the stack

```bash
docker compose --env-file .env.generated up -d
# or, for the multi-node client-VPC tier:
helm install keystone helm/keystone -f helm/keystone/values-client-vpc.yaml \
  --set global.imageRegistry=registry.internal.<client-domain>
```

### 10. Smoke test

```bash
curl -k https://<host>/health
# confirm no outbound DNS/connection attempts during startup or a request:
sudo tcpdump -i any -n 'not (net <internal-subnet>)' &
curl -k https://<host>/v1/chat/completions -H "Authorization: Bearer ks-..." -d '{...}'
# tcpdump should show nothing but internal traffic
```

Also see `docs/TELEMETRY_AUDIT.md`'s "How to re-verify" section for the
static (source-grep) half of this check — the `tcpdump` step above is the
live half, and (like the rest of this section) hasn't been run against a
genuinely network-isolated host in this pass.

## What's genuinely still open

- Steps 2 (model download) and 10 (live smoke test on a real air-gapped
  host) have not been executed end-to-end in this development
  environment — flagged, not hidden. Everything else in this runbook
  (steps 1, 3, 4, 5, 6, 7, and the underlying OpenBao/Grafana/Prometheus
  stack it brings up) has been run for real against live infrastructure
  during this project.
