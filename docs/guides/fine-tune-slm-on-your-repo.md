# Fine-tune a small model on your own repositories (guided)

Describe what the adapter should get better at, review a plan with its cost, approve it, watch it train. No JSONL to prepare, no hyperparameters to pick — the defaults are the ones that work for a repository-conventions adapter (LoRA rank 8, one epoch, QLoRA when the GPU is small, a repository-stratified 20 % holdout), and every number in the plan says where it came from.

Verified on: 2026-09-22 — `tests/test_finetune_guided.py` runs the whole path against a real git repository and a real Postgres: dataset built from real commits, a record carrying a credential dropped, a record carrying an email redacted, a `planned` job created, approved into `pending`, and the same over the HTTP routes.

## What you need

- A Keystone deployment with the `finetune` API scope on your key.
- One or more repositories on your allow-listed git host (`GIT_ALLOWED_HOSTS`); the bot token (`GIT_HOST_TOKEN`) is used to clone them.
- A GPU: the planner detects the server's GPUs, or you state the target (`--gpu "NVIDIA L4:22.5"`) when training will run elsewhere.

## 1. Describe → plan

```bash
keystone finetune guided \
  --repo https://git.internal/acme/payments.git \
  --repo https://git.internal/acme/billing.git \
  --goal "Follow our conventions when fixing bugs in the payments services" \
  --gpu "NVIDIA L4:22.5" --gpu-hourly-cost 0.69
```

The server:

1. clones each repository and turns its commit history into task → diff examples (`src/finetuning/sources/git_history.py`), and adds every agent task this tenant has **accepted or merged** (`sources/trajectories.py`);
2. drops any record that carries a credential (key-shaped content, private-key blocks) and redacts PII (email, phone, SSN, card, IP) in the rest — nothing that leaves your repositories is ever more sensitive than what is in them, and usually less;
3. splits by repository (no repository on both sides) and writes a manifest with content hashes;
4. picks the model (`auto` = the 7B coder, stepped down to the largest that fits your GPU) and the method (QLoRA on 4-bit weights when memory is tight, LoRA on bf16 when it is not), counts the optimizer steps exactly, and estimates time and cost with the basis stated;
5. saves it all as a `planned` job — nothing trains yet.

You see something like:

```
Base model   Qwen/Qwen2.5-Coder-7B-Instruct
Method       qlora  (fits)
Hardware     1 x NVIDIA L4 (22.5 GB; needs ~10.4 GB)
Examples     412 train / 103 held out  (from 515 collected)
Sources      payments.git: 318, billing.git: 171; accepted tasks: 26
Safety       3 record(s) dropped by the secret scan, 11 with PII redacted
Steps        26 optimizer steps (1 epoch(s), LoRA r=8)
Time         ~0.2 h
Cost         ~$0.14
Basis        estimate: ~700 tokens/s per 22 GB-class GPU x 1; the first real run records the measured rate; cost at $0.69/GPU-hour
Approve and start training? [y/N]
```

The same flow in the web UI: **Fine-tune → + New job** (step 1 describe, step 2 review, step 3 watch), and over the API: `GET /v1/finetune/catalog`, `POST /v1/finetune/plans`, `GET /v1/finetune/plans/{id}`, `POST /v1/finetune/plans/{id}/approve`.

## 2. Approve → train

`y` (or `--yes`, or `keystone finetune approve <job-id>` later) flips the job to `pending` and starts it through the same runner every fine-tune uses (Temporal-backed, asyncio fallback). Follow it with `keystone finetune watch <job-id>` or the web UI. A plan that does not fit the hardware is refused unless you pass `--force` (for hardware the planner could not see).

## 3. Verdict → promote → served in seconds

While it trains, `keystone finetune watch <job-id>` and the web UI show every logged step — loss, held-out loss, step/total, elapsed, ETA — from the trainer's own callback, not a status poll.

Before the first optimizer step the trainer evaluates the **untouched base model** on the held-out split; after training it evaluates the adapter on the same split. The job's metrics carry both (`base_eval_loss`, `eval_loss`) and a **verdict** (`src/finetuning/verdict.py`):

| Verdict | Meaning |
|---|---|
| `pass` | the adapter's held-out loss beats the base model's by at least 1 % — worth serving |
| `fail` | it does not — the base model is at least as good on data it never saw |
| `unknown` | no held-out comparison (no holdout split, or a job trained before the base eval existed) |

`keystone finetune promote <job-id>` is gated on it: a `fail` or `unknown` verdict is refused with HTTP 409 and the reason. An **admin** may still promote with `keystone finetune promote <job-id> --force` (`POST /v1/finetune/jobs/{id}/promote?force=true`); the force is written to the audit log (`finetune.promote_forced`, with the verdict it overrode). Nobody else can.

A promotion registers the adapter as this tenant's default for its base model (`ModelAdapter`), and the gateway routes the tenant's requests to it — every gateway call and every agent task, no client change (model names are roles). It is then made **servable at once** (`src/inference/lora_registry.py`):

1. `<FINETUNING_OUTPUT_DIR>/lora_modules.json` is rewritten with every promoted adapter, grouped by base model, plus the ready-made `--lora-modules name=path …` argv. The serving launch reads it, so a restart serves everything that was promoted (Helm: `vllm.<role>.lora.enabled`; compose: `VLLM_CODING_EXTRA_ARGS`).
2. The role serving that base model gets `POST /v1/load_lora_adapter` — vLLM loads it **without a restart** when the server runs with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` (set by the chart and the compose file when LoRA is enabled). A server that refuses answers with its status, which is reported in the promote response's `metrics.serving`, never hidden; the manifest still covers the next restart.

The adapter directory must be visible to the vLLM process: mount the same volume the trainer wrote to (`FINETUNING_OUTPUT_DIR`) into the serving pods.

Verified for real (`tests/test_finetune_promote_gate.py`, real Postgres + the real route): the failing job is refused, a non-admin's `force` is refused, an admin's force is audited, the passing job promotes, the manifest lists both, and the live load is attempted for each. The base-model eval and the per-step progress are exercised by the CPU trainer run in CI (`tests/test_trainer_smoke.py`).

## 4. Export the adapter

What you get: the trained adapter in a form other runtimes load without Keystone — a plain Hugging Face checkpoint with the adapter folded into the base weights (`merged`), a GGUF file for llama.cpp and the CPU demo backend (`gguf`), or an AWQ 4-bit checkpoint for vLLM on a GPU (`awq`). Exports live next to the adapter under `<output_dir>/export/` and are recorded on the job (`metrics.exports.<format>`: path, size in bytes, quant, or the error).

```bash
keystone finetune export <job-id> --format gguf --quant q8_0 --wait
#   Export gguf (q8_0) started for <job-id> via asyncio_fallback — output under /data/finetuning/output/guided/<job-id>/export
#   Done /data/finetuning/output/guided/<job-id>/export/qwen2.5-coder-7b-instruct-q8_0.gguf  (8,101,234,688 bytes)
keystone finetune export <job-id> --format merged
keystone finetune export <job-id> --format awq          # needs a CUDA GPU and the autoawq package
```

Over the API: `POST /v1/finetune/jobs/{id}/export {"format": "gguf", "quant": "q8_0"}` answers 202 and runs the export in the background through the same mechanism as training (Temporal when reachable, the asyncio fallback otherwise); only a `completed` job with an adapter on disk is accepted (409 otherwise), `quant` is one of `f32`, `f16`, `bf16`, `q8_0` (422 otherwise). The merge and the conversion run where the trainers run (the training image, `docker/training.Dockerfile`, which vendors llama.cpp's converter at a pinned commit); the API process itself has no ML stack and records the reason if it is asked to export where none is installed.

Verified for real (`tests/test_trainer_smoke.py`, CPU, the 0.5B model): the smoke's adapter is merged with PEFT, the merged checkpoint is reloaded with plain transformers and runs a forward pass, and llama.cpp's converter writes a GGUF file whose header carries the `GGUF` magic and which is less than half the float32 checkpoint's size at `q8_0`. The route's refusals and the background outcome are verified over real HTTP against a real Postgres (`tests/test_finetune_export.py`). **Not verified**: the AWQ export — it needs a CUDA GPU, which this project's development environment does not have; on a CPU host it refuses with that reason rather than writing anything.

## 5. Train on more than one GPU / on a rented pod

`FINETUNE_BACKEND` chooses where a job's trainer runs; the job, the API, the CLI, the wizard, the verdict and the promotion are the same in every case.

| Backend | What you get | Set |
|---|---|---|
| `inprocess` (default) | the trainer runs in the API/worker process on that host's GPUs — the path every test above exercises | nothing |
| `ray` | a Ray Train job on a KubeRay cluster (ADR 0002): one Ray worker per GPU, data-parallel, the adapter written to the shared fine-tuning volume, progress streamed as usual | `helm ... --set training.ray.enabled=true` (sets `FINETUNE_BACKEND=ray` and `RAY_ADDRESS` in the chart's ConfigMap); `RAY_TRAIN_NUM_WORKERS` (0 = the plan's GPU count) |
| `runpod_pod` | an on-demand RunPod GPU pod runs the training image with the job in its environment and writes the adapter to a RunPod network volume at `/runpod-volume/adapters/<job-id>` — the path a serverless endpoint created with the same volume serves (ADR 0005); the pod is deleted when the result is in, so billing stops | `FINETUNE_BACKEND=runpod_pod`, `RUNPOD_API_KEY`, `RUNPOD_NETWORK_VOLUME_ID`, `RUNPOD_TRAINING_IMAGE` (the training image on a registry RunPod can pull from); GPU tier via `RUNPOD_TRAINING_GPU_TYPE_IDS` |

```bash
# KubeRay: the operator once per cluster, then the chart with the training cluster enabled
helm upgrade --install keystone helm/keystone -f helm/keystone/values-client-vpc.yaml --set training.ray.enabled=true
keystone finetune guided --repo ... --goal ...        # unchanged; the job now runs on the Ray workers

# RunPod: a network volume in the pod's region, the training image on a registry, then the same commands
FINETUNE_BACKEND=runpod_pod RUNPOD_NETWORK_VOLUME_ID=<volume id> RUNPOD_TRAINING_IMAGE=registry.example/keystone-training:2026.09
keystone finetune guided --repo ... --goal ... --gpu "NVIDIA L4:22.5" --gpu-hourly-cost 0.44
```

What is verified, and what is not:

- **Ray**: the `TorchTrainer` configuration is built by pure functions (`src/finetuning/backends/ray_train.py`: `build_scaling_config`, `build_run_config`, `ray_job_spec`) that `tests/test_ray_train_backend.py` asserts exactly, and, where `ray[train]` is installed (the `.venv-train` environment, the training image), constructs the real `ScalingConfig`, `RunConfig` and `TorchTrainer` objects. The submission and polling loop — submit, poll, relay each progress line once, read the metrics line, surface a failure with Ray's message and the log tail, stop on timeout — runs against a real local server speaking Ray's Jobs REST API. The chart renders the `RayCluster` (head, GPU worker group, model cache and fine-tuning output PVCs) in CI. **No real Ray cluster and no GPU were available: a real multi-worker run has not happened**, and the data-parallel behaviour of transformers' Trainer inside a Ray worker is documented upstream, not observed here.
- **RunPod pod**: the `PodCreateInput` the backend sends is asserted against RunPod's published OpenAPI document (`tests/fixtures/runpod_openapi_pods.json`, a verbatim subset), and the whole loop — create, poll the pod and its status endpoint, relay progress, read the metrics, delete the pod on success, failure and timeout — runs against a real local server that speaks those shapes and rejects undocumented fields. The process the pod runs (`src/finetuning/backends/pod_entrypoint.py`) is executed for real on CPU in `tests/test_trainer_smoke.py`: it trains, writes the adapter and `metrics.json`, serves its status, and exits on the shutdown request. **No pod has been launched: the RunPod account has no credit**, so pod scheduling, the image pull, the network volume mount and the proxied status port have not been observed against RunPod itself.

## What "estimate" means here

- **Steps** are exact: `ceil(train_examples / (batch × accumulation × GPUs)) × epochs`.
- **Tokens** use a measured average per git-history record (`AVG_TOKENS_PER_EXAMPLE` in `src/finetuning/planner.py`).
- **Time** divides tokens by a conservative per-GPU-class throughput table until your first real run records a measured rate; pass a measured `tokens/s` to replace it.
- **Cost** appears only when you give a GPU price. Nothing is invented.
