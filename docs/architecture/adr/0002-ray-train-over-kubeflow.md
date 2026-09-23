# ADR 0002 — Multi-GPU training: Ray Train via KubeRay, not Kubeflow

**Status**: accepted (2026-09-22); implementation tracked in ROADMAP Phase 5

## Context

Fine-tuning a 7B–32B model with LoRA/QLoRA on more than one GPU (or node) needs a distributed launcher. Two candidates were on the table:

- **Kubeflow Training Operator** — a `ClusterTrainingRuntime`/`TrainJob` CRD scheduler dedicated to training.
- **Ray Train on KubeRay** — Ray clusters as Kubernetes CRDs, with `TorchTrainer` wrapping an existing training entrypoint.

The Helm chart (`helm/keystone/templates/vllm.yaml`) already bootstraps a Ray cluster for multi-node vLLM serving (`nodeCount > 1`, pipeline parallelism across nodes). Air-gapped deployments bundle every operator image through `airgap/`.

## Decision

Ray Train via KubeRay. One compute stack — the same Ray operator, images and networking — serves models and trains adapters. `src/finetuning/backends/ray_train.py` (Phase 5) wraps the existing trainer entrypoints (`lora_train.py`, `sft_train.py`, `dpo_train.py`) in a `TorchTrainer`; `Settings.finetune_backend` selects `inprocess | ray | runpod_pod`. The Kubeflow runtime template is removed rather than kept as a second, unused scheduler.

## Consequences

- One more operator image in the air-gap bundle (KubeRay), not two.
- CI verifies the chart renders with `training.ray.enabled=true` and unit-tests the `TorchTrainer` config builder; a real multi-node run is GPU-gated and recorded in `docs/deployment/verification-log.md` when it happens.
- Teams standardised on Kubeflow Pipelines can still call Keystone's fine-tune API from a pipeline step; only the in-cluster launcher is Ray.
