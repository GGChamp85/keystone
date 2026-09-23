# Hardware sizing

What one GPU tier serves. VRAM numbers for the Qwen2.5-Coder family come from `src/inference/catalog.py` (bf16 weights × 1.25 for KV-cache headroom; stated as estimates and kept conservative), the large-model rows from the vLLM configs in `src/inference/config.py`. Cloud prices are list prices at the time of writing (September 2026) and belong in your own cost model (`benchmarks/cost_model.py`), not in a decision made from this table alone.

| GPUs | Serves (bf16 unless noted) | Fine-tunes (QLoRA) | Typical instances |
|---|---|---|---|
| 1 × 24 GB (L4, A5000, RTX 4090) | Qwen2.5-Coder-7B (17.5 GB) · 14B in AWQ 4-bit | up to 7B (QLoRA ≈ 10.4 GB) — the default guided fine-tune target | AWS g6.xlarge · Azure NC A10 · GCP g2-standard-8 · RunPod L4 / A5000 (~$0.69/h) |
| 2 × 24 GB | Qwen2.5-Coder-14B (35 GB, tensor-parallel 2) | 14B (QLoRA ≈ 18.8 GB) | 2× the above |
| 1 × 80 GB (A100/H100) | Qwen2.5-Coder-32B (80 GB — tight; prefer AWQ or 2 × 80 GB) | 32B (QLoRA ≈ 40.4 GB) | AWS p4d partition / g6e (L40S 48 GB for AWQ) · Azure NC24ads A100 v4 · GCP a2-highgpu-1g · RunPod A100 (~$2.72/h) |
| 4 × 80 GB | Qwen2.5-Coder-32B-Instruct at 128K context (the `coding_fallback` default) | LoRA on bf16 32B | AWS p4d.24xlarge partition · Azure ND A100 v4 · GCP a2-highgpu-4g |
| 8 × 80 GB | GLM-5.3-Flash (FP8 on disk, ~328 GB of weights; the `coding` default) | — | AWS p4de.24xlarge · Azure ND96amsr A100 v4 · GCP a3-highgpu-8g · RunPod 8×H100 |
| CPU only | Qwen2.5-Coder-0.5B via llama.cpp (the `demo-model` profile) — plumbing and smoke tests, not quality | 0.5B (the CI trainer smoke) | any laptop |

Per-model numbers, including the LoRA-on-bf16 footprint, are on every catalog card in the web UI's Model Library (`GET /v1/keystone/models`), and `keystone doctor` reports the GPUs it detects.

| Model | Params | Serve (bf16) | QLoRA train | LoRA train (bf16) |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct | 0.5 B | 1.2 GB | 2.6 GB | 3.6 GB |
| Qwen2.5-Coder-1.5B-Instruct | 1.5 B | 3.8 GB | 3.8 GB | 6.8 GB |
| Qwen2.5-Coder-3B-Instruct | 3 B | 7.5 GB | 5.6 GB | 11.6 GB |
| Qwen2.5-Coder-7B-Instruct | 7 B | 17.5 GB | 10.4 GB | 24.4 GB |
| Qwen2.5-Coder-14B-Instruct | 14 B | 35.0 GB | 18.8 GB | 46.8 GB |
| Qwen2.5-Coder-32B-Instruct | 32 B | 80.0 GB | 40.4 GB | 104.4 GB |

The formulas: serve = params × 2 × 1.25; QLoRA = params × 0.5 + 0.35 × params × 2 + 2; LoRA bf16 = params × 2 × 1.6 + 2 (GB, params in billions). `src/finetuning/planner.py` applies them to the GPUs it detects and refuses a plan that does not fit unless forced.
