# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — the fine-tune planner: model + hardware + dataset → a plan a
non-ML admin can approve.

Given a catalog entry, the GPUs available (detected, or stated), and the
dataset size, decide the training method (QLoRA on 4-bit weights when
memory is tight, LoRA on bf16 when it is not), say whether it fits, and
count the optimizer steps exactly. Time and cost are ESTIMATES and are
labelled with their basis: the step count is exact, the tokens-per-step
throughput comes from a conservative per-GPU-class table unless the
operator supplies a measured figure, and dollars appear only when an
hourly GPU price is given. Nothing here claims a precision it doesn't have.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.hardware import GPU
from src.inference.catalog import CatalogEntry

# Conservative training throughput (tokens/second, whole GPU, QLoRA-or-LoRA on a 7B-class model)
# by VRAM class. Real numbers vary with sequence length and kernels; the first real run replaces
# this with a measurement (the runner records tokens/sec) — until then the plan says "estimate".
_TOKENS_PER_SECOND_BY_VRAM_CLASS: tuple[tuple[float, float], ...] = (
    (80.0, 2_500.0),  # A100/H100 80 GB
    (48.0, 1_400.0),  # L40S / A6000 48 GB
    (24.0, 700.0),  # L4 / A5000 / 4090 24 GB
    (16.0, 350.0),  # T4 / A4000 16 GB
    (0.0, 150.0),  # anything smaller
)

AVG_TOKENS_PER_EXAMPLE = 1_200  # a task + diff SFT record, measured on the git-history source's output


@dataclass(frozen=True)
class TrainingPlan:
    base_model: str
    method: str  # "qlora" | "lora" | "unfit"
    fits: bool
    gpu_count: int
    gpu_name: str | None
    vram_available_gb: float
    vram_required_gb: float
    train_examples: int
    holdout_examples: int
    epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    total_steps: int
    lora_r: int
    estimated_tokens: int
    estimated_hours: float | None
    estimate_basis: str
    gpu_hourly_cost_usd: float | None
    estimated_cost_usd: float | None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _throughput_for(vram_gb: float) -> float:
    for threshold, tps in _TOKENS_PER_SECOND_BY_VRAM_CLASS:
        if vram_gb >= threshold:
            return tps
    return _TOKENS_PER_SECOND_BY_VRAM_CLASS[-1][1]


def plan_training(
    entry: CatalogEntry,
    gpus: list[GPU],
    *,
    train_examples: int,
    holdout_examples: int,
    epochs: int = 1,
    lora_r: int | None = None,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    gpu_hourly_cost_usd: float | None = None,
    measured_tokens_per_second: float | None = None,
) -> TrainingPlan:
    reasons: list[str] = []
    gpu_count = len(gpus)
    vram = min((g.vram_gb for g in gpus), default=0.0)  # the smallest device bounds a data-parallel run
    gpu_name = gpus[0].name if gpus else None

    if gpu_count == 0:
        method, fits, required = "unfit", False, entry.vram_qlora_gb
        reasons.append("no GPU detected — pass the target hardware explicitly or run on a GPU host")
    elif vram >= entry.vram_lora_bf16_gb:
        method, fits, required = "lora", True, entry.vram_lora_bf16_gb
        reasons.append(f"LoRA on bf16 weights fits: needs ~{required} GB, {vram} GB available per GPU")
    elif vram >= entry.vram_qlora_gb:
        method, fits, required = "qlora", True, entry.vram_qlora_gb
        reasons.append(f"QLoRA (4-bit base) fits: needs ~{required} GB, {vram} GB available per GPU")
    else:
        method, fits, required = "unfit", False, entry.vram_qlora_gb
        reasons.append(f"does not fit: QLoRA needs ~{required} GB per GPU, only {vram} GB available")

    effective_batch = per_device_batch_size * gradient_accumulation_steps * max(gpu_count, 1)
    steps_per_epoch = max(1, -(-train_examples // effective_batch))  # ceil
    total_steps = steps_per_epoch * max(epochs, 1)
    estimated_tokens = train_examples * max(epochs, 1) * AVG_TOKENS_PER_EXAMPLE

    if train_examples < 3:
        reasons.append("fewer than 3 training examples — the minimum for a meaningful run")
    if train_examples and holdout_examples == 0:
        reasons.append("no held-out examples — the verdict gate cannot compare against the base model")

    hours: float | None = None
    cost: float | None = None
    if fits:
        if measured_tokens_per_second:
            tps, basis = measured_tokens_per_second, "measured tokens/second supplied by the operator"
        else:
            tps, basis = (
                _throughput_for(vram) * max(gpu_count, 1),
                (
                    f"estimate: ~{_throughput_for(vram):.0f} tokens/s per {vram:.0f} GB-class GPU x {gpu_count}; "
                    "the first real run records the measured rate"
                ),
            )
        hours = round(estimated_tokens / tps / 3600, 2)
        if gpu_hourly_cost_usd is not None:
            cost = round(hours * gpu_hourly_cost_usd * max(gpu_count, 1), 2)
            basis += f"; cost at ${gpu_hourly_cost_usd}/GPU-hour"
        else:
            basis += "; no GPU price given, so no dollar figure"
    else:
        basis = "no estimate: the plan does not fit the hardware"

    return TrainingPlan(
        base_model=entry.hf_id,
        method=method,
        fits=fits,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        vram_available_gb=vram,
        vram_required_gb=required,
        train_examples=train_examples,
        holdout_examples=holdout_examples,
        epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        total_steps=total_steps,
        lora_r=lora_r or entry.default_lora_r,
        estimated_tokens=estimated_tokens,
        estimated_hours=hours,
        estimate_basis=basis,
        gpu_hourly_cost_usd=gpu_hourly_cost_usd,
        estimated_cost_usd=cost,
        reasons=reasons,
    )
