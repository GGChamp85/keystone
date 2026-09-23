# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Enterprise Fine-Tuning — Phase 1: LoRA / QLoRA continued pre-training.
Domain adaptation on internal codebases using low-rank adaptation.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.finetuning._common import warmup_kwargs

logger = structlog.get_logger(__name__)


ProgressCallback = Callable[[dict[str, Any]], None]


def make_progress_callback(report: ProgressCallback):
    """A transformers TrainerCallback that reports every logged step (loss, eval loss, step/total, elapsed,
    ETA) to `report` — the live view an admin watches in `keystone finetune watch` and the web UI. Built
    inside a factory so this module stays importable without transformers installed."""
    from transformers import TrainerCallback

    started = time.monotonic()

    class _KeystoneProgress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            step = int(getattr(state, "global_step", 0) or 0)
            total = int(getattr(state, "max_steps", 0) or 0)
            elapsed = time.monotonic() - started
            eta = (elapsed / step) * (total - step) if step and total else None
            payload = {
                "step": step,
                "total_steps": total,
                "epoch": logs.get("epoch"),
                "loss": logs.get("loss"),
                "eval_loss": logs.get("eval_loss"),
                "learning_rate": logs.get("learning_rate"),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta, 1) if eta is not None else None,
            }
            try:
                report(payload)
            except Exception as exc:  # a progress hiccup must never fail the training run
                logger.warning("lora.progress_callback_failed", error=str(exc))

    return _KeystoneProgress()


@dataclass
class LoRATrainingConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # the catalog default SLM (src/inference/catalog.py)
    training_data: str = ""
    eval_data: str | None = None  # held-out split (src/finetuning/manifest.py) — without it, a run has no
    # way to tell "training loss went down" from "the model actually got better," and no way to compare
    # against a promotion baseline (base+RAG) on the exact same held-out tasks.
    eval_steps: int = 50
    output_dir: str = "/data/finetuning/output/lora"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    learning_rate: float = 2e-4
    num_epochs: int = 3
    # When set, overrides num_epochs with a hard optimizer-step budget — what a
    # bounded smoke run and a cost-estimated guided run both need.
    max_steps: int | None = None
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_seq_length: int = 8192
    fp16: bool = True
    bf16: bool = False
    use_4bit: bool = True  # QLoRA
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    seed: int = 42
    wandb_project: str | None = "keystone-finetuning"
    hf_token: str | None = None
    # Runtime-only (not part of the persisted job config): called with a progress dict — step, total_steps,
    # loss, eval_loss, elapsed_seconds, eta_seconds — after every logged step, from the training thread.
    progress: ProgressCallback | None = field(default=None, repr=False, compare=False)


def run_lora_training(config: LoRATrainingConfig) -> dict:
    """
    Execute LoRA/QLoRA continued pre-training.
    Requires: torch, transformers, peft, bitsandbytes, accelerate, datasets
    """
    import torch
    from datasets import load_dataset
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("lora.loading_model", model=config.base_model)

    # Quantization config for QLoRA
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        token=config.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=config.hf_token,
        # Match the weights dtype to the requested mixed-precision mode; with
        # neither fp16 nor bf16 set (a CPU run, for instance) the weights must
        # be float32 — forcing float16 there is what made CPU training fail.
        torch_dtype=torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32),
    )

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)

    # LoRA config
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    logger.info("lora.loading_data", path=config.training_data)
    dataset = load_dataset("json", data_files=config.training_data, split="train")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    eval_dataset = None
    if config.eval_data:
        logger.info("lora.loading_eval_data", path=config.eval_data)
        eval_raw = load_dataset("json", data_files=config.eval_data, split="train")
        eval_dataset = eval_raw.map(tokenize_fn, batched=True, remove_columns=eval_raw.column_names)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps if config.max_steps else -1,  # -1 = "use num_train_epochs"
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        **warmup_kwargs(config.warmup_ratio, TrainingArguments),
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        report_to="wandb" if config.wandb_project else "none",
        run_name=f"vs-lora-{config.base_model.split('/')[-1]}",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.eval_steps if eval_dataset is not None else None,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if config.progress is not None:
        trainer.add_callback(make_progress_callback(config.progress))

    # The verdict needs the BASE model's loss on the same held-out data. A freshly initialised LoRA
    # (B = 0) leaves the model's output identical to the base, so evaluating before the first step
    # is exactly the base model's held-out loss — no second model load.
    base_eval_loss = None
    if eval_dataset is not None:
        logger.info("lora.base_eval_start")
        base_eval_loss = trainer.evaluate().get("eval_loss")
        logger.info("lora.base_eval_done", base_eval_loss=base_eval_loss)

    logger.info("lora.training_start")
    train_result = trainer.train()

    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
    train_tokens = int(sum(len(ids) for ids in tokenized["input_ids"])) * max(config.num_epochs, 1)
    runtime = float(train_result.metrics.get("train_runtime", 0) or 0)

    # Save adapter
    adapter_path = os.path.join(config.output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    metrics = {
        "train_loss": train_result.metrics.get("train_loss", 0),
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "eval_loss": eval_metrics.get("eval_loss"),
        "base_eval_loss": base_eval_loss,
        "train_tokens": train_tokens,
        "train_tokens_per_second": round(train_tokens / runtime, 1) if runtime > 0 else None,
        "adapter_path": adapter_path,
    }
    logger.info("lora.training_complete", **metrics)
    return metrics
