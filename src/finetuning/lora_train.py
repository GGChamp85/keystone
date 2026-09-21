# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Enterprise Fine-Tuning — Phase 1: LoRA / QLoRA continued pre-training.
Domain adaptation on internal codebases using low-rank adaptation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LoRATrainingConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    training_data: str = ""
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
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
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

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        report_to="wandb" if config.wandb_project else "none",
        run_name=f"vs-lora-{config.base_model.split('/')[-1]}",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    logger.info("lora.training_start")
    train_result = trainer.train()

    # Save adapter
    adapter_path = os.path.join(config.output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    metrics = {
        "train_loss": train_result.metrics.get("train_loss", 0),
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "adapter_path": adapter_path,
    }
    logger.info("lora.training_complete", **metrics)
    return metrics
