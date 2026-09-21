# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Enterprise Fine-Tuning — Phase 2: Supervised Fine-Tuning (SFT).
Instruction tuning on code instruction-response pairs using TRL's SFTTrainer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SFTTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    adapter_path: str | None = None  # Previous LoRA adapter to continue from
    training_data: str = ""
    output_dir: str = "/data/finetuning/output/sft"
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
    learning_rate: float = 1e-4
    num_epochs: int = 2
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    max_seq_length: int = 8192
    bf16: bool = True
    use_4bit: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42
    wandb_project: str | None = "keystone-finetuning"
    hf_token: str | None = None
    packing: bool = True


def run_sft_training(config: SFTTrainingConfig) -> dict:
    """
    Execute SFT training using TRL's SFTTrainer with LoRA.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    os.makedirs(config.output_dir, exist_ok=True)
    logger.info("sft.loading_model", model=config.base_model, adapter=config.adapter_path)

    # Quantization
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True, token=config.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=config.hf_token,
        torch_dtype=torch.bfloat16,
    )

    # If previous LoRA adapter exists, merge and re-apply LoRA
    if config.adapter_path and os.path.exists(config.adapter_path):
        logger.info("sft.merging_lora_adapter", path=config.adapter_path)
        model = PeftModel.from_pretrained(model, config.adapter_path)
        model = model.merge_and_unload()

    # New LoRA for SFT phase
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset
    logger.info("sft.loading_data", path=config.training_data)
    dataset = load_dataset("json", data_files=config.training_data, split="train")

    def format_chat(example):
        messages = example.get("messages", [])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        report_to="wandb" if config.wandb_project else "none",
        run_name=f"vs-sft-{config.base_model.split('/')[-1]}",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        packing=config.packing,
    )

    logger.info("sft.training_start")
    train_result = trainer.train()

    adapter_path = os.path.join(config.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    metrics = {
        "train_loss": train_result.metrics.get("train_loss", 0),
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "adapter_path": adapter_path,
    }
    logger.info("sft.training_complete", **metrics)
    return metrics
