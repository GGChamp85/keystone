# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Enterprise Fine-Tuning — Phase 3: Direct Preference Optimization (DPO).
Aligns the model using chosen/rejected pairs (sandbox-verified).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import structlog

from src.finetuning._common import warmup_kwargs

logger = structlog.get_logger(__name__)


@dataclass
class DPOTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"  # the catalog default SLM (src/inference/catalog.py)
    sft_adapter_path: str | None = None  # Previous SFT adapter
    training_data: str = ""
    eval_data: str | None = None  # held-out split (src/finetuning/manifest.py)
    eval_steps: int = 50
    output_dir: str = "/data/finetuning/output/dpo"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    learning_rate: float = 5e-5
    num_epochs: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_length: int = 4096
    max_prompt_length: int = 2048
    beta: float = 0.1  # DPO temperature
    bf16: bool = True
    use_4bit: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42
    wandb_project: str | None = "keystone-finetuning"
    hf_token: str | None = None


def run_dpo_training(config: DPOTrainingConfig) -> dict:
    """
    Execute DPO training using TRL's DPOTrainer.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DPOConfig, DPOTrainer

    os.makedirs(config.output_dir, exist_ok=True)
    logger.info("dpo.loading_model", model=config.base_model, sft_adapter=config.sft_adapter_path)

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

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=config.hf_token,
        torch_dtype=torch.bfloat16,
    )

    # Merge SFT adapter if available
    if config.sft_adapter_path and os.path.exists(config.sft_adapter_path):
        logger.info("dpo.merging_sft_adapter", path=config.sft_adapter_path)
        model = PeftModel.from_pretrained(model, config.sft_adapter_path)
        model = model.merge_and_unload()

    # Reference model is a copy of the model (DPOTrainer handles this internally)
    ref_model = None  # DPOTrainer creates ref_model from model when not provided

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load preference dataset
    logger.info("dpo.loading_data", path=config.training_data)
    dataset = load_dataset("json", data_files=config.training_data, split="train")

    def format_dpo(example):
        return {
            "prompt": tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True),
            "chosen": tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False),
            "rejected": tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False),
        }

    dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)

    eval_dataset = None
    if config.eval_data:
        logger.info("dpo.loading_eval_data", path=config.eval_data)
        eval_dataset = load_dataset("json", data_files=config.eval_data, split="train")
        eval_dataset = eval_dataset.map(format_dpo, remove_columns=eval_dataset.column_names)

    dpo_config = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        **warmup_kwargs(config.warmup_ratio, DPOConfig),
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        beta=config.beta,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        report_to="wandb" if config.wandb_project else "none",
        run_name=f"vs-dpo-{config.base_model.split('/')[-1]}",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.eval_steps if eval_dataset is not None else None,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    logger.info("dpo.training_start")
    train_result = trainer.train()

    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}

    adapter_path = os.path.join(config.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    metrics = {
        "train_loss": train_result.metrics.get("train_loss", 0),
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "dpo_rewards_chosen": train_result.metrics.get("rewards/chosen", 0),
        "dpo_rewards_rejected": train_result.metrics.get("rewards/rejected", 0),
        "dpo_rewards_margins": train_result.metrics.get("rewards/margins", 0),
        "eval_loss": eval_metrics.get("eval_loss"),
        "adapter_path": adapter_path,
    }
    logger.info("dpo.training_complete", **metrics)
    return metrics
