# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Enterprise Fine-Tuning Pipeline — Phase 1: Data Preparation.
Processes internal repos and docs into training datasets for LoRA/SFT/DPO.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Keystone Agents, an expert coding assistant by Keystone. "
    "You write production-quality code with proper error handling, type hints, "
    "and documentation. You follow the conventions of the target codebase."
)


def prepare_continued_pretraining_data(
    repo_paths: list[str],
    output_path: str,
    file_extensions: list[str] | None = None,
    max_file_size_kb: int = 500,
) -> dict:
    """
    Phase 1: Domain Adaptation — Unsupervised continued pre-training data.
    Reads all code files from internal repos and creates a JSONL dataset.
    """
    if file_extensions is None:
        file_extensions = [".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c"]

    records = []
    files_processed = 0
    files_skipped = 0

    for repo_path in repo_paths:
        repo = Path(repo_path)
        if not repo.exists():
            logger.warning("data_prep.repo_not_found", path=repo_path)
            continue

        for fpath in repo.rglob("*"):
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in file_extensions:
                continue
            if fpath.stat().st_size > max_file_size_kb * 1024:
                files_skipped += 1
                continue
            # Skip common non-code directories
            parts = fpath.parts
            if any(p in {".git", "node_modules", "__pycache__", ".venv", "dist", "build"} for p in parts):
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                if len(content.strip()) < 50:
                    files_skipped += 1
                    continue
                rel_path = str(fpath.relative_to(repo))
                records.append(
                    {
                        "text": f"# File: {rel_path}\n{content}",
                        "source": str(repo.name),
                        "file_path": rel_path,
                        "language": fpath.suffix.lstrip("."),
                        "sha256": hashlib.sha256(content.encode()).hexdigest()[:16],
                    }
                )
                files_processed += 1
            except Exception as exc:
                logger.warning("data_prep.read_error", path=str(fpath), error=str(exc))
                files_skipped += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)

    stats = {
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "records": len(records),
        "output": output_path,
    }
    logger.info("data_prep.pretraining_complete", **stats)
    return stats


def prepare_sft_data(
    instruction_pairs: list[dict],
    internal_tasks: list[dict],
    output_path: str,
) -> dict:
    """
    Phase 2: Instruction & Tool-Use Tuning — Supervised Fine-Tuning data.

    instruction_pairs: list of {"instruction": str, "input": str, "output": str}
    internal_tasks: list of {"task": str, "context": str, "solution": str}
    """

    def _sft_record(user_content: str, assistant_content: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }

    # Public instruction data (e.g., from OpenCodeInstruct)
    records = [
        _sft_record(
            f"{pair['input']}\n{pair['instruction']}" if pair.get("input") else pair["instruction"],
            pair["output"],
        )
        for pair in instruction_pairs
    ]

    # Internal synthetic tasks
    records.extend(
        _sft_record(
            f"Context:\n{task['context']}\n\nTask:\n{task['task']}" if task.get("context") else task["task"],
            task["solution"],
        )
        for task in internal_tasks
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)

    stats = {
        "total_records": len(records),
        "public_pairs": len(instruction_pairs),
        "internal_tasks": len(internal_tasks),
        "output": output_path,
    }
    logger.info("data_prep.sft_complete", **stats)
    return stats


def prepare_dpo_data(
    preference_pairs: list[dict],
    output_path: str,
) -> dict:
    """
    Phase 3: Preference Alignment — DPO training data.

    preference_pairs: list of {
        "prompt": str,
        "chosen": str,   # passed sandbox tests
        "rejected": str,  # failed or hallucinated
    }
    """
    records = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pair["prompt"]},
            ],
            "chosen": [{"role": "assistant", "content": pair["chosen"]}],
            "rejected": [{"role": "assistant", "content": pair["rejected"]}],
        }
        for pair in preference_pairs
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)

    stats = {"total_pairs": len(records), "output": output_path}
    logger.info("data_prep.dpo_complete", **stats)
    return stats
