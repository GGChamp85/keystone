# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
A real trainer run — the one test in this repository that actually
executes src/finetuning/lora_train.py end to end: a real 0.5B base model
pulled from Hugging Face, a real LoRA adapter trained for a bounded number
of optimizer steps on CPU, a real held-out eval loss, and the saved adapter
loaded back onto the base with PEFT to prove it is a usable artifact — not
a fake runner standing in for the trainer (tests/test_finetuning_runner.py
covers the job-lifecycle plumbing with those; this covers the trainer).
Then the export path on that same adapter (src/finetuning/export.py): the
real PEFT merge reloaded with plain transformers, and llama.cpp's real
converter producing a GGUF file. A second test runs the RunPod pod
entrypoint (src/finetuning/backends/pod_entrypoint.py) in-process.

Gated on KEYSTONE_TRAINING_SMOKE=1 and an importable torch: the ML stack
is the `finetuning` extra, installed only in the training image
(docker/training.Dockerfile, Python 3.12) and CI's `train-smoke` job, not
in the main app environment. Two steps on a 0.5B model is a smoke, not a
training claim — it proves the path works, nothing about model quality.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import time
from pathlib import Path

import pytest

_BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

requires_training_stack = pytest.mark.skipif(
    os.environ.get("KEYSTONE_TRAINING_SMOKE") != "1" or importlib.util.find_spec("torch") is None,
    reason="Needs KEYSTONE_TRAINING_SMOKE=1 and the `finetuning` extra (torch/transformers/peft) installed",
)

_TRAIN_RECORDS = [
    "def add(a: int, b: int) -> int:\n    return a + b\n",
    "def is_even(n: int) -> bool:\n    return n % 2 == 0\n",
    "def clamp(x: float, lo: float, hi: float) -> float:\n    return max(lo, min(hi, x))\n",
    "class Counter:\n    def __init__(self) -> None:\n        self.n = 0\n\n"
    "    def inc(self) -> None:\n        self.n += 1\n",
    "def flatten(xs: list[list[int]]) -> list[int]:\n    return [x for row in xs for x in row]\n",
    "def safe_div(a: float, b: float) -> float | None:\n    return None if b == 0 else a / b\n",
    "def last(xs: list[str]) -> str | None:\n    return xs[-1] if xs else None\n",
    "def squares(n: int) -> list[int]:\n    return [i * i for i in range(n)]\n",
]
_EVAL_RECORDS = [
    "def double(n: int) -> int:\n    return n * 2\n",
    "def is_blank(s: str) -> bool:\n    return not s.strip()\n",
]


def _write_jsonl(path: Path, records: list[str]) -> str:
    path.write_text("".join(json.dumps({"text": r}) + "\n" for r in records))
    return str(path)


@requires_training_stack
def test_lora_trainer_runs_for_real_on_cpu_and_saves_a_loadable_adapter(tmp_path, monkeypatch):
    from src.finetuning.lora_train import LoRATrainingConfig, run_lora_training

    # transformers 5.x materializes weights on a thread pool by default; on
    # macOS arm64 (torch 2.14, transformers 5.17) that path segfaults inside
    # core_model_loading._materialize_copy before any training code runs —
    # observed for real, with a fault stack, on this project's dev machine.
    # HF_DEACTIVATE_ASYNC_LOAD is transformers' own documented switch for the
    # serial loader, harmless on Linux and a few seconds slower for a 0.5B.
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")

    config = LoRATrainingConfig(
        base_model=_BASE_MODEL,
        training_data=_write_jsonl(tmp_path / "train.jsonl", _TRAIN_RECORDS),
        eval_data=_write_jsonl(tmp_path / "eval.jsonl", _EVAL_RECORDS),
        output_dir=str(tmp_path / "out"),
        lora_r=8,
        lora_alpha=16,
        num_epochs=1,
        max_steps=2,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=256,
        fp16=False,
        bf16=False,
        use_4bit=False,
        gradient_checkpointing=False,
        logging_steps=1,
        save_steps=1000,
        wandb_project=None,
    )

    progress: list[dict] = []
    config.progress = progress.append
    metrics = run_lora_training(config)

    adapter_path = Path(metrics["adapter_path"])
    assert (adapter_path / "adapter_config.json").is_file()
    assert any(p.name.startswith("adapter_model") for p in adapter_path.iterdir())
    assert isinstance(metrics["train_loss"], float) and math.isfinite(metrics["train_loss"])
    assert isinstance(metrics["eval_loss"], float) and math.isfinite(metrics["eval_loss"])
    # The base model was evaluated on the same held-out data BEFORE training (the verdict's baseline),
    # and the real throughput was measured (what the planner replaces its estimate with).
    assert isinstance(metrics["base_eval_loss"], float) and math.isfinite(metrics["base_eval_loss"])
    assert metrics["train_tokens"] > 0 and metrics["train_tokens_per_second"] > 0
    # Live progress reached the callback from inside the training thread, one report per logged step.
    assert progress and progress[-1]["step"] == 2 and progress[-1]["total_steps"] == 2
    assert any(p.get("loss") is not None for p in progress)

    # The artifact must be a real, loadable adapter — the thing promote_job
    # registers and vLLM would serve — not just files on disk.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(_BASE_MODEL, torch_dtype=torch.float32)
    # is_trainable=True: PEFT loads an adapter frozen for inference by default,
    # which would leave nothing with requires_grad — the meaningful check is
    # that the saved adapter can be picked up again for continued training.
    loaded = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    param_names = [name for name, _ in loaded.named_parameters()]
    assert any("lora_A" in name for name in param_names), "no LoRA modules were attached from the adapter"
    trainable = sum(p.numel() for p in loaded.parameters() if p.requires_grad)
    assert trainable > 0
    del loaded, base

    # ── Export: merge the real adapter, reload the merged checkpoint, convert it to GGUF ──
    from transformers import AutoTokenizer

    from src.finetuning.export import ExportUnavailable as _ExportUnavailable
    from src.finetuning.export import (
        ensure_converter,
        export_awq,
        export_gguf,
        merge_adapter,
        run_export,
        scratch_converter_dir,
    )

    export_root = tmp_path / "out" / "export"
    merged = merge_adapter(_BASE_MODEL, adapter_path, export_root / "merged")
    merged_dir = Path(merged.path)
    assert (merged_dir / "config.json").is_file() and (merged_dir / "tokenizer_config.json").is_file()
    assert any(p.suffix == ".safetensors" for p in merged_dir.iterdir()), "merged weights must be safetensors"
    assert merged.size_bytes == sum(p.stat().st_size for p in merged_dir.rglob("*") if p.is_file()) > 0

    # A plain transformers load — no PEFT — and a real forward pass: the merge produced a usable model.
    reloaded = AutoModelForCausalLM.from_pretrained(str(merged_dir), torch_dtype=torch.float32)
    assert not any("lora" in name for name, _ in reloaded.named_parameters()), "LoRA modules must be folded in"
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    with torch.no_grad():
        logits = reloaded(**tokenizer("def add(a, b):", return_tensors="pt")).logits
    assert logits.shape[-1] == reloaded.config.vocab_size and bool(torch.isfinite(logits).all())
    del reloaded

    # llama.cpp's converter at the pinned commit: the training image vendors it; here it is fetched once.
    converter = ensure_converter(scratch_converter_dir())
    gguf = export_gguf(merged_dir, export_root / "model-q8_0.gguf", "q8_0", converter=converter)
    gguf_path = Path(gguf.path)
    assert gguf_path.is_file() and gguf.size_bytes == gguf_path.stat().st_size > 0
    with open(gguf_path, "rb") as f:
        assert f.read(4) == b"GGUF"
    assert gguf.quant == "q8_0"
    # Q8_0 is ~8.5 bits per weight: the file must be far smaller than the float32 checkpoint.
    assert gguf.size_bytes < merged.size_bytes / 2

    # The orchestrating call reuses the merged checkpoint rather than merging again.
    monkeypatch.setenv("KEYSTONE_GGUF_CONVERTER_DIR", str(converter.parent))
    second = run_export("gguf", base_model=_BASE_MODEL, adapter_dir=adapter_path, export_root=export_root, quant="f16")
    assert Path(second.path).name.endswith("-f16.gguf") and second.size_bytes > gguf.size_bytes

    # AWQ needs CUDA; on this CPU host the refusal is the verified behaviour.
    if not torch.cuda.is_available():
        with pytest.raises(_ExportUnavailable, match="CUDA"):
            export_awq(merged_dir, export_root / "awq")


@requires_training_stack
def test_the_runpod_pod_entrypoint_trains_for_real_and_serves_its_status(tmp_path, monkeypatch):
    """The process a RunPod training pod runs (src/finetuning/backends/pod_entrypoint.py), executed here
    in-process on CPU with the same 0.5B model: it trains from the job in its environment, writes the
    adapter and metrics.json to the output directory, reports progress then the result over its status
    port, and exits when asked to shut down — the whole contract the control plane's poller relies on,
    short of the pod itself."""
    import json as _json
    import socket
    import threading
    import urllib.request

    from src.finetuning.backends import pod_entrypoint

    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    output_dir = tmp_path / "volume" / "adapters" / "job1"
    job = {
        "id": "job1",
        "job_type": "lora",
        "base_model": _BASE_MODEL,
        "config": {
            "training_data": _write_jsonl(tmp_path / "train.jsonl", _TRAIN_RECORDS),
            "lora_r": 8,
            "lora_alpha": 16,
            "max_steps": 1,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 128,
            "fp16": False,
            "use_4bit": False,
            "gradient_checkpointing": False,
            "logging_steps": 1,
            "save_steps": 1000,
            "wandb_project": None,
        },
    }
    env = {
        pod_entrypoint.JOB_ENV: _json.dumps(job),
        pod_entrypoint.OUTPUT_DIR_ENV: str(output_dir),
        pod_entrypoint.STATUS_PORT_ENV: str(port),
        pod_entrypoint.LINGER_ENV: "600",
    }
    exit_code: list[int] = []
    thread = threading.Thread(target=lambda: exit_code.append(pod_entrypoint.main(env)), daemon=True)
    thread.start()

    def status() -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as resp:
            return _json.loads(resp.read())

    seen_states: set[str] = set()
    for _ in range(600):  # up to 5 minutes on a slow CPU
        try:
            body = status()
        except OSError:
            time.sleep(0.5)
            continue
        seen_states.add(body["state"])
        if body["state"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert body["state"] == "completed", body
    assert "running" in seen_states
    assert body["metrics"]["adapter_path"] == str(output_dir / "adapter")
    assert (output_dir / "adapter" / "adapter_config.json").is_file()
    assert _json.loads((output_dir / "metrics.json").read_text())["train_loss"] == body["metrics"]["train_loss"]
    assert body["progress"]["step"] == 1 and body["progress"]["total_steps"] == 1

    assert thread.is_alive(), "after completion the entrypoint lingers until the control plane has read the result"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/shutdown", method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
        assert _json.loads(resp.read()) == {"ok": True, "state": "completed"}
    thread.join(timeout=30)
    assert not thread.is_alive() and exit_code == [0]
