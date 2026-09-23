# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — export a trained adapter as something other runtimes can load.

Three real exports, each producing a real artifact on disk:

- `merge_adapter`: PEFT's `merge_and_unload` folds the LoRA deltas into the
  base weights and saves a plain Hugging Face checkpoint plus tokenizer —
  what vLLM, TGI, or `transformers` load without PEFT.
- `export_gguf`: llama.cpp's own `convert_hf_to_gguf.py` (pinned to one
  commit of the llama.cpp repository, vendored into the training image by
  docker/training.Dockerfile and fetched into a cache directory here when
  absent) turns the merged checkpoint into a GGUF file — the format the
  CPU demo backend (ADR 0003) and any llama.cpp deployment serve.
- `export_awq`: AWQ 4-bit quantisation through the `autoawq` package. It
  needs a CUDA GPU and that package; on a host without either it raises
  `ExportUnavailable` with the reason rather than writing something that
  looks like an artifact and is not.

Verified: the merge and the GGUF conversion run for real on the CPU smoke's
0.5B adapter (tests/test_trainer_smoke.py, `.venv-train`); the AWQ path
has not run anywhere — this project's development environment has no GPU.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

EXPORT_FORMATS = ("merged", "gguf", "awq")

# llama.cpp commit the converter is taken from — the same pin docker/training.Dockerfile uses
# (tests/test_finetune_export.py asserts the two agree). Bump both together.
LLAMA_CPP_COMMIT = "391fac16460f15233a7740550d858ac96df3419d"
LLAMA_CPP_TARBALL_URL = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.tar.gz"
# At this commit the converter is a script plus the `conversion/` package it imports, and it prefers the
# `gguf-py/` library next to it over an installed `gguf` — all three travel together.
CONVERTER_MEMBERS = ("convert_hf_to_gguf.py", "conversion/", "gguf-py/")
CONVERTER_SCRIPT = "convert_hf_to_gguf.py"
DEFAULT_CONVERTER_DIR = "/opt/llama.cpp"  # where docker/training.Dockerfile puts it
CONVERTER_DIR_ENV = "KEYSTONE_GGUF_CONVERTER_DIR"

# The converter's `--outtype` values that make sense for a served model (its ternary tq1_0/tq2_0 are for
# BitNet-style weights, not a LoRA-merged coder). Finer llama.cpp quantisations (q4_k_m, ...) are a second
# step with llama.cpp's `llama-quantize` binary, which the training image does not build.
GGUF_QUANT_TYPES = ("f32", "f16", "bf16", "q8_0")
DEFAULT_GGUF_QUANT = "q8_0"

AWQ_DEFAULT_QUANT_CONFIG: dict[str, Any] = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}


class ExportUnavailable(RuntimeError):
    """The export cannot run on this host (missing GPU or package); the message says which."""


@dataclass(frozen=True)
class ExportArtifact:
    format: str
    path: str
    size_bytes: int
    quant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def artifact_size(path: str | Path) -> int:
    """Bytes on disk: a file's size, or the sum of every file under a directory."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# ── the llama.cpp converter ─────────────────────────────────────


def converter_dir() -> Path:
    return Path(os.environ.get(CONVERTER_DIR_ENV) or DEFAULT_CONVERTER_DIR)


def ensure_converter(dest_dir: str | Path) -> Path:
    """The pinned converter at `dest_dir/convert_hf_to_gguf.py`, downloading the llama.cpp source
    tarball for LLAMA_CPP_COMMIT and extracting exactly the converter's files when it is not there
    yet — the same three paths docker/training.Dockerfile vendors. Returns the script path."""
    dest = Path(dest_dir)
    script = dest / CONVERTER_SCRIPT
    if script.is_file() and (dest / "conversion").is_dir():
        return script
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("export.fetching_converter", url=LLAMA_CPP_TARBALL_URL, dest=str(dest))
    with urllib.request.urlopen(LLAMA_CPP_TARBALL_URL, timeout=300) as resp:
        data = resp.read()
    prefix = f"llama.cpp-{LLAMA_CPP_COMMIT}/"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for member in tar.getmembers():
            if not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix) :]
            if any(relative == m.rstrip("/") or relative.startswith(m) for m in CONVERTER_MEMBERS):
                member.name = relative
                members.append(member)
        tar.extractall(dest, members=members, filter="data")
    if not script.is_file():
        raise RuntimeError(f"the llama.cpp tarball for {LLAMA_CPP_COMMIT} did not contain {CONVERTER_SCRIPT}")
    return script


# ── exports ─────────────────────────────────────────────────────


def merge_adapter(
    base_model: str,
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    hf_token: str | None = None,
    dtype: str = "auto",
) -> ExportArtifact:
    """Fold a LoRA adapter into its base model and save a plain Hugging Face checkpoint + tokenizer.

    `dtype="auto"` keeps float16 weights on a CUDA host and float32 on CPU (float16 matmuls are not
    implemented for every op on CPU, and the merge is one). Pass "float16"/"bfloat16"/"float32" to choose.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_dir)
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"{adapter_path} is not a PEFT adapter directory (no adapter_config.json)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if dtype == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    logger.info("export.merge_start", base_model=base_model, adapter=str(adapter_path), dtype=str(torch_dtype))
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, trust_remote_code=True, token=hf_token
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)

    # The trainer saves the tokenizer next to the adapter; fall back to the base model's.
    tokenizer_source = str(adapter_path) if (adapter_path / "tokenizer_config.json").is_file() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, token=hf_token)
    tokenizer.save_pretrained(str(out))

    artifact = ExportArtifact(format="merged", path=str(out), size_bytes=artifact_size(out))
    logger.info("export.merge_done", **artifact.to_dict())
    return artifact


def export_gguf(
    merged_dir: str | Path,
    out_file: str | Path,
    quant: str = DEFAULT_GGUF_QUANT,
    *,
    converter: str | Path | None = None,
    python: str | None = None,
) -> ExportArtifact:
    """Run llama.cpp's `convert_hf_to_gguf.py` on a merged checkpoint and check the result is a GGUF file
    (the `GGUF` magic in its first four bytes) rather than trusting the exit code alone."""
    if quant not in GGUF_QUANT_TYPES:
        raise ValueError(f"unsupported GGUF quant {quant!r}; expected one of {GGUF_QUANT_TYPES}")
    script = Path(converter) if converter else converter_dir() / CONVERTER_SCRIPT
    if not script.is_file():
        raise ExportUnavailable(
            f"llama.cpp's converter is not at {script}: set {CONVERTER_DIR_ENV} to a directory holding "
            f"convert_hf_to_gguf.py from llama.cpp commit {LLAMA_CPP_COMMIT} (docker/training.Dockerfile vendors it)"
        )
    merged = Path(merged_dir)
    if not (merged / "config.json").is_file():
        raise FileNotFoundError(f"{merged} is not a Hugging Face checkpoint directory (no config.json)")
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [python or sys.executable, str(script), str(merged), "--outfile", str(out), "--outtype", quant]
    logger.info("export.gguf_start", cmd=cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603 — fixed argv
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-20:]
        raise RuntimeError(f"convert_hf_to_gguf.py exited {result.returncode}:\n" + "\n".join(tail))
    if not out.is_file():
        raise RuntimeError(f"convert_hf_to_gguf.py exited 0 but wrote no file at {out}")
    with open(out, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        raise RuntimeError(f"{out} does not start with the GGUF magic (got {magic!r})")
    artifact = ExportArtifact(format="gguf", path=str(out), size_bytes=out.stat().st_size, quant=quant)
    logger.info("export.gguf_done", **artifact.to_dict())
    return artifact


def export_awq(
    merged_dir: str | Path,
    out_dir: str | Path,
    *,
    quant_config: dict[str, Any] | None = None,
) -> ExportArtifact:
    """AWQ 4-bit quantisation of a merged checkpoint with `autoawq`. Needs a CUDA GPU (the calibration
    pass runs the model) and the package; refuses clearly otherwise — no CPU imitation."""
    try:
        import torch
    except ImportError as exc:
        raise ExportUnavailable("AWQ export needs torch (the `finetuning` extra)") from exc
    if not torch.cuda.is_available():
        raise ExportUnavailable("AWQ export needs a CUDA GPU for the calibration pass; this host has none")
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:
        raise ExportUnavailable("AWQ export needs the `autoawq` package (pip install autoawq) on a CUDA host") from exc
    from transformers import AutoTokenizer

    merged = Path(merged_dir)
    if not (merged / "config.json").is_file():
        raise FileNotFoundError(f"{merged} is not a Hugging Face checkpoint directory (no config.json)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(quant_config or AWQ_DEFAULT_QUANT_CONFIG)

    logger.info("export.awq_start", merged=str(merged), quant_config=config)
    model = AutoAWQForCausalLM.from_pretrained(str(merged), safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(str(merged), trust_remote_code=True)
    model.quantize(tokenizer, quant_config=config)
    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))
    artifact = ExportArtifact(format="awq", path=str(out), size_bytes=artifact_size(out), quant=f"w{config['w_bit']}")
    logger.info("export.awq_done", **artifact.to_dict())
    return artifact


def run_export(
    fmt: str,
    *,
    base_model: str,
    adapter_dir: str | Path,
    export_root: str | Path,
    quant: str | None = None,
    hf_token: str | None = None,
) -> ExportArtifact:
    """One export of `fmt` for a completed job: the merged checkpoint is built once under
    `<export_root>/merged` and reused by the GGUF and AWQ exports."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; expected one of {EXPORT_FORMATS}")
    root = Path(export_root)
    merged_dir = root / "merged"
    if (merged_dir / "config.json").is_file():
        merged = ExportArtifact(format="merged", path=str(merged_dir), size_bytes=artifact_size(merged_dir))
    else:
        merged = merge_adapter(base_model, adapter_dir, merged_dir, hf_token=hf_token)
    if fmt == "merged":
        return merged
    if fmt == "gguf":
        q = quant or DEFAULT_GGUF_QUANT
        name = base_model.split("/")[-1].lower()
        return export_gguf(merged_dir, root / f"{name}-{q}.gguf", q)
    return export_awq(merged_dir, root / "awq")


def scratch_converter_dir() -> Path:
    """Where a host without the training image caches the fetched converter (the test uses it)."""
    return Path(tempfile.gettempdir()) / "keystone-llama.cpp" / LLAMA_CPP_COMMIT
