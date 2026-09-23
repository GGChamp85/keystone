# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — GPU detection, for the fine-tune planner and `keystone doctor`.

Two real sources, in order: `torch.cuda` when torch is importable (the
training image), else `nvidia-smi` (any host with the driver). Neither
present means no GPUs — a laptop planning a run for a GPU box elsewhere
passes the target hardware explicitly instead.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    vram_gb: float

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "vram_gb": self.vram_gb}


def detect_gpus() -> list[GPU]:
    gpus = _from_torch()
    if gpus is None:
        gpus = _from_nvidia_smi()
    return gpus or []


def _from_torch() -> list[GPU] | None:
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            out.append(GPU(name=props.name, vram_gb=round(props.total_memory / 1024**3, 1)))
        return out
    except Exception:
        return None


def _from_nvidia_smi() -> list[GPU] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return [gpu for gpu in (parse_nvidia_smi_line(line) for line in result.stdout.splitlines()) if gpu]


def parse_nvidia_smi_line(line: str) -> GPU | None:
    """`NVIDIA L4, 23034` (memory in MiB with --nounits) -> GPU("NVIDIA L4", 22.5)."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        mib = float(parts[1].split()[0])
    except ValueError:
        return None
    return GPU(name=parts[0], vram_gb=round(mib / 1024, 1))
