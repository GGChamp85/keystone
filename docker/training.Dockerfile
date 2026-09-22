# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0
#
# Training image — the one environment the fine-tuning trainers
# (src/finetuning/*_train.py) actually execute in. Pinned to Python 3.12 on
# purpose: the ML stack (torch/transformers/peft/trl/bitsandbytes, and the
# fused-kernel packages that may join it later) lags the newest CPython
# releases, and the main app's own venv runs a newer interpreter where these
# wheels are not installable at all. Built CPU-only by default so CI and a
# laptop can run a real trainer smoke; a GPU deployment swaps the torch
# index for a CUDA build via TORCH_INDEX_URL.

FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# torch first from the chosen index (CPU by default), then the project's
# own `finetuning` extra resolves the rest against it.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url "${TORCH_INDEX_URL}" \
    && pip install --no-cache-dir -e ".[finetuning]"

RUN useradd --create-home --uid 1000 trainer \
    && mkdir -p /data/finetuning \
    && chown -R trainer:trainer /app /data/finetuning
USER trainer

ENV PYTHONPATH=/app
CMD ["python", "-c", "import torch, transformers, peft, trl; print('training image ready', torch.__version__)"]
