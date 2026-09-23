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
#
# The same image is what a KubeRay worker (helm/keystone/templates/kuberay.yaml,
# ADR 0002) and a RunPod training pod (src/finetuning/backends/runpod_pod.py,
# ADR 0005) run, so it carries Ray (`finetuning-ray` extra; drop it with
# PIP_EXTRAS=finetuning) and llama.cpp's GGUF converter at a pinned commit —
# the pin must equal src/finetuning/export.py's LLAMA_CPP_COMMIT (a test checks).

FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_EXTRAS=finetuning,finetuning-ray
ARG LLAMA_CPP_COMMIT=391fac16460f15233a7740550d858ac96df3419d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    KEYSTONE_GGUF_CONVERTER_DIR=/opt/llama.cpp

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp's convert_hf_to_gguf.py (MIT) with the `conversion/` package it imports and the matching
# `gguf-py/` library, taken from one pinned commit of the llama.cpp repository — nothing floats.
RUN mkdir -p /opt/llama.cpp \
    && curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.tar.gz" \
       | tar -xz -C /opt/llama.cpp --strip-components=1 --wildcards \
         "llama.cpp-${LLAMA_CPP_COMMIT}/convert_hf_to_gguf.py" \
         "llama.cpp-${LLAMA_CPP_COMMIT}/conversion/*" \
         "llama.cpp-${LLAMA_CPP_COMMIT}/gguf-py/*" \
    && test -f /opt/llama.cpp/convert_hf_to_gguf.py

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# torch first from the chosen index (CPU by default), then the project's
# own extras resolve the rest against it.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url "${TORCH_INDEX_URL}" \
    && pip install --no-cache-dir -e ".[${PIP_EXTRAS}]"

RUN useradd --create-home --uid 1000 trainer \
    && mkdir -p /data/finetuning /runpod-volume \
    && chown -R trainer:trainer /app /data/finetuning /runpod-volume /opt/llama.cpp
USER trainer

ENV PYTHONPATH=/app
CMD ["python", "-c", "import torch, transformers, peft, trl; print('training image ready', torch.__version__)"]
