#!/bin/bash
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
# Launch vLLM model servers individually (for non-Docker deployments)

set -euo pipefail

MODEL_ROLE="${1:-all}"
HF_TOKEN="${HF_TOKEN:-}"

case "$MODEL_ROLE" in
  coding)
    echo "Starting GLM-5.3-Flash on GPUs 0-7..."
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
      --model zai-org/GLM-5.3-Flash \
      --served-model-name glm-5.3-flash \
      --tensor-parallel-size 8 \
      --max-model-len 131072 \
      --gpu-memory-utilization 0.92 \
      --max-num-seqs 32 \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --max-num-batched-tokens 16384 \
      --block-size 16 --swap-space 8 \
      --trust-remote-code \
      --enable-auto-tool-choice --tool-call-parser glm47 \
      --host 0.0.0.0 --port 8001
    ;;
  coding-fallback)
    echo "Starting Qwen2.5-Coder-32B (coding fallback) on GPUs 8-11..."
    CUDA_VISIBLE_DEVICES=8,9,10,11 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-Coder-32B-Instruct \
      --served-model-name qwen-coder-32b \
      --tensor-parallel-size 4 \
      --max-model-len 131072 \
      --gpu-memory-utilization 0.92 \
      --max-num-seqs 32 \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --max-num-batched-tokens 16384 \
      --block-size 16 --swap-space 8 \
      --trust-remote-code \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --host 0.0.0.0 --port 8004
    ;;
  reasoning)
    echo "Starting DeepSeek-R1 on GPUs 12-13..."
    CUDA_VISIBLE_DEVICES=12,13 python -m vllm.entrypoints.openai.api_server \
      --model deepseek-ai/DeepSeek-R1 \
      --served-model-name deepseek-r1 \
      --tensor-parallel-size 2 \
      --max-model-len 65536 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 16 \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --max-num-batched-tokens 8192 \
      --block-size 16 --swap-space 4 \
      --trust-remote-code \
      --host 0.0.0.0 --port 8002
    ;;
  all)
    echo "Starting all models in background..."
    $0 coding &
    $0 coding-fallback &
    $0 reasoning &
    wait
    ;;
  *)
    echo "Usage: $0 {coding|coding-fallback|reasoning|all}"
    exit 1
    ;;
esac
