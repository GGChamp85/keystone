#!/usr/bin/env bash
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Run on the BUILD MACHINE (has internet access). Downloads every model
# weight Keystone Inference / Keystone Agents needs — the three vLLM-served
# LLMs plus the BGE embedding model used by src/memory/embeddings.py — into
# a local directory laid out exactly as `huggingface_hub.snapshot_download`
# leaves it, so it can be bind-mounted straight into the air-gapped
# environment with HF_HUB_OFFLINE=1 and no further network access.
#
# Both large models dominate this bundle's total size — budget bandwidth/
# disk accordingly: DeepSeek-R1 is natively FP8 on disk at ~688GB (684B
# total params), and GLM-5.3-Flash is natively FP8 at ~328GB (see
# src/inference/config.py's coding_model_config() docstring for its real
# verified footprint) — both real, verified sizes, not estimates.
#
# Usage:
#   ./airgap/download_models.sh [output_dir]
#   HF_TOKEN=hf_xxx ./airgap/download_models.sh   # DeepSeek-R1/Qwen/GLM are open,
#                                                  # no token needed today, but some
#                                                  # gated models may require one —
#                                                  # keep this path available.
#
# Model IDs match .env.example / docker-compose.yml exactly — keep them in
# sync; this script does not invent its own model list.

set -euo pipefail

OUTPUT_DIR="${1:-./airgap/models}"

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
python3 -c "import huggingface_hub" 2>/dev/null || {
  echo "Installing huggingface_hub (build-machine only; not needed air-gapped) ..."
  pip install -q "huggingface_hub[cli]>=0.24"
}

mkdir -p "$OUTPUT_DIR"

# repo_id|local_dirname — local_dirname is what src/config.py's
# *_model_path / embedding_model_path settings should point at once this
# bundle is imported.
MODELS=(
  "zai-org/GLM-5.3-Flash|glm-5.3-flash"
  "Qwen/Qwen2.5-Coder-32B-Instruct|qwen2.5-coder-32b-instruct"
  "deepseek-ai/DeepSeek-R1|deepseek-r1"
  "BAAI/bge-large-en-v1.5|bge-large-en-v1.5"
)

echo "== Keystone air-gap model bundle download =="
echo "Output dir: $OUTPUT_DIR"
echo

MANIFEST="$OUTPUT_DIR/manifest.txt"
: > "$MANIFEST"

for entry in "${MODELS[@]}"; do
  repo_id="${entry%%|*}"
  local_name="${entry##*|}"
  dest="$OUTPUT_DIR/$local_name"

  echo "--- Downloading $repo_id -> $dest ---"
  python3 - "$repo_id" "$dest" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

repo_id, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo_id,
    local_dir=dest,
    # Some repos ship redundant .pt/.bin/.consolidated variants alongside
    # the safetensors weights vLLM/transformers actually load — skip them
    # so the bundle isn't 2x larger than necessary.
    ignore_patterns=["*.pt", "*.bin", "*.consolidated*", "*.gguf", "original/*"],
)
print(f"Downloaded to: {path}")
PYEOF

  size="$(du -sh "$dest" 2>/dev/null | awk '{print $1}')"
  file_count="$(find "$dest" -type f | wc -l | tr -d ' ')"
  echo "$repo_id|$local_name|$size|${file_count} files" >> "$MANIFEST"
  echo
done

echo "== Done. Manifest: $MANIFEST =="
echo "Total bundle size: $(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')"
echo
echo "Transfer $OUTPUT_DIR into the air-gapped environment (e.g. under"
echo "/opt/keystone/models/), then point the deployment at local paths:"
echo "  - docker-compose.yml vllm-* services: --model /models/glm-5.3-flash (etc.)"
echo "  - .env / Helm values: EMBEDDING_MODEL_PATH=/models/bge-large-en-v1.5"
echo "  - HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 on every service that touches these"
echo "    (see docs/airgap/OFFLINE_INSTALL_RUNBOOK.md step 8 for exact locations)."
