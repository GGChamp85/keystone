#!/usr/bin/env bash
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Run on the BUILD MACHINE (has internet access). Pulls every container
# image Keystone needs, tags each with the internal registry prefix, and
# saves them as tarballs for transfer into the air-gapped environment.
# Companion to import_bundle.sh, which runs on the air-gapped side.
#
# Usage:
#   ./airgap/build_image_bundle.sh [output_dir]
#
# Every image is pinned to an exact tag — never `:latest` — so the bundle
# that ships to a client is reproducible and so a later re-run of this
# script doesn't silently swap in a newer, unvetted image.

set -euo pipefail

OUTPUT_DIR="${1:-./airgap/output/images}"
REGISTRY_PREFIX="${KEYSTONE_INTERNAL_REGISTRY:-registry.internal.keystone.local}"

mkdir -p "$OUTPUT_DIR"

# name:published_tag — the exact, pinned versions this repo actually
# depends on (docker-compose.yml, helm/keystone/values.yaml, Dockerfiles).
# Keep this list in sync with those files; do not add a bare `:latest`.
IMAGES=(
  "postgres:16-alpine"
  "valkey/valkey:7.2-alpine" # BSD-3-Clause Redis fork — see docker-compose.yml's redis service comment
  "qdrant/qdrant:v1.11.0"
  "vllm/vllm-openai:v0.29.0" # reasoning/coding-fallback roles
  "vllm/vllm-openai:glm53-flash" # coding role (GLM-5.3-Flash) — vLLM's own dedicated image for this model
  "temporalio/auto-setup:1.24"
  "temporalio/ui:2.27.0"
  "nginx:1.27-alpine"
  "openbao/openbao:2.1"
  "prom/prometheus:v2.55.0"
  "prom/alertmanager:v0.27.0"
  "grafana/grafana:11.2.0"
  "grafana/loki:3.2.0"
  "grafana/promtail:3.2.0"
  "quay.io/kuberay/operator:v1.7.1" # KubeRay operator — the RayCluster in helm/keystone/templates/kuberay.yaml (ADR 0002); installed once per cluster
)

echo "== Keystone air-gap image bundle build =="
echo "Output dir: $OUTPUT_DIR"
echo "Registry prefix for re-tagging: $REGISTRY_PREFIX"
echo

MANIFEST="$OUTPUT_DIR/manifest.txt"
: > "$MANIFEST"

for image in "${IMAGES[@]}"; do
  echo "--- Pulling $image ---"
  docker pull "$image"

  retagged="$REGISTRY_PREFIX/$image"
  docker tag "$image" "$retagged"

  safe_name="$(echo "$image" | tr '/:' '__')"
  tarball="$OUTPUT_DIR/${safe_name}.tar"
  echo "Saving to $tarball ..."
  docker save "$image" -o "$tarball"

  digest="$(docker inspect --format='{{index .RepoDigests 0}}' "$image" 2>/dev/null || echo 'unknown')"
  sha256="$(shasum -a 256 "$tarball" | awk '{print $1}')"
  echo "$image|$retagged|$tarball|$digest|$sha256" >> "$MANIFEST"
  echo
done

# The Keystone-built images (app, sandbox-daemon, and the three sandbox runtime images the daemon
# starts tasks in) aren't pulled from a registry — build them locally first (`make build`, which
# runs `make sandbox-images`) then include here so the whole bundle is one self-consistent artifact.
for local_image in "keystone-app:latest" "keystone-sandbox-daemon:latest" \
  "keystone-sandbox-python:latest" "keystone-sandbox-node:latest" "keystone-sandbox-go:latest"; do
  if docker image inspect "$local_image" >/dev/null 2>&1; then
    echo "--- Including locally-built $local_image ---"
    safe_name="$(echo "$local_image" | tr '/:' '__')"
    tarball="$OUTPUT_DIR/${safe_name}.tar"
    docker save "$local_image" -o "$tarball"
    sha256="$(shasum -a 256 "$tarball" | awk '{print $1}')"
    echo "$local_image|$REGISTRY_PREFIX/$local_image|$tarball||$sha256" >> "$MANIFEST"
  else
    echo "WARNING: $local_image not found locally — build it first (docker build ...) before bundling." >&2
  fi
done

echo
echo "== Done. Manifest: $MANIFEST =="
echo "Transfer the entire $OUTPUT_DIR directory into the air-gapped environment,"
echo "then run airgap/import_bundle.sh there."
