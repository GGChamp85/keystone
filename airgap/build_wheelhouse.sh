#!/usr/bin/env bash
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Run on the BUILD MACHINE (has internet access). Downloads every Python
# wheel Keystone's app/worker/sandbox-daemon images need — core deps plus,
# optionally, the `finetuning` extra (torch/transformers/etc., large) — into
# a local wheelhouse directory. The Dockerfile's air-gapped build stage
# installs with `pip install --no-index --find-links=/wheelhouse ...`
# against this directory instead of reaching PyPI.
#
# Usage:
#   ./airgap/build_wheelhouse.sh [output_dir] [--with-finetuning]
#
# Platform note: wheels are platform/ABI-specific. The Dockerfile builds on
# python:3.12-slim (Debian, linux/x86_64) regardless of what OS/arch this
# script runs on (e.g. macOS/ARM during development) — so by default this
# targets that exact platform/ABI via `pip download --platform
# manylinux2014_x86_64 --python-version 312 --only-binary=:all:` rather than
# whatever interpreter happens to run this script. Verified: downloading
# for linux/x86_64 py3.12 from a macOS/ARM build machine works cleanly for
# every dependency in pyproject.toml (no sdist-only stragglers). Override
# with TARGET_PLATFORM/TARGET_PYTHON_VERSION env vars if the Dockerfile's
# base image ever changes.

set -euo pipefail

OUTPUT_DIR="${1:-./airgap/wheelhouse}"
TARGET_PLATFORM="${TARGET_PLATFORM:-manylinux2014_x86_64}"
TARGET_PYTHON_VERSION="${TARGET_PYTHON_VERSION:-312}"
WITH_FINETUNING=false
for arg in "$@"; do
  [[ "$arg" == "--with-finetuning" ]] && WITH_FINETUNING=true
done

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
PIP=(python3 -m pip)

mkdir -p "$OUTPUT_DIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== Keystone air-gap Python wheelhouse build =="
echo "Output dir: $OUTPUT_DIR"
echo "Target: ${TARGET_PLATFORM} / cp${TARGET_PYTHON_VERSION}"
echo "Include finetuning extra (torch/transformers/trl/...): $WITH_FINETUNING"
echo

DOWNLOAD_ARGS=(
  --dest "$OUTPUT_DIR"
  --platform "$TARGET_PLATFORM"
  --python-version "$TARGET_PYTHON_VERSION"
  --implementation cp
  --abi "cp${TARGET_PYTHON_VERSION}"
  --only-binary=:all:
)

echo "--- Downloading core + dev dependencies from pyproject.toml ---"
"${PIP[@]}" download "${DOWNLOAD_ARGS[@]}" "${REPO_ROOT}[dev]"

if [[ "$WITH_FINETUNING" == "true" ]]; then
  echo "--- Downloading finetuning extra (this is large — several GB, includes torch) ---"
  "${PIP[@]}" download "${DOWNLOAD_ARGS[@]}" "${REPO_ROOT}[finetuning]"
fi

echo "--- Building local package index metadata ---"
# A plain flat directory of wheels is enough for `--find-links` — no index
# server needed for the Dockerfile's install step. If serving via devpi
# (a real self-hosted PyPI mirror, for `pip install` ergonomics inside the
# air-gapped dev environment rather than just Docker builds) is wanted,
# point devpi's import at this same $OUTPUT_DIR.
MANIFEST="$OUTPUT_DIR/manifest.txt"
find "$OUTPUT_DIR" -maxdepth 1 -name '*.whl' -o -name '*.tar.gz' | sort > "$MANIFEST"
wheel_count="$(wc -l < "$MANIFEST" | tr -d ' ')"

echo
echo "== Done. $wheel_count packages in $OUTPUT_DIR (manifest: $MANIFEST) =="
echo "Total size: $(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')"
echo
echo "In the air-gapped Dockerfile build stage:"
echo "  COPY airgap/wheelhouse /wheelhouse"
echo "  RUN pip install --no-index --find-links=/wheelhouse -e \".[dev]\""
