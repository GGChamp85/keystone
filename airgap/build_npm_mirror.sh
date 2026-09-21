#!/usr/bin/env bash
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Run on the BUILD MACHINE (has internet access). Resolves and downloads
# opencode-ai (the Keystone Agents interactive CLI, see
# docs/deployment/OPENCODE_SETUP.md) plus its full transitive dependency
# tree into a self-contained node_modules directory — no npm registry
# access needed once this is transferred into the air-gapped environment.
#
# IMPORTANT — real bug found and worked around here: opencode-ai ships its
# actual CLI as a platform-specific native (Bun-compiled) binary,
# distributed via per-platform optionalDependencies (opencode-linux-x64,
# opencode-darwin-arm64, etc.) and copied into place by a `postinstall`
# script. That postinstall script decides which binary to copy using the
# BUILD MACHINE's own `os.platform()/os.arch()` — it does NOT respect
# `npm install --os=... --cpu=...` — and when the matching platform package
# isn't present locally, it silently runs its OWN nested `npm install`
# straight against the live registry as a fallback. That means a naive
# `npm install --os=linux --cpu=x64 opencode-ai` run on a macOS build
# machine would appear to work but actually reach back out to npm at
# postinstall time and stage a macOS binary into the bundle — exactly the
# kind of hidden network dependency this whole exercise exists to catch.
#
# The fix, verified against the real published package: install with
# `--ignore-scripts` (so postinstall never runs and never phones home),
# then manually copy the correct platform/arch optional-dependency
# package's binary into node_modules/opencode-ai/bin/opencode.exe
# ourselves — reproducing exactly what a working postinstall does, for the
# TARGET platform rather than the build machine's. Verified working
# end-to-end for both a same-platform round trip (`opencode --version`
# actually ran) and a cross-platform build (macOS ARM build machine ->
# staged a genuine linux-x64 ELF binary, confirmed via `file`).
#
# Usage:
#   ./airgap/build_npm_mirror.sh [output_dir] [opencode-ai version] [target platform-arch[-variant]]
#
# target platform-arch[-variant] defaults to linux-x64 (the common client-VPC
# server target) with no -baseline/-musl suffix (assumes a post-2013 x86_64
# CPU with AVX2, glibc). Pass e.g. "linux-x64-baseline" for older hardware,
# "linux-x64-musl" for Alpine, or "linux-arm64" for ARM servers — must match
# one of the opencode-<name> keys in opencode-ai's package.json
# optionalDependencies (run `npm view opencode-ai optionalDependencies` on
# the build machine to see the current list for the version being bundled).

set -euo pipefail

OUTPUT_DIR="${1:-./airgap/npm-mirror}"
OPENCODE_VERSION="${2:-latest}"
TARGET="${3:-linux-x64}"
TARGET_OS="${TARGET%%-*}"
TARGET_REST="${TARGET#*-}"
TARGET_CPU="${TARGET_REST%%-*}"
TARGET_PKG="opencode-${TARGET}"

command -v npm >/dev/null || { echo "npm is required on the build machine" >&2; exit 1; }
command -v node >/dev/null || { echo "node is required on the build machine" >&2; exit 1; }
command -v file >/dev/null || { echo "file is required (for the binary sanity check)" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "== Keystone air-gap npm mirror build (opencode-ai@${OPENCODE_VERSION}, target ${TARGET}) =="
echo "Output dir: $OUTPUT_DIR"
echo

cd "$STAGE_DIR"
npm init -y >/dev/null

echo "--- Installing opencode-ai@${OPENCODE_VERSION} + dependency tree for ${TARGET_OS}/${TARGET_CPU} (--ignore-scripts: no postinstall network fallback) ---"
npm install --ignore-scripts --no-audit --no-fund \
  --os="$TARGET_OS" --cpu="$TARGET_CPU" \
  "opencode-ai@${OPENCODE_VERSION}"

resolved_version="$(node -p "require('./node_modules/opencode-ai/package.json').version")"
echo "Resolved version: $resolved_version"

if [[ ! -d "node_modules/${TARGET_PKG}" ]]; then
  echo "FATAL: node_modules/${TARGET_PKG} was not installed — check the target name against:" >&2
  echo "  npm view opencode-ai@${OPENCODE_VERSION} optionalDependencies" >&2
  exit 1
fi

echo "--- License check (must stay MIT/Apache-class to redistribute) ---"
license="$(node -p "require('./node_modules/opencode-ai/package.json').license || 'UNKNOWN'")"
echo "opencode-ai license: $license"
if [[ "$license" != "MIT" && "$license" != "Apache-2.0" ]]; then
  echo "WARNING: opencode-ai license is '$license', not MIT/Apache-2.0 — re-verify docs/LICENSES_AND_COMPLIANCE.md before shipping this bundle." >&2
fi

echo "--- Staging the ${TARGET} binary (postinstall never ran — doing its job for the real target instead of the build machine) ---"
source_binary="node_modules/${TARGET_PKG}/bin/opencode"
[[ "$TARGET_OS" == "windows" ]] && source_binary="node_modules/${TARGET_PKG}/bin/opencode.exe"
target_binary="node_modules/opencode-ai/bin/opencode.exe"
cp "$source_binary" "$target_binary"
chmod +x "$target_binary"

file_output="$(file -b "$target_binary")"
echo "Staged binary: $file_output"
case "$TARGET_OS" in
  linux) [[ "$file_output" == *ELF* ]] || { echo "FATAL: expected an ELF binary for target linux, got: $file_output" >&2; exit 1; } ;;
  darwin) [[ "$file_output" == *Mach-O* ]] || { echo "FATAL: expected a Mach-O binary for target darwin, got: $file_output" >&2; exit 1; } ;;
  windows) [[ "$file_output" == *PE32* || "$file_output" == *MS-DOS* ]] || echo "WARNING: unexpected file type for target windows: $file_output" >&2 ;;
esac

echo "--- Copying resolved node_modules + package.json/package-lock.json to $OUTPUT_DIR ---"
rm -rf "${OUTPUT_DIR:?}"/*
cp -R node_modules "$OUTPUT_DIR/"
cp package.json package-lock.json "$OUTPUT_DIR/"

MANIFEST="$OUTPUT_DIR/manifest.txt"
{
  echo "opencode-ai|${resolved_version}|${license}|target=${TARGET}"
  echo "staged binary: ${file_output}"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "package count: $(find "$OUTPUT_DIR/node_modules" -maxdepth 2 -name package.json | wc -l | tr -d ' ')"
} > "$MANIFEST"

echo
echo "== Done. Manifest: $MANIFEST =="
echo "Total size: $(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')"
echo
echo "In the air-gapped environment (no npm/network access needed — the binary is self-contained):"
echo "  cp -R $OUTPUT_DIR /opt/keystone/opencode"
echo "  alias opencode='/opt/keystone/opencode/node_modules/.bin/opencode'   # run directly, NOT via 'node ...'"
