#!/usr/bin/env bash
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Run on the AIR-GAPPED TARGET (zero internet access). Loads every image
# tarball produced by build_image_bundle.sh, re-tags each to the internal
# registry prefix, and (if reachable from here — internal-network only)
# pushes to that registry. Companion to build_image_bundle.sh.
#
# Usage:
#   ./airgap/import_bundle.sh [bundle_dir] [--push]
#
# Verifies each tarball's sha256 against the manifest before loading it —
# a bundle carried across an air gap on physical media (USB/tape) is
# exactly the scenario where silent transfer corruption is a real risk,
# not a theoretical one.

set -euo pipefail

BUNDLE_DIR="${1:-./airgap/output/images}"
DO_PUSH=false
for arg in "$@"; do
  [[ "$arg" == "--push" ]] && DO_PUSH=true
done

MANIFEST="$BUNDLE_DIR/manifest.txt"
[[ -f "$MANIFEST" ]] || { echo "FATAL: $MANIFEST not found — run build_image_bundle.sh on the build machine first and transfer its output here." >&2; exit 1; }

command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }

echo "== Keystone air-gap image bundle import =="
echo "Bundle dir: $BUNDLE_DIR"
echo "Push to registry after load: $DO_PUSH"
echo

failures=0
while IFS='|' read -r image retagged tarball _digest sha256; do
  [[ -z "$image" ]] && continue

  echo "--- $image ---"
  if [[ ! -f "$tarball" ]]; then
    echo "FATAL: $tarball listed in manifest but missing from bundle — transfer incomplete." >&2
    failures=$((failures + 1))
    continue
  fi

  actual_sha256="$(shasum -a 256 "$tarball" | awk '{print $1}')"
  if [[ -n "$sha256" && "$actual_sha256" != "$sha256" ]]; then
    echo "FATAL: sha256 mismatch for $tarball" >&2
    echo "  expected: $sha256" >&2
    echo "  actual:   $actual_sha256" >&2
    failures=$((failures + 1))
    continue
  fi
  echo "sha256 verified: $actual_sha256"

  docker load -i "$tarball"

  if [[ -n "$retagged" && "$retagged" != "$image" ]]; then
    docker tag "$image" "$retagged"
    echo "Tagged as $retagged"
    if [[ "$DO_PUSH" == "true" ]]; then
      docker push "$retagged"
    fi
  fi
  echo
done < "$MANIFEST"

if [[ "$failures" -gt 0 ]]; then
  echo "== FAILED: $failures image(s) failed verification/load. Re-transfer the bundle and retry. ==" >&2
  exit 1
fi

echo "== Done. All images loaded$( [[ "$DO_PUSH" == "true" ]] && echo " and pushed to the internal registry" ). =="
echo "Verify with: docker images | grep keystone"
