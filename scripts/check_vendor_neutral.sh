#!/usr/bin/env bash
# ADR 0006: repo-facing text (README, CHANGELOG, ROADMAP, docs/) names no frontier vendor's
# product in prose — "a frontier model" / "the frontier proxy" is the term. Internal
# identifiers such as model keys in benchmarks/cost_model.py are not covered by this check.
# Usage: scripts/check_vendor_neutral.sh FILE...   (exit 1 with the offending lines)
set -euo pipefail

pattern='\b(Claude|Vouchstone)\b'
status=0
for file in "$@"; do
  # Allow model-id strings such as claude-opus-4-6 (an identifier, not prose) by stripping them first.
  if hits=$(sed -E 's/claude-[a-z0-9.-]+//g' "$file" | grep -n -E "$pattern"); then
    echo "$file:"
    while IFS= read -r line; do echo "  $line"; done <<<"$hits"
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  echo "Repo-facing text must stay vendor-neutral (see docs/architecture/adr/0006-vendor-neutral-display-strings.md)." >&2
fi
exit "$status"
