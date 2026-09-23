#!/usr/bin/env bash
# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0
#
# Load-test smoke: start the real application against the configured services, mint a tenant and key
# through the real admin API, run benchmarks/load/k6_chat.js against it, write the summary JSON, stop.
# CI runs this against the CPU demo model to keep the script and the k6 test honest; the numbers that
# mean something come from running the same k6 script against a GPU deployment (docs/benchmarks/load-test.md).
#
#   scripts/k6_smoke.sh [summary.json]
# Needs: k6 on PATH, the Python venv active, DATABASE_URL/REDIS_URL/VLLM_CODING_URL/KEYSTONE_ROOT_ADMIN_TOKEN
# and VS_SECRET_KEY exported (the same environment the test suite uses).
set -euo pipefail

SUMMARY="${1:-k6-summary.json}"
PORT="${K6_APP_PORT:-18090}"
BASE="http://127.0.0.1:$PORT"
: "${KEYSTONE_ROOT_ADMIN_TOKEN:?export KEYSTONE_ROOT_ADMIN_TOKEN (the bootstrap admin token)}"
command -v k6 >/dev/null || { echo "k6 is not installed (https://k6.io)"; exit 2; }

python -m uvicorn src.main:app --host 127.0.0.1 --port "$PORT" --log-level warning &
APP_PID=$!
trap 'kill "$APP_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if curl -sf "$BASE/health/live" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -sf "$BASE/health/live" >/dev/null || { echo "app did not start"; exit 1; }

ADMIN=(-H "Authorization: Bearer $KEYSTONE_ROOT_ADMIN_TOKEN" -H "Content-Type: application/json")
STAMP=$(date +%s)
TENANT=$(curl -sf "$BASE/v1/admin/tenants" "${ADMIN[@]}" -d "{\"name\":\"k6-$STAMP\",\"email\":\"k6-$STAMP@load.test\"}" | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
KEY=$(curl -sf "$BASE/v1/admin/tenants/$TENANT/keys" "${ADMIN[@]}" -d '{"name":"k6","scopes":["inference"]}' | python -c 'import json,sys; print(json.load(sys.stdin)["key"])')

echo "== k6 against $BASE (model=${MODEL:-coding}, vus=${VUS:-2}, duration=${DURATION:-20s})"
KEYSTONE_URL="$BASE" KEYSTONE_API_KEY="$KEY" SUMMARY_JSON="$SUMMARY" \
  k6 run --quiet -e KEYSTONE_URL="$BASE" -e KEYSTONE_API_KEY="$KEY" -e SUMMARY_JSON="$SUMMARY" \
  -e MODEL="${MODEL:-coding}" -e VUS="${VUS:-2}" -e DURATION="${DURATION:-20s}" -e MAX_TOKENS="${MAX_TOKENS:-32}" \
  -e STREAM="${STREAM:-0}" benchmarks/load/k6_chat.js
echo "== summary written to $SUMMARY"
