#!/usr/bin/env bash
# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0
#
# Restore drill: prove a backup of the platform database restores into a fresh Postgres and is the same
# database — same Alembic revision, same row counts in the tables that matter. Run it against a live
# instance (it takes the dump itself) or against an existing dump file.
#
#   scripts/restore_drill.sh --from-url postgresql://keystone:pw@host:5432/keystone
#   scripts/restore_drill.sh --dump /backups/keystone-20260922T020000Z.dump --from-url <same url, for comparison>
#
# Needs: docker (a throwaway postgres:16-alpine container is started and removed), pg_dump/pg_restore/psql
# on PATH or inside that container (the script uses the container's tools, so nothing local is required
# beyond docker). Exit code 0 only when every check passes. CI runs this against its Postgres service.
set -euo pipefail

FROM_URL=""
DUMP=""
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from-url) FROM_URL="$2"; shift 2 ;;
    --dump) DUMP="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$FROM_URL" ]; then
  echo "usage: $0 --from-url postgresql://user:pw@host:port/db [--dump file.dump] [--keep]" >&2
  exit 2
fi

IMAGE="${PG_IMAGE:-postgres:16-alpine}"
NAME="keystone-restore-drill-$$"
WORK="$(mktemp -d)"
cleanup() {
  if [ "$KEEP" = "0" ]; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    rm -rf "$WORK"
  else
    echo "kept container $NAME and $WORK"
  fi
}
trap cleanup EXIT

TABLES="alembic_version tenants api_keys users agent_tasks usage_records model_adapters finetune_jobs"

# pg tools run inside a helper container on the host network so a URL to localhost/LAN works unchanged
pg() { docker run --rm --network host -v "$WORK:/work" "$IMAGE" "$@"; }

if [ -z "$DUMP" ]; then
  echo "== taking a dump from the source"
  pg pg_dump --format=custom --no-owner --file=/work/source.dump "$FROM_URL"
  DUMP="$WORK/source.dump"
else
  cp "$DUMP" "$WORK/source.dump"
fi
echo "== dump: $(stat -f %z "$WORK/source.dump" 2>/dev/null || stat -c %s "$WORK/source.dump") bytes"

echo "== starting a fresh postgres ($IMAGE)"
PORT="${DRILL_PORT:-55432}"
docker run -d --name "$NAME" -e POSTGRES_PASSWORD=drill -e POSTGRES_USER=keystone -e POSTGRES_DB=keystone -p "$PORT:5432" "$IMAGE" >/dev/null
for _ in $(seq 1 60); do
  if docker exec "$NAME" pg_isready -U keystone -d keystone >/dev/null 2>&1; then break; fi
  sleep 1
done
TARGET_URL="postgresql://keystone:drill@127.0.0.1:$PORT/keystone"

echo "== restoring"
pg pg_restore --no-owner --no-privileges --dbname="$TARGET_URL" /work/source.dump

echo "== comparing"
fail=0
for t in $TABLES; do
  src=$(pg psql -tA "$FROM_URL" -c "select count(*) from $t" 2>/dev/null || echo "missing")
  dst=$(pg psql -tA "$TARGET_URL" -c "select count(*) from $t" 2>/dev/null || echo "missing")
  if [ "$src" = "$dst" ]; then
    echo "   $t: $src rows (match)"
  else
    echo "   $t: source=$src restored=$dst  MISMATCH"; fail=1
  fi
done
src_rev=$(pg psql -tA "$FROM_URL" -c "select version_num from alembic_version")
dst_rev=$(pg psql -tA "$TARGET_URL" -c "select version_num from alembic_version")
if [ "$src_rev" = "$dst_rev" ] && [ -n "$dst_rev" ]; then
  echo "   alembic revision: $dst_rev (match)"
else
  echo "   alembic revision: source=$src_rev restored=$dst_rev  MISMATCH"; fail=1
fi

if [ "$fail" = "0" ]; then
  echo "RESTORE DRILL PASSED"
else
  echo "RESTORE DRILL FAILED" >&2
  exit 1
fi
