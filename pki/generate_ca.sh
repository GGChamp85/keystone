#!/usr/bin/env bash
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Generates a real internal Certificate Authority for air-gapped/client-VPC
# deployments — no ACME, no Let's Encrypt, nothing that needs internet
# access or a public DNS record. This CA's cert (ca.crt) is what gets
# distributed to every client machine/service that needs to trust
# Keystone's internally-issued TLS certs (nginx, OpenBao, internal mTLS).
#
# This replaces `make certs`'s bare self-signed leaf cert (still fine for
# a single developer's local `docker compose up`, kept as-is for that) with
# a proper two-tier CA -> leaf-cert chain for anything beyond a laptop:
# one CA generated once and distributed for trust, then `issue_cert.sh`
# mints as many short-lived leaf certs off it as needed without
# regenerating trust anchors each time.
#
# Usage:
#   ./pki/generate_ca.sh [output_dir]

set -euo pipefail

OUTPUT_DIR="${1:-./pki/ca}"
mkdir -p "$OUTPUT_DIR"

CA_KEY="$OUTPUT_DIR/ca.key"
CA_CERT="$OUTPUT_DIR/ca.crt"

if [[ -f "$CA_KEY" || -f "$CA_CERT" ]]; then
  echo "FATAL: $CA_KEY or $CA_CERT already exists — refusing to overwrite an existing CA." >&2
  echo "(Regenerating the CA invalidates every cert issued from the old one.)" >&2
  echo "Delete both first if this is deliberate, or pass a different output_dir." >&2
  exit 1
fi

echo "== Keystone internal CA generation =="
echo "Output dir: $OUTPUT_DIR"

openssl genrsa -out "$CA_KEY" 4096
chmod 600 "$CA_KEY"

openssl req -x509 -new -nodes \
  -key "$CA_KEY" \
  -sha256 \
  -days 3650 \
  -out "$CA_CERT" \
  -subj "/O=Keystone/CN=Keystone Internal CA" \
  -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
  -addext "keyUsage=critical,keyCertSign,cRLSign" \
  -addext "subjectKeyIdentifier=hash"

echo
echo "== Done =="
openssl x509 -in "$CA_CERT" -noout -subject -issuer -dates
echo
echo "Distribute $CA_CERT (NOT $CA_KEY) to every machine/service that must"
echo "trust Keystone's internally-issued certs. Keep $CA_KEY offline/access-"
echo "controlled — anyone holding it can mint a trusted cert for any hostname."
echo
echo "Issue a leaf certificate with:"
echo "  ./pki/issue_cert.sh keystone.local $OUTPUT_DIR"
