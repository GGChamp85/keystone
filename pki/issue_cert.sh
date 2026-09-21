#!/usr/bin/env bash
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0
#
# Issues a leaf TLS certificate signed by the internal CA from
# generate_ca.sh — for nginx, OpenBao, or any internal-mTLS need. Includes
# a proper SAN (Subject Alternative Name) extension, not just a CN —
# modern TLS clients (curl, browsers, most language HTTP libraries) will
# not accept certs that only have a CN, only SANs.
#
# Usage:
#   ./pki/issue_cert.sh <primary-hostname> [ca_dir] [output_dir] [extra SAN ...]
#
# Examples:
#   ./pki/issue_cert.sh keystone.local
#   ./pki/issue_cert.sh keystone.internal.client-domain.com ./pki/ca ./pki/certs \
#       DNS:keystone-app DNS:localhost IP:127.0.0.1

set -euo pipefail

PRIMARY_HOST="${1:?Usage: issue_cert.sh <primary-hostname> [ca_dir] [output_dir] [extra SAN ...]}"
CA_DIR="${2:-./pki/ca}"
OUTPUT_DIR="${3:-./pki/certs}"
shift $(( $# >= 3 ? 3 : $# ))
EXTRA_SANS=("$@")

CA_KEY="$CA_DIR/ca.key"
CA_CERT="$CA_DIR/ca.crt"

[[ -f "$CA_KEY" && -f "$CA_CERT" ]] || {
  echo "FATAL: CA not found at $CA_DIR (expected ca.key + ca.crt) — run generate_ca.sh first." >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"
SAFE_NAME="$(echo "$PRIMARY_HOST" | tr '.*' '__')"
LEAF_KEY="$OUTPUT_DIR/${SAFE_NAME}.key"
LEAF_CSR="$OUTPUT_DIR/${SAFE_NAME}.csr"
LEAF_CERT="$OUTPUT_DIR/${SAFE_NAME}.crt"
LEAF_FULLCHAIN="$OUTPUT_DIR/${SAFE_NAME}.fullchain.crt"

SAN="DNS:${PRIMARY_HOST}"
for extra in "${EXTRA_SANS[@]}"; do
  SAN="${SAN},${extra}"
done

echo "== Issuing cert for ${PRIMARY_HOST} (SAN: ${SAN}) =="

openssl genrsa -out "$LEAF_KEY" 2048
chmod 600 "$LEAF_KEY"

openssl req -new \
  -key "$LEAF_KEY" \
  -out "$LEAF_CSR" \
  -subj "/O=Keystone/CN=${PRIMARY_HOST}"

EXT_FILE="$(mktemp)"
trap 'rm -f "$EXT_FILE"' EXIT
cat > "$EXT_FILE" <<EOF
basicConstraints=CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth,clientAuth
subjectAltName=${SAN}
EOF

openssl x509 -req \
  -in "$LEAF_CSR" \
  -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial \
  -out "$LEAF_CERT" \
  -days 397 \
  -sha256 \
  -extfile "$EXT_FILE"

cat "$LEAF_CERT" "$CA_CERT" > "$LEAF_FULLCHAIN"
rm -f "$LEAF_CSR"

echo "== Verifying the issued cert against the CA =="
openssl verify -CAfile "$CA_CERT" "$LEAF_CERT"

echo
echo "== Done =="
openssl x509 -in "$LEAF_CERT" -noout -subject -ext subjectAltName -dates
echo
echo "Leaf key:        $LEAF_KEY"
echo "Leaf cert:       $LEAF_CERT"
echo "Fullchain (nginx wants this as ssl_certificate): $LEAF_FULLCHAIN"
echo
echo "nginx.conf:"
echo "  ssl_certificate     $LEAF_FULLCHAIN;"
echo "  ssl_certificate_key $LEAF_KEY;"
