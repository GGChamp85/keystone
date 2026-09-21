# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0
#
# keystoned — the self-hosted sandbox execution daemon. Built as its own
# image (never colocated with the main app image) because it needs
# privileged host access (the Docker socket for the gVisor backend, or
# /dev/kvm for the Firecracker backend) that the main API tier must not have.

FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="keystoned"
LABEL org.opencontainers.image.description="Keystone self-hosted sandbox execution daemon"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# nftables/iptables for host-side egress policy enforcement; firecracker/jailer
# binaries are expected to be provided via the base image or a volume mount
# in the offline bundle (airgap/build_image_bundle.sh) rather than fetched here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    iptables nftables curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
WORKDIR /app
COPY src/ ./src/
COPY pyproject.toml ./

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 9000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:9000/health || exit 1
CMD ["python", "-m", "uvicorn", "src.sandbox.daemon:app", "--host", "0.0.0.0", "--port", "9000"]
