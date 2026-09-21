# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir --prefix=/install .
COPY src/ ./src/

# Keystone Agents' plan/execution-trace UI (web/) — built once here so the
# runtime image never needs Node.js at all, matching this Dockerfile's
# existing pattern of a slim runtime stage with only what's needed to run.
FROM node:22-slim AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
LABEL maintainer="Gaurav Gupta <gauravright@gmail.com>"
LABEL org.opencontainers.image.title="Keystone"
LABEL org.opencontainers.image.description="Keystone Inference & Keystone Agents"
LABEL org.opencontainers.image.licenses="Apache-2.0"
# Numeric UID/GID, not a name: Kubernetes' `runAsNonRoot: true` pod security
# check can't resolve a username to a UID without starting the container, so
# `USER vsuser` (unlike `USER 1000`) fails that check with "image has
# non-numeric user, cannot verify user is non-root" — caught by actually
# deploying this image to a real cluster, not by docker build/run alone.
RUN groupadd -r -g 1000 vsuser && useradd -r -u 1000 -g vsuser -s /sbin/nologin vsuser
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
WORKDIR /app
COPY src/ ./src/
COPY LICENSE NOTICE pyproject.toml ./
COPY scripts/ ./scripts/
COPY --from=web-builder /web/dist ./web/dist
RUN mkdir -p /data/finetuning /data/repos /tmp/sandboxes && chown -R 1000:1000 /app /data /tmp/sandboxes
USER 1000:1000
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 VS_ENV=production
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:8080/health || exit 1
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4", "--loop", "uvloop", "--http", "httptools", "--access-log"]
