# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0
#
# Keystone sandbox runtime image: Go. Chosen by the sandbox daemon for a repository whose tooling is
# Go (go.mod) — src/orchestrator/nodes/_shared.py switches the task's sandbox to it after the clone
# identifies the ecosystem. Same rules as python.Dockerfile: the toolchain is baked in at build time
# (no network inside most sandboxes, air-gap operation); the target repository's own modules are
# fetched at task time through GOPROXY.
#
# Build: docker build -f docker/sandbox-runtimes/go.Dockerfile -t keystone-sandbox-go:latest .
# In an air-gapped build, --build-arg GOPROXY=https://goproxy.internal.keystone.local

FROM golang:1.23-bookworm

ARG GOPROXY=https://proxy.golang.org,direct
ENV GOPROXY=${GOPROXY} \
    GOFLAGS=-mod=mod \
    GOTOOLCHAIN=local \
    CGO_ENABLED=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    universal-ctags \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# staticcheck for the quality gate next to `go vet`; pinned so the image is reproducible.
RUN go install honnef.co/go/tools/cmd/staticcheck@2024.1.1 && cp /go/bin/staticcheck /usr/local/bin/

COPY docker/sandbox-runtimes/ca-certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates

RUN groupadd -r sandbox && useradd -r -g sandbox -d /workspace sandbox \
    && mkdir -p /workspace /go/pkg && chown -R sandbox:sandbox /workspace /go
USER sandbox
WORKDIR /workspace
ENV GOPATH=/go GOCACHE=/workspace/.gocache
CMD ["sleep", "infinity"]
