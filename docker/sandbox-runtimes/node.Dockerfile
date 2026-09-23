# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0
#
# Keystone sandbox runtime image: Node.js. Chosen by the sandbox daemon for a repository whose
# tooling is Node (package.json) — src/orchestrator/nodes/_shared.py switches the task's sandbox to
# it after the clone identifies the ecosystem. Same rules as python.Dockerfile: everything the agent
# needs is baked in at build time (no network inside most sandboxes, air-gap operation); only the
# target repository's own dependencies are installed at task time through the configured mirror.
#
# Build: docker build -f docker/sandbox-runtimes/node.Dockerfile -t keystone-sandbox-node:latest .
# In an air-gapped build, --build-arg NPM_CONFIG_REGISTRY=https://npm.internal.keystone.local

FROM node:22-bookworm-slim

ARG NPM_CONFIG_REGISTRY=https://registry.npmjs.org/
ENV NPM_CONFIG_REGISTRY=${NPM_CONFIG_REGISTRY} \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false

# git: the real git workflow. curl: git's https transport. universal-ctags: the repository map.
# python3: repositories with a small Python helper script alongside their JS, and the daemon's own
# bookkeeping commands, which are shell-portable but occasionally shell out to python for JSON.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    universal-ctags \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# eslint and typescript for the quality gate on repositories that do not pin their own; a repository's
# own devDependencies win when present (the profile runs `npm run lint` / `npx tsc`).
RUN npm install -g --no-audit --no-fund eslint@9 typescript@5 && npm cache clean --force

COPY docker/sandbox-runtimes/ca-certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates

RUN groupadd -r sandbox && useradd -r -g sandbox -d /workspace sandbox \
    && mkdir -p /workspace && chown sandbox:sandbox /workspace
USER sandbox
WORKDIR /workspace
CMD ["sleep", "infinity"]
