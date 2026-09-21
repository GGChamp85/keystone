# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0
#
# Keystone sandbox runtime image: Python.
#
# Test/lint tooling is baked in at BUILD time (on the build machine, which
# has internet access), not installed inside the sandbox at CREATE time —
# most sandboxes default to network_enabled=False (src/sandbox/runtime.py's
# ResourceLimits), so a runtime `pip install` would silently fail with no
# network to reach PyPI on for those. Baking dependencies into the image is
# also what makes sub-second sandbox creation possible (nothing to fetch at
# boot) and is required for air-gapped operation in the first place. This
# holds even for the git-workflow sandboxes that DO get network_enabled=True
# (src/orchestrator/workspace.py) — `git`/`ruff`/`mypy`/`bandit` are still
# baked in here rather than pip-installed per task, for the same speed and
# air-gap reasons; only the *target repo's own* dependencies get installed
# at runtime via the internal mirrors, since those can't be known ahead of
# time.
#
# Build: docker build -f docker/sandbox-runtimes/python.Dockerfile -t keystone-sandbox-python:latest .
# In an air-gapped build, --build-arg PIP_INDEX_URL=https://pypi.internal.keystone.local/simple
#
# To trust an internal CA (e.g. for an internal git host whose cert is
# signed by pki/generate_ca.sh's CA, not a public one) drop its root cert(s)
# into docker/sandbox-runtimes/ca-certs/ before building — see that
# directory's own .gitkeep for the real failure this closes. Empty by
# default; a normal build adds zero extra trusted roots.

FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# git: the real git workflow (clone/branch/commit/push, workspace.py).
# curl: git's own https transport plus general debugging/tool use.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pytest>=8.3.0 \
    pytest-asyncio>=0.24.0 \
    httpx>=0.27.0 \
    pydantic>=2.8.0 \
    structlog>=24.4.0 \
    ruff>=0.6.0 \
    mypy>=1.11.0 \
    bandit>=1.7.9

COPY docker/sandbox-runtimes/ca-certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates

RUN groupadd -r sandbox && useradd -r -g sandbox -d /workspace sandbox \
    && mkdir -p /workspace && chown -R sandbox:sandbox /workspace

WORKDIR /workspace
USER sandbox
CMD ["sleep", "infinity"]
