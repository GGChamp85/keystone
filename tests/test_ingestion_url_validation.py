# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real, no-mock tests for the repository_url SSRF guard
(src/memory/ingestion.py::_validate_repo_url) — this is a security-critical
fix, so it's tested against real DNS resolution, not mocked sockets.
"""

from __future__ import annotations

import pytest

from src.memory.ingestion import InvalidRepositoryURLError, _validate_repo_url


def test_rejects_file_scheme():
    with pytest.raises(InvalidRepositoryURLError, match="scheme"):
        _validate_repo_url("file:///etc/passwd", allowed_hosts=[])


def test_rejects_bare_git_scheme():
    with pytest.raises(InvalidRepositoryURLError, match="scheme"):
        _validate_repo_url("git://example.com/repo.git", allowed_hosts=[])


def test_allows_https_to_real_public_host_when_no_allowlist():
    # github.com resolves to a real public IP — must not be blocked as "private"
    _validate_repo_url("https://github.com/octocat/Hello-World.git", allowed_hosts=[])


def test_blocks_loopback_when_no_allowlist():
    with pytest.raises(InvalidRepositoryURLError, match="disallowed address"):
        _validate_repo_url("https://127.0.0.1/repo.git", allowed_hosts=[])


def test_blocks_localhost_hostname_when_no_allowlist():
    with pytest.raises(InvalidRepositoryURLError, match="disallowed address"):
        _validate_repo_url("https://localhost/repo.git", allowed_hosts=[])


def test_blocks_cloud_metadata_endpoint_when_no_allowlist():
    with pytest.raises(InvalidRepositoryURLError, match="disallowed address"):
        _validate_repo_url("https://169.254.169.254/latest/meta-data/", allowed_hosts=[])


def test_blocks_private_range_when_no_allowlist():
    with pytest.raises(InvalidRepositoryURLError, match="disallowed address"):
        _validate_repo_url("https://10.0.0.5/repo.git", allowed_hosts=[])


def test_allowlist_rejects_hosts_not_on_the_list():
    with pytest.raises(InvalidRepositoryURLError, match="not in the allowlisted"):
        _validate_repo_url("https://evil.example.com/repo.git", allowed_hosts=["gitea.internal.keystone.local"])


def test_allowlisted_internal_host_is_accepted_even_if_it_would_resolve_private():
    # An operator-approved internal Gitea host is EXPECTED to be a private
    # address in an air-gapped deployment — that must not be rejected.
    # (localhost stands in for "resolves privately" here since we can't
    # control DNS for a real internal hostname in this test environment.)
    _validate_repo_url("https://localhost/repo.git", allowed_hosts=["localhost"])


def test_missing_host_rejected():
    with pytest.raises(InvalidRepositoryURLError, match="no resolvable host"):
        _validate_repo_url("https:///repo.git", allowed_hosts=[])
