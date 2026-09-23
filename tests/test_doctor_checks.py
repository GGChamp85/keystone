# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""`keystone doctor`'s newer checks against real things: a real self-signed certificate (openssl), the real
migration revision of the test database, real disk usage, and the readiness probe's skip/unreachable
paths. The database check needs DATABASE_URL."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.cli.doctor import CheckStatus, check_app_ready, check_disk, check_migrations, check_tls
from src.config import Settings

from .conftest import requires_integration_env


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, vs_secret_key="x" * 64, **overrides)


def _self_signed(tmp_path: Path, days: int) -> tuple[Path, Path]:
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    openssl = shutil.which("openssl")
    assert openssl, "openssl is required for this test"
    subprocess.run(  # noqa: S603
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            str(days),
            "-subj",
            "/CN=keystone.test",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def test_tls_check_reads_a_real_certificate_and_reports_its_remaining_validity(tmp_path):
    assert check_tls(_settings()).status == CheckStatus.SKIP  # TLS terminated elsewhere

    cert, key = _self_signed(tmp_path, days=365)
    ok = check_tls(_settings(tls_cert_path=str(cert), tls_key_path=str(key)))
    assert ok.status == CheckStatus.OK and "CN=keystone.test" in ok.message and "36" in ok.message

    (tmp_path / "soon").mkdir()
    soon_cert, soon_key = _self_signed(tmp_path / "soon", days=3)
    soon = check_tls(_settings(tls_cert_path=str(soon_cert), tls_key_path=str(soon_key)))
    assert soon.status == CheckStatus.WARN and "expires in" in soon.message

    missing = check_tls(_settings(tls_cert_path=str(tmp_path / "nope.pem"), tls_key_path=str(key)))
    assert missing.status == CheckStatus.FAIL and "make certs" in missing.message

    (tmp_path / "junk.pem").write_text("not a certificate")
    junk = check_tls(_settings(tls_cert_path=str(tmp_path / "junk.pem"), tls_key_path=str(key)))
    assert junk.status == CheckStatus.FAIL and "not a PEM certificate" in junk.message


def test_disk_check_reports_real_free_space_for_existing_directories(tmp_path):
    results = check_disk(_settings(finetuning_output_dir=str(tmp_path), finetuning_data_dir="/definitely/not/here"))
    names = [r.name for r in results]
    assert f"Disk ({tmp_path})" in names and "Disk (working directory)" in names
    assert all("GB free" in r.message for r in results)


@pytest.mark.integration
@requires_integration_env
async def test_migration_check_agrees_with_the_real_database():
    import os

    result = await check_migrations(_settings(database_url=os.environ["DATABASE_URL"]))
    assert result.status == CheckStatus.OK, result.message
    assert "database at head" in result.message


async def test_app_ready_check_skips_without_a_url_and_fails_on_an_unreachable_one():
    assert (await check_app_ready(None)).status == CheckStatus.SKIP
    dead = await check_app_ready("http://127.0.0.1:1")
    assert dead.status == CheckStatus.FAIL and "unreachable" in dead.message
