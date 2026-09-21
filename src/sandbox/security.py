"""
Keystone — Sandbox Security Module
Zero-trust sandbox networking: egress filtering, ephemeral secret injection,
environment sanitization, and content scanning.

Copyright 2024-2026 Gaurav Gupta — Apache 2.0 License
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import secrets
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger()


# ── Egress Policy ────────────────────────────────────────────────────


class EgressAction(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"


@dataclass(frozen=True)
class EgressRule:
    """Single egress rule for sandbox network filtering."""

    destination: str  # CIDR, hostname, or "*"
    port: int | None = None  # None = all ports
    protocol: str = "tcp"  # tcp | udp | icmp
    action: EgressAction = EgressAction.DENY
    description: str = ""


# Default allowlist: internal registries only, deny everything else
DEFAULT_EGRESS_RULES: list[EgressRule] = [
    # Allow DNS resolution
    EgressRule(
        destination="*",
        port=53,
        protocol="udp",
        action=EgressAction.ALLOW,
        description="DNS resolution",
    ),
    # Allow internal PyPI mirror (devpi — airgap/build_wheelhouse.sh)
    EgressRule(
        destination="pypi.internal.keystone.local",
        port=443,
        action=EgressAction.ALLOW,
        description="Internal PyPI mirror",
    ),
    # Allow internal NPM registry
    EgressRule(
        destination="npm.internal.keystone.local",
        port=443,
        action=EgressAction.ALLOW,
        description="Internal NPM registry",
    ),
    # Allow internal container registry (Harbor)
    EgressRule(
        destination="registry.internal.keystone.local",
        port=443,
        action=EgressAction.ALLOW,
        description="Internal container registry",
    ),
    # Allow internal Gitea (repository checkout during sandboxed builds)
    EgressRule(
        destination="gitea.internal.keystone.local",
        port=443,
        action=EgressAction.ALLOW,
        description="Internal Git server",
    ),
    # Deny all public internet by default
    EgressRule(
        destination="0.0.0.0/0",
        port=None,
        action=EgressAction.DENY,
        description="Block all public internet egress",
    ),
    # Deny all private ranges except our own
    EgressRule(
        destination="10.0.0.0/8",
        port=None,
        action=EgressAction.DENY,
        description="Block private range 10.x",
    ),
    EgressRule(
        destination="172.16.0.0/12",
        port=None,
        action=EgressAction.DENY,
        description="Block private range 172.16-31.x",
    ),
    EgressRule(
        destination="192.168.0.0/16",
        port=None,
        action=EgressAction.DENY,
        description="Block private range 192.168.x",
    ),
    # Deny metadata endpoint (cloud provider SSRF vector)
    EgressRule(
        destination="169.254.169.254/32",
        port=None,
        action=EgressAction.DENY,
        description="Block cloud metadata endpoint (SSRF protection)",
    ),
]


@dataclass
class EgressPolicy:
    """
    Network egress policy for a sandbox instance.
    First-match semantics: rules are evaluated in order.
    """

    rules: list[EgressRule] = field(default_factory=lambda: list(DEFAULT_EGRESS_RULES))
    default_action: EgressAction = EgressAction.DENY
    log_denied: bool = True

    def add_allow_rule(
        self,
        destination: str,
        port: int | None = None,
        protocol: str = "tcp",
        description: str = "",
    ) -> None:
        """Add an allow rule at the beginning (highest priority)."""
        rule = EgressRule(
            destination=destination,
            port=port,
            protocol=protocol,
            action=EgressAction.ALLOW,
            description=description,
        )
        self.rules.insert(0, rule)

    def to_iptables_commands(self, interface: str = "eth0") -> list[str]:
        """
        Generate iptables commands to enforce this policy.
        Applied inside the sandbox/microVM network namespace.
        """
        commands = [
            # Flush existing OUTPUT rules
            "iptables -F OUTPUT",
            # Allow loopback
            "iptables -A OUTPUT -o lo -j ACCEPT",
            # Allow established connections
            "iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT",
        ]

        for rule in self.rules:
            cmd = self._rule_to_iptables(rule, interface)
            if cmd:
                commands.append(cmd)

        # Default deny at the end
        commands.append(f"iptables -A OUTPUT -o {interface} -j DROP")

        # IPv6: deny all by default
        commands.extend(
            [
                "ip6tables -F OUTPUT",
                "ip6tables -A OUTPUT -o lo -j ACCEPT",
                "ip6tables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT",
                f"ip6tables -A OUTPUT -o {interface} -j DROP",
            ]
        )

        return commands

    def _rule_to_iptables(self, rule: EgressRule, interface: str) -> str | None:
        """Convert a single rule to an iptables command."""
        target = {
            EgressAction.ALLOW: "ACCEPT",
            EgressAction.DENY: "DROP",
            EgressAction.LOG: "LOG --log-prefix 'SANDBOX_EGRESS: '",
        }[rule.action]

        parts = [f"iptables -A OUTPUT -o {interface}"]

        # Protocol
        if rule.protocol in ("tcp", "udp"):
            parts.append(f"-p {rule.protocol}")
        elif rule.protocol == "icmp":
            parts.append("-p icmp")

        # Destination
        if rule.destination != "*":
            try:
                ipaddress.ip_network(rule.destination, strict=False)
                parts.append(f"-d {rule.destination}")
            except ValueError:
                # It's a hostname — we can't use hostnames directly in iptables
                # Log a warning; hostname resolution must be done beforehand
                logger.warning(
                    "iptables_hostname_skip",
                    hostname=rule.destination,
                    description=rule.description,
                )
                return None

        # Port
        if rule.port is not None and rule.protocol in ("tcp", "udp"):
            parts.append(f"--dport {rule.port}")

        # Comment
        if rule.description:
            safe_desc = rule.description[:255].replace('"', "")
            parts.append(f'-m comment --comment "{safe_desc}"')

        parts.append(f"-j {target}")
        return " ".join(parts)


# ── Environment Sanitization ─────────────────────────────────────────

# Environment variables that must NEVER leak into the sandbox
BLOCKED_ENV_VARS: set[str] = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "AZURE_CLIENT_SECRET",
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "DATABASE_URL",
    "DB_PASSWORD",
    "POSTGRES_PASSWORD",
    "REDIS_URL",
    "REDIS_PASSWORD",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITLAB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "SLACK_TOKEN",
    "SLACK_WEBHOOK",
    "STRIPE_SECRET_KEY",
    "STRIPE_PUBLISHABLE_KEY",
    "SENDGRID_API_KEY",
    "MAILGUN_API_KEY",
    "JWT_SECRET",
    "SECRET_KEY",
    "ENCRYPTION_KEY",
    "QDRANT_API_KEY",
    "KEYSTONE_ROOT_ADMIN_TOKEN",
    "VS_DATABASE_URL",
    "VS_REDIS_URL",
    "VS_QDRANT_API_KEY",
    "VS_ENCRYPTION_KEY",
    "VS_SECRET_KEY",
}

# Patterns that indicate a secret value
SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:password|passwd|pwd|secret|token|key|credential)", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|access[_-]?key|auth[_-]?token)", re.IGNORECASE),
    re.compile(r"^sk[-_]", re.IGNORECASE),  # Stripe, OpenAI-style keys
    re.compile(r"^ghp_", re.IGNORECASE),  # GitHub PATs
    re.compile(r"^gho_", re.IGNORECASE),  # GitHub OAuth
    re.compile(r"^glpat-", re.IGNORECASE),  # GitLab PATs
    re.compile(r"^xox[bprs]-", re.IGNORECASE),  # Slack tokens
]


def sanitize_environment(
    env: dict[str, str],
    extra_blocked: set[str] | None = None,
) -> dict[str, str]:
    """
    Remove all known secrets and sensitive variables from an environment dict.
    Returns a clean copy safe for sandbox injection.
    """
    blocked = BLOCKED_ENV_VARS | (extra_blocked or set())
    clean = {}

    for key, value in env.items():
        # Block known dangerous variables
        if key.upper() in blocked:
            logger.debug("env_var_blocked", key=key, reason="blocklist")
            continue

        # Block variables whose names match secret patterns
        if any(p.search(key) for p in SECRET_PATTERNS):
            logger.debug("env_var_blocked", key=key, reason="pattern_match")
            continue

        # Block variables whose values look like secrets
        if _looks_like_secret(value):
            logger.debug("env_var_blocked", key=key, reason="value_heuristic")
            continue

        clean[key] = value

    return clean


def _looks_like_secret(value: str) -> bool:
    """Heuristic: does this value look like a secret/token/key?"""
    if len(value) < 16:
        return False

    # High entropy string (base64-ish or hex)
    if len(value) >= 32:
        alphanum = sum(1 for c in value if c.isalnum())
        if alphanum / max(len(value), 1) > 0.85:
            return True

    # Starts with known prefixes
    prefixes = ("sk-", "pk-", "ghp_", "gho_", "glpat-", "xoxb-", "xoxp-", "Bearer ", "Basic ", "eyJ")  # JWT prefix
    return value.startswith(prefixes)


# ── Ephemeral Secret Injection ───────────────────────────────────────


@dataclass
class EphemeralSecret:
    """
    A secret scoped to a single sandbox execution lifecycle.
    Automatically expired after max_ttl_seconds.
    """

    name: str
    value: str
    created_at: float = field(default_factory=time.time)
    max_ttl_seconds: int = 300  # 5 minutes default
    access_count: int = 0
    max_accesses: int = 10
    _revoked: bool = field(default=False, repr=False)

    @property
    def is_expired(self) -> bool:
        if self._revoked:
            return True
        if time.time() - self.created_at > self.max_ttl_seconds:
            return True
        return self.access_count >= self.max_accesses

    def read(self) -> str | None:
        """Read the secret value. Returns None if expired."""
        if self.is_expired:
            return None
        self.access_count += 1
        return self.value

    def revoke(self) -> None:
        """Immediately revoke this secret."""
        self._revoked = True
        self.value = ""  # Zero out the value


class SecretVault:
    """
    Manages ephemeral secrets for sandbox execution.
    All secrets are wiped when the vault is destroyed.
    """

    def __init__(self) -> None:
        self._secrets: dict[str, EphemeralSecret] = {}
        self._vault_id = secrets.token_hex(8)

    def inject(
        self,
        name: str,
        value: str,
        ttl_seconds: int = 300,
        max_accesses: int = 10,
    ) -> str:
        """
        Inject an ephemeral secret. Returns a reference token
        (not the actual value) for the sandbox to use.
        """
        ref_token = f"ks-secret-{self._vault_id}-{secrets.token_hex(8)}"
        self._secrets[ref_token] = EphemeralSecret(
            name=name,
            value=value,
            max_ttl_seconds=ttl_seconds,
            max_accesses=max_accesses,
        )
        logger.info(
            "secret_injected",
            name=name,
            ttl=ttl_seconds,
            vault=self._vault_id,
        )
        return ref_token

    def resolve(self, ref_token: str) -> str | None:
        """Resolve a reference token to the actual secret value."""
        secret = self._secrets.get(ref_token)
        if secret is None:
            return None
        return secret.read()

    def to_env_vars(self) -> dict[str, str]:
        """
        Export all live secrets as environment variables.
        Used for sandbox environment injection.
        """
        env = {}
        for secret in self._secrets.values():
            if not secret.is_expired:
                env[secret.name] = secret.value
        return env

    def revoke_all(self) -> None:
        """Revoke all secrets immediately."""
        for secret in self._secrets.values():
            secret.revoke()
        self._secrets.clear()
        logger.info("vault_revoked", vault=self._vault_id)

    def cleanup_expired(self) -> int:
        """Remove expired secrets. Returns count of cleaned entries."""
        expired = [k for k, v in self._secrets.items() if v.is_expired]
        for key in expired:
            self._secrets[key].revoke()
            del self._secrets[key]
        return len(expired)

    def __del__(self) -> None:
        """Ensure secrets are wiped on garbage collection."""
        self.revoke_all()


# ── Output Scanning ──────────────────────────────────────────────────

# Patterns that indicate possible data exfiltration in sandbox output
EXFIL_PATTERNS: list[re.Pattern] = [
    re.compile(r"curl\s+.*-d\s+.*\$\{?\w+", re.IGNORECASE),
    re.compile(r"wget\s+.*--post-data", re.IGNORECASE),
    re.compile(r"nc\s+-[a-z]*\s+\d+\.\d+\.\d+\.\d+", re.IGNORECASE),
    re.compile(r"python.*-c.*socket\.connect", re.IGNORECASE),
    re.compile(r"base64.*\|\s*curl", re.IGNORECASE),
    re.compile(r"/etc/passwd|/etc/shadow", re.IGNORECASE),
    re.compile(r"eval\s*\(\s*base64", re.IGNORECASE),
    re.compile(r"subprocess.*shell\s*=\s*True", re.IGNORECASE),
    re.compile(r"os\.system\s*\(", re.IGNORECASE),
    re.compile(r"__import__\s*\(\s*['\"]os['\"]", re.IGNORECASE),
]


@dataclass
class ScanResult:
    """Result of scanning sandbox output for security issues."""

    is_safe: bool
    findings: list[dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0-1.0


def scan_sandbox_output(
    stdout: str,
    stderr: str,
    code: str | None = None,
) -> ScanResult:
    """
    Scan sandbox output and generated code for potential security issues.
    Returns a ScanResult with findings.
    """
    findings = []
    combined = f"{stdout}\n{stderr}\n{code or ''}"

    for pattern in EXFIL_PATTERNS:
        matches = pattern.findall(combined)
        if matches:
            findings.append(
                {
                    "type": "exfiltration_pattern",
                    "pattern": pattern.pattern,
                    "matches": matches[:5],  # Limit to first 5
                    "severity": "high",
                }
            )

    # Check for unexpected network activity in output
    ip_pattern = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    found_ips = set(ip_pattern.findall(combined))
    external_ips = []
    for ip_str in found_ips:
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_global:
                external_ips.append(ip_str)
        except ValueError:
            continue

    if external_ips:
        findings.append(
            {
                "type": "external_ip_reference",
                "ips": external_ips[:10],
                "severity": "medium",
            }
        )

    # Check for potential prompt injection in output
    injection_markers = [
        "ignore previous instructions",
        "disregard the above",
        "new system prompt",
        "you are now",
        "act as",
    ]
    combined_lower = combined.lower()
    findings.extend(
        {"type": "prompt_injection_attempt", "marker": marker, "severity": "critical"}
        for marker in injection_markers
        if marker in combined_lower
    )

    # Calculate risk score
    severity_weights = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}
    if findings:
        total_weight = sum(severity_weights.get(f.get("severity", "low"), 0.1) for f in findings)
        risk_score = min(total_weight / len(findings), 1.0)
    else:
        risk_score = 0.0

    is_safe = risk_score < 0.5 and not any(f.get("severity") == "critical" for f in findings)

    if findings:
        logger.warning(
            "sandbox_scan_findings",
            count=len(findings),
            risk_score=risk_score,
            is_safe=is_safe,
        )

    return ScanResult(
        is_safe=is_safe,
        findings=findings,
        risk_score=risk_score,
    )


# ── Content Hash Verification ────────────────────────────────────────


def hash_file_content(content: str) -> str:
    """SHA-256 hash of file content for integrity verification."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify_file_integrity(
    original_hash: str,
    current_content: str,
) -> bool:
    """Verify a file hasn't been tampered with."""
    return hash_file_content(current_content) == original_hash
