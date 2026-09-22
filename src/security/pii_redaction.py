# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — PII redaction.

Real, deterministic pattern matching for common personal-data formats
(email addresses, phone numbers, US SSNs, credit card numbers, IPv4
addresses) — not an ML/NER classifier. That's a deliberate trade-off, not
a shortcut: a model-based detector (e.g. Presidio + spaCy) pulls in a
large dependency and non-deterministic behavior into an air-gapped image
for a feature whose whole point is a predictable, auditable guarantee.

Honest limitation, stated plainly: regex-only detection misses PII with
no fixed format — names, street addresses, free-text medical or financial
details. This catches the structured, high-confidence cases; it is not a
general-purpose PII classifier, and should not be represented as one.

This is intentionally a separate module from src/sandbox/security.py's
SECRET_PATTERNS/EXFIL_PATTERNS — those target credentials and sandbox
exfiltration attempts, a different threat model from personal data.
"""

from __future__ import annotations

import dataclasses
import re
from enum import StrEnum


class PIIType(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IPV4 = "ipv4"


@dataclasses.dataclass(frozen=True)
class PIIMatch:
    """Records what kind of PII was found and where — never the matched
    value itself, so a PIIMatch is always safe to log or persist."""

    type: PIIType
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    matches: tuple[PIIMatch, ...]

    @property
    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.matches:
            counts[m.type.value] = counts.get(m.type.value, 0) + 1
        return counts

    @property
    def found_any(self) -> bool:
        return bool(self.matches)


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_IPV4_RE = re.compile(r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?!\d)")
# Candidate spans for credit-card numbers — 13-19 digits, optionally grouped
# by spaces or hyphens (real card formatting); each candidate is verified
# against the real Luhn checksum before being treated as a match, since a
# bare "13-19 digit" pattern alone would false-positive constantly on
# ordinary numeric IDs, phone numbers, and hashes appearing in code/logs.
_CREDIT_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

_REPLACEMENT = {
    PIIType.EMAIL: "[EMAIL_REDACTED]",
    PIIType.PHONE: "[PHONE_REDACTED]",
    PIIType.SSN: "[SSN_REDACTED]",
    PIIType.CREDIT_CARD: "[CREDIT_CARD_REDACTED]",
    PIIType.IPV4: "[IPV4_REDACTED]",
}


def _luhn_valid(digits: str) -> bool:
    """Real Luhn checksum (ISO/IEC 7812) — the standard credit-card
    check-digit algorithm, not a heuristic approximation."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _find_credit_cards(text: str) -> list[re.Match]:
    matches = []
    for m in _CREDIT_CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            matches.append(m)
    return matches


_DETECTORS: dict[PIIType, re.Pattern[str] | None] = {
    PIIType.EMAIL: _EMAIL_RE,
    PIIType.PHONE: _PHONE_RE,
    PIIType.SSN: _SSN_RE,
    PIIType.IPV4: _IPV4_RE,
    PIIType.CREDIT_CARD: None,  # handled by _find_credit_cards (needs Luhn verification)
}


def redact_pii(text: str, *, types: set[PIIType] | None = None) -> RedactionResult:
    """
    Finds every match for the requested PII types (default: all) and
    replaces each with a type-labeled placeholder, left to right,
    non-overlapping — the first pattern to claim a span wins, checked in
    a fixed order (email, SSN, credit card, phone, IPv4) chosen so a more
    specific format (e.g. an SSN's ddd-dd-dddd) is never swallowed by a
    looser one (e.g. the credit-card candidate scan).
    """
    active = types or set(PIIType)
    order = [PIIType.EMAIL, PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PHONE, PIIType.IPV4]

    spans: list[tuple[int, int, PIIType]] = []
    for pii_type in order:
        if pii_type not in active:
            continue
        if pii_type == PIIType.CREDIT_CARD:
            found = _find_credit_cards(text)
        else:
            pattern = _DETECTORS[pii_type]
            found = list(pattern.finditer(text))
        for m in found:
            start, end = m.span()
            if any(start < e and end > s for s, e, _ in spans):
                continue  # overlaps a higher-priority match already claimed
            spans.append((start, end, pii_type))

    spans.sort(key=lambda s: s[0])

    out_parts: list[str] = []
    matches: list[PIIMatch] = []
    cursor = 0
    for start, end, pii_type in spans:
        out_parts.append(text[cursor:start])
        out_parts.append(_REPLACEMENT[pii_type])
        matches.append(PIIMatch(type=pii_type, start=start, end=end))
        cursor = end
    out_parts.append(text[cursor:])

    return RedactionResult(redacted_text="".join(out_parts), matches=tuple(matches))
