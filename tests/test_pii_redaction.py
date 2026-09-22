# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Real tests for src/security/pii_redaction.py — pattern matching against
real-format PII (including real Luhn-valid test credit card numbers, the
published test numbers vendors themselves use), not mocked detection."""

from __future__ import annotations

from src.security.pii_redaction import PIIType, redact_pii

# Real published test card numbers (Visa/Mastercard/Amex test ranges) —
# these are Luhn-valid by construction, same numbers Stripe/vendors
# document for test-mode use, never real cardholder data.
_VISA_TEST = "4242424242424242"
_MASTERCARD_TEST = "5555555555554444"
_AMEX_TEST = "378282246310005"


def test_redacts_an_email_address():
    result = redact_pii("Contact jane.doe@example.com for details.")
    assert "[EMAIL_REDACTED]" in result.redacted_text
    assert "jane.doe@example.com" not in result.redacted_text
    assert result.counts_by_type == {"email": 1}


def test_redacts_a_us_phone_number():
    result = redact_pii("Call me at 415-555-0132 tomorrow.")
    assert "[PHONE_REDACTED]" in result.redacted_text
    assert "415-555-0132" not in result.redacted_text


def test_redacts_a_ssn():
    result = redact_pii("SSN on file: 078-05-1120.")
    assert "[SSN_REDACTED]" in result.redacted_text
    assert "078-05-1120" not in result.redacted_text


def test_redacts_a_real_luhn_valid_credit_card_number():
    result = redact_pii(f"Card: {_VISA_TEST}")
    assert "[CREDIT_CARD_REDACTED]" in result.redacted_text
    assert _VISA_TEST not in result.redacted_text
    assert result.matches[0].type == PIIType.CREDIT_CARD


def test_redacts_credit_cards_across_vendors():
    for card in (_VISA_TEST, _MASTERCARD_TEST, _AMEX_TEST):
        result = redact_pii(card)
        assert card not in result.redacted_text, f"{card} should have been redacted"


def test_does_not_flag_a_luhn_invalid_number_as_a_credit_card():
    not_a_card = "1234567890123456"  # 16 digits, fails Luhn
    result = redact_pii(not_a_card)
    assert not_a_card in result.redacted_text
    assert "credit_card" not in result.counts_by_type


def test_redacts_an_ipv4_address():
    result = redact_pii("Connect to 203.0.113.42 on port 8080.")
    assert "[IPV4_REDACTED]" in result.redacted_text
    assert "203.0.113.42" not in result.redacted_text


def test_leaves_ordinary_text_untouched():
    text = "The quick brown fox jumps over the lazy dog. def foo(): return 42"
    result = redact_pii(text)
    assert result.redacted_text == text
    assert not result.found_any


def test_types_filter_restricts_detection():
    text = "Email jane@example.com or call 415-555-0132."
    result = redact_pii(text, types={PIIType.EMAIL})
    assert "[EMAIL_REDACTED]" in result.redacted_text
    assert "415-555-0132" in result.redacted_text  # phone not requested, left alone


def test_multiple_pii_types_in_one_string_all_get_redacted():
    text = f"jane@example.com, 415-555-0132, 078-05-1120, {_VISA_TEST}, 203.0.113.42"
    result = redact_pii(text)
    assert result.counts_by_type == {"email": 1, "phone": 1, "ssn": 1, "credit_card": 1, "ipv4": 1}
    for raw in ("jane@example.com", "415-555-0132", "078-05-1120", _VISA_TEST, "203.0.113.42"):
        assert raw not in result.redacted_text


def test_ssn_pattern_does_not_swallow_a_luhn_valid_card_number_looking_span():
    """SSN's ddd-dd-dddd format is checked before the looser credit-card
    candidate scan, so a real SSN is never misclassified as a card."""
    result = redact_pii("078-05-1120")
    assert result.matches[0].type == PIIType.SSN
    assert "credit_card" not in result.counts_by_type


def test_match_objects_never_expose_the_matched_value():
    result = redact_pii("jane@example.com")
    match = result.matches[0]
    assert not hasattr(match, "value")
    assert not hasattr(match, "text")
