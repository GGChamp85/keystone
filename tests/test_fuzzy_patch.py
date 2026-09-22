# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
`find_fuzzy_match` (src/orchestrator/tools/impl.py) — the pure locator
behind apply_patch's tolerance for a `search` block that is not byte-exact.
The contract under test: a whitespace-only difference matches; a near-miss
above the similarity bar matches; two candidate places are refused as
ambiguous rather than guessed; nothing close is an empty result.
"""

from __future__ import annotations

from src.orchestrator.tools.impl import FuzzyMatch, find_fuzzy_match

_FILE = """def load(path):
    with open(path) as fh:
        return fh.read()


def save(path, data):
    with open(path, "w") as fh:
        fh.write(data)
"""


def test_whitespace_only_difference_matches_the_right_lines():
    # The model re-indented and dropped the blank-line structure inside its search block.
    search = "def load(path):\n  with open(path) as fh:\n      return fh.read()"
    found = find_fuzzy_match(_FILE, search)
    assert isinstance(found, FuzzyMatch)
    assert (found.start_line, found.end_line) == (1, 3)
    assert found.exact_after_whitespace is True


def test_near_miss_above_threshold_matches():
    # One character dropped ("retrn") — 97% similar.
    search = "def load(path):\n    with open(path) as fh:\n        retrn fh.read()"
    found = find_fuzzy_match(_FILE, search)
    assert isinstance(found, FuzzyMatch)
    assert (found.start_line, found.end_line) == (1, 3)
    assert found.exact_after_whitespace is False
    assert found.ratio >= 0.92


def test_a_whitespace_match_beats_a_merely_similar_line():
    # Line 2 matches after whitespace normalisation; line 7 (`open(path, "w")`) is only similar.
    # The exact-after-whitespace candidate wins outright — one match, no ambiguity.
    found = find_fuzzy_match(_FILE, "with open(path)  as fh:")
    assert isinstance(found, FuzzyMatch)
    assert (found.start_line, found.end_line) == (2, 2)
    assert found.exact_after_whitespace is True


def test_two_equally_good_places_are_refused_as_ambiguous():
    # Two identical lines, both whitespace-equal to the search — exactly the case a fuzzy edit must not guess at.
    found = find_fuzzy_match("a = 1\nb = 2\na = 1\n", "a  =  1")
    assert isinstance(found, list)
    assert [(m.start_line, m.end_line) for m in found] == [(1, 1), (3, 3)]


def test_nothing_close_is_an_empty_result():
    assert find_fuzzy_match(_FILE, "class Completely:\n    unrelated = True") == []


def test_search_longer_than_the_file_cannot_match():
    assert find_fuzzy_match("one line\n", "a\nb\nc\nd") == []
