from slug import slugify


def test_lowercases_and_replaces_spaces():
    assert slugify("Hello World") == "hello-world"


def test_keeps_digits():
    assert slugify("Release 2024") == "release-2024"


def test_collapses_runs_of_separators():
    assert slugify("Hello,   World!!!  Again") == "hello-world-again"


def test_no_leading_or_trailing_dash():
    assert slugify("  --Hello World--  ") == "hello-world"
