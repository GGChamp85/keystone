from csv_line import parse_line


def test_simple_fields():
    assert parse_line("a,b,c\n") == ["a", "b", "c"]


def test_empty_fields_are_preserved():
    assert parse_line("a,,c") == ["a", "", "c"]


def test_quoted_field_with_comma_stays_one_field():
    assert parse_line('1,"Doe, Jane",42') == ["1", "Doe, Jane", "42"]


def test_quotes_are_removed_from_a_plain_quoted_field():
    assert parse_line('"x","y"') == ["x", "y"]
