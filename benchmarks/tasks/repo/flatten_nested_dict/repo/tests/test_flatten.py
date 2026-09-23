from flatten import flatten


def test_flattens_nested_keys_with_dot_paths():
    assert flatten({"a": {"b": 1, "c": 2}}) == {"a.b": 1, "a.c": 2}


def test_non_dict_values_at_top_level_are_kept_as_is():
    assert flatten({"x": 1, "y": "z"}) == {"x": 1, "y": "z"}


def test_an_empty_nested_dict_is_preserved_not_dropped():
    assert flatten({"a": {}, "b": 1}) == {"a": {}, "b": 1}


def test_deeply_nested_empty_dict_is_preserved():
    assert flatten({"a": {"b": {}}}) == {"a.b": {}}
