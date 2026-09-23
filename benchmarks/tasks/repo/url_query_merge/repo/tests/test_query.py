from query import merge_query


def test_merges_new_param_into_existing_query():
    assert merge_query("https://x.test/path?a=1", {"b": "2"}) == "https://x.test/path?a=1&b=2"


def test_new_param_overrides_existing_same_key():
    assert merge_query("https://x.test/path?a=1", {"a": "9"}) == "https://x.test/path?a=9"


def test_special_characters_in_values_are_percent_encoded():
    assert merge_query("https://x.test/search", {"q": "a b&c=d"}) == "https://x.test/search?q=a+b%26c%3Dd"


def test_a_hash_character_in_a_value_is_also_encoded():
    assert merge_query("https://x.test/x", {"tag": "c#1"}) == "https://x.test/x?tag=c%231"
