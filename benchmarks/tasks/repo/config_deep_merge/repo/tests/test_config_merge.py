from config_merge import merge_config


def test_top_level_override_replaces_value():
    assert merge_config({"debug": False, "port": 80}, {"debug": True}) == {"debug": True, "port": 80}


def test_new_keys_are_added():
    assert merge_config({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_nested_dicts_are_merged_not_replaced():
    base = {"database": {"host": "db", "port": 5432, "pool": {"size": 5, "timeout": 30}}}
    override = {"database": {"port": 6543, "pool": {"size": 20}}}
    assert merge_config(base, override) == {
        "database": {"host": "db", "port": 6543, "pool": {"size": 20, "timeout": 30}}
    }


def test_inputs_are_not_mutated():
    base = {"database": {"host": "db", "port": 5432}}
    override = {"database": {"port": 6543}}
    merge_config(base, override)
    assert base == {"database": {"host": "db", "port": 5432}}
    assert override == {"database": {"port": 6543}}
