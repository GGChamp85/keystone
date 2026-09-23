from lru_cache import LRUCache


def test_put_evicts_the_least_recently_inserted_when_full():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2 and cache.get("c") == 3


def test_put_of_existing_key_updates_value_and_recency():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 10


def test_get_refreshes_recency_so_a_just_read_key_survives_eviction():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # "a" is now the most recently used
    cache.put("c", 3)  # must evict "b", not "a"
    assert cache.get("b") is None
    assert cache.get("a") == 1
