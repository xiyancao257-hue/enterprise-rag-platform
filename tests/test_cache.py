from enterprise_rag.cache.in_memory import InMemoryCache


def test_in_memory_cache_returns_values_before_ttl_expires() -> None:
    now = iter([100.0, 105.0])
    cache = InMemoryCache(now=lambda: next(now))

    cache.set("key", "value", ttl_seconds=10)

    assert cache.get("key") == "value"
    assert cache.hits == 1
    assert cache.misses == 0


def test_in_memory_cache_expires_values_after_ttl() -> None:
    now = iter([100.0, 111.0])
    cache = InMemoryCache(now=lambda: next(now))

    cache.set("key", "value", ttl_seconds=10)

    assert cache.get("key") is None
    assert cache.hits == 0
    assert cache.misses == 1
