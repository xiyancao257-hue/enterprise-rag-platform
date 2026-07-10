from enterprise_rag.cache.in_memory import InMemoryCache
from enterprise_rag.cache.redis_cache import RedisCache


class FakeRedisClient:
    def __init__(self) -> None:
        self.values = {}
        self.set_calls = []
        self.setex_calls = []

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.set_calls.append((key, value))
        self.values[key] = value.encode("utf-8")

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        self.setex_calls.append((key, ttl_seconds, value))
        self.values[key] = value.encode("utf-8")


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


def test_redis_cache_namespaces_keys_and_round_trips_json_values() -> None:
    client = FakeRedisClient()
    cache = RedisCache(prefix="tenant-a", client=client)

    cache.set("embedding:abc", [0.1, 0.2], ttl_seconds=30)

    assert client.setex_calls == [("tenant-a:embedding:abc", 30, "[0.1,0.2]")]
    assert cache.get("embedding:abc") == [0.1, 0.2]


def test_redis_cache_without_positive_ttl_uses_plain_set() -> None:
    client = FakeRedisClient()
    cache = RedisCache(prefix="rag:", client=client)

    cache.set("query:abc", {"answer": "ok"}, ttl_seconds=0)

    assert client.set_calls == [("rag:query:abc", '{"answer":"ok"}')]
    assert cache.get("query:abc") == {"answer": "ok"}
