from enterprise_rag.config import LeaseConfig
from enterprise_rag.leases.factory import create_lease_store
from enterprise_rag.leases.in_memory import InMemoryLeaseStore
from enterprise_rag.leases.redis_lease import RELEASE_IF_OWNER_SCRIPT, RedisLeaseStore


class FakeRedisLeaseClient:
    def __init__(self) -> None:
        self.values = {}
        self.set_calls = []
        self.eval_calls = []

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None) -> bool:
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.values:
            return False
        self.values[key] = value.encode("utf-8")
        return True

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def eval(self, script: str, num_keys: int, key: str, owner: str) -> int:
        self.eval_calls.append((script, num_keys, key, owner))
        if script != RELEASE_IF_OWNER_SCRIPT or num_keys != 1:
            raise AssertionError("Unexpected Redis Lua script call")
        if self.values.get(key) == owner.encode("utf-8"):
            del self.values[key]
            return 1
        return 0


def test_in_memory_lease_store_blocks_duplicate_owner_until_expiry() -> None:
    clock = iter([100.0, 100.0, 111.0, 111.0])
    leases = InMemoryLeaseStore(now=lambda: next(clock))

    assert leases.acquire("ingest-job:1", "worker-a", ttl_seconds=10) is True
    assert leases.acquire("ingest-job:1", "worker-b", ttl_seconds=10) is False
    assert leases.acquire("ingest-job:1", "worker-b", ttl_seconds=10) is True
    assert leases.get_owner("ingest-job:1") == "worker-b"


def test_in_memory_lease_store_releases_only_matching_owner() -> None:
    leases = InMemoryLeaseStore(now=lambda: 100.0)

    assert leases.acquire("ingest-job:1", "worker-a", ttl_seconds=10) is True
    assert leases.release("ingest-job:1", "worker-b") is False
    assert leases.get_owner("ingest-job:1") == "worker-a"
    assert leases.release("ingest-job:1", "worker-a") is True
    assert leases.get_owner("ingest-job:1") is None


def test_redis_lease_store_uses_namespaced_set_nx_with_ttl() -> None:
    client = FakeRedisLeaseClient()
    leases = RedisLeaseStore(prefix="rag", client=client)

    assert leases.acquire("ingest-job:1", "worker-a", ttl_seconds=30) is True
    assert leases.acquire("ingest-job:1", "worker-b", ttl_seconds=30) is False

    assert client.set_calls == [
        ("rag:lease:ingest-job:1", "worker-a", True, 30),
        ("rag:lease:ingest-job:1", "worker-b", True, 30),
    ]
    assert leases.get_owner("ingest-job:1") == "worker-a"


def test_redis_lease_store_releases_with_atomic_owner_check() -> None:
    client = FakeRedisLeaseClient()
    leases = RedisLeaseStore(prefix="rag", client=client)

    assert leases.acquire("ingest-job:1", "worker-a", ttl_seconds=30) is True
    assert leases.release("ingest-job:1", "worker-b") is False
    assert leases.get_owner("ingest-job:1") == "worker-a"
    assert leases.release("ingest-job:1", "worker-a") is True
    assert leases.get_owner("ingest-job:1") is None


def test_lease_factory_creates_memory_store_and_rejects_unknown_provider() -> None:
    assert isinstance(create_lease_store(LeaseConfig(provider="memory")), InMemoryLeaseStore)

    try:
        create_lease_store(LeaseConfig(provider="unknown"))
    except ValueError as exc:
        assert "Unsupported lease provider" in str(exc)
    else:
        raise AssertionError("Expected unsupported lease provider to raise ValueError")
