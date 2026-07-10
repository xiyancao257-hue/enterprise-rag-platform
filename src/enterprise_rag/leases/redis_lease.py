from __future__ import annotations

from typing import Any

RELEASE_IF_OWNER_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
else
  return 0
end
"""


class RedisLeaseNotConfiguredError(RuntimeError):
    pass


class RedisLeaseStore:
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "enterprise-rag",
        client: Any | None = None,
    ) -> None:
        self.prefix = prefix.rstrip(":")
        self.client = client or self._create_client(url)

    def acquire(self, name: str, owner: str, ttl_seconds: int) -> bool:
        return bool(self.client.set(self._key(name), owner, nx=True, ex=max(1, ttl_seconds)))

    def release(self, name: str, owner: str) -> bool:
        key = self._key(name)
        return bool(self.client.eval(RELEASE_IF_OWNER_SCRIPT, 1, key, owner))

    def get_owner(self, name: str) -> str | None:
        owner = self.client.get(self._key(name))
        if owner is None:
            return None
        if isinstance(owner, bytes):
            return owner.decode("utf-8")
        return str(owner)

    def _key(self, name: str) -> str:
        return f"{self.prefix}:lease:{name}"

    def _create_client(self, url: str) -> Any:
        try:
            import redis
        except ImportError as exc:
            raise RedisLeaseNotConfiguredError(
                "Redis lease support requires the optional dependency: install enterprise-rag[redis]."
            ) from exc
        return redis.Redis.from_url(url)
