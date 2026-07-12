from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


class ProviderCircuitOpenError(RuntimeError):
    pass


@dataclass
class ProviderResiliencePolicy:
    max_retries: int = 0
    retry_backoff_seconds: float = 0.0
    circuit_breaker_failure_threshold: int = 0
    circuit_breaker_reset_seconds: float = 30.0
    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    def call(self, operation: Callable[[], T], provider_name: str) -> T:
        self._raise_if_circuit_open(provider_name)

        attempts = max(0, self.max_retries) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                result = operation()
            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1 and self.retry_backoff_seconds > 0:
                    self.sleep(self.retry_backoff_seconds)
                continue
            self._record_success()
            return result

        self._record_failure()
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Provider `{provider_name}` failed without an exception.")

    def _raise_if_circuit_open(self, provider_name: str) -> None:
        if self.circuit_breaker_failure_threshold <= 0 or self._opened_at is None:
            return
        if self.now() - self._opened_at >= self.circuit_breaker_reset_seconds:
            self._opened_at = None
            return
        raise ProviderCircuitOpenError(f"Provider circuit is open for `{provider_name}`.")

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if (
            self.circuit_breaker_failure_threshold > 0
            and self._consecutive_failures >= self.circuit_breaker_failure_threshold
        ):
            self._opened_at = self.now()
