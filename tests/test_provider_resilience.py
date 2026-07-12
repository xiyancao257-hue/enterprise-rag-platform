import pytest

from enterprise_rag.providers.resilience import ProviderCircuitOpenError, ProviderResiliencePolicy


def test_provider_resilience_retries_then_succeeds() -> None:
    calls = 0

    def flaky_operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary outage")
        return "ok"

    policy = ProviderResiliencePolicy(max_retries=1)

    assert policy.call(flaky_operation, provider_name="test_provider") == "ok"
    assert calls == 2


def test_provider_resilience_raises_last_error_after_retries() -> None:
    calls = 0

    def failing_operation() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider down")

    policy = ProviderResiliencePolicy(max_retries=2)

    with pytest.raises(RuntimeError, match="provider down"):
        policy.call(failing_operation, provider_name="test_provider")
    assert calls == 3


def test_provider_resilience_opens_circuit_after_failure_threshold() -> None:
    current_time = 100.0
    policy = ProviderResiliencePolicy(
        max_retries=0,
        circuit_breaker_failure_threshold=2,
        circuit_breaker_reset_seconds=30,
        now=lambda: current_time,
    )

    with pytest.raises(RuntimeError):
        policy.call(lambda: (_ for _ in ()).throw(RuntimeError("first failure")), provider_name="test_provider")
    with pytest.raises(RuntimeError):
        policy.call(lambda: (_ for _ in ()).throw(RuntimeError("second failure")), provider_name="test_provider")
    with pytest.raises(ProviderCircuitOpenError, match="test_provider"):
        policy.call(lambda: "should not run", provider_name="test_provider")


def test_provider_resilience_allows_probe_after_reset_window() -> None:
    current_time = 100.0

    def now() -> float:
        return current_time

    policy = ProviderResiliencePolicy(
        max_retries=0,
        circuit_breaker_failure_threshold=1,
        circuit_breaker_reset_seconds=30,
        now=now,
    )

    with pytest.raises(RuntimeError):
        policy.call(lambda: (_ for _ in ()).throw(RuntimeError("failure")), provider_name="test_provider")

    with pytest.raises(ProviderCircuitOpenError):
        policy.call(lambda: "blocked", provider_name="test_provider")

    current_time = 131.0

    assert policy.call(lambda: "recovered", provider_name="test_provider") == "recovered"
