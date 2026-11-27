import threading
import time
from enum import IntEnum


class CircuitState(IntEnum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitBreaker:
    """Thread-safe circuit breaker guarding Tier-2 CLIP escalation calls."""

    def __init__(
        self, failure_threshold: int = 3, recovery_timeout_sec: float = 30.0
    ) -> None:
        self._lock = threading.RLock()
        self._failure_threshold = failure_threshold
        self._recovery_timeout_sec = recovery_timeout_sec
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def reconfigure(self, failure_threshold: int, recovery_timeout_sec: float) -> None:
        with self._lock:
            self._failure_threshold = failure_threshold
            self._recovery_timeout_sec = recovery_timeout_sec

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and self._opened_at is not None
                and (time.monotonic() - self._opened_at) >= self._recovery_timeout_sec
            ):
                self._state = CircuitState.HALF_OPEN
            return self._state

    def allow_request(self) -> bool:
        return self.state != CircuitState.OPEN

    def record_success(self) -> CircuitState:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._state = CircuitState.CLOSED
            return self._state

    def record_failure(self) -> CircuitState:
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._state == CircuitState.HALF_OPEN
                or self._consecutive_failures >= self._failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
            return self._state
