"""Bounded, latest-value exchanges for workers that must never accumulate latency."""

from __future__ import annotations

from threading import Lock


class LatestValue[T]:
    """One completed immutable value; replacing it never blocks a producer."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._value: T | None = None

    def publish(self, value: T) -> None:
        with self._lock:
            self._value = value

    def read(self) -> T | None:
        with self._lock:
            return self._value

    def clear(self) -> None:
        with self._lock:
            self._value = None

