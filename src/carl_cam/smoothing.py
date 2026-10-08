"""Time-based adaptive low-pass smoothing for tracked points."""

from __future__ import annotations

import math

import numpy as np


class OneEuro:
    """Filter slow jitter more than deliberate motion; never extrapolate points."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.015, derivative_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._time: float | None = None
        self._value: np.ndarray | None = None
        self._raw: np.ndarray | None = None
        self._derivative: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

    def reset(self) -> None:
        self._time = self._value = self._raw = self._derivative = None

    def update(self, value: np.ndarray, time_seconds: float) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("Cannot smooth invalid coordinates")
        if self._time is None or time_seconds <= self._time or self._value.shape != raw.shape:
            self._time = time_seconds
            self._value = self._raw = raw.copy()
            self._derivative = np.zeros_like(raw)
            return raw.copy()
        dt = max(time_seconds - self._time, 1e-5)
        derivative = (raw - self._raw) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        self._derivative = derivative_alpha * derivative + (1 - derivative_alpha) * self._derivative
        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)
        alpha = self._alpha(cutoff, dt)
        self._value = alpha * raw + (1 - alpha) * self._value
        self._raw = raw.copy()
        self._time = time_seconds
        return self._value.copy()

