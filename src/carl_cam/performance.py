"""Approved five-second throughput/latency fallback; never upgrades silently."""

from collections import deque
from math import isfinite
from statistics import median


class PerformanceFallback:
    """Observe cumulative distinct renders, not frame IDs or output duplicates.

    A five-second rolling window includes its preceding observation for an
    honest count/time difference. Only timing scalars are retained. Private
    modes and camera loss discard the window without undoing the cooldown.
    """

    def __init__(self):
        self.samples = deque(maxlen=512)
        self.last_change = float('-inf')

    def reset(self):
        self.samples.clear()

    def changed(self, now):
        self.reset()
        self.last_change = now

    def observe(self, now, sequence, delay_ms, level, *, enabled=True):
        if (not enabled or not isfinite(now) or type(sequence) is not int or sequence < 0
                or type(level) is not int or not 0 <= level <= 2
                or delay_ms is not None and (not isfinite(delay_ms) or delay_ms < 0)):
            self.reset()
            return None
        if self.samples and (now <= self.samples[-1][0] or sequence < self.samples[-1][1]):
            self.reset()
        self.samples.append((now, sequence, delay_ms))
        while len(self.samples) > 1 and self.samples[1][0] <= now - 5:
            self.samples.popleft()
        span = now - self.samples[0][0]
        if span < 5 or now - self.last_change < 10 or level == 2:
            return None
        throughput = (sequence - self.samples[0][1]) / span
        # Duplicate observations do not overweight a completed frame's delay.
        delays = {}
        for _, count, delay in self.samples:
            if delay is not None:
                delays[count] = delay
        slow = throughput < 27 or bool(delays) and median(delays.values()) > 150
        if slow:
            self.changed(now)
            return level + 1
        return None
