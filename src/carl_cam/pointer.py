"""A menu pointer keeps its opening hand instead of swapping on every frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carl_cam.features import palm_ruler
from carl_cam.records import HandObservation


@dataclass(frozen=True, slots=True)
class PointerObservation:
    position: tuple[float, float]
    pinch_ratio: float
    owner_changed: bool


class HandPointer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._wrist = None
        self._handedness = None
        self._last_seen = None

    def begin(self, hand: HandObservation, now: float) -> None:
        self._wrist = np.array(hand.landmarks[0][:2])
        self._handedness = hand.handedness
        self._last_seen = now

    def update(self, hands: tuple[HandObservation, ...], now: float) -> PointerObservation | None:
        usable = []
        for hand in hands:
            points = np.asarray(hand.landmarks)
            if points.shape == (21, 3) and np.isfinite(points[:, :2]).all() and np.isfinite(hand.score) and hand.score >= 0.5:
                ruler = palm_ruler(points)
                if ruler >= 36:
                    usable.append((hand, points, ruler))
        matching = []
        if self._wrist is not None:
            matching = [item for item in usable if item[0].handedness == self._handedness
                        and np.linalg.norm(item[1][0, :2] - self._wrist) <= 2 * item[2]]
        changed = False
        if matching:
            selected = min(matching, key=lambda item: np.linalg.norm(item[1][0, :2] - self._wrist))
        elif usable and (self._last_seen is None or now - self._last_seen >= 0.200):
            selected = max(usable, key=lambda item: item[2])
            changed = True
        else:
            return None
        hand, points, ruler = selected
        self.begin(hand, now)
        return PointerObservation(tuple(float(value) for value in points[8, :2]),
                                  float(np.linalg.norm(points[4, :2] - points[8, :2]) / ruler), changed)
