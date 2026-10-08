"""Head-relative command geometry, stabilized and frozen during interaction.

This module owns positions and hit tests, independent of the Figma artwork.
The same geometry is mirrored once for output; labels are painted separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from carl_cam.geometry import Point


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float = 300
    height: float = 100

    def contains(self, point: Point) -> bool:
        return self.x <= point[0] <= self.x + self.width and self.y <= point[1] <= self.y + self.height

    def overlaps(self, other: Rect, clearance: float = 0) -> bool:
        return (self.x < other.x + other.width + clearance and self.x + self.width + clearance > other.x
                and self.y < other.y + other.height + clearance and self.y + self.height + clearance > other.y)

    def moved(self, dx: float, dy: float) -> Rect:
        return Rect(self.x + dx, self.y + dy, self.width, self.height)

    def output(self) -> Rect:
        return Rect(1280 - self.x - self.width, self.y, self.width, self.height)


@dataclass(frozen=True, slots=True)
class MenuLayout:
    cards: tuple[tuple[str, Rect], ...]
    fixed: bool
    frozen: bool = False
    needs_space: bool = False

    def hit(self, point: Point | None) -> str | None:
        if point is None or not np.isfinite(point).all():
            return None
        return next((name for name, rect in self.cards if rect.contains(point)), None)

    def output(self) -> MenuLayout:
        return MenuLayout(tuple((name, rect.output()) for name, rect in self.cards), self.fixed, self.frozen, self.needs_space)


def perimeter(face: Rect | None = None) -> MenuLayout:
    slots = [Rect(24, 24), Rect(24, 280), Rect(24, 596),
             Rect(956, 24), Rect(956, 280), Rect(956, 596)]
    chosen = []
    for name, preferred in (("red", (1, 0, 2, 4, 3, 5)), ("yellow", (3, 4, 0, 1, 5, 2)),
                            ("blue", (5, 4, 2, 1, 3, 0))):
        rect = next((slots[index] for index in preferred
                     if slots[index] not in chosen and (face is None or not slots[index].overlaps(face, 24))), None)
        if rect is None:
            # No compliant orbit/perimeter exists at this camera distance. Do
            # not paint a misleading choice across the face. Native desktop
            # selection and Back/Escape remain available with move-back guidance.
            return MenuLayout((), True, needs_space=True)
        chosen.append(rect)
    back = Rect(525, 596, 230, 100)
    if face and back.overlaps(face, 24):
        back = Rect(525, 24, 230, 100)
        if back.overlaps(face, 24):
            return MenuLayout((), True, needs_space=True)
    return MenuLayout(tuple(zip(("red", "yellow", "blue"), chosen, strict=True)) + (("back", back),), True)


def orbit(center: Point, face: Rect) -> MenuLayout:
    cards = (Rect(face.x - 324, center[1] - 50), Rect(face.x + face.width + 24, center[1] - 160),
             Rect(face.x + face.width + 24, center[1] + 60))
    dx = max(24 - min(rect.x for rect in cards), min(0, 1256 - max(rect.x + rect.width for rect in cards)))
    dy = max(24 - min(rect.y for rect in cards), min(0, 696 - max(rect.y + rect.height for rect in cards)))
    moved = tuple(rect.moved(dx, dy) for rect in cards)
    if any(rect.x < 24 or rect.y < 24 or rect.x + rect.width > 1256 or rect.y + rect.height > 696
           or rect.overlaps(face, 24) for rect in moved):
        return perimeter(face)
    back = Rect(525, 596, 230, 100)
    if back.overlaps(face, 24) or any(back.overlaps(rect) for rect in moved):
        back = Rect(525, 24, 230, 100)
    if back.overlaps(face, 24) or any(back.overlaps(rect) for rect in moved):
        return perimeter(face)
    return MenuLayout(tuple(zip(("red", "yellow", "blue"), moved, strict=True)) + (("back", back),), False)


class MenuFollower:
    def __init__(self) -> None:
        self.layout = perimeter()
        self._smoothed: np.ndarray | None = None
        self._time: float | None = None
        self._last_head: float | None = None

    def reset(self) -> None:
        self.layout = perimeter()
        self._smoothed = self._time = self._last_head = None

    def update(self, head: Point | None, face: Rect | None, now: float, *, hover: bool, pinching: bool) -> MenuLayout:
        if head is not None and face is not None and np.isfinite((*head, face.x, face.y, face.width, face.height)).all():
            self._last_head = now
        else:
            head = face = None
        if hover or pinching:
            self.layout = MenuLayout(self.layout.cards, self.layout.fixed, True, self.layout.needs_space)
            self._time = now
            return self.layout
        if head is None:
            if self._last_head is None or now - self._last_head >= 0.300:
                self.layout = perimeter()
                self._smoothed = None
            self._time = now
            self.layout = MenuLayout(self.layout.cards, self.layout.fixed, needs_space=self.layout.needs_space)
            return self.layout
        values = np.array((*head, face.x, face.y, face.width, face.height), dtype=float)
        if self._smoothed is None or self._time is None:
            self._smoothed = values
        else:
            alpha = 1 - exp(-max(0, now - self._time) / 0.120)
            self._smoothed += alpha * (values - self._smoothed)
        self._time = now
        smoothed = self._smoothed
        self.layout = orbit(tuple(smoothed[:2]), Rect(*smoothed[2:]))
        return self.layout
