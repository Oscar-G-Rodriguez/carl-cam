"""One time-based owner for gesture dwell, frame grace, and pinch selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carl_cam.geometry import Point, Quad, validate_quad
from carl_cam.records import InteractionSnapshot
from carl_cam.storage import LABELS, valid_assignments


@dataclass(frozen=True, slots=True)
class InputObservation:
    source_time: float
    scores: tuple[float, ...] | None = None
    quad: Quad | None = None
    pointer: Point | None = None
    pinch_ratio: float | None = None
    hover: str | None = None
    two_hand_stable: bool = False
    released: bool = False


def winning_pose(scores: tuple[float, ...] | None, *, threshold: float = 0.80, margin: float = 0.15) -> str | None:
    if scores is None or len(scores) != len(LABELS):
        return None
    values = np.asarray(scores)
    if not np.isfinite(values).all():
        return None
    order = np.argsort(values)
    winner = int(order[-1])
    if winner == len(LABELS) - 1 or values[winner] < threshold or values[winner] - values[order[-2]] < margin:
        return None
    return LABELS[winner]


class InteractionController:
    def __init__(self, *, theme="red", hold_ms=300, assignments=None, threshold=0.80, margin=0.15) -> None:
        self.mode = "camera"
        self.theme = theme
        self.assignments = assignments or {}
        if not valid_assignments(self.assignments):
            raise ValueError("Each direct pose must be assigned to one distinct theme")
        self.hold_ms = hold_ms
        self.threshold, self.margin = threshold, margin
        self.quad: Quad | None = None
        self.pointer: Point | None = None
        self._pending: str | None = None
        self._pending_since = 0.0
        self._last_valid_frame = 0.0
        self._needs_release = False
        self._release_since: float | None = None
        self.pinching = False
        self._pinch_target: str | None = None
        self._pinch_armed = True
        self._pinch_release_since: float | None = None
        self._last_pointer_time = 0.0

    def enter_mode(self, mode: str) -> None:
        self.mode = mode
        self.quad = None
        self._pending = None
        self._cancel_pinch()

    def dispatch(self, action: str) -> str | None:
        """All mouse, keyboard, and completed pinch actions share this dispatcher."""
        if action == "menu" and self.mode in ("camera", "framed"):
            self.enter_mode("menu")
            return "menu"
        if self.mode == "menu" and action in ("red", "yellow", "blue", "back"):
            if action != "back":
                self.theme = action
            self.enter_mode("camera")
            self._needs_release = True
            return action
        return None

    def _cancel_pinch(self) -> None:
        self.pinching = False
        self._pinch_target = None
        self._pinch_armed = False
        self._pinch_release_since = None

    def _snapshot(self, command: str | None = None) -> InteractionSnapshot:
        return InteractionSnapshot(self.mode, self.theme, self.quad, self.pointer, command=command)

    def update(self, observation: InputObservation | None, now: float) -> InteractionSnapshot:
        if self.mode not in ("camera", "framed", "menu"):
            return self._snapshot()
        position_fresh = observation is not None and 0 <= now - observation.source_time <= 0.100
        command_fresh = observation is not None and 0 <= now - observation.source_time <= 0.080
        if self.mode == "menu":
            return self._update_menu(observation if position_fresh else None, now)
        pose = winning_pose(observation.scores, threshold=self.threshold, margin=self.margin) if command_fresh else None
        if pose == "frame" and (not observation.two_hand_stable or observation.quad is None or not validate_quad(observation.quad, (1280, 720))):
            pose = None
        release_observed = command_fresh and (observation.released or (
            observation.scores is not None and len(observation.scores) == 6
            and np.isfinite(observation.scores).all() and observation.scores[5] >= 0.8
        ))
        if release_observed:
            if self._release_since is None:
                self._release_since = now
            elif now - self._release_since >= 0.100:
                self._needs_release = False
        else:
            self._release_since = None
        if self.mode == "framed":
            # Existing graphics have a 100 ms positioning window. A new command
            # still uses the stricter 80 ms gate above; these are not aliases.
            positioned_pose = winning_pose(observation.scores, threshold=self.threshold, margin=self.margin) if position_fresh else None
            if (positioned_pose == "frame" and observation.two_hand_stable
                    and observation.quad is not None and validate_quad(observation.quad, (1280, 720))):
                self.quad = observation.quad
                self._last_valid_frame = now
            elif now - self._last_valid_frame >= 0.150:
                self.quad = None
                self.mode = "camera"
        allowed = pose == "peace" or (self.mode == "camera" and pose is not None)
        if not allowed or (self._needs_release and pose != "frame"):
            self._pending = None
            return self._snapshot()
        if self._pending != pose:
            self._pending, self._pending_since = pose, now
            return self._snapshot()
        dwell = (0.250 if pose == "peace" else 0.300) if self.hold_ms == 300 else self.hold_ms / 1000
        if now - self._pending_since < dwell:
            return self._snapshot()
        self._pending = None
        if pose == "peace":
            self._needs_release = True
            self.enter_mode("menu")
            return self._snapshot("menu")
        if pose == "frame":
            self.mode = "framed"
            self.quad = observation.quad
            self._last_valid_frame = now
            return self._snapshot("frame")
        if pose in self.assignments:
            self.theme = self.assignments[pose]
            self._needs_release = True
            return self._snapshot(f"select:{self.theme}")
        return self._snapshot()

    def _update_menu(self, observation: InputObservation | None, now: float) -> InteractionSnapshot:
        if observation is None or observation.pointer is None or observation.pinch_ratio is None:
            self.pointer = None
            if now - self._last_pointer_time >= 0.100:
                self._cancel_pinch()
            return self._snapshot()
        self.pointer = observation.pointer
        self._last_pointer_time = now
        ratio = observation.pinch_ratio
        if not np.isfinite(ratio):
            self._cancel_pinch()
            return self._snapshot()
        if ratio > 0.50:
            if self.pinching:
                target = self._pinch_target
                self.pinching = False
                self._pinch_target = None
                self._pinch_release_since = now
                if target is not None and observation.hover == target:
                    return self._snapshot(self.dispatch(target))
            if self._pinch_release_since is None:
                self._pinch_release_since = now
            if now - self._pinch_release_since >= 0.100:
                self._pinch_armed = True
        else:
            self._pinch_release_since = None
            if self.pinching and observation.hover != self._pinch_target:
                self._pinch_target = None
            if ratio < 0.35 and self._pinch_armed and not self.pinching:
                self.pinching = True
                self._pinch_armed = False
                self._pinch_target = observation.hover
        return self._snapshot()
