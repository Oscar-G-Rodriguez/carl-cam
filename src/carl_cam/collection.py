"""Guided rounds that save at most five numerical observations each second."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from uuid import uuid4

import numpy as np

from carl_cam.features import FeatureResult, palm_ruler
from carl_cam.storage import LABELS

VARIATIONS = ("center", "shift", "angle", "distance", "lighting")
NEGATIVE_VARIATIONS = ("rest_typing", "touch_face", "ordinary_movement", "point_pinch", "entry_exit")


@dataclass(frozen=True, slots=True)
class RoundPrompt:
    label: str
    round_index: int
    variations: tuple[str, str, str]
    target_seconds: int


def collection_plan() -> tuple[RoundPrompt, ...]:
    rounds = []
    for label in LABELS:
        for index in range(5):
            sequence = NEGATIVE_VARIATIONS if label == "other" else VARIATIONS
            variations = tuple(sequence[(index + shift) % len(sequence)] for shift in range(3))
            rounds.append(RoundPrompt(label, index, variations, 15 if label == "other" else 12))
    return tuple(rounds)


class CollectionRound:
    def __init__(self, prompt: RoundPrompt, *, dataset_id: str, start_seconds: float) -> None:
        self.prompt = prompt
        self.dataset_id = dataset_id
        self.session_id = uuid4().hex
        self.start_seconds = start_seconds
        self.rows: list[dict] = []
        self._last_accepted: float | None = None
        self._variation_seconds = {name: 0.0 for name in prompt.variations}
        self._last_shape: np.ndarray | None = None
        self._last_ruler: float | None = None
        self.last_hint = ""

    def accept(self, result: FeatureResult, now: float, variation: str, *, valid_frame: bool = False) -> bool:
        self.last_hint = ""
        if variation not in self._variation_seconds or not result.vectors:
            return False
        if self.prompt.label == "frame" and (len(result.hands) != 2 or not result.two_hand_stable or not valid_frame):
            return False
        if self._last_accepted is not None and now - self._last_accepted < 0.2 - 1e-8:
            return False
        shape = np.asarray([hand.landmarks for hand in result.hands], np.float32)
        ruler = float(np.mean([palm_ruler(points) for points in shape]))
        # Prototype duplicate gate: 1% of a palm ruler RMS movement in current
        # image coordinates. Translation/scale variation remains useful even
        # though those factors are normalized out of classifier features.
        # This threshold still needs real-hand jitter/collection validation.
        if self._last_shape is not None and shape.shape == self._last_shape.shape:
            displacement = np.sqrt(np.mean((shape[..., :2] - self._last_shape[..., :2]) ** 2))
            if displacement / max(ruler, self._last_ruler) < .01:
                self.last_hint = "This example is nearly identical. Keep the pose and gently vary its position, angle or distance."
                return False
        # A missing-tracking gap contributes no usable collection time.
        elapsed = 0 if self._last_accepted is None else now - self._last_accepted
        contribution = 0.2 if 0.2 - 1e-8 <= elapsed <= 0.35 else 0
        self._variation_seconds[variation] += contribution
        self.rows.append({
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "label": self.prompt.label,
            "round_index": self.prompt.round_index,
            "time_seconds": now - self.start_seconds,
            "variation": variation,
            "features": {str(width): vector.tolist() for width, vector in result.vectors.items()},
            "landmarks": [tuple((float(x), float(y), float(z) if isfinite(z) else None) for x, y, z in hand.landmarks) for hand in result.hands],
            "handedness": [hand.handedness for hand in result.hands],
            "quality": {"two_hand_stable": result.two_hand_stable, "valid_frame": valid_frame},
        })
        self._last_accepted = now
        self._last_shape, self._last_ruler = shape.copy(), ruler
        return True

    @property
    def usable_seconds(self) -> float:
        return sum(self._variation_seconds.values())

    def variation_seconds(self, name: str) -> float:
        return self._variation_seconds.get(name, 0.0)

    @property
    def complete(self) -> bool:
        return self.usable_seconds >= self.prompt.target_seconds - 1e-6 and all(seconds >= 2 - 1e-6 for seconds in self._variation_seconds.values())
