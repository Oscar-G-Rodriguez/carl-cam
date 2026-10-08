"""Versioned 112/138/180-value gesture measurements, derived from the approved formulas.

x/y are mirrored canvas pixels. z is normalized to canvas width, so relative
depth times canvas width uses the same pixel ruler as x/y. Ratios use the larger
of wrist→middle-MCP and index-MCP→pinky-MCP. Angles are radians divided by pi.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carl_cam.geometry import quad_from_opposing_hands, validate_quad
from carl_cam.records import HandObservation

FEATURE_SCHEMAS = {112: "carlcam-a-112-v1", 138: "carlcam-b-138-v1", 180: "carlcam-c-180-v1"}
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
TIPS = (4, 8, 12, 16, 20)


@dataclass(frozen=True, slots=True)
class FeatureResult:
    vectors: dict[int, np.ndarray]
    hands: tuple[HandObservation, ...]
    two_hand_stable: bool
    message: str = ""


def observed_quad(result: FeatureResult):
    """Recover only a valid photographed two-hand frame from stable observations."""
    if len(result.hands) != 2 or not result.two_hand_stable:
        return None
    first, second = sorted(result.hands, key=lambda hand: hand.landmarks[1][0])
    quad = quad_from_opposing_hands(tuple(point[:2] for point in first.landmarks), tuple(point[:2] for point in second.landmarks))
    return quad if quad is not None and validate_quad(quad, (1280, 720)) else None


def palm_ruler(points: np.ndarray) -> float:
    return float(max(np.linalg.norm(points[9, :2] - points[0, :2]), np.linalg.norm(points[5, :2] - points[17, :2])))


def _unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length < 1e-6:
        raise ValueError("Finger direction is too short to measure")
    return vector / length


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(_unit(first), _unit(second)), -1, 1)) / np.pi)


def _hand_values(points: np.ndarray, ruler: float, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    coordinates = np.clip((points[:, :2] - points[0, :2]) / ruler, -8, 8).flatten()
    straightness, bends, directions = [], [], []
    for indices in FINGERS:
        finger = points[list(indices), :2]
        path = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if path < 1e-6:
            raise ValueError("Finger joints are not usable")
        straightness.append(float(np.clip(np.linalg.norm(finger[-1] - finger[0]) / path, 0, 1)))
        # Thumb IP is landmark 3; the four other fingers use their PIP joint.
        joint = 2 if indices[0] == 1 else 1
        bends.append(_angle(finger[joint - 1] - finger[joint], finger[joint + 1] - finger[joint]))
        directions.append(_unit(finger[-1] - finger[0]))
    span = float(np.clip(np.linalg.norm(points[4, :2] - points[8, :2]) / ruler, 0, 8))
    base = np.concatenate((coordinates, straightness, bends, [span])).astype(np.float32)
    distances = [float(np.clip(np.linalg.norm(points[a, :2] - points[b, :2]) / ruler, 0, 8)) for a, b in ((8, 12), (12, 16), (16, 20), (4, 20))]
    splay = [_angle(directions[index], directions[index + 1]) for index in range(4)]
    palm_axis = _unit(points[9, :2] - points[0, :2])
    additional = np.array([*distances, *splay, palm_axis[1], palm_axis[0]], dtype=np.float32)
    depth = None
    if np.isfinite(points[:, 2]).all():
        values = (points[:, 2] - points[0, 2]) * width / ruler
        if np.max(np.abs(values)) <= 8:
            depth = values.astype(np.float32)
    return base, additional, depth


class FeatureExtractor:
    def __init__(self, canvas_size: tuple[int, int] = (1280, 720)) -> None:
        self.width, self.height = canvas_size
        self._last_order: tuple[str, str] | None = None
        self._stable_frames = 0

    def extract(self, hands: tuple[HandObservation, ...]) -> FeatureResult:
        if not hands:
            self._stable_frames = 0
            return FeatureResult({}, (), False, "Show your hand in the camera.")
        if len(hands) > 2:
            return FeatureResult({}, (), False, "Only two hands are supported.")
        try:
            ordered = tuple(sorted(hands, key=lambda hand: hand.landmarks[0][0]))
            arrays = [np.asarray(hand.landmarks, dtype=np.float64) for hand in ordered]
            if any(points.shape != (21, 3) or not np.isfinite(points[:, :2]).all() for points in arrays):
                raise ValueError("Tracking is incomplete. Keep your fingers visible.")
            rulers = [palm_ruler(points) for points in arrays]
            if min(rulers) < 0.05 * self.height:
                raise ValueError("Move your hand closer to the camera.")
            if any(hand.score < 0.5 for hand in ordered):
                raise ValueError("Tracking is uncertain. Show your hand more clearly.")
            values = [_hand_values(points, ruler, self.width) for points, ruler in zip(arrays, rulers, strict=True)]
        except (ValueError, IndexError, TypeError) as error:
            self._stable_frames = 0
            return FeatureResult({}, hands, False, str(error))
        pair_base = np.zeros(4, dtype=np.float32)
        pair_extra = np.zeros(6, dtype=np.float32)
        two_hand_stable = False
        if len(ordered) == 2:
            first, second = arrays
            ruler = float(np.mean(rulers))
            order = (ordered[0].handedness, ordered[1].handedness)
            separated = second[0, 0] - first[0, 0] >= 0.5 * ruler
            if separated and (self._last_order is None or self._last_order == order):
                self._stable_frames += 1
            else:
                self._stable_frames = 0
            self._last_order = order
            two_hand_stable = self._stable_frames >= 3
            displacement = (second[0, :2] - first[0, :2]) / ruler
            axis_a = first[9, :2] - first[0, :2]
            axis_b = second[9, :2] - second[0, :2]
            pair_base = np.array([*np.clip(displacement, -16, 16), min(np.linalg.norm(displacement), 16), _angle(axis_a, axis_b)], dtype=np.float32)
            cmc = np.clip((second[1, :2] - first[1, :2]) / ruler, -16, 16)
            ray_a_thumb = _unit(first[4, :2] - first[1, :2])
            ray_a_index = _unit(first[8, :2] - first[5, :2])
            ray_b_thumb = _unit(second[4, :2] - second[1, :2])
            ray_b_index = _unit(second[8, :2] - second[5, :2])
            relations = []
            for a, b in ((ray_a_thumb, ray_b_index), (ray_a_index, ray_b_thumb)):
                relations.extend((float(np.dot(a, b)), float(a[0] * b[1] - a[1] * b[0])))
            pair_extra = np.array([*cmc, *relations], dtype=np.float32)
        else:
            self._last_order = None
            self._stable_frames = 0
        bases = [value[0] for value in values] + ([np.zeros(53, dtype=np.float32)] if len(values) == 1 else [])
        extras = [value[1] for value in values] + ([np.zeros(10, dtype=np.float32)] if len(values) == 1 else [])
        presence = np.array([1, int(len(values) == 2)], dtype=np.float32)
        base = np.concatenate((*bases, pair_base, presence))
        expanded = np.concatenate((base, *extras, pair_extra))
        vectors = {112: base, 138: expanded}
        if all(value[2] is not None for value in values):
            depths = [value[2] for value in values] + ([np.zeros(21, dtype=np.float32)] if len(values) == 1 else [])
            vectors[180] = np.concatenate((expanded, *depths))
        for width, vector in vectors.items():
            if vector.shape != (width,) or not np.isfinite(vector).all():
                return FeatureResult({}, ordered, False, "Measurements are unusable. Try a clearer pose.")
            vector.flags.writeable = False
        return FeatureResult(vectors, ordered, two_hand_stable)
