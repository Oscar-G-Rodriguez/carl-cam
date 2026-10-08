"""Picture-frame ray intersections, validation, coordinates, and mask generation.

Camera-facing code gives this module preview-space pixel landmarks. The two thumb
CMC anchors remain actual corners. The other corners are intersections of the
opposing thumb/index rays; fingertips are never substituted for corners.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

import numpy as np

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]  # upper-left, upper-right, lower-right, lower-left


@dataclass(frozen=True, slots=True)
class Ray:
    origin: Point
    direction: Point


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def ray_intersection(first: Ray, second: Ray, *, epsilon: float = 1e-5) -> Point | None:
    """Intersect two forward rays; reject parallel, unstable, or rear intersections."""
    p, r, q, s = first.origin, first.direction, second.origin, second.direction
    norm = hypot(*r) * hypot(*s)
    determinant = _cross(r, s)
    if norm < epsilon or abs(determinant) / norm < epsilon:
        return None
    displacement = (q[0] - p[0], q[1] - p[1])
    t = _cross(displacement, s) / determinant
    u = _cross(displacement, r) / determinant
    if t < 0 or u < 0:
        return None
    return p[0] + t * r[0], p[1] + t * r[1]


def _ray_from_hand(landmarks: Sequence[Point], start: int, end: int) -> Ray:
    anchor = landmarks[1]
    return Ray(anchor, (landmarks[end][0] - landmarks[start][0], landmarks[end][1] - landmarks[start][1]))


def quad_from_opposing_hands(
    lower_left_hand: Sequence[Point], upper_right_hand: Sequence[Point]
) -> Quad | None:
    """Use CMC anchors and cross-hand rays to recover the photographed frame pose."""
    if len(lower_left_hand) < 9 or len(upper_right_hand) < 9:
        return None
    a = lower_left_hand[1]
    b = upper_right_hand[1]
    a_thumb = _ray_from_hand(lower_left_hand, 2, 4)
    a_index = _ray_from_hand(lower_left_hand, 5, 8)
    b_thumb = _ray_from_hand(upper_right_hand, 2, 4)
    b_index = _ray_from_hand(upper_right_hand, 5, 8)
    upper_left = ray_intersection(a_thumb, b_index)
    lower_right = ray_intersection(a_index, b_thumb)
    if upper_left is None or lower_right is None:
        return None
    return upper_left, b, lower_right, a


def polygon_area(points: Sequence[Point]) -> float:
    return abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )) / 2


def validate_quad(quad: Sequence[Point], image_size: tuple[int, int]) -> bool:
    """Apply the approved convexity, visibility, area, edge, and ratio gates."""
    if len(quad) != 4:
        return False
    width, height = image_size
    if width <= 0 or height <= 0:
        return False
    points = np.asarray(quad, dtype=np.float64)
    if not np.isfinite(points).all():
        return False
    if (points[:, 0] < -0.02 * width).any() or (points[:, 0] > 1.02 * width).any():
        return False
    if (points[:, 1] < -0.02 * height).any() or (points[:, 1] > 1.02 * height).any():
        return False
    edges = np.roll(points, -1, axis=0) - points
    if (np.linalg.norm(edges, axis=1) < 0.10 * min(width, height)).any():
        return False
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(turns > 1e-6) or np.all(turns < -1e-6)):
        return False
    if polygon_area(quad) < 0.08 * width * height:
        return False
    bounds = np.ptp(points, axis=0)
    return bounds[1] > 0 and 0.6 <= bounds[0] / bounds[1] <= 2.5


def mirror_point(point: Point, width: int) -> Point:
    """Map pixel-center coordinates between unmirrored and mirrored views."""
    return width - 1 - point[0], point[1]


def letterbox_transform(source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[float, float, float]:
    """Return uniform scale and x/y offsets; never crop hands to fill output."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = min(target_width / source_width, target_height / source_height)
    return scale, (target_width - scale * source_width) / 2, (target_height - scale * source_height) / 2


def polygon_mask(quad: Quad, image_size: tuple[int, int], feather_px: int = 3) -> np.ndarray:
    """Return a 0..1 antialiased mask for a validated preview-space quadrilateral."""
    import cv2

    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255, lineType=cv2.LINE_AA)
    if feather_px > 0:
        sigma = max(0.1, feather_px / 2)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    return mask.astype(np.float32) / 255
