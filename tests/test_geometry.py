import numpy as np

from carl_cam.geometry import (
    Ray,
    letterbox_transform,
    mirror_point,
    polygon_area,
    ray_intersection,
    validate_quad,
)


def test_forward_ray_intersection() -> None:
    point = ray_intersection(Ray((0, 0), (1, 1)), Ray((4, 0), (-1, 1)))
    assert point == (2, 2)


def test_parallel_or_behind_rays_rejected() -> None:
    assert ray_intersection(Ray((0, 0), (1, 0)), Ray((0, 1), (1, 0))) is None
    assert ray_intersection(Ray((0, 0), (1, 0)), Ray((-1, -1), (0, 1))) is None


def test_valid_skewed_quad() -> None:
    quad = ((230, 120), (980, 155), (1020, 590), (200, 570))
    assert validate_quad(quad, (1280, 720))
    assert polygon_area(quad) > 0


def test_crossed_and_tiny_quads_rejected() -> None:
    assert not validate_quad(((200, 100), (900, 600), (900, 100), (200, 600)), (1280, 720))
    assert not validate_quad(((10, 10), (30, 10), (30, 30), (10, 30)), (1280, 720))


def test_mirror_and_letterbox() -> None:
    assert mirror_point((0, 12), 1280) == (1279, 12)
    assert letterbox_transform((640, 480), (1280, 720)) == (1.5, 160.0, 0.0)
    assert np.isfinite(letterbox_transform((640, 480), (1280, 720))).all()
