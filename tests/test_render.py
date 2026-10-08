import numpy as np

from carl_cam.render import SceneState, compose_palette


def test_filter_preserves_exterior_and_changes_only_valid_region() -> None:
    frame = np.full((720, 1280, 3), (150, 150, 150), dtype=np.uint8)
    state = SceneState("red", ((300, 150), (980, 150), (980, 570), (300, 570)))
    final, _ = compose_palette(frame, state)
    assert np.array_equal(final[:, :290], frame[:, :290])
    assert np.array_equal(final[:140], frame[:140])
    assert not np.array_equal(final[200:500, 400:800], frame[200:500, 400:800])


def test_invalid_frame_and_no_frame_produce_plain_camera() -> None:
    frame = np.full((720, 1280, 3), (70, 130, 230), dtype=np.uint8)
    for quad in (None, ((10, 10), (20, 10), (20, 20), (10, 20))):
        final, _ = compose_palette(frame, SceneState("blue", quad))
        assert np.array_equal(final, frame)
