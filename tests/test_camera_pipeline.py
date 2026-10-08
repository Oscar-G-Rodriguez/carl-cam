import numpy as np

from carl_cam.camera import canvas_rgb
from carl_cam.channels import LatestValue
from carl_cam.smoothing import OneEuro


def test_letterbox_preserves_full_image_and_does_not_stretch() -> None:
    source = np.full((480, 640, 3), (110, 160, 210), dtype=np.uint8)
    canvas = canvas_rgb(source)
    assert canvas.shape == (720, 1280, 3)
    assert not canvas[:, :160].any() and not canvas[:, 1120:].any()
    assert np.all(canvas[:, 160:1120] == (110, 160, 210))


def test_latest_channel_replaces_old_unconsumed_frames() -> None:
    channel: LatestValue[int] = LatestValue()
    for index in range(1000):
        channel.publish(index)
    assert channel.read() == 999
    channel.clear()
    assert channel.read() is None


def test_smoother_resets_on_identity_change_and_does_not_extrapolate() -> None:
    smoothing = OneEuro()
    assert np.array_equal(smoothing.update(np.array([10, 20]), 1), [10, 20])
    moved = smoothing.update(np.array([30, 40]), 1.033)
    assert np.all(moved >= [10, 20]) and np.all(moved <= [30, 40])
    smoothing.reset()
    assert np.array_equal(smoothing.update(np.array([100, 200]), 2), [100, 200])
