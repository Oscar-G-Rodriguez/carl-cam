from time import monotonic_ns, sleep
from types import SimpleNamespace

import numpy as np

from carl_cam.camera import available_devices
from carl_cam.channels import LatestValue
from carl_cam.output import OutputWorker, safe_output
from carl_cam.records import RenderPacket


def packet(timestamp):
    image = np.full((720, 1280, 3), (10, 50, 90), np.uint8)
    return RenderPacket(1, image[:, ::-1], image, timestamp, 10, timestamp)


def test_output_uses_capture_age_and_never_falls_back_to_preview_or_old_frames():
    fresh = packet(1_000_000_000)
    np.testing.assert_array_equal(safe_output(fresh, 1_249_000_000), fresh.output_rgb)
    assert not safe_output(fresh, 1_251_000_000).any()
    assert not safe_output(fresh, 999_000_000).any()
    assert not safe_output(None, 1_100_000_000).any()


def test_output_is_explicit_obs_black_first_and_closes_without_camera_access():
    frames = LatestValue()
    frames.publish(packet(monotonic_ns()))
    seen, options = [], {}

    class Sender:
        closed = False

        def send(self, rgb):
            seen.append(rgb.copy())

        def close(self):
            self.closed = True

    sender = Sender()

    def factory(**kwargs):
        options.update(kwargs)
        return sender

    worker = OutputWorker(frames, factory=factory)
    assert not worker.running and not seen  # Construction cannot broadcast.
    worker.start()
    try:
        for _ in range(100):
            if len(seen) >= 2:
                break
            sleep(.005)
        assert options["backend"] == "obs" and options["device"] == "OBS Virtual Camera"
        assert options["fps"] == 30 and not seen[0].any()
        np.testing.assert_array_equal(seen[1][0, 0], (10, 50, 90))
        assert len(seen) < 10  # No unbounded busy-loop output.
    finally:
        assert worker.stop() and sender.closed


def test_driver_failure_is_recoverable_and_does_not_mutate_render_results():
    frames = LatestValue()
    source = packet(monotonic_ns())
    frames.publish(source)

    def unavailable(**kwargs):
        raise RuntimeError("No OBS driver")

    worker = OutputWorker(frames, factory=unavailable)
    worker.start()
    for _ in range(100):
        if not worker.running:
            break
        sleep(.002)
    assert "Install OBS" in worker.status.read() and frames.read() is source
    assert worker.stop()


def test_camera_names_exclude_obs_without_opening_any_stream():
    records = [SimpleNamespace(index=0, name="USB Camera"),
               SimpleNamespace(index=1, name="OBS Virtual Camera"),
               SimpleNamespace(index=2, name="Integrated Camera")]
    devices = available_devices(enumerator=lambda backend: records)
    assert [(item.index, item.name) for item in devices] == [(0, "USB Camera"), (2, "Integrated Camera")]


def test_camera_enumeration_failure_does_not_guess_numeric_inputs():
    def failure(backend):
        raise RuntimeError("Enumeration unavailable")
    assert available_devices(enumerator=failure) == ()
