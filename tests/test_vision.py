from time import monotonic_ns, sleep
from types import SimpleNamespace

import numpy as np
import pytest

from carl_cam.channels import LatestValue
from carl_cam.records import FramePacket, PerceptionPacket
from carl_cam.vision import HandHeadEngine, RegionEngine, VisionWorker, convert_hands
from carl_cam.vision_privacy import VisionPrivacyError, require_verified_vision_runtime


def test_normalized_coordinates_use_mirrored_canvas_rulers():
    points = [SimpleNamespace(x=0.25, y=0.5, z=-0.02) for _ in range(21)]
    raw = SimpleNamespace(hand_landmarks=[points], handedness=[[SimpleNamespace(category_name="Left", score=0.9)]])
    hand, = convert_hands(raw)
    assert hand.landmarks[0] == (320, 360, -0.02)
    assert hand.handedness == "Left"


def test_bundled_models_load_and_process_without_a_camera():
    try:
        require_verified_vision_runtime()
    except VisionPrivacyError:
        pytest.skip('Required real-model integration check held pending a verified telemetry-free SDK')
    packet = FramePacket(1, monotonic_ns(), np.zeros((360, 640, 3), dtype=np.uint8))
    hands = HandHeadEngine()
    regions = RegionEngine()
    try:
        perception = hands.process(packet)
        segmented = regions.process(packet)
        assert perception.frame_id == segmented.frame_id == packet.frame_id
        assert perception.capture_time_ns == segmented.capture_time_ns == packet.capture_time_ns
        assert not perception.hands and perception.head_center is None
        assert segmented.confidence_masks.shape == (360, 640, 6)
        assert np.isfinite(segmented.confidence_masks).all()
        assert not segmented.confidence_masks.flags.writeable
    finally:
        hands.close()
        regions.close()


def test_worker_replaces_latest_result_and_discards_stale_source():
    class Engine:
        closed = False

        def process(self, packet):
            return PerceptionPacket(packet.frame_id, packet.capture_time_ns, ())

        def close(self):
            self.closed = True

    frames = LatestValue()
    engine = Engine()
    worker = VisionWorker(frames, factory=lambda: engine)
    worker.start()
    try:
        frames.publish(FramePacket(7, monotonic_ns(), np.zeros((20, 20, 3), dtype=np.uint8)))
        for _ in range(100):
            if worker.results.read() is not None:
                break
            sleep(0.005)
        assert worker.results.read().frame_id == 7
        assert worker.source.read().frame_id == 7
        frames.publish(FramePacket(8, monotonic_ns() - 300_000_000, np.zeros((20, 20, 3), dtype=np.uint8)))
        for _ in range(100):
            if worker.results.read() is None:
                break
            sleep(0.005)
        assert worker.results.read() is None and worker.source.read() is None
    finally:
        assert worker.stop() and engine.closed


def test_worker_cancel_token_closes_engine_during_initialization():
    from threading import Event
    entered = Event()
    closed = Event()

    class Engine:
        def initialize(self, cancel):
            entered.set()
            assert cancel.wait(1)

        def process(self, packet):
            pytest.fail('Cancelled startup must never process a frame')

        def close(self):
            closed.set()

    worker = VisionWorker(LatestValue(), factory=Engine)
    worker.start()
    assert entered.wait(1)
    assert worker.stop(timeout=1) and closed.is_set()
    assert worker.backend_status.read() is None


def test_worker_publishes_changed_backend_after_gpu_fallback():
    class Engine:
        backend_status = 'GPU'

        def initialize(self, cancel):
            pass

        def process(self, packet):
            self.backend_status = 'CPU — GPU stopped'
            return PerceptionPacket(packet.frame_id, packet.capture_time_ns, ())

        def close(self):
            pass

    frames = LatestValue()
    worker = VisionWorker(frames, factory=Engine)
    worker.start()
    try:
        frames.publish(FramePacket(9, monotonic_ns(), np.zeros((20, 20, 3), np.uint8)))
        for _ in range(100):
            if worker.completed.read() is not None:
                break
            sleep(.005)
        assert worker.completed.read()[0].frame_id == 9
        assert worker.backend_status.read() == 'CPU — GPU stopped'
    finally:
        assert worker.stop()
