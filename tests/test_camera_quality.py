from threading import get_ident

import cv2
import numpy as np
import pytest

from carl_cam import camera as module
from carl_cam.camera import CameraDevice, CameraWorker
from carl_cam.channels import LatestValue


@pytest.mark.parametrize('level', [-1, 3, True, 1.5])
def test_invalid_capture_quality_is_rejected_without_opening_a_camera(level):
    with pytest.raises(ValueError):
        CameraWorker().request_quality_level(level)


@pytest.mark.parametrize('failed_sizes', [(), (960,), (960, 640)])
def test_quality_queue_changes_only_capture_owner_and_acknowledges_request_ids(monkeypatch, failed_sizes):
    worker = CameraWorker()
    dimensions, events, statuses = [], [], []

    class Capture:
        width, height = 1280, 720
        closed = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            events.append(get_ident())
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self.width = value
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = value

        def read(self):
            if self.width in failed_sizes:
                return False, None
            return True, np.zeros((self.height, self.width, 3), np.uint8)

        def release(self):
            self.closed = True

    class Frames(LatestValue):
        def publish(self, packet):
            dimensions.append((packet.frame_id, packet.rgb.shape[:2]))
            assert not packet.rgb.flags.writeable
            if packet.frame_id == 0:
                assert worker.request_quality_level(1) == 1
            elif packet.frame_id == 1:
                assert worker.request_quality_level(0) == 2
            else:
                worker._stop.set()
            super().publish(packet)

    class Statuses(LatestValue):
        def publish(self, status):
            if status.state == 'live':
                statuses.append((status.quality_level, status.quality_request_id))
            super().publish(status)

    capture = Capture()
    monkeypatch.setattr(module, 'available_devices', lambda: (CameraDevice(0, 'Fake'),))
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda *args: capture)
    worker.frames, worker.status = Frames(), Statuses()
    worker._run()
    if failed_sizes == (960, 640):
        assert dimensions == [(0, (720, 1280))]
        assert statuses == [(0, 0)]
        assert worker.status.read().state == 'error'
    else:
        reduced_size, level = ((360, 640), 2) if failed_sizes else ((540, 960), 1)
        assert dimensions == [(0, (720, 1280)), (1, reduced_size), (2, (720, 1280))]
        assert statuses == [(0, 0), (level, 1), (0, 2)]
    assert set(events) == {get_ident()}
    assert capture.closed and worker.frames.read() is None
