from time import monotonic_ns

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from test_ui import FakeCamera

from carl_cam.comparison_ui import ComparisonDialog
from carl_cam.comparison_vision import ComparisonEngine
from carl_cam.records import FramePacket
from carl_cam.storage import Store
from carl_cam.training import GestureNetwork, NeuralCandidate, Standardizer
from carl_cam.ui import CameraWindow
from carl_cam.vision_privacy import VisionPrivacyError, require_verified_vision_runtime


def candidate():
    return NeuralCandidate(112, (64, 32), GestureNetwork(112, (64, 32)).eval(),
        Standardizer(np.zeros(112, np.float32), np.ones(112, np.float32)),
        .5, (.9,) * 6, 1, {'train': (), 'validation': (), 'sealed': ()})


def test_actual_offline_canned_model_receives_paired_frame_ids_without_retaining_pixels():
    try:
        require_verified_vision_runtime()
    except VisionPrivacyError:
        pytest.skip('Required real-model comparison check held pending a verified telemetry-free SDK')
    engine = ComparisonEngine(candidate())
    try:
        for index in range(2):
            pair = engine.process(FramePacket(index, monotonic_ns(), np.zeros((360, 640, 3), np.uint8)))
            assert pair.frame_id == index
            assert pair.neural.outcome == pair.canned.outcome == 'no_hand'
            assert pair.neural.completed_time_ns >= pair.capture_time_ns
            assert pair.canned.completed_time_ns >= pair.capture_time_ns
            assert not hasattr(pair, 'rgb') and not hasattr(pair, 'landmarks')
    finally:
        engine.close()


@pytest.fixture(scope='module')
def qt_app():
    return QApplication.instance() or QApplication([])


def test_optional_comparison_does_not_start_without_model_or_save_without_consent(qt_app, tmp_path):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.mode = 'inspect'
    dialog = ComparisonDialog(host)
    dialog.show()
    qt_app.processEvents()
    assert not dialog.save_consent.isChecked() and dialog.worker is None
    dialog.handle_action()
    assert dialog.exercise is None and dialog.worker is None
    assert 'required' in dialog.status.text()
    assert not host.store.datasets.exists()
    dialog.reject()
    host.close()
