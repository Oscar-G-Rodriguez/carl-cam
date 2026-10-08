"""Optional diagnostics cannot open hardware, dispatch commands or trap Back."""

import pytest
from PySide6.QtWidgets import QApplication
from test_ui import FakeCamera

from carl_cam.comparison_ui import ComparisonDialog
from carl_cam.storage import Store
from carl_cam.ui import CameraWindow


def test_comparison_privacy_hold_does_not_suggest_reinstalling_the_same_sdk(application, tmp_path):
    from types import SimpleNamespace

    from carl_cam.channels import LatestValue
    from carl_cam.vision_privacy import PRIVACY_HOLD
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.mode = 'inspect'
    dialog = ComparisonDialog(host)
    status = LatestValue()
    status.publish('Vision unavailable: ' + PRIVACY_HOLD)
    dialog.worker = SimpleNamespace(status=status, results=LatestValue(), stop=lambda: True)
    dialog.exercise = SimpleNamespace(phase='validation')
    try:
        dialog.refresh()
        assert 'missing or unsupported' in dialog.status.text()
        assert 'Repair' not in dialog.status.text() and not dialog.action.isEnabled()
    finally:
        dialog.reject()
        host.close()


@pytest.fixture(scope='module')
def application():
    return QApplication.instance() or QApplication([])


def test_comparison_without_camera_and_tested_model_stays_optional(application, tmp_path):
    camera = FakeCamera()
    store = Store(tmp_path / 'CarlCam')
    host = CameraWindow(test_camera=camera, store=store)
    host.mode = 'inspect'
    host.show()
    def unexpected_worker(*args, **kwargs):
        pytest.fail('A blocked exercise must not create a vision worker')
    dialog = ComparisonDialog(host, worker_factory=unexpected_worker)
    dialog.show()
    assert not dialog.save_consent.isChecked()
    dialog.action.click()
    assert dialog.exercise is None and dialog.worker is None
    assert not camera.running and not store.datasets.exists()
    assert 'required' in dialog.status.text()
    dialog.back.click()
    assert not dialog.isVisible() and host.isVisible()
    host.close()


def test_back_refuses_to_abandon_a_running_comparison_worker(application, tmp_path):
    class SlowWorker:
        stopped = False
        def stop(self):
            return self.stopped
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.mode = 'inspect'
    host.show()
    dialog = ComparisonDialog(host)
    worker = SlowWorker()
    dialog.worker = worker
    dialog.show()
    dialog.back.click()
    assert dialog.isVisible() and dialog.timer.isActive()
    assert 'still stopping' in dialog.status.text()
    assert host.isVisible()
    worker.stopped = True
    dialog.back.click()
    assert not dialog.isVisible() and not dialog.timer.isActive()
    assert host.isVisible()
    host.close()
