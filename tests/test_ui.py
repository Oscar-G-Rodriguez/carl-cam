from time import monotonic_ns
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from carl_cam.camera import CameraStatus
from carl_cam.channels import LatestValue
from carl_cam.records import FramePacket, PerceptionPacket, RenderPacket
from carl_cam.storage import Store
from carl_cam.ui import CameraWindow


class FakeCamera:
    def __init__(self) -> None:
        self.frames = LatestValue()
        self.status = LatestValue()
        self.running = False

    def start(self) -> None:
        self.running = True
        image = np.full((360, 640, 3), (130, 160, 180), dtype=np.uint8)
        self.frames.publish(FramePacket(1, monotonic_ns(), image))
        self.status.publish(CameraStatus("live", "Test camera", (640, 360), 30))

    def stop(self, timeout=3) -> bool:
        self.running = False
        self.frames.clear()
        return True


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_camera_is_off_until_started_and_keyboard_frame_is_operable(qt_app, tmp_path) -> None:
    fake = FakeCamera()
    window = CameraWindow(test_camera=fake, store=Store(tmp_path / 'CarlCam'))
    window.show()
    assert not fake.running
    window.start_camera()
    QTest.qWait(100)
    assert fake.running and window.preview._image is not None
    window.enable_mouse_frame()
    before = window.preview.quad[0]
    QTest.keyClick(window.preview, Qt.Key.Key_1)
    QTest.keyClick(window.preview, Qt.Key.Key_Right)
    assert window.preview.quad[0] == (before[0] + 8, before[1])
    QTest.keyClick(window.preview, Qt.Key.Key_Escape)
    assert window.preview.quad is None
    window.close()
    assert not fake.running


def test_stopped_camera_clears_preview(qt_app, tmp_path) -> None:
    fake = FakeCamera()
    window = CameraWindow(test_camera=fake, store=Store(tmp_path / 'CarlCam'))
    window.start_camera()
    QTest.qWait(100)
    assert window.stop_camera()
    assert window.preview._image.pixelColor(100, 100).red() == 0
    window.close()


@pytest.mark.parametrize('size', [(1280, 800), (1024, 640)])
def test_camera_screen_controls_fit_supported_sizes(qt_app, tmp_path, size):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    try:
        window.resize(*size)
        window.show()
        qt_app.processEvents()
        assert (window.width(), window.height()) == size
        controls = (window.setup_button, window.training_button, window.inspect_button,
                    window.settings_button, window.output_button, window.help_button,
                    window.devices, window.start_button, window.stop_button, window.themes,
                    window.refresh_devices_button, window.menu_button, window.frame_button,
                    window.clear_button, window.quality_button)
        for widget in controls:
            assert widget.isVisible()
            bounds = widget.geometry()
            assert bounds.width() >= widget.minimumSizeHint().width()
            assert bounds.height() >= 44
            assert window.centralWidget().rect().contains(bounds)
        assert [window.navigation_layout.itemAt(i).widget().text() for i in range(6)] == [
            'Camera setup', 'Train my gestures', 'Inspect', 'Settings', 'Start virtual camera', 'Help']
        assert all(not button.icon().isNull() for button in
                   (window.start_button, window.training_button, window.inspect_button))
        scene = window.preview.scene_rect()
        assert scene.width() / scene.height() == pytest.approx(16 / 9)
        assert window.preview.rect().contains(scene.toAlignedRect())
        center = scene.center().toPoint()
        assert window.preview.grab().toImage().pixelColor(center).name() == '#000000'
        assert window.start_button.isEnabled() and not window.stop_button.isEnabled()
        assert qt_app.font().pixelSize() == 14
        assert window.title_bar.height() == 56
        assert all((button.width(), button.height()) == (44, 44)
                   for button in window.title_bar.buttons.values())
        window.enable_mouse_frame()
        window.asset_warning.show()
        qt_app.processEvents()
        assert (window.width(), window.height()) == size
        assert window.frame_instructions.isVisible() and window.asset_warning.isVisible()
        assert window.centralWidget().rect().contains(window.frame_instructions.geometry())
        assert window.centralWidget().rect().contains(window.asset_warning.geometry())
    finally:
        window.close()


def test_manual_frame_instructions_and_camera_buttons_follow_state(qt_app, tmp_path):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    try:
        window.show()
        assert not window.frame_instructions.isVisible()
        window.start_camera()
        assert not window.start_button.isEnabled() and window.stop_button.isEnabled()
        window.enable_mouse_frame()
        assert window.frame_instructions.isVisible()
        window.open_menu()
        assert not window.frame_instructions.isVisible()
        window.dispatch_menu('back')
        window.enable_mouse_frame()
        window._set_quad(None)
        assert not window.frame_instructions.isVisible()
        window.enable_mouse_frame()
        assert window.stop_camera()
        assert not window.frame_instructions.isVisible()
        assert not window._manual_frame and window.preview.quad is None
        assert window.start_button.isEnabled() and not window.stop_button.isEnabled()
    finally:
        window.close()


def test_privacy_hold_is_visible_and_cannot_open_a_training_dead_end(qt_app, tmp_path, monkeypatch):
    from carl_cam import training_ui
    from carl_cam.vision import VisionWorker
    from carl_cam.vision_privacy import PRIVACY_HOLD, VisionPrivacyError
    fake = FakeCamera()
    window = CameraWindow(test_camera=fake, store=Store(tmp_path / 'CarlCam'))
    def no_training(*args, **kwargs):
        pytest.fail('An unverified SDK must not enter guided collection')
    monkeypatch.setattr(training_ui, 'TrainingDialog', no_training)
    def held():
        raise VisionPrivacyError(PRIVACY_HOLD)
    monkeypatch.setattr('carl_cam.vision_privacy.require_verified_vision_runtime', held)
    try:
        window.start_camera()
        window.vision = VisionWorker(fake.frames, factory=held)
        window.vision.status.publish('Vision unavailable: ' + PRIVACY_HOLD)
        window.refresh()
        assert PRIVACY_HOLD in window.status.text()
        window.open_training()
        assert window.status.text() == PRIVACY_HOLD
        assert window.mode == 'camera' and fake.running
    finally:
        window.close()


def test_reduced_input_quality_is_visible_and_clears_at_full_size(qt_app, tmp_path):
    fake = FakeCamera()
    window = CameraWindow(test_camera=fake, store=Store(tmp_path / 'CarlCam'))
    try:
        window.start_camera()
        window.refresh()
        assert '640×360 input' in window.status.text()
        assert 'Reduced input quality' in window.status.text()
        fake.status.publish(CameraStatus('live', 'Test camera', (1280, 720), 30))
        window.refresh()
        assert '1280×720 input' in window.status.text()
        assert 'Reduced input quality' not in window.status.text()
    finally:
        window.close()


class QualityCamera(FakeCamera):
    def __init__(self):
        super().__init__()
        self.requests = []

    def request_quality_level(self, level):
        self.requests.append(level)
        return len(self.requests)


def test_window_automatic_fallback_and_manual_restore_wait_for_exact_ack(qt_app, tmp_path, monkeypatch):
    from carl_cam import ui

    clock = [100.]
    monkeypatch.setattr(ui, 'monotonic', lambda: clock[0])
    fake = QualityCamera()
    fake.running = True
    window = CameraWindow(test_camera=fake, store=Store(tmp_path / 'CarlCam'))
    pixels = np.zeros((2, 2, 3), np.uint8)
    live = CameraStatus('live', 'Fake', (1280, 720), 30)
    try:
        for index in range(11):
            clock[0] = 100 + index / 2
            now_ns = int(clock[0] * 1e9)
            fake.frames.publish(FramePacket(index, now_ns, pixels))
            fake.status.publish(live)
            result = RenderPacket(index, pixels, pixels, now_ns, 10, now_ns - 100_000_000, index * 10)
            window._observe_performance(result, live)
        assert fake.requests == [1]
        assert window._quality_pending_request == 1
        assert window._latest_input is None
        clock[0] += .1
        fake.frames.publish(FramePacket(200, int(clock[0] * 1e9), pixels))
        reduced = CameraStatus('live', 'Fake', (960, 540), 30, 1, 1)
        fake.status.publish(reduced)
        window._observe_performance(result, reduced)
        assert window._quality_pending_request is None
        assert window._quality_capture_barrier == 200
        assert window.quality_button.isEnabled()
        # An old-format hand result can be recent in time but still unusable.
        old_hands = LatestValue()
        old_hands.publish(PerceptionPacket(150, int(clock[0] * 1e9), ()))
        window.vision = SimpleNamespace(results=old_hands, stop=lambda: True)
        window._update_live_controls()
        assert window._latest_input is None
        window.restore_input_quality()
        assert fake.requests == [1, 0]
        assert window._quality_pending_request == 2
        window._observe_performance(result, reduced)
        assert window._quality_pending_request == 2  # Old ack cannot authorize restore.
        restored = CameraStatus('live', 'Fake', (1280, 720), 30, 0, 2)
        fake.status.publish(restored)
        window._observe_performance(result, restored)
        assert window._quality_pending_request is None
        assert not window.quality_button.isEnabled()
    finally:
        window.close()


def test_window_quality_policy_is_inactive_in_private_modes_and_camera_loss(qt_app, tmp_path, monkeypatch):
    from carl_cam import ui

    clock = [100.]
    monkeypatch.setattr(ui, 'monotonic', lambda: clock[0])
    fake = QualityCamera()
    fake.running = True
    window = CameraWindow(test_camera=fake, store=Store(tmp_path / 'CarlCam'))
    pixels = np.zeros((2, 2, 3), np.uint8)
    try:
        for mode, state in (('training', 'live'), ('inspect', 'live'), ('settings', 'live'), ('camera', 'error')):
            window.mode = mode
            for index in range(12):
                clock[0] += .5
                fake.frames.publish(FramePacket(index, int(clock[0] * 1e9), pixels))
                status = CameraStatus(state, 'Fake', (640, 360), 30, 2)
                fake.status.publish(status)
                window._observe_performance(None, status)
            window.restore_input_quality()
        assert fake.requests == []
    finally:
        window.close()
