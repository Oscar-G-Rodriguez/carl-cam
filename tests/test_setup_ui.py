import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton, QStyle, QStyleOptionButton
from test_ui import FakeCamera

from carl_cam.setup_ui import SetupDialog
from carl_cam.storage import Store
from carl_cam.ui import CameraWindow


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_setup_requires_privacy_camera_and_explicit_gesture_choice(qt_app, tmp_path):
    camera = FakeCamera()
    host = CameraWindow(test_camera=camera, store=Store(tmp_path / "CarlCam"))
    dialog = SetupDialog(host)
    assert not camera.running and dialog.stack.currentIndex() == 0
    dialog.advance()
    assert dialog.stack.currentIndex() == 0 and "checkbox" in dialog.message.text()
    dialog.consent.setChecked(True)
    dialog.advance()
    assert dialog.stack.currentIndex() == 1 and host.settings.privacy_accepted
    dialog.advance()
    assert dialog.stack.currentIndex() == 1 and "working camera" in dialog.message.text()
    host.start_camera()
    dialog.advance()
    assert dialog.stack.currentIndex() == 2
    dialog.advance()
    assert dialog.stack.currentIndex() == 2 and not host.settings.setup_complete
    dialog.use_mouse()
    dialog.advance()
    assert dialog.stack.currentIndex() == 3 and host.output is None
    dialog.advance()
    assert host.settings.setup_complete and host.store.load_settings()[0].setup_complete
    assert host.output is None and host.model is None
    host.close()


def test_setup_can_be_revisited_and_closed_without_starting_camera(qt_app, tmp_path):
    camera = FakeCamera()
    host = CameraWindow(test_camera=camera, store=Store(tmp_path / "CarlCam"))
    dialog = SetupDialog(host)
    dialog.show()
    dialog.resize(1024, 640)
    QTest.qWait(20)
    assert dialog.width() == 1024 and not camera.running
    dialog.consent.setChecked(True)
    dialog.advance()
    dialog.back()
    assert dialog.stack.currentIndex() == 0
    dialog.reject()
    assert not camera.running and not host.settings.setup_complete
    host.close()


def test_settings_and_setup_screen_wiring_return_to_camera(qt_app, tmp_path, monkeypatch):
    from carl_cam.settings_ui import SettingsDialog
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    monkeypatch.setattr(SettingsDialog, "exec", lambda self: 0)
    host.open_settings()
    assert host.mode == "camera"
    monkeypatch.setattr(SetupDialog, "exec", lambda self: 0)
    host.open_setup()
    assert host.mode == "camera" and host.preview.quad is None
    host.close()


@pytest.mark.parametrize('size', [(1024, 700), (1024, 640)])
@pytest.mark.parametrize('step', range(4))
def test_all_setup_views_fit_without_starting_capture(qt_app, tmp_path, size, step):
    camera = FakeCamera()
    host = CameraWindow(test_camera=camera, store=Store(tmp_path / 'CarlCam'))
    dialog = SetupDialog(host)
    try:
        dialog.resize(*size)
        dialog.stack.setCurrentIndex(step)  # Layout fixture only; gates are tested above.
        dialog.show_step()
        dialog.show()
        qt_app.processEvents()
        assert (dialog.width(), dialog.height()) == size
        assert not dialog.message.isVisible()
        assert not camera.running and host.output is None and host.model is None
        assert not host.settings.setup_complete
        assert dialog.title_bar.height() == 56
        controls = dialog.stack.currentWidget().findChildren(QPushButton)
        controls += dialog.stack.currentWidget().findChildren(QCheckBox)
        controls += [dialog.previous, dialog.next]
        for widget in controls:
            assert widget.isVisible() and widget.height() >= 44
            top_left = widget.mapTo(dialog, widget.rect().topLeft())
            bottom_right = widget.mapTo(dialog, widget.rect().bottomRight())
            assert dialog.rect().contains(top_left) and dialog.rect().contains(bottom_right)
        if step == 0:
            assert not dialog.vision_notice.isVisible()
            assert 'MediaPipe usage metrics' in dialog.consent.text()
            assert dialog.heading.text().endswith('Your camera stays on this computer')
        if step == 1:
            scene = dialog.preview.scene_rect()
            assert scene.width() / scene.height() == pytest.approx(16 / 9)
            if size == (1024, 700):
                assert scene.height() == pytest.approx(400)
            assert dialog.devices.width() == 240
            assert dialog.preview.grab().toImage().pixelColor(scene.center().toPoint()).name() == '#000000'
        if step == 3:
            assert dialog.practice_choice.isChecked() and dialog.next.text() == 'Finish setup'
    finally:
        dialog.reject()
        host.close()


def test_setup_messages_reveal_recovery_without_hiding_footer(qt_app, tmp_path):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    dialog = SetupDialog(host)
    try:
        dialog.resize(1024, 640)
        dialog.show()
        dialog.advance()
        qt_app.processEvents()
        assert dialog.message.isVisible() and 'checkbox' in dialog.message.text()
        assert dialog.next.isVisible() and dialog.next.height() >= 44
        dialog.set_message('')
        qt_app.processEvents()
        assert not dialog.message.isVisible()
    finally:
        dialog.reject()
        host.close()


def test_setup_checkbox_is_keyboard_operable_and_uses_18_pixel_indicator(qt_app, tmp_path):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    dialog = SetupDialog(host)
    try:
        dialog.show()
        dialog.consent.setFocus()
        qt_app.processEvents()
        assert dialog.consent.height() >= 44 and not dialog.consent.isChecked()
        option = QStyleOptionButton()
        option.initFrom(dialog.consent)
        indicator = dialog.consent.style().subElementRect(
            QStyle.SubElement.SE_CheckBoxIndicator, option, dialog.consent)
        assert (indicator.width(), indicator.height()) == (18, 18)
        QTest.keyClick(dialog.consent, Qt.Key.Key_Space)
        assert dialog.consent.isChecked() and not host.camera.running
    finally:
        dialog.reject()
        host.close()
