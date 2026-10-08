import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout
from test_ui import FakeCamera

from carl_cam.storage import Store
from carl_cam.ui import CameraWindow
from carl_cam.window_chrome import install_dialog_chrome


@pytest.fixture(scope="module")
def application():
    yield QApplication.instance() or QApplication([])


def test_custom_controls_are_named_and_keyboard_operable(application, tmp_path):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    window.show()
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    for button in window.title_bar.buttons.values():
        assert button.width() >= 44 and button.height() >= 44
        assert button.accessibleName() and not button.icon().isNull()
    maximize = window.title_bar.buttons["maximize"]
    maximize.setFocus()
    QTest.keyClick(maximize, Qt.Key.Key_Space)
    assert window.isMaximized() and "Restore" in maximize.accessibleName()
    QTest.keyClick(maximize, Qt.Key.Key_Space)
    assert not window.isMaximized()
    window.start_camera()
    camera = window.camera
    assert camera.running
    close = window.title_bar.buttons["close"]
    close.setFocus()
    QTest.keyClick(close, Qt.Key.Key_Space)
    assert not window.isVisible() and window.camera is None and not camera.running


def test_missing_assets_leave_readable_recovery_controls_and_explicit_repair_notice(application, tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise ValueError('Missing test graphic')
    monkeypatch.setattr('carl_cam.ui.asset_pixels', missing)
    monkeypatch.setattr('carl_cam.window_chrome.verified_svg', missing)
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    window.show()
    assert window.asset_warning.isVisible() and 'repair' in window.asset_warning.text()
    assert [button.text() for button in window.title_bar.buttons.values()] == ['_', '□', 'X']
    assert window.start_button.isEnabled() and window.help_button.isEnabled()
    window.close()


def test_missing_font_recovery_is_readable_but_strict_verification_rejects_it(application, tmp_path, monkeypatch):
    from PySide6.QtGui import QFontDatabase

    from carl_cam.typography import configure_typography

    previous = application.property('carlCamFontLoaded')
    application.setProperty('carlCamFontLoaded', False)
    monkeypatch.setattr(QFontDatabase, 'addApplicationFont', lambda path: -1)
    try:
        assert configure_typography(strict=False) is False
        with pytest.raises(RuntimeError, match='font is missing'):
            configure_typography(strict=True)
        window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
        window.show()
        assert window.asset_warning.isVisible()
        assert window.start_button.isEnabled() and window.help_button.isEnabled()
        window.close()
    finally:
        application.setProperty('carlCamFontLoaded', previous)


@pytest.mark.parametrize('result', [QDialog.DialogCode.Accepted, QDialog.DialogCode.Rejected])
def test_dialog_completion_retains_host_without_explicit_exit_intent(application, tmp_path, result):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    host.show()
    dialog = QDialog(host)
    install_dialog_chrome(dialog, QVBoxLayout(dialog), host)
    dialog.show()
    application.processEvents()
    try:
        dialog.done(result)
        application.processEvents()
        assert not dialog.isVisible() and host.isVisible()
        assert not dialog.title_bar._exit_requested
        assert host.vision is None and host.output is None and not host.camera.running
    finally:
        dialog.reject()
        host.close()
