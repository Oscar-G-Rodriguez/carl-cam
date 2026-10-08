"""Camera-free Help layout and privacy-hold guidance."""

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from test_ui import FakeCamera

from carl_cam.help_ui import HelpDialog
from carl_cam.storage import Store
from carl_cam.training_ui import POSE_CROPS, POSE_NAMES
from carl_cam.ui import CameraWindow
from carl_cam.vision_privacy import VisionPrivacyError


@pytest.fixture(scope='module')
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def help_fixture(qt_app, tmp_path, monkeypatch):
    def held():
        raise VisionPrivacyError('fixture hold')
    monkeypatch.setattr('carl_cam.help_ui.require_verified_vision_runtime', held)
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    window.timer.stop()
    yield window
    window.close()


@pytest.mark.parametrize('size', [(1024, 700), (1024, 640)])
def test_help_readable_privacy_hold_illustrations_and_persistent_controls(qt_app, help_fixture, size):
    dialog = HelpDialog(help_fixture)
    try:
        dialog.resize(*size)
        dialog.show()
        qt_app.processEvents()
        assert (dialog.width(), dialog.height()) == size
        assert dialog.runtime_notice.isVisible() and 'on hold' in dialog.runtime_notice.text()
        assert 'mouse/keyboard' in dialog.runtime_notice.text()
        assert dialog.scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert dialog.scroll.horizontalScrollBar().maximum() == 0
        for button in (dialog.practice, dialog.back):
            origin = button.mapTo(dialog, QPoint())
            assert button.isVisible() and button.height() >= 44
            assert origin.y() + button.height() <= dialog.height() - 16
        for image in dialog.guide_images.values():
            assert image.accessibleName() and not image.pixmap().isNull()
        assert len(dialog.headings) == 8
        dialog.scroll.verticalScrollBar().setValue(dialog.scroll.verticalScrollBar().maximum())
        qt_app.processEvents()
        assert dialog.back.isVisible() and dialog.practice.isVisible()
        assert help_fixture.vision is None and help_fixture.model is None
        assert not help_fixture.camera.running and help_fixture.output is None
    finally:
        dialog.reject()


def test_verified_runtime_notice_is_not_fabricated_or_persisted(qt_app, help_fixture, monkeypatch):
    calls = []
    monkeypatch.setattr('carl_cam.help_ui.require_verified_vision_runtime', lambda: calls.append('hash-check') or 'fixture-hash')
    dialog = HelpDialog(help_fixture)
    try:
        dialog.show()
        qt_app.processEvents()
        assert calls == ['hash-check']
        assert dialog.runtime_notice.text() == '' and not dialog.runtime_notice.isVisible()
        assert help_fixture.vision is None and not help_fixture.camera.running
    finally:
        dialog.reject()


def test_help_missing_illustration_has_written_fallback(qt_app, help_fixture, monkeypatch):
    monkeypatch.setattr('carl_cam.help_ui.QPixmap', lambda *args: QPixmap())
    dialog = HelpDialog(help_fixture)
    try:
        assert all('written instructions' in image.text() for image in dialog.guide_images.values())
        assert all(image.accessibleName() for image in dialog.guide_images.values())
    finally:
        dialog.reject()


def test_help_keyboard_return_does_not_request_practice(qt_app, help_fixture):
    dialog = HelpDialog(help_fixture)
    dialog.show()
    qt_app.processEvents()
    dialog.back.setFocus()
    QTest.keyClick(dialog.back, Qt.Key.Key_Space)
    assert not dialog.isVisible() and not dialog.practice_requested


def test_help_keyboard_practice_is_explicit_request_only(qt_app, help_fixture):
    dialog = HelpDialog(help_fixture)
    dialog.show()
    qt_app.processEvents()
    dialog.practice.setFocus()
    QTest.keyClick(dialog.practice, Qt.Key.Key_Space)
    assert not dialog.isVisible() and dialog.practice_requested
    assert not help_fixture.camera.running and help_fixture.vision is None


@pytest.mark.parametrize('size', [(1024, 700), (1024, 640)])
@pytest.mark.parametrize('pose,drawn_width', [
    ('frame', 208), ('peace', 138), ('thumbs_up', 141),
    ('open_palm', 153), ('ily', 160),
])
def test_help_guide_uses_exact_owner_crop_and_centered_design_slot(
    qt_app, help_fixture, size, pose, drawn_width,
):
    sheet = QPixmap(str(Path(__file__).parents[1] / 'src/carl_cam/assets/guides/gesture-sheet.png'))
    assert not sheet.isNull()
    expected = sheet.copy(*POSE_CROPS[pose]).scaled(
        240, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
    )
    dialog = HelpDialog(help_fixture)
    try:
        dialog.resize(*size)
        dialog.show()
        qt_app.processEvents()
        image = dialog.guide_images[pose]
        actual = image.pixmap()
        assert actual.toImage() == expected.toImage()
        assert (actual.width(), actual.height()) == (drawn_width, 180)
        assert (image.width(), image.height()) == (288, 180)
        assert image.alignment() == Qt.AlignmentFlag.AlignCenter
        assert image.accessibleName() == POSE_NAMES[pose] + ' original drawing'
        cell = image.parentWidget()
        assert (cell.width(), cell.height()) == (304, 220)
        assert cell.layout().itemAt(1).widget().text() == POSE_NAMES[pose]
        assert help_fixture.vision is None and not help_fixture.camera.running
    finally:
        dialog.reject()
