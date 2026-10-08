from time import monotonic_ns, sleep
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from test_ui import FakeCamera

from carl_cam.channels import LatestValue
from carl_cam.help_ui import HelpDialog
from carl_cam.inspection_ui import InspectionDialog
from carl_cam.records import FramePacket
from carl_cam.render import InspectionSnapshot, RenderWorker, SceneState
from carl_cam.storage import Store
from carl_cam.ui import CameraWindow


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_inspection_publishes_matched_panels_but_broadcasts_only_plain_camera(monkeypatch):
    now = monotonic_ns()
    monkeypatch.setattr("carl_cam.render.monotonic_ns", lambda: now)
    rgb = np.full((720, 1280, 3), (180, 140, 120), np.uint8)
    frames = LatestValue()
    frames.publish(FramePacket(8, now, rgb))
    worker = RenderWorker(frames)
    worker.scene.publish(SceneState("red", ((300, 150), (980, 150), (980, 570), (300, 570)), "inspect"))
    worker.start()
    try:
        for _ in range(100):
            if worker.results.read() is not None:
                break
            sleep(.002)
        result = worker.results.read()
        snapshot = worker.inspection.read()
        assert snapshot.frame_id == result.frame_id == 8
        np.testing.assert_array_equal(result.output_rgb, rgb)
        assert not np.array_equal(snapshot.final_rgb[300, 500], rgb[300, 500])
        assert not np.array_equal(snapshot.overlay_rgb, snapshot.original_rgb)
        assert not snapshot.original_rgb.flags.writeable
        assert not snapshot.stages.edge_magnitude.flags.writeable
    finally:
        assert worker.stop()


def test_help_inspection_and_camera_layout_at_minimum_size(qt_app, tmp_path):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    window.resize(1024, 640)
    window.show()
    QTest.qWait(20)
    assert window.width() == 1024 and window.height() == 640
    assert window.grab().save("build/camera-screen.png")
    help_dialog = HelpDialog(window)
    help_dialog.show()
    assert help_dialog.findChildren(type(window.status))
    assert help_dialog.grab().save("build/help-screen.png")
    help_dialog.reject()
    inspection = InspectionDialog(window)
    inspection.resize(1024, 640)
    inspection.show()
    QTest.qWait(20)
    assert len(inspection.cells) == 6 and inspection.width() == 1024 and inspection.height() == 640
    assert inspection.grab().save("build/inspection-screen.png")
    inspection.reject()
    window.close()


@pytest.fixture
def inspection_fixture(qt_app, tmp_path, monkeypatch):
    """Numerical/color fixtures only; no camera or SDK starts."""
    now = 5_000_000_000
    monkeypatch.setattr('carl_cam.inspection_ui.monotonic_ns', lambda: now)
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    renderer = SimpleNamespace(inspection=LatestValue(), results=LatestValue())
    host.renderer = renderer
    dialog = InspectionDialog(host)
    dialog.timer.stop()
    rgb = np.full((8, 12, 3), (180, 140, 120), np.uint8)
    stages = SimpleNamespace(luminance=np.full((8, 12), .25), edge_magnitude=np.full((8, 12), .5))
    snapshot = InspectionSnapshot(8, now, rgb, stages, np.ones((8, 12)),
                                  np.full_like(rgb, 100), np.full_like(rgb, 200), False)
    renderer.inspection.publish(snapshot)
    renderer.results.publish(SimpleNamespace(frame_id=8, capture_time_ns=now, render_ms=41.0))
    try:
        yield host, dialog, renderer, snapshot, now
    finally:
        dialog.reject()
        host.renderer = host.vision = host.model = None
        host.close()


def test_inspection_reads_six_matched_signals_and_timing(inspection_fixture):
    _, dialog, _, _, _ = inspection_fixture
    dialog.refresh()
    assert [p._image.pixelColor(0, 0).red() for p in dialog.cells] == [180, 64, 128, 255, 100, 200]
    assert 'Frame 8' in dialog.metrics.text() and '41 ms' in dialog.metrics.text()
    assert 'palette only' in dialog.metrics.text()


def test_inspection_does_not_attach_another_frames_timing(inspection_fixture):
    _, dialog, renderer, _, now = inspection_fixture
    renderer.results.publish(SimpleNamespace(frame_id=9, capture_time_ns=now, render_ms=999.0))
    dialog.refresh()
    assert 'Frame 8' in dialog.metrics.text() and 'ms processing' not in dialog.metrics.text()
    renderer.results.publish(SimpleNamespace(frame_id=8, capture_time_ns=now - 1, render_ms=999.0))
    dialog.refresh()
    assert 'ms processing' not in dialog.metrics.text()


@pytest.mark.parametrize('age', [250_000_001, -1])
def test_inspection_clears_stale_or_future_snapshots_and_restores_same_frame(inspection_fixture, age):
    host, dialog, renderer, snapshot, now = inspection_fixture
    dialog.refresh()
    host.camera.running = True  # State fixture; no start() call or device access.
    renderer.inspection.publish(InspectionSnapshot(8, now - age, snapshot.original_rgb, snapshot.stages,
                                                 snapshot.person_mask, snapshot.overlay_rgb,
                                                 snapshot.final_rgb, False))
    dialog.refresh()
    assert all(p._image.pixelColor(0, 0).red() == 0 for p in dialog.cells)
    assert dialog.last_id == -1 and 'cleared' in dialog.metrics.text()
    renderer.inspection.publish(snapshot)
    dialog.refresh()
    assert dialog.cells[0]._image.pixelColor(0, 0).red() == 180
    host.camera.running = False


def test_inspection_classifier_readout_is_same_frame_only(inspection_fixture):
    host, dialog, _, _, now = inspection_fixture
    calls = []
    host.vision = SimpleNamespace(results=LatestValue())
    host.model = SimpleNamespace(width=112, threshold=.8, margin=.15,
                                 scores=lambda vector: calls.append(vector) or np.array([[.9, .02, .02, .02, .02, .02]]))
    dialog.extractor = SimpleNamespace(extract=lambda hands: SimpleNamespace(vectors={112: np.zeros(112)}))
    for frame, capture in ((9, now), (8, now - 1)):
        host.vision.results.publish(SimpleNamespace(frame_id=frame, capture_time_ns=capture, hands=()))
        dialog.refresh()
        assert not calls and 'predicted command' not in dialog.metrics.text()
    host.vision.results.publish(SimpleNamespace(frame_id=8, capture_time_ns=now, hands=()))
    dialog.refresh()
    assert len(calls) == 1 and 'predicted command: frame' in dialog.metrics.text()
    assert 'uncalibrated' in dialog.metrics.text()


def test_inspection_recovery_clears_partial_panels(inspection_fixture):
    _, dialog, renderer, snapshot, now = inspection_fixture
    dialog.refresh()
    # The first panel updates, then an empty luminance image makes cv2.resize
    # fail. Recovery must clear all panels, including that partial update.
    broken = SimpleNamespace(luminance=np.zeros((0, 0)), edge_magnitude=np.zeros((8, 12)))
    renderer.inspection.publish(InspectionSnapshot(9, now, snapshot.original_rgb, broken,
                                                 snapshot.person_mask, snapshot.overlay_rgb,
                                                 snapshot.final_rgb, False))
    dialog.safe_refresh()
    assert all(p._image.pixelColor(0, 0).red() == 0 for p in dialog.cells)
    assert 'could not be inspected' in dialog.metrics.text()


@pytest.mark.parametrize('size', [(1200, 780), (1024, 640)])
def test_inspection_grid_and_controls_fit_supported_sizes(qt_app, tmp_path, size):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = InspectionDialog(host)
    dialog.timer.stop()
    dialog.resize(*size)
    dialog.show()
    QTest.qWait(30)
    assert (dialog.width(), dialog.height()) == size
    assert len(dialog.panel_labels) == 6 and len(dialog.cells) == 6
    for cell, label in zip(dialog.cells, dialog.panel_labels, strict=True):
        scene = cell.scene_rect()
        assert abs(scene.width() / scene.height() - 16 / 9) < .001
        assert cell.width() >= 240 and cell.height() >= 135
        assert label.fontMetrics().horizontalAdvance(label.text()) <= label.width()
    for button in (dialog.back, dialog.compare):
        assert button.height() >= 44 and dialog.rect().contains(button.geometry())
    assert not host.camera.running and host.vision is None and host.output is None
    dialog.reject()
    host.close()
