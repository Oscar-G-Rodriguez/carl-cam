from time import monotonic_ns
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from test_ui import FakeCamera

from carl_cam.channels import LatestValue
from carl_cam.evaluation import Exercise
from carl_cam.records import FramePacket, PerceptionPacket
from carl_cam.storage import LABELS, Store
from carl_cam.training_flow import TrainingFlow
from carl_cam.training_ui import POSE_CROPS, TrainingDialog
from carl_cam.ui import CameraWindow


class FakeVision:
    def __init__(self):
        self.results = LatestValue()

    def start(self):
        self.results.publish(PerceptionPacket(1, monotonic_ns(), ()))

    def stop(self):
        self.results.clear()
        return True


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('available,message', [(True, 'NVIDIA training is available.'),
    (False, 'No compatible NVIDIA GPU and driver are available. CPU training is available.')])
def test_review_explains_gpu_availability_and_keeps_cpu_default(qt_app, tmp_path, monkeypatch, available, message):
    from carl_cam import training_ui
    monkeypatch.setattr(training_ui, 'gpu_training_status', lambda: (available, message))
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    fit_rows = [{'label': label, 'round_index': index, 'features': {'180': [0.] * 180}}
                for label in LABELS for index in (0, 3) for _ in range(10)]
    dialog.flow = SimpleNamespace(phase='review', rows=lambda: fit_rows, stop=lambda **_: True)
    try:
        dialog.show()
        dialog._show_phase(force=True)
        qt_app.processEvents()
        assert dialog.cuda.isVisible() and dialog.cuda.isEnabled() is available
        assert not dialog.cuda.isChecked()
        assert dialog.cuda_info.isVisible() and dialog.cuda_info.text() == message
        assert dialog.cuda.toolTip() == message
        assert not host.camera.running and host.output is None
    finally:
        dialog.reject()
        host.close()


@pytest.mark.parametrize('size', [(1180, 780), (1024, 640)])
def test_started_movement_check_label_fits_and_explains_camera_pause(qt_app, tmp_path, size):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    candidate = SimpleNamespace(width=112, threshold=0.8, margin=0.15)
    exercise = Exercise((candidate,), attempts_per_pose=5)
    for pose in LABELS[:5]:
        for _ in range(5):
            exercise.start_trial(pose, 0.0)
            exercise.finish_trial()
    dialog.flow = SimpleNamespace(phase='comparison', exercise=exercise, stop=lambda **_: True)
    dialog._trial_index = 25
    for widget in (dialog.recording_choice, dialog.mapping_widget, dialog.consent, dialog.consent_info):
        widget.hide()
    try:
        dialog._show_phase(force=True)
        dialog.resize(*size)
        dialog.show()
        qt_app.processEvents()
        dialog.handle_action()
        dialog.safe_refresh()
        assert dialog.action.text() == 'Checking movement…' and not dialog.action.isEnabled()
        assert QFontMetrics(dialog.action.font()).horizontalAdvance(dialog.action.text()) <= dialog.action.width() - 20
        assert dialog._negative_started and exercise.negative_seconds == 0
        assert 'Movement check is paused' in dialog.detail.text()
        assert dialog.back.isVisible() and dialog.rect().contains(dialog.back.geometry())
        assert not host.camera.running and host.vision is None and host.output is None
        assert host.model is None and not host.store.datasets.exists()
        assert not host.store.models.exists()
    finally:
        dialog.reject()
        host.close()


def test_training_requires_consent_and_distinct_pose_assignments(qt_app, tmp_path):
    store = Store(tmp_path / "CarlCam")
    host = CameraWindow(test_camera=FakeCamera(), test_vision=FakeVision(), store=store)
    host.start_camera()
    dialog = TrainingDialog(host)
    dialog.show()
    QTest.qWait(30)
    dialog.handle_action()
    assert dialog.flow is None and not store.datasets.exists()
    dialog.consent.setChecked(True)
    dialog.mapping["open_palm"].setCurrentIndex(0)
    dialog.handle_action()
    assert dialog.flow is None and "exactly once" in dialog.detail.text()
    dialog.mapping["open_palm"].setCurrentIndex(1)
    dialog.handle_action()
    assert dialog.flow.phase == "collection"
    assert not dialog.drawing.pixmap().isNull()
    assert dialog.action.text() == "Start round"
    dialog.reject()
    host.close()


def test_training_layout_and_owner_drawing_render_at_supported_size(qt_app, tmp_path):
    host = CameraWindow(test_camera=FakeCamera(), test_vision=FakeVision(), store=Store(tmp_path / "CarlCam"))
    host.start_camera()
    dialog = TrainingDialog(host)
    dialog.show()
    dialog.consent.setChecked(True)
    dialog.handle_action()
    dialog.resize(1024, 640)
    QTest.qWait(50)
    assert dialog.width() == 1024 and dialog.height() == 640
    assert dialog.preview.width() >= 480
    assert dialog.action.isVisible()
    assert dialog.grab().save("build/training-screen.png")
    dialog.reject()
    host.close()


def test_drawings_keep_the_approved_crop_shapes(qt_app, tmp_path):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    dialog = TrainingDialog(host)
    assert dialog.sheet.width() == 1126 and dialog.sheet.height() == 908
    for pose, (_, _, width, height) in POSE_CROPS.items():
        dialog.show_pose(pose)
        pixmap = dialog.drawing.pixmap()
        assert abs(pixmap.width() / pixmap.height() - width / height) < 0.02
    dialog.reject()
    host.close()


def test_menu_is_keyboard_operable_without_a_personal_model(qt_app, tmp_path):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    window.show()
    window.start_camera()
    window.open_menu()
    assert window.mode == "menu" and window.model is None
    QTest.keyClick(window.preview, Qt.Key.Key_Right)
    QTest.keyClick(window.preview, Qt.Key.Key_Return)
    assert window.mode == "camera" and window.theme == "yellow"
    assert window.themes.currentData() == "yellow"
    window.open_menu()
    QTest.keyClick(window.preview, Qt.Key.Key_Escape)
    assert window.mode == "camera"
    window.close()


@pytest.mark.parametrize("size", [(1180, 780), (1024, 640)])
@pytest.mark.parametrize("collection", [False, True])
def test_training_panels_scroll_vertically_without_clipped_consent(qt_app, tmp_path, size, collection):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    if collection:
        dialog.consent.setChecked(True)
        dialog.handle_action()
        assert dialog.flow.current is None and dialog.flow.rows() == []
    dialog.resize(*size)
    dialog.show()
    QTest.qWait(30)
    assert (dialog.width(), dialog.height()) == size
    assert dialog.preview.width() >= 480
    assert dialog.left_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.right_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.back.isVisible() and dialog.back.height() >= 44
    assert dialog.rect().contains(dialog.back.geometry())
    if not collection:
        assert not dialog.drawing.isVisible()
        dialog.left_scroll.ensureWidgetVisible(dialog.consent)
        QTest.qWait(10)
        assert dialog.consent.fontMetrics().horizontalAdvance(dialog.consent.text()) + 26 <= dialog.consent.width()
        assert not dialog.consent.isChecked()
        dialog.consent.setFocus()
        QTest.keyClick(dialog.consent, Qt.Key.Key_Space)
        assert dialog.consent.isChecked() and dialog.flow is None
    else:
        assert dialog.drawing.isVisible()
        assert dialog.drawing.pixmap().width() <= 206
    assert not host.camera.running and host.vision is None and host.model is None and host.output is None
    dialog.reject()
    host.close()


def test_training_back_remains_visible_during_long_reports(qt_app, tmp_path):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    dialog.detail.setText("Long comparison and recovery report.\n" * 100)
    dialog.resize(1024, 640)
    dialog.show()
    QTest.qWait(30)
    assert dialog.right_scroll.verticalScrollBar().maximum() > 0
    assert dialog.back.isVisible() and dialog.rect().contains(dialog.back.geometry())
    QTest.mouseClick(dialog.back, Qt.MouseButton.LeftButton)
    assert not dialog.isVisible() and not host.store.datasets.exists()
    host.close()


def test_resume_description_does_not_expand_training_panel(qt_app, tmp_path):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    host.timer.stop()
    empty_collection = TrainingFlow(host.store, host.settings,
                                    assignments={"thumbs_up": "red", "open_palm": "yellow", "ily": "blue"})
    assert empty_collection.rows() == []
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    assert dialog.recording_choice.count() == 2
    dialog.recording_choice.setCurrentIndex(1)
    dialog.resize(1024, 640)
    dialog.show()
    QTest.qWait(30)
    assert dialog.left_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.recording_choice.width() <= dialog.left_scroll.viewport().width() - 16
    assert dialog.flow is None and not host.camera.running and host.vision is None
    dialog.reject()
    host.close()


@pytest.mark.parametrize('capture_state', [
    'stale', 'future', 'missing', 'stopped',
    'tracking_stale', 'tracking_future', 'tracking_missing',
    'tracking_late',
])
def test_negative_exercise_pauses_when_capture_is_not_live(
    qt_app, tmp_path, monkeypatch, capture_state,
):
    camera, vision = FakeCamera(), FakeVision()
    host = CameraWindow(test_camera=camera, test_vision=vision, store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    host.vision = vision  # Fixture channel only; no worker or native SDK starts.
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    candidate = SimpleNamespace(width=112, threshold=0.8, margin=0.15)
    exercise = Exercise((candidate,), attempts_per_pose=5)
    for pose in LABELS[:5]:
        for _ in range(5):
            exercise.start_trial(pose, 99)
            exercise.finish_trial()
    exercise.start_negative()
    dialog.flow = SimpleNamespace(phase='comparison', exercise=exercise, stop=lambda **_: True)
    dialog._negative_started = True
    clock = [100.0]
    monkeypatch.setattr('carl_cam.training_ui.monotonic', lambda: clock[0])
    pixels = np.zeros((360, 640, 3), dtype=np.uint8)

    def publish(frame_id, *, capture_time=None):
        clock[0] = round(clock[0], 9)
        stamp = round(clock[0] * 1e9)
        camera.frames.publish(FramePacket(frame_id, stamp if capture_time is None else capture_time, pixels))
        vision.results.publish(PerceptionPacket(frame_id, stamp, ()))

    try:
        camera.running = True
        publish(1)
        dialog.refresh()
        clock[0] += 0.05
        publish(2)
        dialog.refresh()
        assert exercise.negative_seconds == pytest.approx(0.05)
        clock[0] += 0.02
        publish(3)
        if capture_state == 'stale':
            camera.frames.publish(FramePacket(3, round((clock[0] - 1) * 1e9), pixels))
        elif capture_state == 'future':
            camera.frames.publish(FramePacket(3, round((clock[0] + 1) * 1e9), pixels))
        elif capture_state == 'missing':
            camera.frames.clear()
        elif capture_state == 'stopped':
            camera.running = False
        elif capture_state == 'tracking_stale':
            vision.results.publish(PerceptionPacket(3, round((clock[0] - 1) * 1e9), ()))
        elif capture_state == 'tracking_future':
            vision.results.publish(PerceptionPacket(3, round((clock[0] + 1) * 1e9), ()))
        elif capture_state == 'tracking_late':
            vision.results.publish(PerceptionPacket(3, round((clock[0] - .09) * 1e9), ()))
        else:
            vision.results.clear()
        dialog.preview.hands = ('old overlay',)
        dialog.preview.quad = ((0, 0), (1, 0), (1, 1), (0, 1))
        dialog.refresh()
        assert exercise.negative_seconds == pytest.approx(0.05)
        assert dialog.preview.hands == () and dialog.preview.quad is None
        assert 'Movement check is paused' in dialog.detail.text()
        assert dialog.preview._image.pixelColor(100, 100).red() == 0
        clock[0] += 0.02
        camera.running = True
        publish(4)
        dialog.refresh()
        assert exercise.negative_seconds == pytest.approx(0.05)
        clock[0] += 0.05
        publish(5)
        dialog.refresh()
        assert exercise.negative_seconds == pytest.approx(0.10)
        assert 'Movement check is paused' not in dialog.detail.text()
        assert exercise.reports()[0].false_commands == 0
        assert host.model is None and host.output is None and not host.store.datasets.exists()
    finally:
        dialog.reject()
        host.close()


@pytest.mark.parametrize('capture_state', ['stale', 'future', 'missing', 'stopped'])
def test_collection_does_not_accept_tracking_without_live_capture(
    qt_app, tmp_path, monkeypatch, capture_state,
):
    camera, vision = FakeCamera(), FakeVision()
    host = CameraWindow(test_camera=camera, store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    host.vision = vision
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    clock = [100.0]
    monkeypatch.setattr('carl_cam.training_ui.monotonic', lambda: clock[0])
    dialog.consent.setChecked(True)
    dialog.handle_action()
    dialog.handle_action()
    assert dialog.flow.current is not None
    assert dialog.action.text() == 'Recording round…' and not dialog.action.isEnabled()
    calls = []
    observe = dialog.flow.observe
    monkeypatch.setattr(dialog.flow, 'observe', lambda *a, **kw: calls.append(kw) or observe(*a, **kw))
    camera.running = capture_state != 'stopped'
    pixels = np.zeros((360, 640, 3), dtype=np.uint8)
    if capture_state != 'missing':
        offset = -1 if capture_state == 'stale' else 1 if capture_state == 'future' else 0
        camera.frames.publish(FramePacket(1, round((clock[0] + offset) * 1e9), pixels))
    vision.results.publish(PerceptionPacket(1, round(clock[0] * 1e9), ()))
    try:
        dialog.refresh()
        assert calls == [] and dialog.flow.current.rows == []
        assert dialog.flow.current.usable_seconds == 0
        assert 'paused' in dialog.detail.text()
        camera.running = True
        clock[0] = 100.05
        camera.frames.publish(FramePacket(2, round(clock[0] * 1e9), pixels))
        vision.results.publish(PerceptionPacket(2, round(clock[0] * 1e9), ()))
        dialog.refresh()
        assert len(calls) == 1
        assert host.model is None and host.output is None
    finally:
        dialog.reject()
        host.close()


@pytest.mark.parametrize('handler,method,phase,active', [
    ('redo_round', 'interrupt_round', 'collection', True),
    ('skip_lighting', 'skip_lighting', 'collection', False),
    ('handle_depth_retry', 'retry_depth_rounds', 'review', False),
])
@pytest.mark.parametrize('error_type', [ValueError, OSError, RuntimeError, KeyError, TypeError])
def test_auxiliary_training_errors_stay_in_app_without_overlapping_action(
    qt_app, tmp_path, handler, method, phase, active, error_type,
):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()

    def fail():
        raise error_type('Fixture recovery control failed')

    current = object() if active else None
    dialog.flow = SimpleNamespace(phase=phase, current=current, stop=lambda **_: True,
                                  **{method: fail})
    # Existing presentation is retained on a failed active-round interruption.
    dialog._displayed_phase = phase
    dialog.action.setText('Recording round…' if active else 'Retry fixture step')
    dialog.action.setEnabled(not active)
    if phase == 'collection' and not active:
        dialog.flow.prompt = SimpleNamespace(label='frame', round_index=0, target_seconds=12,
                                            variations=('center', 'shift', 'angle'))
        dialog.flow.index, dialog.flow.plan = 0, [dialog.flow.prompt]
    try:
        getattr(dialog, handler)()
        assert 'Fixture recovery control failed' in dialog.detail.text()
        assert 'previous model is kept' in dialog.detail.text()
        assert dialog.flow.current is current
        assert dialog.action.isEnabled() is not active
        assert dialog.action.text() == ('Recording round…' if active else 'Start round'
                                        if phase == 'collection' else 'Retry fixture step')
        assert host.model is None and host.vision is None and host.output is None
        assert not host.camera.running and not host.store.datasets.exists()
        assert not host.store.models.exists()
    finally:
        dialog.reject()
        host.close()


def test_redo_manifest_failure_resyncs_round_controls_without_saving_examples(
    qt_app, tmp_path, monkeypatch,
):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    dialog.consent.setChecked(True)
    dialog.handle_action()
    dialog.handle_action()
    original_round = dialog.flow.current

    def fail():
        raise OSError('Fixture manifest save failure')

    monkeypatch.setattr(dialog.flow, '_save_manifest', fail)
    try:
        dialog.redo_round()
        assert dialog.flow.current is None and original_round.rows == []
        assert dialog.action.text() == 'Start round' and dialog.action.isEnabled()
        assert not dialog.redo.isVisible()
        assert 'Fixture manifest save failure' in dialog.detail.text()
        assert 'previous model is kept' in dialog.detail.text()
        assert dialog.flow.rows() == [] and host.model is None
        assert host.vision is None and host.output is None and not host.camera.running
    finally:
        monkeypatch.undo()
        dialog.reject()
        host.close()


@pytest.mark.parametrize('handler,method', [
    ('redo_round', 'interrupt_round'), ('skip_lighting', 'skip_lighting'),
    ('handle_depth_retry', 'retry_depth_rounds'),
])
def test_auxiliary_controls_cannot_change_workflow_during_pending_close(
    qt_app, tmp_path, handler, method,
):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    calls = []
    dialog.flow = SimpleNamespace(phase='collection', stop=lambda **_: True,
                                  **{method: lambda: calls.append(method)})
    dialog._pending_done = 0
    try:
        getattr(dialog, handler)()
        assert calls == [] and dialog._pending_done == 0
        assert host.model is None and host.vision is None and host.output is None
        assert not host.camera.running and not host.store.datasets.exists()
    finally:
        dialog.reject()
        host.close()


@pytest.mark.parametrize('size', [(1180, 780), (1024, 640)])
@pytest.mark.parametrize('phase', [
    'review', 'fitting', 'comparison', 'movement', 'sealed', 'needs_examples', 'ready',
])
def test_late_training_presentation_keeps_controls_reachable_without_activation(
    qt_app, tmp_path, size, phase,
):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    dialog.flow = SimpleNamespace(
        phase='comparison' if phase == 'movement' else phase, rows=list, stop=lambda **_: True,
        exercise=SimpleNamespace(attempts_per_pose=5),
        winner=SimpleNamespace(name='Fixture-only candidate'),
        action_reports=None, sealed_report=None, rehearsal_report=None,
    )
    for widget in (dialog.recording_choice, dialog.mapping_widget, dialog.consent, dialog.consent_info):
        widget.hide()
    try:
        if phase == 'movement':
            dialog._trial_index = 25
        dialog._show_phase(force=True)
        dialog.resize(*size)
        dialog.show()
        qt_app.processEvents()
        assert (dialog.width(), dialog.height()) == size
        assert dialog.back.isVisible() and dialog.back.height() >= 44
        assert dialog.rect().contains(dialog.back.geometry())
        for scroll in (dialog.left_scroll, dialog.right_scroll):
            assert scroll.horizontalScrollBar().maximum() == 0
        dialog.right_scroll.ensureWidgetVisible(dialog.action)
        qt_app.processEvents()
        position = dialog.action.mapTo(dialog.right_scroll.viewport(), QPoint())
        assert position.y() >= 0
        assert position.y() + dialog.action.height() <= dialog.right_scroll.viewport().height()
        if phase == 'fitting':
            assert (dialog.progress.minimum(), dialog.progress.maximum()) == (0, 0)
            assert not dialog.action.isEnabled()
            assert dialog.action.text() == 'Training in progress…'
            assert 'cancellation' in dialog.detail.text()
        else:
            assert dialog.action.isEnabled()
        if phase == 'review':
            assert dialog.depth.isVisible() and dialog.retry_depth.isVisible()
        if phase == 'ready':
            assert 'Activate' in dialog.action.text()
            assert dialog.progress.value() == 100
        elif phase == 'movement':
            assert dialog.progress.value() == 100
            assert dialog.action.text() == 'Start movement check'
            assert not dialog.drawing.isVisible()
            assert 'Camera gaps pause the timer' in dialog.instructions.text()
        assert host.model is None and host.output is None and host.vision is None
        assert not host.camera.running and not host.store.datasets.exists()
        assert not host.store.models.exists()
    finally:
        dialog.reject()
        host.close()


@pytest.mark.parametrize('state', [
    'between_attempts', 'fitting', 'collection', 'scoring', 'movement', 'cancel_pending',
])
def test_refresh_error_does_not_enable_an_overlapping_action(qt_app, tmp_path, monkeypatch, state):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    phase = state if state in ('fitting', 'collection') else 'comparison'
    dialog.flow = SimpleNamespace(phase=phase, current=object() if state == 'collection' else None,
                                  stop=lambda **_: True)
    dialog._displayed_phase = phase
    dialog._trial_until = 100 if state == 'scoring' else None
    dialog._negative_started = state == 'movement'

    def fail(*args, **kwargs):
        raise OSError('Fixture recoverable step failure')

    monkeypatch.setattr(dialog, 'refresh', fail)
    if state == 'cancel_pending':
        dialog._pending_done = 0
        dialog.flow.stop = fail
    try:
        dialog.safe_refresh()
        assert 'Fixture recoverable step failure' in dialog.detail.text()
        assert dialog.action.isEnabled() is (state == 'between_attempts')
        assert host.model is None and host.vision is None and host.output is None
        assert not host.camera.running and not host.store.models.exists()
    finally:
        dialog.flow.stop = lambda **_: True
        dialog.reject()
        host.close()


def test_fitting_failure_stays_visible_after_return_to_review(qt_app, tmp_path):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    dialog.consent.setChecked(True)
    dialog.handle_action()
    dialog.flow.phase = 'fitting'
    error = LatestValue()
    error.publish('Fixture training failure: retry on CPU. Your previous model is kept.')
    dialog.flow.worker = SimpleNamespace(result=LatestValue(), error=error,
                                         progress=LatestValue(), stop=lambda **_: True)
    try:
        dialog._show_phase(force=True)
        dialog.refresh()
        assert dialog.flow.phase == 'review'
        assert dialog.detail.text() == error.read()
        assert dialog.action.text() == 'Train and compare' and dialog.action.isEnabled()
        assert host.model is None and host.vision is None and not host.store.models.exists()
        assert not host.camera.running
    finally:
        dialog.reject()
        host.close()


@pytest.mark.parametrize('approve', [False, True])
def test_training_activation_requires_explicit_confirmation(qt_app, tmp_path, monkeypatch, approve):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    activations, confirmations = [], []

    def ask(parent, title, body, options, default):
        confirmations.append((options, default, body))
        return QMessageBox.StandardButton.Yes if approve else QMessageBox.StandardButton.No

    def activate(*, user_confirmed):
        activations.append(user_confirmed)
        return host.settings  # Fixture delegate; no model/storage mutation.

    dialog.flow = SimpleNamespace(phase='ready', activate=activate, stop=lambda **_: True)
    monkeypatch.setattr(QMessageBox, 'question', ask)
    try:
        dialog.show()
        qt_app.processEvents()
        dialog.handle_action()
        assert len(confirmations) == 1
        assert confirmations[0][1] == QMessageBox.StandardButton.No
        assert 'previous version is kept' in confirmations[0][2]
        assert activations == ([True] if approve else [])
        assert dialog.isVisible() is not approve
        assert host.model is None and not host.store.models.exists()
    finally:
        dialog.reject()
        host.close()


@pytest.mark.parametrize('entry', ['button', 'refresh'])
def test_final_check_failure_keeps_error_and_retry_visible(qt_app, tmp_path, monkeypatch, entry):
    from carl_cam.model_store import ModelStore

    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    dialog.consent.setChecked(True)
    dialog.handle_action()
    dialog.flow.phase = 'sealed'
    dialog.flow.version = 'fixedcandidate'
    dialog.flow.winner = SimpleNamespace(name='Fixture-only candidate',
                                        session_ids={'sealed': ('untouchedsession',)})

    def fail(*args):
        raise OSError('Fixture unreadable candidate. Your previous model is kept.')

    monkeypatch.setattr(ModelStore, 'load', fail)
    dialog._show_phase(force=True)
    try:
        if entry == 'button':
            dialog.handle_action()
        else:
            monkeypatch.setattr(dialog, 'refresh', dialog._handle_action)
            dialog.safe_refresh()
        assert dialog.flow.phase == 'needs_examples'
        assert 'Fixture unreadable candidate' in dialog.detail.text()
        assert dialog.action.text() == 'Collect extra examples' and dialog.action.isEnabled()
        assert dialog.heading.text() == 'More examples are needed'
        assert host.model is None and not host.store.models.exists()
        assert not host.camera.running and host.vision is None and host.output is None
    finally:
        dialog.reject()
        host.close()


@pytest.mark.parametrize('close_source', ['back', 'x', 'alt-f4', 'escape'])
def test_pending_training_close_finishes_once_without_blocking_or_restarting(
    qt_app, tmp_path, monkeypatch, close_source,
):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    host.show()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    dialog.consent.setChecked(True)
    dialog.handle_action()
    ready = [False]
    stop_timeouts = []

    def stop(*, timeout):
        stop_timeouts.append(timeout)
        return ready[0]

    dialog.flow.worker = SimpleNamespace(stop=stop)
    dialog.flow.phase = 'fitting'
    dialog._show_phase(force=True)
    dialog.show()
    qt_app.processEvents()
    try:
        if close_source == 'back':
            dialog.back.click()
        elif close_source == 'x':
            dialog.title_bar.buttons['close'].click()
        elif close_source == 'alt-f4':
            QTest.keyClick(dialog, Qt.Key.Key_F4, Qt.KeyboardModifier.AltModifier)
        else:
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
        qt_app.processEvents()
        assert dialog.isVisible() and host.isVisible()
        assert dialog._pending_done is not None
        assert not dialog.action.isEnabled() and not dialog.depth.isEnabled()
        assert 'Cancel requested' in dialog.detail.text()

        def no_work(*args, **kwargs):
            pytest.fail('A pending close must not refresh training or start another fit')

        monkeypatch.setattr(dialog, 'refresh', no_work)
        monkeypatch.setattr(dialog.flow, 'begin_fit', no_work)
        dialog.handle_action()
        dialog.safe_refresh()
        assert dialog.isVisible() and host.isVisible()
        ready[0] = True
        dialog.safe_refresh()
        qt_app.processEvents()
        assert not dialog.isVisible() and dialog._pending_done is None
        assert host.isVisible() is (close_source in ('back', 'escape'))
        assert stop_timeouts and set(stop_timeouts) == {0}
        assert not dialog.timer.isActive()
        assert host.model is None and host.vision is None and host.output is None
        assert not host.store.models.exists()
    finally:
        ready[0] = True
        dialog.reject()
        host.close()


@pytest.mark.parametrize('size', [(1180, 780), (1024, 640)])
@pytest.mark.parametrize('phase,count,recheck,title', [
    ('comparison', 5, False, 'Candidate comparison'),
    ('rehearsal', 10, False, 'Activation rehearsal'),
    ('rehearsal', 5, True, 'Camera recheck'),
])
def test_pose_marker_and_four_second_attempt_do_not_activate_commands(
    qt_app, tmp_path, monkeypatch, size, phase, count, recheck, title,
):
    host = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    host.timer.stop()
    dialog = TrainingDialog(host)
    dialog.timer.stop()
    dialog.recheck_version = 'fixtureversion' if recheck else None
    candidate = SimpleNamespace(width=112, threshold=0.8, margin=0.15)
    exercise = Exercise((candidate,), attempts_per_pose=count)
    dialog.flow = SimpleNamespace(phase=phase, exercise=exercise, stop=lambda **_: True)
    for widget in (dialog.recording_choice, dialog.mapping_widget, dialog.consent, dialog.consent_info):
        widget.hide()
    clock = [100.0]
    monkeypatch.setattr('carl_cam.training_ui.monotonic', lambda: clock[0])
    try:
        dialog._show_phase(force=True)
        dialog.resize(*size)
        dialog.show()
        qt_app.processEvents()
        assert dialog.heading.text().startswith(title) and f'1/{count}' in dialog.heading.text()
        assert dialog.drawing.isVisible() and dialog.action.text() == 'Prepare this attempt'
        dialog.handle_action()
        assert dialog._waiting_marker and dialog.action.shortcut().toString() == 'Space'
        assert 'Pose ready' in dialog.action.text() and exercise._trial is None
        dialog.handle_action()
        assert exercise._trial == ('frame', 100.0) and dialog._trial_until == 104.0
        assert not dialog.action.isEnabled() and not dialog.action.shortcut().toString()
        assert dialog.action.text() == 'Scoring attempt…'
        assert 'four-second' in dialog.instructions.text()
        clock[0] = 104.0
        dialog.safe_refresh()
        assert dialog._trial_until is None and dialog._trial_index == 1
        assert exercise._trial is None and exercise._attempts[0] == 1
        assert exercise.reports()[0].pose_successes == (0, 0, 0, 0, 0)
        assert dialog.action.isEnabled() and dialog.action.text() == 'Prepare this attempt'
        assert dialog.progress.value() == round(100 / (count * 5))
        assert dialog.back.isVisible() and dialog.rect().contains(dialog.back.geometry())
        assert host.model is None and host.vision is None and host.output is None
        assert not host.camera.running and not host.store.datasets.exists()
        assert not host.store.models.exists()
    finally:
        dialog.reject()
        host.close()
