from dataclasses import replace

import pytest
from test_model_store import candidate, passed_gate

from carl_cam.camera_recheck import CameraRecheckFlow
from carl_cam.evaluation import ExerciseReport
from carl_cam.model_store import ModelStore, fingerprint_dataset
from carl_cam.storage import Settings, Store

FIRST, SECOND = "a" * 64, "b" * 64


def make_tested_model(store):
    models = ModelStore(store)
    version = models.stage(candidate(), dataset_fingerprint=fingerprint_dataset([]), report={})
    settings = models.activate(version, Settings(camera_key=FIRST), passed_gate(), user_confirmed=True)
    return models, version, settings


def test_camera_change_preserves_model_but_needs_its_own_confirmation(tmp_path):
    store = Store(tmp_path / 'CarlCam')
    models, version, settings = make_tested_model(store)
    assert models.camera_verified(version, FIRST)
    assert not models.camera_verified(version, SECOND)
    before = (store.models / version / 'weights.pt').read_bytes()
    flow = CameraRecheckFlow(store, replace(settings, camera_key=SECOND), version)
    flow.exercise.negative_seconds = 120
    flow.exercise.reports = lambda: (ExerciseReport((5,) * 5, 5, 0, 120, (300,) * 25, 0, 0),)
    assert flow.finish_rehearsal()
    with pytest.raises(ValueError, match='confirmation'):
        flow.activate(user_confirmed=False)
    assert not models.camera_verified(version, SECOND)
    updated = flow.activate(user_confirmed=True)
    assert updated.active_model_version == version and models.camera_verified(version, SECOND)
    assert models.camera_verified(version, FIRST)
    assert (store.models / version / 'weights.pt').read_bytes() == before
    assert not store.datasets.exists()


def test_failed_recheck_keeps_commands_disabled_without_training_or_model_changes(tmp_path):
    store = Store(tmp_path / 'CarlCam')
    models, version, settings = make_tested_model(store)
    flow = CameraRecheckFlow(store, replace(settings, camera_key=SECOND), version)
    flow.exercise.negative_seconds = 120
    flow.exercise.reports = lambda: (ExerciseReport((5, 5, 3, 5, 5), 5, 1, 120, (300,), 0, 0),)
    assert not flow.finish_rehearsal() and flow.phase == 'needs_examples'
    with pytest.raises(ValueError, match='Finish'):
        flow.activate(user_confirmed=True)
    assert not models.camera_verified(version, SECOND)
    flow.retry_weak_cases()
    assert flow.phase == 'rehearsal' and flow.exercise.negative_seconds == 0
    assert flow.stop() and models.load_tested(version)
    assert not store.datasets.exists()


def test_camera_recheck_reuses_private_training_screen_with_explicit_consent(tmp_path):
    from PySide6.QtWidgets import QApplication
    from test_ui import FakeCamera

    from carl_cam.training_ui import TrainingDialog
    from carl_cam.ui import CameraWindow

    app = QApplication.instance() or QApplication([])
    store = Store(tmp_path / 'CarlCam')
    _, version, settings = make_tested_model(store)
    window = CameraWindow(test_camera=FakeCamera(), store=store)
    window._current_camera_key = SECOND
    window.settings = replace(settings, camera_key=SECOND)
    window._load_active_model()
    assert window.model is None and window._camera_recheck_needed
    assert window.training_button.text() == 'Recheck saved gestures'
    dialog = TrainingDialog(window, recheck_version=version)
    dialog.handle_action()
    assert dialog.flow is None
    dialog.consent.setChecked(True)
    dialog.handle_action()
    assert dialog.flow.phase == 'rehearsal' and dialog.flow.exercise.attempts_per_pose == 5
    assert 'Camera recheck' in dialog.heading.text()
    assert not store.datasets.exists()
    dialog.reject()
    window.close()
    app.processEvents()
