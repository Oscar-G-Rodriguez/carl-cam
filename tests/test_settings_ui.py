import json
from zipfile import ZipFile

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFormLayout, QMessageBox
from test_model_store import candidate, passed_gate
from test_ui import FakeCamera

from carl_cam.creator import export_candidate
from carl_cam.desktop_tabs import TAB_HEIGHT, SettingsTabBar
from carl_cam.model_store import ModelStore, fingerprint_dataset
from carl_cam.settings_ui import SettingsDialog
from carl_cam.storage import Settings, Store
from carl_cam.ui import CameraWindow


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def make_tested_model(store):
    models = ModelStore(store)
    version = models.stage(candidate(), dataset_fingerprint=fingerprint_dataset([]), report={})
    models.activate(version, Settings(), passed_gate(), user_confirmed=True)
    return version


def test_settings_changes_assignment_without_changing_current_filter(qt_app, tmp_path):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    window.theme = "blue"
    dialog = SettingsDialog(window)
    dialog.mapping["thumbs_up"].setCurrentIndex(1)
    dialog.save()
    assert "exactly once" in dialog.message.text()
    assert window.settings.assignments == {}
    dialog.mapping["open_palm"].setCurrentIndex(0)
    dialog.reduced.setChecked(True)
    dialog.contrast.setChecked(True)
    dialog.hold.setCurrentIndex(2)
    dialog.save()
    assert window.settings.reduced_motion and window.settings.high_contrast
    assert window.interaction.hold_ms == 600 and window.theme == "blue"
    assert window.store.load_settings()[0].assignments["thumbs_up"] == "yellow"
    dialog.reject()
    window.close()


def test_gpu_choice_is_saved_without_starting_camera_or_switching_active_backend(qt_app, tmp_path):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    dialog = SettingsDialog(window)
    dialog.region_backend.setCurrentIndex(dialog.region_backend.findData('gpu'))
    dialog.save()
    assert window.store.load_settings()[0].region_backend == 'gpu'
    assert not window.camera.running and window.regions is None
    assert 'restart the camera' in dialog.message.text()
    dialog.reject()
    window.close()


def test_sample_deletion_preserves_models_and_unrelated_files(qt_app, tmp_path, monkeypatch):
    store = Store(tmp_path / "CarlCam")
    version = make_tested_model(store)
    store.datasets.mkdir()
    (store.datasets / "recording.json").write_text("{}")
    unrelated = store.root / "keep-me.txt"
    unrelated.write_text("kept")
    window = CameraWindow(test_camera=FakeCamera(), store=store)
    dialog = SettingsDialog(window)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    dialog.delete_data("samples")
    assert store.datasets.exists()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    dialog.delete_data("samples")
    assert not store.datasets.exists() and store.models.joinpath(version).exists()
    assert window.settings.active_model_version == version and unrelated.read_text() == "kept"
    dialog.reject()
    window.close()


def test_creator_export_excludes_samples_and_cannot_claim_release(tmp_path):
    store = Store(tmp_path / "CarlCam")
    version = make_tested_model(store)
    target = tmp_path / "candidate.zip"
    with pytest.raises(ValueError, match="Confirm"):
        export_candidate(store, version, target, user_confirmed=False)
    assert not target.exists()
    export_candidate(store, version, target, user_confirmed=True)
    with ZipFile(target) as archive:
        assert set(archive.namelist()) == {"weights.pt", "scaler.json", "manifest.json", "candidate.json", "README.txt"}
        report = json.loads(archive.read("candidate.json"))
        assert report["candidate_only"] and not report["public_release_approved"]
        assert not report["contains_training_recordings"] and archive.testzip() is None
    original = target.read_bytes()
    with pytest.raises(ValueError, match="overwritten"):
        export_candidate(store, version, target, user_confirmed=True)
    assert target.read_bytes() == original


def test_staged_model_is_not_offered_as_a_tested_control_choice(tmp_path):
    store = Store(tmp_path / "CarlCam")
    models = ModelStore(store)
    version = models.stage(candidate(), dataset_fingerprint=fingerprint_dataset([]), report={})
    assert models.tested_versions() == ()
    with pytest.raises(ValueError, match="activation"):
        models.load_tested(version)
    models.activate(version, Settings(), passed_gate(), user_confirmed=True)
    assert models.tested_versions()[0][0] == version


def test_creator_export_failure_leaves_no_partial_chosen_archive(tmp_path, monkeypatch):
    store = Store(tmp_path / "CarlCam")
    version = make_tested_model(store)
    target = tmp_path / "candidate.zip"
    def fail(*args):
        raise OSError("simulated disk error")
    monkeypatch.setattr(ZipFile, "write", fail)
    with pytest.raises(OSError, match="disk error"):
        export_candidate(store, version, target, user_confirmed=True)
    assert not target.exists() and not list(tmp_path.glob(".candidate.zip.*.tmp"))


@pytest.mark.parametrize('size', [(1024, 700), (1024, 640)])
@pytest.mark.parametrize('group', range(4))
def test_settings_groups_have_persistent_actions_and_vertical_only_content(qt_app, tmp_path, size, group):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    window.timer.stop()
    dialog = SettingsDialog(window)
    try:
        dialog.resize(*size)
        dialog.tabs.setCurrentIndex(group)
        dialog.show()
        qt_app.processEvents()
        assert (dialog.width(), dialog.height()) == size
        assert not dialog.message.isVisible()
        page = dialog.pages[group]
        assert page.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert page.horizontalScrollBar().maximum() == 0
        for button in (dialog.save_button, dialog.back):
            origin = button.mapTo(dialog, QPoint())
            assert button.isVisible() and button.height() >= 44
            assert origin.y() + button.height() <= dialog.height() - 16
        for index in range(4):
            assert dialog.tabs.tabBar().tabRect(index).height() >= 44
        assert dialog.hold.accessibleName() and dialog.tabs.accessibleName()
        if group == 1:
            form = dialog.hold.parent().layout().itemAt(2).layout()
            assert isinstance(form, QFormLayout)
            for row in range(form.rowCount()):
                label = form.itemAt(row, QFormLayout.ItemRole.LabelRole).widget()
                field = form.itemAt(row, QFormLayout.ItemRole.FieldRole).widget()
                assert label.height() == field.height() == 46
                assert label.geometry().center().y() == field.geometry().center().y()
        assert not window.camera.running and window.model is None
        assert window.vision is None and window.output is None
    finally:
        dialog.reject()
        window.close()


def test_settings_long_recovery_and_model_description_do_not_hide_return(qt_app, tmp_path):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    window.timer.stop()
    dialog = SettingsDialog(window)
    try:
        dialog.resize(1024, 640)
        dialog.tabs.setCurrentIndex(2)
        dialog.models.addItem('Personal model · ' + 'recorded session description ' * 60, 'fixture')
        dialog.models.setCurrentIndex(dialog.models.count() - 1)
        dialog.set_message('Deletion did not fully finish. Some named app data may remain; unrelated files were not targeted. Camera and output are off.')
        dialog.show()
        qt_app.processEvents()
        assert dialog.width() == 1024 and dialog.message.isVisible()
        assert dialog.pages[2].horizontalScrollBar().maximum() == 0
        assert dialog.back.mapTo(dialog, QPoint()).y() + dialog.back.height() <= 624
        dialog.set_message('')
        qt_app.processEvents()
        assert not dialog.message.isVisible()
    finally:
        dialog.reject()
        window.close()


@pytest.mark.parametrize('scope', ['models', 'both'])
def test_model_deletion_refreshes_visible_choices_and_preserves_unrelated_files(qt_app, tmp_path, monkeypatch, scope):
    store = Store(tmp_path / 'CarlCam')
    version = make_tested_model(store)
    unrelated = store.root / 'keep-me.txt'
    unrelated.write_text('kept')
    window = CameraWindow(test_camera=FakeCamera(), store=store)
    window.timer.stop()
    dialog = SettingsDialog(window)
    try:
        assert dialog.models.findData(version) >= 0
        monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
        dialog.delete_data(scope)
        assert dialog.models.count() == 1 and dialog.models.currentData() is None
        assert version not in dialog.model_info.text()
        assert window.settings.active_model_kind == 'none' and window.model is None
        assert not store.models.exists() and unrelated.read_text() == 'kept'
    finally:
        dialog.reject()
        window.close()


def test_settings_save_failure_keeps_existing_settings_and_explains_recovery(qt_app, tmp_path, monkeypatch):
    store = Store(tmp_path / 'CarlCam')
    window = CameraWindow(test_camera=FakeCamera(), store=store)
    window.timer.stop()
    dialog = SettingsDialog(window)
    try:
        previous = window.settings
        dialog.reduced.setChecked(not previous.reduced_motion)
        def fail(updated):
            raise OSError('fixture disk failure')
        monkeypatch.setattr(store, 'save_settings', fail)
        dialog.show()
        dialog.save()
        assert window.settings is previous
        assert window.interaction.assignments == previous.assignments
        assert dialog.message.isVisible() and 'Existing settings and data were kept' in dialog.message.text()
    finally:
        dialog.reject()
        window.close()


def test_settings_tab_navigation_preserves_unsaved_choices(qt_app, tmp_path):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    window.timer.stop()
    dialog = SettingsDialog(window)
    try:
        dialog.show()
        qt_app.processEvents()
        dialog.tabs.setCurrentIndex(1)
        dialog.reduced.setChecked(not window.settings.reduced_motion)
        value = dialog.reduced.isChecked()
        dialog.tabs.tabBar().setFocus()
        QTest.keyClick(dialog.tabs.tabBar(), Qt.Key.Key_Right)
        assert dialog.tabs.currentIndex() == 2
        QTest.keyClick(dialog.tabs.tabBar(), Qt.Key.Key_Left)
        assert dialog.tabs.currentIndex() == 1 and dialog.reduced.isChecked() == value
        assert window.settings.reduced_motion != value
    finally:
        dialog.reject()
        window.close()


@pytest.mark.parametrize('high_contrast', [False, True])
def test_original_tabs_render_selection_and_distinct_native_focus(qt_app, tmp_path, high_contrast):
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / 'CarlCam'))
    window.timer.stop()
    window.settings = Settings(high_contrast=high_contrast)
    dialog = SettingsDialog(window)
    try:
        dialog.show()
        dialog.tabs.setCurrentIndex(1)
        dialog.save_button.setFocus()
        qt_app.processEvents()
        bar = dialog.tabs.tabBar()
        assert isinstance(bar, SettingsTabBar) and bar.accessibleName() == 'Settings groups'
        selected = bar.tabRect(1)
        other = bar.tabRect(0)
        assert selected.height() == TAB_HEIGHT
        assert selected.width() == bar.fontMetrics().horizontalAdvance(bar.tabText(1)) + 36
        before = bar.grab().toImage()
        assert before.pixelColor(selected.center().x(), selected.bottom() - 1).name() == '#4ac9e8'
        expected = '#000000' if high_contrast else '#151823'
        assert before.pixelColor(selected.left() + 12, 12).name() == expected
        expected = '#000000' if high_contrast else '#10121b'
        assert before.pixelColor(other.left() + 12, 12).name() == expected
        assert before.pixelColor(selected.center().x(), 5).name() != '#f5c915'
        bar.setFocus(Qt.FocusReason.TabFocusReason)
        qt_app.processEvents()
        focused = bar.grab().toImage()
        assert bar.hasFocus()
        assert focused.pixelColor(selected.center().x(), 5).name() == '#f5c915'
        assert focused.pixelColor(other.center().x(), 5).name() != '#f5c915'
        QTest.keyClick(bar, Qt.Key.Key_Right)
        qt_app.processEvents()
        assert dialog.tabs.currentIndex() == 2
        moved = bar.grab().toImage()
        current = bar.tabRect(2)
        assert moved.pixelColor(current.center().x(), 5).name() == '#f5c915'
        assert moved.pixelColor(selected.center().x(), 5).name() != '#f5c915'
        dialog.save_button.setFocus()
        qt_app.processEvents()
        assert bar.grab().toImage().pixelColor(current.center().x(), 5).name() != '#f5c915'
        assert not window.camera.running and window.vision is None and window.output is None
    finally:
        dialog.reject()
        window.close()


def test_rejected_model_choice_restores_selector_to_actual_active_model(qt_app, tmp_path, monkeypatch):
    store = Store(tmp_path / 'CarlCam')
    version = make_tested_model(store)
    window = CameraWindow(test_camera=FakeCamera(), store=store)
    window.timer.stop()
    dialog = SettingsDialog(window)
    try:
        previous = window.settings
        previous_model = window.model
        dialog.models.addItem('Unavailable fixture candidate', 'missing-version')
        dialog.models.setCurrentIndex(dialog.models.count() - 1)
        def fail(*args):
            raise ValueError('fixture verification failed')
        monkeypatch.setattr(ModelStore, 'load_tested', fail)
        dialog.choose_model()
        assert window.settings is previous and window.model is previous_model
        assert dialog.models.currentData() == version
        assert 'previous selection was kept' in dialog.message.text()
    finally:
        dialog.reject()
        window.close()


def test_reset_refuses_deletion_when_worker_has_not_stopped(qt_app, tmp_path, monkeypatch):
    store = Store(tmp_path / 'CarlCam')
    version = make_tested_model(store)
    window = CameraWindow(test_camera=FakeCamera(), store=store)
    window.timer.stop()
    dialog = SettingsDialog(window)
    try:
        previous = window.settings
        monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
        monkeypatch.setattr(window, 'stop_camera', lambda: False)
        dialog.reset_all()
        assert window.settings is previous and store.models.joinpath(version).exists()
        assert dialog.models.currentData() == version
        assert 'Nothing was deleted' in dialog.message.text()
    finally:
        dialog.reject()
        monkeypatch.undo()
        window.close()
