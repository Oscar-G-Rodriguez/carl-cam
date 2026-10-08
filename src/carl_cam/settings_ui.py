"""Accessible local settings, scoped data controls and creator export."""

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from carl_cam.accessibility import desktop_style
from carl_cam.desktop_tabs import SettingsTabBar
from carl_cam.model_store import ModelStore
from carl_cam.storage import valid_assignments
from carl_cam.window_chrome import install_dialog_chrome


class SettingsDialog(QDialog):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.setWindowTitle("Carl-Cam — Settings")
        self.resize(1024, 700)
        self.setMinimumSize(1024, 640)
        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("Settings groups")
        self.tabs.setTabBar(SettingsTabBar(high_contrast=host.settings.high_contrast))
        self.pages = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self.tabs)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.hide()
        layout.addWidget(self.message)
        self.save_button = QPushButton("Save settings")
        self.save_button.clicked.connect(self.save)
        self.back = QPushButton("Back to camera")
        self.back.clicked.connect(self.reject)
        self.back.setFixedWidth(240)
        actions = QHBoxLayout()
        actions.setSpacing(16)
        actions.addWidget(self.save_button, 1)
        actions.addWidget(self.back)
        layout.addLayout(actions)
        camera = self.page("Camera")
        camera.addWidget(QLabel("Stop the camera, choose an input on the Camera screen, then start it. OBS Virtual Camera is excluded to prevent a feedback loop."))
        camera.addWidget(QLabel("Frames stay live on this computer. Carl-Cam does not save or upload photos or video."))
        quality = QLabel("Sustained slow processing can automatically request a smaller camera input. The Camera screen shows actual dimensions and offers Try higher input quality. Training and other private screens do not trigger automatic format changes; the final canvas remains 1280×720.")
        quality.setWordWrap(True)
        camera.addWidget(quality)
        self.region_backend = QComboBox()
        self.region_backend.setAccessibleName('Segmentation processor')
        self.region_backend.addItem('CPU — available on every supported computer', 'cpu')
        self.region_backend.addItem('NVIDIA GPU — experimental, with CPU fallback', 'gpu')
        self.region_backend.setCurrentIndex(self.region_backend.findData(host.settings.region_backend))
        camera.addWidget(QLabel('Segmentation processor'))
        camera.addWidget(self.region_backend)
        backend_help = QLabel('This changes hair, skin and clothing masking only. Hand tracking and your gesture model stay separate. Save, then stop and restart the camera to apply. If the verified GPU component is unavailable or fails, Carl-Cam shows the CPU fallback. The current GPU component is available in the source development build; installer integration is still being verified.')
        backend_help.setWordWrap(True)
        camera.addWidget(backend_help)
        camera.addStretch()
        interaction = self.page("Interaction and accessibility")
        self.reduced = QCheckBox("Reduce motion — keep decorative layers still")
        self.reduced.setChecked(host.settings.reduced_motion)
        self.contrast = QCheckBox("High-contrast desktop controls")
        self.contrast.setChecked(host.settings.high_contrast)
        interaction.addWidget(self.reduced)
        interaction.addWidget(self.contrast)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.hold = QComboBox()
        self.hold.setAccessibleName("Gesture hold duration")
        for ms in (300, 400, 600):
            self.hold.addItem(f"{ms} ms gesture hold", ms)
        self.hold.setCurrentIndex(self.hold.findData(host.settings.hold_ms))
        form.addRow("Gesture dwell", self.hold)
        self.mapping = {}
        defaults = host.settings.assignments or {"thumbs_up": "red", "open_palm": "yellow", "ily": "blue"}
        for pose, label in (("thumbs_up", "Thumbs up"), ("open_palm", "Open palm"), ("ily", "I love you")):
            combo = QComboBox()
            for theme in ("red", "yellow", "blue"):
                combo.addItem(theme.title(), theme)
            combo.setCurrentIndex(combo.findData(defaults[pose]))
            combo.setAccessibleName(f"{label} filter assignment")
            form.addRow(label, combo)
            self.mapping[pose] = combo
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.ItemRole.LabelRole).widget()
            label.setMinimumHeight(46)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        interaction.addLayout(form)
        interaction.addWidget(QLabel("Peace opens the menu. The two-hand frame shows the effect. Assignment changes apply to future commands without relabeling or retraining."))
        interaction.addStretch()
        data = self.page("Personal data and models")
        self.space = QLabel()
        data.addWidget(self.space)
        self.model_info = QLabel(host.model_message + (f"\nActive version: {host.settings.active_model_version}" if host.settings.active_model_version else ""))
        self.model_info.setWordWrap(True)
        data.addWidget(self.model_info)
        self.models = QComboBox()
        self.models.setAccessibleName("Available tested personal models")
        self.models.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.models.setMinimumContentsLength(24)
        self.refresh_models()
        data.addWidget(self.models)
        choose = QPushButton("Use selected control model")
        choose.clicked.connect(self.choose_model)
        data.addWidget(choose)
        retrain = QPushButton("Collect and train a new personal model")
        retrain.clicked.connect(self.retrain)
        data.addWidget(retrain)
        creator = QPushButton("Prepare selected creator-demo candidate…")
        creator.clicked.connect(self.prepare_creator)
        data.addWidget(creator)
        for text, scope in (("Delete numerical training samples…", "samples"), ("Reset personal gesture models…", "models"), ("Delete my training data…", "both")):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, selected=scope: self.delete_data(selected))
            data.addWidget(button)
        reset = QPushButton("Reset all Carl-Cam local data…")
        reset.clicked.connect(self.reset_all)
        data.addWidget(reset)
        data.addStretch()
        output = self.page("Virtual camera")
        explanation = QLabel("OBS Virtual Camera is optional. Install OBS yourself if needed; Carl-Cam does not install drivers or start OBS. Start your physical camera, then explicitly press Start virtual camera on the Camera screen. In the receiving app choose OBS Virtual Camera. Do not start OBS's own virtual-camera sender at the same time. Output always starts off on each launch, and becomes black if the source is too old.")
        explanation.setWordWrap(True)
        output.addWidget(explanation)
        output.addStretch()
        self.update_space()
        for label in self.findChildren(QLabel):
            label.setWordWrap(True)
        install_dialog_chrome(self, layout, host)

    def page(self, name):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setAccessibleName(name + " settings")
        scroll.setWidget(page)
        self.tabs.addTab(scroll, name)
        self.pages.append(scroll)
        return layout

    def set_message(self, text):
        self.message.setText(text)
        self.message.setVisible(bool(text))

    def refresh_models(self):
        """Keep choices consistent with explicit selection and scoped deletion."""
        self.models.clear()
        self.models.addItem("Mouse/keyboard — learned commands off", None)
        for version, label in ModelStore(self.host.store).tested_versions():
            self.models.addItem(label, version)
        self.models.setCurrentIndex(max(0, self.models.findData(self.host.settings.active_model_version)))

    def update_space(self):
        self.space.setText(f"Carl-Cam local storage: {self.host.store.occupied_bytes():,} bytes.\nOnly numerical recordings, model artifacts, settings and local diagnostics are retained.")

    def retrain(self):
        self.reject()
        QTimer.singleShot(0, lambda: self.host.open_training(force_collection=True))

    def save(self):
        assignments = {pose: combo.currentData() for pose, combo in self.mapping.items()}
        if not valid_assignments(assignments):
            self.set_message("Assign Red, Yellow and Blue exactly once.")
            return
        updated = replace(self.host.settings, assignments=assignments, hold_ms=self.hold.currentData(),
                          reduced_motion=self.reduced.isChecked(), high_contrast=self.contrast.isChecked(),
                          region_backend=self.region_backend.currentData())
        try:
            self.host.store.save_settings(updated)
        except (OSError, ValueError):
            self.set_message("Settings could not be saved. Existing settings and data were kept.")
            return
        self.host.settings = updated
        self.host.interaction.assignments = dict(assignments)
        self.host.interaction.hold_ms = updated.hold_ms
        self.host.setStyleSheet(desktop_style(updated.high_contrast))
        self.tabs.tabBar().high_contrast = updated.high_contrast
        self.tabs.tabBar().update()
        self.host._publish_scene()
        self.set_message("Settings saved. The current filter stays selected. Stop and restart the camera to apply a changed segmentation processor.")

    def choose_model(self):
        version = self.models.currentData()
        try:
            if version is not None:
                ModelStore(self.host.store).load_tested(version)
            updated = replace(self.host.settings, active_model_kind="personal" if version else "none", active_model_version=version)
            self.host.store.save_settings(updated)
            self.host.settings = updated
            self.host._load_active_model()
            self.model_info.setText(self.host.model_message + (f"\nActive version: {version}" if version else ""))
            self.set_message(self.host.model_message)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            self.refresh_models()
            self.set_message("That model could not be verified. Your previous selection was kept; retrain or choose mouse/keyboard.")

    def delete_data(self, scope):
        consequence = {"samples": "Delete numerical training samples only? Your tested models stay usable; retraining will need new recordings.",
                       "models": "Delete all personal model versions? Personal learned commands stop; numerical samples are kept.",
                       "both": "Delete numerical samples and all personal model versions? Personal learned commands stop until you train again."}[scope]
        answer = QMessageBox.question(self, "Delete named Carl-Cam data?", consequence + " No unrelated files or bundled demo are removed. This deletion cannot be undone here.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            if scope in ("models", "both"):
                updated = replace(self.host.settings, active_model_kind="none", active_model_version=None)
                self.host.store.save_settings(updated)
                self.host.settings = updated
                self.host._load_active_model()
                self.host.store.delete_personal_models()
            if scope in ("samples", "both"):
                self.host.store.delete_samples()
            self.set_message("The named data was deleted. Unrelated files were kept. Deleted data cannot be recovered through Carl-Cam.")
            self.update_space()
        except (OSError, ValueError):
            self.set_message("Deletion did not fully finish. Some data may remain; your files outside Carl-Cam were not targeted.")
        finally:
            self.refresh_models()
            version = self.host.settings.active_model_version
            self.model_info.setText(self.host.model_message + (f"\nActive version: {version}" if version else ""))

    def prepare_creator(self):
        version = self.models.currentData()
        if version is None:
            self.set_message("Select a tested personal model candidate first.")
            return
        answer = QMessageBox.question(self, "Prepare creator candidate?", "Export this selected model's weights, normalizer and integrity metadata without recordings, photos or video? This is a candidate only; unseen-person tests and owner release approval are still required.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save creator candidate without recordings", f"carl-cam-creator-{version[:8]}.zip", "ZIP archive (*.zip)")
        if not filename:
            return
        try:
            from carl_cam.creator import export_candidate
            export_candidate(self.host.store, version, Path(filename), user_confirmed=True)
            self.set_message("Creator candidate saved. It is not bundled or approved for public release.")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            self.set_message("The candidate could not be exported. Choose a new filename and verify the selected model; existing files were kept.")

    def reset_all(self):
        answer = QMessageBox.question(self, "Reset all named Carl-Cam data?",
            "Stop the camera and optional output, then delete numerical samples, all personal models, settings/backups and local diagnostic logs? Camera consent and setup choices reset. Learned commands stay off. Unrelated files and bundled assets stay untouched. This cannot be undone here.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self.host.stop_camera():
            self.set_message("A worker is still closing. Nothing was deleted; retry after it stops.")
            return
        try:
            self.host.store.reset_all()
            self.host.settings, self.host.settings_warnings = self.host.store.load_settings()
            self.host._current_camera_key = None
            self.host._load_active_model()
            self.host._diagnostic_states.clear()
            self.host.interaction.assignments = {}
            self.host.interaction.hold_ms = self.host.settings.hold_ms
            self.host.theme = self.host.settings.selected_theme
            self.host.interaction.theme = self.host.theme
            self.host.setStyleSheet(desktop_style(self.host.settings.high_contrast))
            self.host._publish_scene()
            self.reject()
            self.host.status.setText("Named Carl-Cam data reset. Camera and output are off; privacy/setup choices reset. Unrelated files were kept. Deleted data cannot be recovered through Carl-Cam.")
        except (OSError, ValueError):
            self.set_message("Reset did not fully finish. Some named app data may remain; unrelated files were not targeted. Camera and output are off.")
