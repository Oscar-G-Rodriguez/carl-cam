"""Four-step, camera-off-by-default onboarding; no developer commands needed."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from carl_cam.ui import Preview
from carl_cam.window_chrome import install_dialog_chrome


class SetupDialog(QDialog):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.setWindowTitle("Carl-Cam — Setup")
        self.resize(1024, 700)
        self.setMinimumSize(1024, 640)
        self.gesture_chosen = False
        self.practice_requested = False
        self.stack = QStackedWidget()
        self.heading = QLabel("Step 1 of 4 — Your camera stays on this computer")
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(self.heading)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.message)
        buttons = QHBoxLayout()
        self.previous = QPushButton("Previous step")
        self.previous.clicked.connect(self.back)
        self.next = QPushButton("Continue")
        self.next.clicked.connect(self.advance)
        leave = QPushButton("Back to camera")
        leave.clicked.connect(self.reject)
        for button in (self.previous, self.next, leave):
            buttons.addWidget(button, 1)
        layout.addLayout(buttons)
        privacy = self.page()
        text = QLabel("Carl-Cam processes camera images locally and does not save photos or video. MediaPipe may send performance and usage metrics to Google; its camera images and video stay on this computer. Models are bundled. Personal training asks separately before saving numerical hand measurements and labels. Settings can delete samples and personal models. Virtual-camera output starts only when you explicitly request it in this launch.")
        text.setWordWrap(True)
        privacy.addWidget(text)
        self.consent = QCheckBox("I understand local processing and MediaPipe usage metrics")
        self.consent.setChecked(host.settings.privacy_accepted)
        privacy.addWidget(self.consent)
        from carl_cam.vision_privacy import (
            PRIVACY_HOLD,
            VisionPrivacyError,
            require_verified_vision_runtime,
        )
        self.vision_notice = QLabel()
        try:
            require_verified_vision_runtime()
        except VisionPrivacyError:
            self.vision_notice.setText(PRIVACY_HOLD)
        self.vision_notice.setVisible(bool(self.vision_notice.text()))
        privacy.addWidget(self.vision_notice)
        privacy.addStretch()
        camera = self.page()
        camera.addWidget(QLabel("Choose a physical camera. Stop other camera applications if it is busy, then test it here. OBS Virtual Camera is excluded from inputs."))
        self.devices = QComboBox()
        self.devices.setAccessibleName("Setup physical camera")
        self.devices.setMinimumWidth(240)
        self.devices.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.devices.setMinimumContentsLength(16)
        self.devices.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.refresh_devices()
        camera_controls = QHBoxLayout()
        camera_controls.setSpacing(8)
        camera_controls.addWidget(self.devices)
        self.devices.currentIndexChanged.connect(self.select_device)
        refresh = QPushButton("Refresh cameras")
        refresh.clicked.connect(self.refresh_devices)
        camera_controls.addWidget(refresh)
        self.preview = Preview()
        self.preview.setMinimumSize(480, 270)
        camera.addWidget(self.preview, 1)
        test = QPushButton("Test selected camera")
        test.clicked.connect(host.start_camera)
        camera_controls.addWidget(test)
        stop = QPushButton("Stop camera to change device")
        stop.clicked.connect(self.stop_camera)
        camera_controls.addWidget(stop)
        camera_controls.addStretch()
        camera.insertLayout(1, camera_controls)
        self.readiness = QLabel("Camera has not been started")
        self.readiness.setWordWrap(True)
        camera.addWidget(self.readiness)
        gestures = self.page()
        instructions = QLabel("You can try all three filters immediately with mouse and keyboard. Personal training introduces the five supported poses, records prompted variations, fits local neural candidates, checks them, and asks before activation. There is no bundled creator demo in this development build.")
        instructions.setWordWrap(True)
        gestures.addWidget(instructions)
        personal = QPushButton("Train my gestures")
        personal.clicked.connect(self.train)
        mouse = QPushButton("Use mouse/keyboard for now")
        mouse.clicked.connect(self.use_mouse)
        gestures.addWidget(personal)
        gestures.addWidget(mouse)
        if host.model is not None:
            existing = QPushButton("Keep my current tested personal model")
            existing.clicked.connect(self.keep_model)
            gestures.addWidget(existing)
        gestures.addStretch()
        output = self.page()
        explanation = QLabel("Optional output lets another desktop app receive Carl-Cam through OBS Virtual Camera. Install OBS yourself if needed, then return to the Camera screen and press Start virtual camera. Select OBS Virtual Camera in the receiving app. Carl-Cam never installs OBS or starts output automatically. Skip this step if you only want the desktop preview.")
        explanation.setWordWrap(True)
        output.addWidget(explanation)
        self.practice_choice = QCheckBox("Practice both filter-switching paths after setup (optional)")
        self.practice_choice.setChecked(True)
        output.addWidget(self.practice_choice)
        output.addStretch()
        self.previous.setEnabled(False)
        for label in self.findChildren(QLabel):
            label.setWordWrap(True)
        install_dialog_chrome(self, layout, host)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(30)

    def page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self.stack.addWidget(page)
        return layout

    def refresh_devices(self):
        self.host.refresh_devices()
        self.devices.blockSignals(True)
        self.devices.clear()
        for index in range(self.host.devices.count()):
            self.devices.addItem(self.host.devices.itemText(index), self.host.devices.itemData(index))
        self.devices.setCurrentIndex(self.host.devices.currentIndex())
        self.devices.blockSignals(False)

    def select_device(self):
        index = self.host.devices.findData(self.devices.currentData())
        if index >= 0:
            self.host.devices.setCurrentIndex(index)

    def stop_camera(self):
        if self.host.stop_camera():
            self.host.mode = "setup"
            self.host.interaction.enter_mode("setup")

    def save_choices(self):
        try:
            self.host.store.save_settings(self.host.settings)
        except (OSError, ValueError):
            self.set_message("Your choices apply in this launch, but settings could not be saved. Existing data was kept.")

    def set_message(self, text):
        """Reserve space for meaningful recovery text, never an empty label."""
        self.message.setText(text)
        self.message.setVisible(bool(text))

    def advance(self):
        index = self.stack.currentIndex()
        if index == 0:
            if not self.consent.isChecked():
                self.set_message("Read the privacy explanation and choose the checkbox before camera access. You can leave without starting the camera.")
                return
            self.host.settings.privacy_accepted = True
            self.save_choices()
        elif index == 1:
            state = self.host.camera.status.read() if self.host.camera else None
            if not state or state.state != "live":
                self.set_message("Test a working camera before continuing. If it cannot start, return to Camera, refresh devices, and check Windows camera access.")
                return
        elif index == 2 and not self.gesture_chosen:
            self.set_message("Choose personal training, your existing tested model, or mouse/keyboard. A failed new candidate never activates partway.")
            return
        elif index == 3:
            self.host.settings.setup_complete = True
            self.save_choices()
            self.practice_requested = self.practice_choice.isChecked()
            self.accept()
            return
        self.stack.setCurrentIndex(index + 1)
        self.show_step()

    def back(self):
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self.show_step()

    def show_step(self):
        index = self.stack.currentIndex()
        names = ("Your camera stays on this computer", "Choose and test camera",
                 "Choose gesture controls", "Optional output")
        self.heading.setText(f"Step {index + 1} of 4 — {names[index]}")
        self.previous.setEnabled(index > 0)
        self.next.setText("Finish setup" if index == 3 else "Continue")

    def use_mouse(self):
        self.host.settings.active_model_kind = "none"
        self.host.settings.active_model_version = None
        self.host._load_active_model()
        self.gesture_chosen = True
        self.save_choices()
        self.set_message("Mouse/keyboard selected. Your earlier model files remain available in Settings.")

    def keep_model(self):
        self.host._load_active_model()
        if self.host.model is not None:
            self.gesture_chosen = True
            self.set_message("Current tested personal model selected.")

    def train(self):
        self.host.open_training()
        self.host.mode = "setup"
        self.host.interaction.enter_mode("setup")
        self.host._publish_scene()
        if self.host.model is not None:
            self.gesture_chosen = True
            self.set_message("Your tested personal model is ready. Continue to optional output.")
        else:
            self.set_message("No new model was activated. You can resume training later or choose mouse/keyboard now.")

    def refresh(self):
        self.devices.setEnabled(not self.host.camera or not self.host.camera.running)
        if self.stack.currentIndex() != 1:
            return
        result = self.host.renderer.results.read() if self.host.renderer else None
        from time import monotonic_ns

        import numpy as np
        if result:
            if monotonic_ns() - (result.capture_time_ns or result.render_time_ns) <= 250_000_000:
                self.preview.set_rgb(result.preview_rgb)
            else:
                self.preview.set_rgb(np.zeros((720, 1280, 3), np.uint8))
        else:
            self.preview.set_rgb(np.zeros((720, 1280, 3), np.uint8))
        camera = self.host.camera.status.read() if self.host.camera else None
        observation = self.host.vision.results.read() if self.host.vision else None
        self.readiness.setText((camera.message if camera else "Camera has not been started") +
                               (" · Hands visible" if observation and observation.hands else " · Put your hands in view if you want gesture controls") +
                               " · Virtual output is optional")

    def done(self, result):
        self.timer.stop()
        super().done(result)
