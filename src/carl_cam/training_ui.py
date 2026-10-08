"""Guided local gesture training, operable without gesture control or a terminal.

The desktop layout follows the editable Figma assignment/collection views.
Camera pixels are displayed only; only TrainingFlow persists numerical rows.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from time import monotonic

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from carl_cam.features import FeatureExtractor, FeatureResult, observed_quad
from carl_cam.records import HandObservation
from carl_cam.storage import LABELS, valid_assignments
from carl_cam.training import gpu_training_status
from carl_cam.training_flow import TrainingFlow, resumable_recordings
from carl_cam.ui import Preview
from carl_cam.vision import mirrored_canvas
from carl_cam.window_chrome import install_dialog_chrome

POSE_NAMES = {"frame": "Picture frame", "peace": "Peace sign — open menu", "thumbs_up": "Thumbs up",
              "open_palm": "Open palm", "ily": "I love you", "other": "Ordinary movement"}
POSE_CROPS = {"frame": (20, 20, 365, 315), "peace": (440, 85, 265, 345),
              "thumbs_up": (100, 465, 185, 235), "open_palm": (345, 490, 265, 310),
              "ily": (645, 480, 320, 360)}
VARIATION_TEXT = {
    "center": "Hold the pose near the center of the camera.",
    "shift": "Move the pose gently left and right, keeping the fingers visible.",
    "angle": "Tilt and rotate the pose a little. Avoid hiding your fingers.",
    "distance": "Move slightly closer and farther away while your hand stays visible.",
    "lighting": "Try a small lighting change, such as turning toward a lamp. Do not block the camera.",
    "rest_typing": "Rest your hands and type naturally, keeping at least one hand visible.",
    "touch_face": "Touch your face or adjust your glasses naturally.",
    "ordinary_movement": "Move your hands naturally without intentionally making a command pose.",
    "point_pinch": "Point and pinch in ordinary ways, rather than holding a command pose.",
    "entry_exit": "Move your hand in and out of view. Only visible, usable measurements advance this recording.",
}


class TrainingPreview(Preview):
    def __init__(self) -> None:
        super().__init__()
        self.hands: tuple[HandObservation, ...] = ()
        self.setAccessibleName("Training camera with tracked hand landmarks")

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        rect = self.scene_rect()
        painter.translate(rect.x(), rect.y())
        painter.scale(rect.width() / 1280, rect.height() / 720)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#4AC9E8"), 2))
        painter.setBrush(QColor("#FFF4C8"))
        for hand in self.hands:
            points = hand.landmarks
            for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 10, 11, 12),
                          (9, 13, 14, 15, 16), (13, 17, 18, 19, 20), (0, 17)):
                for first, second in pairwise(chain):
                    painter.drawLine(QPointF(*points[first][:2]), QPointF(*points[second][:2]))
            for point in points:
                painter.drawEllipse(QPointF(*point[:2]), 3, 3)


class TrainingDialog(QDialog):
    def __init__(self, parent, *, recheck_version=None) -> None:
        super().__init__(parent)
        self.host = parent
        self.recheck_version = recheck_version
        self.setWindowTitle("Carl-Cam — Train my gestures")
        self.resize(1180, 780)
        self.setMinimumSize(1024, 640)
        self.flow: TrainingFlow | None = None
        self.extractor = FeatureExtractor()
        self._last_perception = -1
        self._features = FeatureResult({}, (), False)
        self._source_time = 0.0
        self._trial_index = 0
        self._trial_until: float | None = None
        self._waiting_marker = False
        self._negative_started = False
        self._displayed_phase = ""
        self._pending_done: int | None = None
        self.preview = TrainingPreview()
        self.preview.setMinimumSize(480, 270)
        self.heading = QLabel("Choose which pose selects each filter")
        self.heading.setWordWrap(True)
        self.instructions = QLabel("Frame and peace have fixed roles. Assign the other three poses to different filters.")
        self.instructions.setWordWrap(True)
        self.instructions.setTextFormat(Qt.TextFormat.PlainText)
        self.drawing = QLabel()
        self.drawing.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drawing.setMinimumHeight(130)
        self.drawing.hide()
        self.sheet = QPixmap(str(Path(__file__).resolve().parent / "assets/guides/gesture-sheet.png"))
        self.mapping_widget = QWidget()
        mapping_form = QFormLayout(self.mapping_widget)
        mapping_form.setContentsMargins(0, 0, 0, 0)
        mapping_form.setSpacing(8)
        mapping_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        mapping_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.mapping = {}
        self.recording_choice = QComboBox()
        # Resume descriptions can be long. Keep the native selector inside the
        # instruction panel; its popup and tooltip retain the full description.
        self.recording_choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.recording_choice.setMinimumContentsLength(16)
        self.recording_choice.setAccessibleName("Start new or resume local gesture recordings")
        self.recording_choice.addItem("Start a new collection", None)
        for dataset_id, label in resumable_recordings(parent.store):
            self.recording_choice.addItem(label, dataset_id)
        self.recording_choice.setToolTip("Completed rounds are kept. Interrupted fitting/comparison restarts; a revealed final check requires fresh held-out rounds. Saved pose assignments are restored.")
        defaults = parent.settings.assignments or {"thumbs_up": "red", "open_palm": "yellow", "ily": "blue"}
        for pose in ("thumbs_up", "open_palm", "ily"):
            combo = QComboBox()
            for theme in ("red", "yellow", "blue"):
                combo.addItem(theme.title(), theme)
            combo.setCurrentIndex(combo.findData(defaults[pose]))
            combo.setAccessibleName(f"Filter assigned to {POSE_NAMES[pose]}")
            mapping_form.addRow(POSE_NAMES[pose], combo)
            self.mapping[pose] = combo
        self.consent = QCheckBox("Save hand measurements")
        self.consent.setAccessibleName("Consent to save numerical hand measurements")
        self.consent.setChecked(False)
        self.consent.setToolTip("You can delete these recordings and personal models separately in Settings.")
        self.consent_info = QLabel("Only anonymous numerical hand measurements are saved on this computer. No photos or video are saved. You can delete measurements and models separately in Settings.")
        self.consent_info.setWordWrap(True)
        self.depth = QCheckBox("Include depth (180 values)")
        self.depth.setAccessibleName("Compare the preferred 180 measurements, including depth")
        self.depth.setChecked(True)
        self.depth.hide()
        self.retry_depth = QPushButton("Retry depth rounds")
        self.retry_depth.clicked.connect(self.handle_depth_retry)
        self.retry_depth.hide()
        self.cuda = QCheckBox("Train on NVIDIA GPU")
        gpu_available, gpu_message = gpu_training_status()
        self.cuda.setEnabled(gpu_available)
        self.cuda.setToolTip(gpu_message)
        self.cuda.hide()
        self.cuda_info = QLabel(gpu_message)
        self.cuda_info.setWordWrap(True)
        self.cuda_info.setTextFormat(Qt.TextFormat.PlainText)
        self.cuda_info.hide()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setAccessibleName("Current training step progress")
        self.detail = QLabel("The camera is displayed live only. The full collection requires at least 6 minutes 15 seconds of usable measurements, plus instructions and tests.")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.action = QPushButton("Begin guided collection")
        self.action.clicked.connect(self.handle_action)
        self.action.setDefault(True)
        self.redo = QPushButton("Redo this round")
        self.redo.clicked.connect(self.redo_round)
        self.redo.hide()
        self.skip = QPushButton("Skip lighting change")
        self.skip.clicked.connect(self.skip_lighting)
        self.skip.hide()
        self.back = QPushButton("Back to camera")
        self.back.clicked.connect(self.reject)
        left = QVBoxLayout()
        for widget in (self.heading, self.drawing, self.instructions, self.recording_choice, self.mapping_widget, self.consent, self.consent_info, self.depth, self.cuda, self.cuda_info):
            left.addWidget(widget)
        left.addStretch()
        right = QVBoxLayout()
        right.addWidget(self.progress)
        right.addWidget(self.detail)
        right.addWidget(self.action)
        right.addWidget(self.redo)
        right.addWidget(self.skip)
        right.addWidget(self.retry_depth)
        right.addStretch()
        left_widget, right_widget = QWidget(), QWidget()
        for panel in (left, right):
            panel.setContentsMargins(8, 8, 8, 8)
            panel.setSpacing(8)
        left_widget.setLayout(left)
        right_widget.setLayout(right)
        self.left_scroll, self.right_scroll = QScrollArea(), QScrollArea()
        for scroll, widget in ((self.left_scroll, left_widget), (self.right_scroll, right_widget)):
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            scroll.setFixedWidth(240)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        center = QHBoxLayout()
        center.setSpacing(16)
        center.addWidget(self.left_scroll)
        center.addWidget(self.preview, 1)
        center.addWidget(self.right_scroll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.addLayout(center, 1)
        footer = QHBoxLayout()
        footer.setSpacing(16)
        self.safety_notice = QLabel("Live commands stay disabled throughout training. A previous working model is kept unless you explicitly activate a tested replacement.")
        self.safety_notice.setWordWrap(True)
        footer.addWidget(self.safety_notice, 1)
        self.back.setFixedWidth(240)
        footer.addWidget(self.back)
        layout.addLayout(footer)
        install_dialog_chrome(self, layout, parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.safe_refresh)
        self.timer.start(30)
        if self.recheck_version:
            self.recording_choice.hide()
            self.heading.setText("Recheck saved gestures on this camera")
            self.consent.setText("Save camera recheck result")
            self.consent.setAccessibleName("Consent to save this camera's aggregate recheck result")
            self.consent_info.setText("This keeps your tested weights unchanged. Only aggregate successes/delays are saved, not new training rows, landmarks, photos or video.")
            self.instructions.setText("Assign the three filters, then check all five poses with commands disabled: five attempts each and two minutes of ordinary movement. If it fails, use mouse/keyboard or collect a new model.")
            self.detail.setText("Your existing personal model and recordings are kept.")
            self.action.setText("Begin camera recheck")

    def assignments(self) -> dict[str, str]:
        return {pose: combo.currentData() for pose, combo in self.mapping.items()}

    def show_pose(self, pose: str) -> None:
        crop = POSE_CROPS.get(pose)
        if crop and not self.sheet.isNull():
            pixmap = self.sheet.copy(*crop).scaled(206, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.drawing.setPixmap(pixmap)
            self.drawing.show()
        else:
            self.drawing.clear()
            self.drawing.hide()

    def handle_action(self) -> None:
        try:
            self._handle_action()
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            self._show_error(error)

    def _show_error(self, error: Exception) -> None:
        # Reconcile a changed workflow first, then retain the reason alongside
        # its available recovery controls. Never enable an overlapping action.
        self._show_phase(force=bool(self.flow and self.flow.phase == "collection"
                                   and self.flow.current is None))
        self.detail.setText(f"This training step could not finish: {error}. Your previous model is kept.")
        self.action.setEnabled(self._action_available())

    def _action_available(self) -> bool:
        """Errors cannot authorize a second action over an unfinished step."""
        return (self._pending_done is None and self._trial_until is None
                and not self._negative_started
                and (self.flow is None or (self.flow.phase != "fitting"
                     and getattr(self.flow, "current", None) is None)))

    def handle_depth_retry(self) -> None:
        try:
            if self.flow and self._pending_done is None:
                self.flow.retry_depth_rounds()
                self._show_phase(force=True)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            self._show_error(error)

    def _handle_action(self) -> None:
        if self._pending_done is not None:
            return  # A pending close cannot start another recording or model.
        now = monotonic()
        if self.flow is None:
            if not self.consent.isChecked():
                self.detail.setText("Choose the numerical-data consent checkbox before recording. You can also return to the camera without training.")
                return
            if not valid_assignments(self.assignments()):
                self.detail.setText("Use each filter exactly once: Red, Yellow, and Blue.")
                return
            if self.recheck_version:
                from carl_cam.camera_recheck import CameraRecheckFlow
                settings = replace(self.host.settings, assignments=self.assignments())
                self.flow = CameraRecheckFlow(self.host.store, settings, self.recheck_version)
            else:
                self.flow = TrainingFlow(self.host.store, self.host.settings, assignments=self.assignments(), resume_id=self.recording_choice.currentData())
            self.recording_choice.hide()
            self.mapping_widget.hide()
            self.consent.hide()
            self.consent_info.hide()
            self._show_phase(force=True)
            if self.recheck_version:
                self._reset_exercise()
            return
        phase = self.flow.phase
        if phase == "collection":
            if self.flow.current is None:
                self.flow.start_round(now)
                self.action.setEnabled(False)
                self.action.setText("Recording round…")
                self.redo.show()
                self.skip.hide()
        elif phase == "review":
            self.flow.begin_fit(include_depth=self.depth.isChecked(), use_cuda=self.cuda.isChecked())
            self.depth.hide()
            self.cuda.hide()
            self.action.setEnabled(False)
            self._show_phase(force=True)
        elif phase in ("comparison", "rehearsal"):
            total = self.flow.exercise.attempts_per_pose * 5
            if self._trial_index >= total:
                self.flow.exercise.start_negative()
                self._negative_started = True
                self.action.setEnabled(False)
                self.action.setText("Checking movement…")
                return
            pose = LABELS[self._trial_index // self.flow.exercise.attempts_per_pose]
            if not self._waiting_marker:
                self._waiting_marker = True
                self.instructions.setText(f"Form {POSE_NAMES[pose]}. When it is formed, click Pose ready or press Space to mark that moment, then keep holding. For the two-hand frame, a helper can mark the moment; a keyboard mark may add hand-return time to the measured delay.")
                self.action.setText("Pose ready — start scoring")
                self.action.setShortcut("Space")
            else:
                self.flow.exercise.start_trial(pose, now)
                self._trial_until = now + 4
                self._waiting_marker = False
                self.action.setShortcut("")
                self.action.setEnabled(False)
                self.action.setText("Scoring attempt…")
                self.instructions.setText("Hold the pose for this four-second attempt. No live command will execute. Release it before the next attempt.")
        elif phase == "sealed":
            self.action.setEnabled(False)
            if self.flow.check_sealed():
                self._reset_exercise()
            self._show_phase(force=True)
        elif phase == "needs_examples":
            self.flow.retry_weak_cases()
            if self.recheck_version:
                self._reset_exercise()
            self._show_phase(force=True)
        elif phase == "ready":
            choice = QMessageBox.question(self, "Activate tested personal gestures?",
                                         "This candidate passed your local checks. Activate all five learned commands? Your previous version is kept.",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if choice == QMessageBox.StandardButton.Yes:
                self.host.settings = self.flow.activate(user_confirmed=True)
                self.accept()

    def redo_round(self) -> None:
        try:
            if self.flow and self.flow.phase == "collection" and self._pending_done is None:
                self.flow.interrupt_round()
                self._show_phase(force=True)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            self._show_error(error)

    def skip_lighting(self) -> None:
        try:
            if self.flow and self._pending_done is None:
                self.flow.skip_lighting()
                self._show_phase(force=True)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            self._show_error(error)

    def _reset_exercise(self) -> None:
        self._trial_index = 0
        self._trial_until = None
        self._waiting_marker = self._negative_started = False

    def _show_phase(self, *, force: bool = False) -> None:
        if not self.flow or not force and self.flow.phase == self._displayed_phase:
            return
        self._displayed_phase = self.flow.phase
        phase = self.flow.phase
        self.drawing.clear()
        self.drawing.hide()
        self.redo.hide()
        self.skip.hide()
        self.depth.hide()
        self.retry_depth.hide()
        self.cuda.hide()
        self.cuda_info.hide()
        self.action.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        if phase == "collection":
            prompt = self.flow.prompt
            self.heading.setText(f"{POSE_NAMES[prompt.label]} · round {prompt.round_index + 1}")
            self.show_pose(prompt.label)
            self.instructions.setText("Keep the required hand(s) clearly visible. Follow three small variations. Progress counts usable measurements, not waiting time. For the frame, both L-hands and a valid four-corner shape are required.")
            self.detail.setText(f"Recording {self.flow.index + 1} of {len(self.flow.plan)} · {prompt.target_seconds} usable seconds. The app is learning your prompted label; an untrained classifier does not certify it.")
            self.action.setText("Start round")
            self.skip.setVisible("lighting" in prompt.variations)
            self.progress.setValue(0)
        elif phase == "review":
            self.heading.setText("Review recordings, then train")
            self.drawing.clear()
            self.drawing.hide()
            rows = self.flow.rows()
            counts = [f"{POSE_NAMES[label]}: {sum(row['label'] == label for row in rows)} measurements" for label in LABELS]
            self.detail.setText("\n".join(counts))
            from carl_cam.training import depth_coverage
            enough_depth = all(min(counts) >= 10 for counts in depth_coverage(rows).values())
            self.depth.setChecked(True)
            self.depth.setEnabled(True)
            self.retry_depth.setVisible(not enough_depth)
            self.depth.show()
            self.cuda.show()
            gpu_available, gpu_message = gpu_training_status()
            self.cuda.setEnabled(gpu_available)
            if not gpu_available:
                self.cuda.setChecked(False)
            self.cuda.setToolTip(gpu_message)
            self.cuda_info.setText(gpu_message)
            self.cuda_info.show()
            self.instructions.setText("Three whole recording rounds train the model. The fourth checks each candidate; the fifth stays untouched until a winner is fixed. PyTorch is primary; logistic regression is a comparison only." + (" Depth needs a retry. Keep hands visible and try another angle/lighting; if unavailable depth persists, explicitly uncheck depth for four 112/138-value candidates." if not enough_depth else " All six candidates can use the same valid-depth fit rows; isolated missing depth rows are excluded consistently."))
            self.action.setText("Train and compare")
        elif phase == "fitting":
            self.heading.setText("Training local neural candidates")
            self.instructions.setText("This stays on your computer. You can cancel and keep your recordings and previous model.")
            self.progress.setRange(0, 0)
            self.detail.setText("Preparing local candidate training. Progress will appear here; Back to camera requests cancellation.")
            self.action.setText("Training in progress…")
            self.action.setEnabled(False)
        elif phase in ("comparison", "rehearsal"):
            self._show_attempt()
        elif phase == "sealed":
            self.heading.setText("Fixed winner — final recording check")
            self.instructions.setText(f"Selected {self.flow.winner.name}. The fifth recordings are checked once, with this model and these thresholds fixed. A failure needs a new final check, not tuning against these recordings.")
            self.detail.setText(self._comparison_summary())
            self.action.setText("Run untouched final check")
        elif phase == "needs_examples":
            self.heading.setText("More examples are needed")
            self.instructions.setText("No candidate commands have been activated. Keep using your previous model or mouse/keyboard. The retry targets weak cases and keeps earlier training recordings; a revealed final test is replaced.")
            self.detail.setText(self._result_summary())
            self.action.setText("Collect extra examples")
            if self.recheck_version:
                self.heading.setText("Camera recheck did not pass")
                self.instructions.setText("Learned commands stay off on this camera. Retry the unchanged model, use mouse/keyboard, or collect a new model from Settings. Existing recordings and models are kept.")
                self.action.setText("Retry camera recheck")
        elif phase == "ready":
            self.progress.setValue(100)
            self.heading.setText("Personal checks passed")
            self.instructions.setText("All five commands can be activated together. These checks concern your own recordings and rehearsal; they do not certify recognition for other people. After activation, Help offers guided practice of direct switching and menu selection on the live camera.")
            self.detail.setText(self._result_summary())
            self.action.setText("Activate tested gestures")

    def _show_attempt(self) -> None:
        exercise = self.flow.exercise
        count = exercise.attempts_per_pose
        title = "Camera recheck" if self.recheck_version else "Candidate comparison" if count == 5 else "Activation rehearsal"
        if self._trial_index >= count * 5:
            self.heading.setText(f"{title} — ordinary movement")
            self.drawing.clear()
            self.drawing.hide()
            self.instructions.setText("Spend two usable minutes typing, touching your face, pointing/pinching, and moving hands into and out of view. Commands remain disabled. Camera gaps pause the timer.")
            self.action.setText("Start movement check")
        else:
            pose = LABELS[self._trial_index // count]
            self.heading.setText(f"{title}: {POSE_NAMES[pose]} · attempt {self._trial_index % count + 1}/{count}")
            self.show_pose(pose)
            self.instructions.setText("Release the previous pose. Prepare this pose, then mark when it is formed. The same observations are checked for every candidate; no filter/menu action runs.")
            self.action.setText("Prepare this attempt")
        self.detail.setText("Timing uses your explicit Pose ready marker. It includes any extra hand-return time after a keyboard mark.")
        self.progress.setValue(round(100 * self._trial_index / (count * 5)))

    def _comparison_summary(self) -> str:
        if not self.flow.action_reports:
            return ""
        lines = [self.flow.fitted.backend_summary]
        for candidate, report in zip(self.flow.fitted.candidates, self.flow.action_reports, strict=True):
            delay = f"{report.p95_delay_ms:.0f} ms" if report.delays_ms else "no successful attempts"
            lines.append(f"Neural {candidate.name}: successes {report.pose_successes}; accidental commands {report.false_commands}; 95% delay {delay}")
        for baseline in self.flow.fitted.baselines:
            lines.append(f"Comparison-only logistic regression ({baseline.width} measurements): held-out recall " + ", ".join(f"{value:.0%}" for value in baseline.validation_recall))
        return "\n".join(lines)

    def _result_summary(self) -> str:
        lines = [self._comparison_summary()]
        if self.flow.sealed_report:
            report = self.flow.sealed_report
            lines.append("Final recording recall: " + ", ".join(f"{POSE_NAMES[label]} {value:.0%}" for label, value in zip(LABELS, report.recalls, strict=True)))
            lines.append(f"Final negative-recording accidental commands: {report.false_commands}")
        if self.flow.rehearsal_report:
            report = self.flow.rehearsal_report
            lines.append(f"Rehearsal successes: {report.pose_successes} / {report.attempts_per_pose} each; accidental commands: {report.false_commands}.")
        return "\n".join(lines)

    def refresh(self) -> None:
        now = monotonic()
        packet = self.host.camera.frames.read() if self.host.camera else None
        camera_live = bool(self.host.camera and self.host.camera.running and packet
                           and 0 <= now - packet.capture_time_ns / 1e9 < 0.2)
        if camera_live:
            self.preview.set_rgb(mirrored_canvas(packet))
        else:
            self.preview.set_rgb(np.zeros((720, 1280, 3), dtype=np.uint8))
        observation = self.host.vision.results.read() if self.host.vision else None
        # A recent worker result cannot keep collection or scoring alive after
        # capture stops. A retained camera packet is not proof of live capture.
        fresh = camera_live and observation is not None and 0 <= now - observation.capture_time_ns / 1e9 <= 0.100
        if fresh and observation.frame_id != self._last_perception:
            self._last_perception = observation.frame_id
            self._source_time = observation.capture_time_ns / 1e9
            self._features = self.extractor.extract(observation.hands)
            quad = observed_quad(self._features)
            self.preview.hands = observation.hands
            self.preview.quad = quad
            if self.flow and self.flow.phase == "collection" and self.flow.current:
                accepted = self.flow.observe(self._features, self._source_time, now, valid_frame=quad is not None)
                round_ = self.flow.current
                self.progress.setValue(round(100 * min(1, round_.usable_seconds / round_.prompt.target_seconds)))
                self.instructions.setText(VARIATION_TEXT[self.flow.variation])
                self.detail.setText(f"{round_.usable_seconds:.1f}/{round_.prompt.target_seconds} usable seconds · {len(round_.rows)} diverse numerical examples" + ("" if accepted else "\n" + (round_.last_hint or self._features.message or "Keep the requested pose visible; frame geometry must be valid.")))
                if round_.complete:
                    self.flow.finish_round()
                    self._show_phase(force=True)
            if self.flow and self.flow.phase in ("comparison", "rehearsal") and (self._trial_until is not None or self._negative_started):
                self.flow.exercise.observe(self._features, self._source_time, now, valid_frame=quad is not None,
                                           camera_live=camera_live)
        elif not fresh:
            self.preview.hands = ()
            self.preview.quad = None
            if self.flow and self.flow.phase == "collection" and self.flow.current:
                self.detail.setText("Tracking is paused. Keep your hands visible and try better lighting; only usable measurements advance this round.")
            if self.flow and self.flow.phase in ("comparison", "rehearsal") and (self._trial_until is not None or self._negative_started):
                # Break the usable-time chain even if capture recovers within
                # one measurement interval. This never executes a command.
                self.flow.exercise.observe(self._features, self._source_time, now,
                                           valid_frame=False, camera_live=False)
        if not self.flow:
            return
        if self.flow.phase == "fitting":
            message = self.flow.poll_fit()
            if self.flow.phase == "comparison":
                self._reset_exercise()
            self._show_phase()
            if message:
                # Phase presentation sets its own summary. Apply the worker's
                # result afterward so a fitting failure is not overwritten by
                # recording counts when the workflow returns to Review.
                self.detail.setText(message)
        if self._trial_until is not None and now >= self._trial_until:
            self.flow.exercise.finish_trial()
            self._trial_until = None
            self._trial_index += 1
            self.action.setEnabled(True)
            self._show_attempt()
        if self._negative_started:
            seconds = self.flow.exercise.negative_seconds
            self.progress.setValue(round(100 * min(seconds / 120, 1)))
            exercise_live = fresh and 0 <= now - observation.capture_time_ns / 1e9 <= 0.080
            pause_hint = "" if exercise_live else "\nMovement check is paused until live camera tracking returns."
            self.detail.setText(f"{seconds:.1f}/120 usable seconds. No-hand time counts when the camera is live; stale camera time does not." + pause_hint)
            if seconds >= 120:
                self._negative_started = False
                if self.flow.phase == "comparison":
                    self.flow.finish_comparison()
                else:
                    self.flow.finish_rehearsal()
                self._show_phase(force=True)

    def safe_refresh(self) -> None:
        try:
            if self._pending_done is not None:
                self.done(self._pending_done)
                return
            self.refresh()
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            self._show_error(error)

    def done(self, result: int) -> None:
        if self.flow and not self.flow.stop(timeout=0):
            self._pending_done = result
            for widget in (self.action, self.redo, self.skip, self.depth, self.cuda, self.retry_depth):
                widget.setEnabled(False)
            self.action.setShortcut("")
            self.detail.setText("Cancel requested. Waiting for the current training step to finish; your previous model is kept.")
            return
        self._pending_done = None
        self.timer.stop()
        super().done(result)
