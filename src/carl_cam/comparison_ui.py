"""Private optional classifier comparison, usable without commands or terminals."""

from pathlib import Path
from time import monotonic_ns

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from carl_cam.comparison import PAIRED_POSES, ComparisonExercise
from carl_cam.comparison_vision import ComparisonEngine
from carl_cam.training_ui import POSE_CROPS, POSE_NAMES
from carl_cam.ui import Preview
from carl_cam.vision import VisionWorker, mirrored_canvas
from carl_cam.vision_privacy import PRIVACY_HOLD
from carl_cam.window_chrome import install_dialog_chrome


class ComparisonDialog(QDialog):
    def __init__(self, host, *, worker_factory=VisionWorker):
        super().__init__(host)
        self.host, self.worker_factory = host, worker_factory
        self.worker = self.exercise = None
        self.trial_until = None
        self.pose_ready = False
        self.last_id = -1
        self.setWindowTitle('Carl-Cam — Compare with MediaPipe')
        self.resize(1024, 750)
        self.setMinimumSize(900, 640)
        layout = QVBoxLayout(self)
        self.instructions = QLabel(
            'This optional diagnostic compares your selected neural model with MediaPipe on the same frames. '
            'It never executes commands or changes your model. Each portion has five attempts per paired pose '
            'and one minute of ordinary movement. Validation tunes independent score thresholds; the second '
            'portion is untouched until those thresholds are frozen. No photos, video or landmarks are saved.')
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        row = QHBoxLayout()
        self.preview = Preview()
        self.preview.setAccessibleName('Private mirrored comparison camera')
        self.preview.setMinimumSize(320, 180)
        self.preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(self.preview, 2)
        self.drawing = QLabel()
        self.drawing.setMaximumWidth(230)
        row.addWidget(self.drawing, 1)
        layout.addLayout(row, 1)
        self.sheet = QPixmap(str(Path(__file__).resolve().parent / 'assets/guides/gesture-sheet.png'))
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status = QLabel('Start your physical camera and select a tested personal model before this exercise.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.save_consent = QCheckBox('Save the numerical categories, scores, frame IDs and report locally when complete')
        self.save_consent.setChecked(False)
        layout.addWidget(self.save_consent)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setAccessibleName('Preliminary comparison results and outcome confusion counts')
        self.report.hide()
        layout.addWidget(self.report, 1)
        self.action = QPushButton('Begin optional comparison')
        self.action.clicked.connect(self.handle_action)
        layout.addWidget(self.action)
        self.back = QPushButton('Back to Inspection')
        self.back.clicked.connect(self.reject)
        layout.addWidget(self.back)
        install_dialog_chrome(self, layout, host)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.safe_refresh)
        self.timer.start(25)

    def _camera_live(self):
        packet = self.host.camera.frames.read() if self.host.camera else None
        state = self.host.camera.status.read() if self.host.camera else None
        return bool(packet is not None and state and state.state == 'live'
                    and 0 <= monotonic_ns() - packet.capture_time_ns <= 200_000_000)

    def handle_action(self):
        try:
            if self.exercise is None:
                if self.host.model is None or not self._camera_live() or self.host.mode != 'inspect':
                    self.status.setText('A fresh camera, selected tested model and private Inspection mode are required. Return to the camera to set them up.')
                    return
                self.exercise = ComparisonExercise(threshold=self.host.model.threshold,
                                                   margin=self.host.model.margin, hold_ms=self.host.settings.hold_ms)
                self.worker = self.worker_factory(self.host.camera.frames,
                                                  factory=lambda: ComparisonEngine(self.host.model))
                self.worker.start()
                self.save_consent.setEnabled(False)
                self.show_step()
            elif self.exercise.phase == 'complete':
                self.reject()
            elif all(count == 5 for count in self.exercise.attempts.values()):
                self.exercise.start_negative()
                self.action.setEnabled(False)
                self.instructions.setText('Type, touch your face, point, pinch and move naturally without deliberately making a command. Only fresh paired camera time counts. Both systems remain commands-disabled.')
            elif not self.pose_ready:
                if not self.exercise.ready:
                    return
                self.pose_ready = True
                self.instructions.setText(f'Form {POSE_NAMES[self.next_pose()]}. Click Pose ready or press Space when formed. Keep it held for four seconds. A helper can mark the moment; keyboard marking may add hand-return delay.')
                self.action.setText('Pose ready — start scoring')
                self.action.setShortcut('Space')
            else:
                now = monotonic_ns()
                self.exercise.start_trial(self.next_pose(), now)
                self.trial_until = now + 4_000_000_000
                self.action.setShortcut('')
                self.action.setEnabled(False)
                self.instructions.setText('Keep the pose formed until this four-second attempt ends. No filter or menu action executes.')
        except (ValueError, OSError, RuntimeError) as error:
            self.status.setText(str(error))

    def next_pose(self):
        return next(pose for pose in PAIRED_POSES if self.exercise.attempts[pose] < 5)

    def show_step(self):
        self.pose_ready = False
        self.action.setShortcut('')
        self.progress.setValue(round(sum(self.exercise.attempts.values()) / 20 * 100))
        if all(count == 5 for count in self.exercise.attempts.values()):
            self.drawing.clear()
            self.instructions.setText(f'{self.exercise.phase.title()} attempts complete. Begin one minute of ordinary movement; no-hand time counts only with a fresh paired camera result.')
            self.action.setText('Begin ordinary movement')
            self.action.setEnabled(True)
            return
        pose = self.next_pose()
        if not self.sheet.isNull():
            self.drawing.setPixmap(self.sheet.copy(*POSE_CROPS[pose]).scaled(220, 170, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.instructions.setText(f'{self.exercise.phase.title()} · {POSE_NAMES[pose]} · attempt {self.exercise.attempts[pose] + 1}/5. First take both hands out of view for release, then prepare this one-hand pose.')
        self.action.setText('Hands released — prepare pose')
        self.action.setEnabled(False)

    def safe_refresh(self):
        try:
            self.refresh()
        except (ValueError, OSError, RuntimeError, TypeError) as error:
            self.timer.stop()
            self.action.setEnabled(False)
            self.status.setText(f'The diagnostic stopped: {type(error).__name__}. Return to Inspection; your active model was not changed.')

    def refresh(self):
        packet = self.host.camera.frames.read() if self.host.camera else None
        live = self._camera_live()
        self.preview.set_rgb(mirrored_canvas(packet) if live else np.zeros((360, 640, 3), np.uint8))
        if self.worker is None or self.exercise.phase == 'complete':
            return
        pair = self.worker.results.read()
        status = self.worker.status.read()
        if status and status.startswith('Vision unavailable'):
            if status == 'Vision unavailable: ' + PRIVACY_HOLD:
                self.status.setText('Comparison is on hold because the vision runtime is missing or unsupported. Return to Inspection; your model was not changed.')
            else:
                self.status.setText('The local comparison asset could not load. Repair the installation or return to Inspection; no classifier was substituted.')
            self.action.setEnabled(False)
            return
        now = monotonic_ns()
        if pair is not None and pair.frame_id != self.last_id:
            self.last_id = pair.frame_id
            self.exercise.observe(pair, now_ns=now, camera_live=live)
            if not self.pose_ready and self.trial_until is None and not self.exercise.negative:
                self.action.setEnabled(self.exercise.ready or all(count == 5 for count in self.exercise.attempts.values()))
            self.status.setText(f'Paired frame {pair.frame_id}: neural {pair.neural.outcome}, MediaPipe {pair.canned.outcome}. '
                                + ('Fresh for scoring.' if pair.fresh and now - pair.capture_time_ns <= 80_000_000 else 'Too old for scoring; this time does not advance progress.'))
        elif not live:
            self.exercise.pause()
            self.status.setText('Camera is unavailable or stale. Comparison progress is paused; old frames do not count.')
        if self.trial_until is not None and now >= self.trial_until:
            self.exercise.finish_trial()
            self.trial_until = None
            self.show_step()
        if self.exercise.negative:
            seconds = self.exercise.negative_seconds
            self.progress.setValue(round(min(1., seconds / 60) * 100))
            self.status.setText(f'{seconds:.1f}/60 usable ordinary-movement seconds. Raw None/unknown/no-hand outcomes are distinct, not automatically correct other labels.')
            if seconds >= 60:
                self.exercise.finish_portion()
                if self.exercise.phase == 'test':
                    self.status.setText('Validation thresholds are frozen. The untouched test uses new attempts and never updates those thresholds.')
                    self.show_step()
                else:
                    self.show_report()

    def show_report(self):
        self.worker.stop()
        self.instructions.setText('Preliminary paired diagnostic complete. This is not release certification or proof of open-set recognition. The active model, assignments and command thresholds are unchanged.')
        lines = []
        for system, report in self.exercise.reports.items():
            lines.append(f'{"Carl-Cam neural" if system == "neural" else "MediaPipe canned"} · threshold {report["score_threshold"]:.2f}, margin {report["margin"]:.2f}')
            for pose in PAIRED_POSES:
                lines.append(f'  {POSE_NAMES[pose]}: {report["successes"][pose]}/5 successes, {report["misses"][pose]} misses')
            delay = f'{report["p95_delay_ms"]:.0f} ms' if report['p95_delay_ms'] is not None else 'no successful attempt'
            lines.append(f'  Accidental paired commands: {report["paired_accidental_commands"]}; wrong actions: {report["wrong_actions"]}; 95% completed-response delay: {delay}')
            if report['mean_inference_ms'] is not None:
                lines.append(f'  Mean perception + classification processing: {report["mean_inference_ms"]:.1f} ms')
            lines.append(f'  Missing/abstention outcomes: {report["missing_outcomes"]}')
            lines.append('  Observed-frame outcome counts by true pose (uncalibrated scores):')
            for truth, counts in report['confusion'].items():
                lines.append(f'    {truth}: ' + ', '.join(f'{outcome}={count}' for outcome, count in counts.items() if count))
        lines.append(f'Carl-Cam frame commands, reported separately with no canned equivalent: {self.exercise.reports["neural"]["frame_commands_separate"]}. This four-pose exercise does not measure deliberate frame-pose success.')
        lines.append('Alternating inference order reduces a fixed first/second-system delay advantage. These timings include this extra diagnostic workload, not normal Camera performance.')
        if self.save_consent.isChecked():
            try:
                self.exercise.save(self.host.store, user_consented=True)
                lines.append('The consented numerical report is saved locally with recordings. Delete recordings or Reset all local data removes it.')
            except (OSError, ValueError):
                lines.append('The numerical report could not be saved. Results remain visible here; no model/settings were changed.')
        else:
            lines.append('Nothing was saved. Closing this screen discards these transient numerical results.')
        lines.append('A tie or MediaPipe win is shown as observed. Carl-Cam learns from MediaPipe-provided landmarks plus our own features/network; it does not build the hand detector from scratch.')
        self.report.setPlainText('\n'.join(lines))
        self.report.show()
        self.drawing.hide()
        self.preview.hide()
        self.action.setEnabled(True)
        self.action.setText('Back to Inspection')

    def done(self, result):
        if self.worker and not self.worker.stop():
            self.status.setText('The comparison worker is still stopping. Please retry Back; the camera remains privately controlled.')
            return
        self.timer.stop()
        super().done(result)
