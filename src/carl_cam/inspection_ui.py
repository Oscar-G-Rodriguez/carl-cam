"""Transient six-panel inspection. No diagnostic pixels are saved or broadcast."""

from time import monotonic_ns

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from carl_cam.features import FeatureExtractor
from carl_cam.interaction import winning_pose
from carl_cam.ui import Preview
from carl_cam.window_chrome import install_dialog_chrome

PANEL_NAMES = ("Original (mirrored)", "Linear luminance", "Sobel edge strength", "Person mask",
               "Hand landmarks and quadrilateral", "Final graphic composite")


class InspectionDialog(QDialog):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.extractor = FeatureExtractor()
        self.setWindowTitle("Carl-Cam — Inspect live processing")
        self.resize(1200, 780)
        self.setMinimumSize(1024, 640)
        self.cells = []
        self.panel_labels = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        grid = QGridLayout()
        grid.setSpacing(16)
        for index, name in enumerate(PANEL_NAMES):
            cell = QWidget()
            column = QVBoxLayout(cell)
            column.setContentsMargins(8, 8, 8, 8)
            column.setSpacing(8)
            label = QLabel(name)
            label.setWordWrap(True)
            self.panel_labels.append(label)
            column.addWidget(label)
            preview = Preview()
            preview.setMinimumSize(240, 135)
            preview.setAccessibleName(name)
            preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            column.addWidget(preview, 1)
            self.cells.append(preview)
            grid.addWidget(cell, index // 3, index % 3)
        for index in range(3):
            grid.setColumnStretch(index, 1)
        for index in range(2):
            grid.setRowStretch(index, 1)
        layout.addLayout(grid, 1)
        self.metrics = QLabel("Start a camera to inspect its current signals.")
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        self.explanation = QLabel("All panels are transient. Hand commands are disabled here, and virtual output receives only the ordinary unmirrored camera—not these diagnostics. A missing exact-frame mask disables background motifs rather than using an old silhouette.")
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        self.compare = QPushButton('Compare with MediaPipe (optional)')
        self.compare.clicked.connect(self.open_comparison)
        self.back = QPushButton("Back to camera")
        self.back.clicked.connect(self.reject)
        self.back.setFixedWidth(240)
        actions = QHBoxLayout()
        actions.setSpacing(16)
        actions.addWidget(self.compare, 1)
        actions.addWidget(self.back)
        layout.addLayout(actions)
        install_dialog_chrome(self, layout, host)
        self.last_id = -1
        self.last_capture_time_ns = -1
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.safe_refresh)
        self.timer.start(50)

    def open_comparison(self):
        from carl_cam.comparison_ui import ComparisonDialog
        # The host stays in commands-disabled Inspection mode throughout.
        ComparisonDialog(self.host).exec()

    def refresh(self):
        now = monotonic_ns()
        snapshot = self.host.renderer.inspection.read() if self.host.renderer else None
        if snapshot is None or not 0 <= now - snapshot.capture_time_ns <= 250_000_000:
            self.clear_panels()
            self.metrics.setText("Waiting for fresh live processing. Old panels have been cleared."
                                 if self.host.camera and self.host.camera.running
                                 else "Start a camera to inspect its current signals.")
            return
        if (snapshot.frame_id, snapshot.capture_time_ns) != (self.last_id, self.last_capture_time_ns):
            grey = lambda signal: np.repeat(np.rint(signal.clip(0, 1) * 255).astype(np.uint8)[..., None], 3, axis=-1)
            panels = (snapshot.original_rgb, grey(snapshot.stages.luminance), grey(snapshot.stages.edge_magnitude),
                      grey(snapshot.person_mask), snapshot.overlay_rgb, snapshot.final_rgb)
            for preview, image in zip(self.cells, panels, strict=True):
                preview.set_rgb(cv2.resize(image, (400, 225), interpolation=cv2.INTER_AREA))
            self.last_id = snapshot.frame_id
            self.last_capture_time_ns = snapshot.capture_time_ns
        renderer = self.host.renderer.results.read()
        text = f"Frame {snapshot.frame_id} · "
        if (renderer is not None and renderer.frame_id == snapshot.frame_id
                and renderer.capture_time_ns == snapshot.capture_time_ns):
            text += f"{renderer.render_ms:.0f} ms processing · "
        text += "Region mask reliable" if snapshot.segmentation_reliable else "Region mask unavailable/uncertain — palette only"
        observation = self.host.vision.results.read() if self.host.vision else None
        if (observation and observation.frame_id == snapshot.frame_id
                and observation.capture_time_ns == snapshot.capture_time_ns
                and 0 <= now - observation.capture_time_ns <= 100_000_000 and self.host.model):
            result = self.extractor.extract(observation.hands)
            vector = result.vectors.get(self.host.model.width)
            if vector is not None:
                scores = tuple(float(value) for value in self.host.model.scores(vector)[0])
                pose = winning_pose(scores, threshold=self.host.model.threshold, margin=self.host.model.margin)
                text += f" · predicted command: {pose or 'abstain'} · highest raw score {max(scores):.2f} (uncalibrated)"
        self.metrics.setText(text)

    def clear_panels(self):
        blank = np.zeros((180, 320, 3), np.uint8)
        for preview in self.cells:
            preview.set_rgb(blank)
        self.last_id = self.last_capture_time_ns = -1

    def safe_refresh(self):
        try:
            self.refresh()
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, cv2.error):
            self.clear_panels()
            self.metrics.setText("This frame could not be inspected. Return to the camera and try again; your saved model is unchanged.")

    def done(self, result):
        self.timer.stop()
        super().done(result)
