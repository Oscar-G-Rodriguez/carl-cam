"""Offline illustrated quick start, with original owner-drawn pose guides."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from carl_cam.training_ui import POSE_CROPS, POSE_NAMES
from carl_cam.vision_privacy import VisionPrivacyError, require_verified_vision_runtime
from carl_cam.window_chrome import install_dialog_chrome


class HelpDialog(QDialog):
    def __init__(self, host):
        super().__init__(host)
        self.practice_requested = False
        self.setWindowTitle("Carl-Cam — Help and quick start")
        self.resize(1024, 700)
        self.setMinimumSize(1024, 640)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        self.runtime_notice = QLabel()
        self.runtime_notice.setWordWrap(True)
        try:
            # Hash/provenance check only: Help never initializes the native SDK.
            require_verified_vision_runtime()
        except VisionPrivacyError:
            self.runtime_notice.setText("Hand controls and training are on hold because the vision runtime is missing or unsupported. Use mouse/keyboard for now. Saved recordings and models are unchanged.")
        self.runtime_notice.setVisible(bool(self.runtime_notice.text()))
        layout.addWidget(self.runtime_notice)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setAccessibleName("Illustrated quick start, troubleshooting and privacy")
        body = QWidget()
        content = QVBoxLayout(body)
        content.setContentsMargins(16, 16, 16, 16)
        content.setSpacing(8)
        self.headings = []
        for title, paragraph in (
            ("Start here", "Open Setup. Read the privacy explanation, select a physical camera and test it. Choose personal training or mouse/keyboard; you can train later. Optional output needs an installed OBS Virtual Camera driver."),
            ("Try a filter without training", "Choose Red, Yellow or Blue. Press Mouse frame, drag its corners, or focus the preview and choose a corner with 1–4, then use arrow keys. The effect appears only inside a valid frame. Clear frame or Escape removes it. Open menu supports mouse clicks or arrow keys and Enter; Back or Escape cancels."),
            ("Use trained hands", "Train all five supported poses in one guided workflow. Assign thumbs up, open palm and I love you to the three filters. Peace opens the head-following menu. Point with the opening hand's index finger and pinch/release over a choice. Hovering or pinching pauses menu movement. A direct-selection pose chooses its assigned filter; release, then form the two-hand frame again to see it."),
            ("Form the hand-defined frame", "Use two opposing L-shaped hands. The thumb/palm junctions are opposite corners. Thumb and index directions extend as rays; their cross-hand intersections make the other corners. The four fingertips are not the corners. Keep both hands visible, separate them, and avoid crossing or almost parallel rays. Hold a valid frame to keep the filter visible; releasing it removes the effect."),
            ("Training and breaks", "Follow each drawn pose and the small position, angle, distance and lighting variations. Only usable hand measurements advance collection. Completed rounds can resume later; an interrupted round restarts. Training and checks do not execute live commands. A candidate activates only after its checks and your explicit approval. A failed replacement keeps the previous model."),
            ("If something fails", "Camera missing: connect it, refresh cameras, check Windows camera permission, and close other apps using it. Hands missing: improve steady lighting, move closer and keep fingers visible without crossing. Mask uncertain: the palette remains while background decorations are omitted. Output missing: install OBS's virtual-camera driver and close other senders. Lower detail or frame rate is not proof that a GPU or model is broken."),
            ("Input quality", "After five seconds of sustained slow processing, Carl-Cam may request a smaller camera input. It waits at least ten seconds before another reduction. The picture frame and outgoing canvas stay the same size. Actual input dimensions and reduced quality are shown below the preview. Use Try higher input quality to request the preferred size again; there is no automatic upgrade. Some cameras ignore unsupported sizes. Hand commands pause while formats change; form a fresh pose afterward. This fallback cannot guarantee 30 frames per second."),
            ("Privacy and local data", "Processing and training run locally. MediaPipe may send performance and usage metrics to Google; its input images and video are not sent to Google. Carl-Cam does not save photos/video, upload training samples, or start output automatically. Numerical rows and models stay in your local data folder. Settings can delete samples or models after confirmation. Creator export contains models without recordings and remains a candidate pending release tests."),
        ):
            heading = QLabel(title)
            heading.setStyleSheet("font-size: 18px; font-weight: bold;")
            text = QLabel(paragraph)
            text.setWordWrap(True)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            content.addWidget(heading)
            content.addWidget(text)
            self.headings.append(heading)
        guides = QGridLayout()
        guides.setSpacing(16)
        self.guide_images = {}
        sheet = QPixmap(str(Path(__file__).parent / "assets/guides/gesture-sheet.png"))
        for index, pose in enumerate(("frame", "peace", "thumbs_up", "open_palm", "ily")):
            cell = QWidget()
            column = QVBoxLayout(cell)
            column.setContentsMargins(8, 8, 8, 8)
            column.setSpacing(8)
            image = QLabel()
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if sheet.isNull():
                image.setText("Pose illustration unavailable. Follow the written instructions above.")
                image.setWordWrap(True)
            else:
                image.setPixmap(sheet.copy(*POSE_CROPS[pose]).scaled(240, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            image.setAccessibleName(POSE_NAMES[pose] + " original drawing")
            self.guide_images[pose] = image
            column.addWidget(image)
            label = QLabel(POSE_NAMES[pose])
            label.setWordWrap(True)
            column.addWidget(label)
            guides.addWidget(cell, index // 3, index % 3)
        content.addLayout(guides)
        self.scroll.setWidget(body)
        layout.addWidget(self.scroll, 1)
        self.practice = QPushButton("Practice both switching paths on my live camera")
        self.practice.clicked.connect(self.start_practice)
        self.back = QPushButton("Back to camera")
        self.back.clicked.connect(self.reject)
        self.back.setFixedWidth(240)
        actions = QHBoxLayout()
        actions.setSpacing(16)
        actions.addWidget(self.practice, 1)
        actions.addWidget(self.back)
        layout.addLayout(actions)
        install_dialog_chrome(self, layout, host)

    def start_practice(self):
        self.practice_requested = True
        self.accept()
