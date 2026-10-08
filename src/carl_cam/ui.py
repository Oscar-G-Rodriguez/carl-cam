"""Desktop camera shell and accessible manual framing controls.

Artwork is integrated after the Figma source is available. Live processing stays
on workers; this module paints completed frames and dispatches user intent.
"""

from __future__ import annotations

from dataclasses import replace
from time import monotonic, monotonic_ns

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from carl_cam.accessibility import desktop_style
from carl_cam.camera import CameraDevice, CameraWorker, available_devices
from carl_cam.diagnostics import LocalDiagnostics
from carl_cam.features import FeatureExtractor, observed_quad
from carl_cam.geometry import Quad, validate_quad
from carl_cam.gpu_process import AdaptiveRegionEngine
from carl_cam.graphics import ROOT, asset_pixels, menu_card, verified_svg
from carl_cam.interaction import InputObservation, InteractionController
from carl_cam.menu import MenuFollower, Rect
from carl_cam.performance import PerformanceFallback
from carl_cam.pointer import HandPointer
from carl_cam.render import RenderWorker, SceneState
from carl_cam.smoothing import OneEuro
from carl_cam.storage import Store
from carl_cam.typography import configure_typography
from carl_cam.vision import VisionWorker
from carl_cam.window_chrome import TitleBar


class Preview(QWidget):
    quad_changed = Signal(object)
    menu_action = Signal(str)
    menu_hover = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        # Presentation space can shrink at the supported 1024×640 window size.
        # The scene rectangle remains 16:9; capture and processing stay unchanged.
        self.setMinimumSize(320, 180)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Live camera preview and manual picture frame")
        self._image: QImage | None = None
        self.quad: Quad | None = None
        self._drag: int | None = None
        self._selected_corner = 0
        self.menu_layout = None
        self.menu_focus_index = 0
        self.setMouseTracking(True)

    def set_rgb(self, rgb: np.ndarray) -> None:
        height, width = rgb.shape[:2]
        self._image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def scene_rect(self) -> QRectF:
        scale = min(self.width() / 1280, self.height() / 720)
        return QRectF((self.width() - 1280 * scale) / 2, (self.height() - 720 * scale) / 2, 1280 * scale, 720 * scale)

    def scene_point(self, point: QPointF) -> tuple[float, float]:
        rect = self.scene_rect()
        return ((point.x() - rect.x()) * 1280 / rect.width(), (point.y() - rect.y()) * 720 / rect.height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        rect = self.scene_rect()
        painter.fillRect(rect, Qt.GlobalColor.black)
        if self._image:
            painter.drawImage(rect, self._image)
        if self.quad:
            painter.save()
            painter.translate(rect.x(), rect.y())
            painter.scale(rect.width() / 1280, rect.height() / 720)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            color = "#61d9ef" if validate_quad(self.quad, (1280, 720)) else "#ff7380"
            painter.setPen(QPen(QColor(color), 3))
            for index, point in enumerate(self.quad):
                next_point = self.quad[(index + 1) % 4]
                painter.drawLine(QPointF(*point), QPointF(*next_point))
                painter.setBrush(QColor(color))
                painter.drawEllipse(QPointF(*point), 10, 10)
            painter.restore()

    def mousePressEvent(self, event) -> None:
        if self.menu_layout and event.button() == Qt.MouseButton.LeftButton:
            target = self.menu_layout.hit(self.scene_point(event.position()))
            if target:
                self.menu_action.emit(target)
            return
        if event.button() != Qt.MouseButton.LeftButton or not self.quad:
            return
        position = np.array(self.scene_point(event.position()))
        distances = np.linalg.norm(np.array(self.quad) - position, axis=1)
        closest = int(np.argmin(distances))
        if distances[closest] < 35:
            self._drag = closest

    def mouseMoveEvent(self, event) -> None:
        if self.menu_layout:
            self.menu_hover.emit(self.scene_point(event.position()))
            return
        if self._drag is None or self.quad is None:
            return
        x, y = self.scene_point(event.position())
        points = list(self.quad)
        points[self._drag] = (float(np.clip(x, 0, 1279)), float(np.clip(y, 0, 719)))
        self.quad = tuple(points)
        self.quad_changed.emit(self.quad)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None

    def keyPressEvent(self, event) -> None:
        if self.menu_layout:
            if event.key() == Qt.Key.Key_Escape:
                self.menu_action.emit("back")
            elif self.menu_layout.cards and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_Right, Qt.Key.Key_Down):
                self.menu_focus_index = (self.menu_focus_index + (-1 if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up) else 1)) % len(self.menu_layout.cards)
            elif self.menu_layout.cards and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                self.menu_action.emit(self.menu_layout.cards[self.menu_focus_index][0])
            else:
                super().keyPressEvent(event)
            return
        if event.key() == Qt.Key.Key_Escape:
            self.quad = None
            self.quad_changed.emit(None)
            self.update()
        elif self.quad and Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_4:
            self._selected_corner = event.key() - Qt.Key.Key_1
        elif self.quad and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            step = 1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 8
            dx = step * ((event.key() == Qt.Key.Key_Right) - (event.key() == Qt.Key.Key_Left))
            dy = step * ((event.key() == Qt.Key.Key_Down) - (event.key() == Qt.Key.Key_Up))
            points = list(self.quad)
            x, y = points[self._selected_corner]
            points[self._selected_corner] = (float(np.clip(x + dx, 0, 1279)), float(np.clip(y + dy, 0, 719)))
            self.quad = tuple(points)
            self.quad_changed.emit(self.quad)
            self.update()
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event) -> None:
        self.menu_hover.emit(None)
        super().leaveEvent(event)


class CameraWindow(QMainWindow):
    def __init__(self, *, test_camera=None, test_vision=None, store=None) -> None:
        typography_ready = configure_typography(strict=False)
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Carl-Cam")
        # Fonts and menu text are prepared on the UI thread. Workers only read
        # immutable rendered assets; they do not change application typography.
        artwork_ready = True
        for theme in ("red", "yellow", "blue"):
            try:
                asset_pixels("border-" + theme)
                menu_card(theme)
                menu_card(theme, True)
            except (OSError, ValueError, RuntimeError, cv2.error):
                artwork_ready = False
        self.store = store or Store()
        try:
            verified_svg('icon-check.svg')
        except (OSError, ValueError):
            artwork_ready = False
        self.diagnostics = LocalDiagnostics(self.store)
        self._diagnostic_states = set()
        self.settings, self.settings_warnings = self.store.load_settings()
        self.setStyleSheet(desktop_style(self.settings.high_contrast))
        self.resize(self.settings.window_width, self.settings.window_height)
        self.setMinimumSize(1024, 640)
        self.camera = test_camera
        self._test_camera = test_camera is not None
        self._test_vision = test_vision
        self.vision = None
        self.regions = None
        self.output = None
        self.mode = "camera"
        self.model = None
        self._current_camera_key = None
        self._camera_recheck_needed = False
        self.model_message = "Mouse/keyboard controls"
        self.extractor = FeatureExtractor()
        self.interaction = InteractionController(theme=self.settings.selected_theme, hold_ms=self.settings.hold_ms, assignments=self.settings.assignments)
        self.menu = MenuFollower()
        self.pointer = HandPointer()
        self.corner_smoother = OneEuro()
        self._last_perception = -1
        self._latest_input = None
        self._mouse_pointer = None
        self._manual_frame = False
        self.practice = None
        self.renderer: RenderWorker | None = None
        self._last_render = -2
        self.performance = PerformanceFallback()
        self._quality_pending_request = None
        self._quality_capture_barrier = -1
        self.theme = self.settings.selected_theme
        self.preview = Preview()
        self.preview.quad_changed.connect(self._set_quad)
        self.preview.menu_action.connect(self.dispatch_menu)
        self.preview.menu_hover.connect(lambda position: setattr(self, "_mouse_pointer", position))
        self.status = QLabel("Choose a camera, then press Start camera.")
        self.status.setWordWrap(True)
        self.segmentation_status = QLabel('Masking: not started')
        self.segmentation_status.setWordWrap(True)
        self.segmentation_status.setAccessibleName('Segmentation processor and fallback status')
        self.segmentation_status.hide()
        self.devices = QComboBox()
        self.devices.setAccessibleName("Camera device")
        self.devices.setMinimumWidth(240)
        self.devices.setMinimumContentsLength(16)
        self.devices.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.refresh_devices()
        self.refresh_devices_button = QPushButton("Refresh cameras")
        self.refresh_devices_button.clicked.connect(self.refresh_devices)
        self.start_button = QPushButton("Start camera")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton("Stop camera")
        self.stop_button.setEnabled(bool(self.camera and self.camera.running))
        self.stop_button.clicked.connect(self.stop_camera)
        self.quality_button = QPushButton('Try higher input quality')
        self.quality_button.setAccessibleName('Request preferred camera input quality')
        self.quality_button.setEnabled(False)
        self.quality_button.clicked.connect(self.restore_input_quality)
        self.frame_button = QPushButton("Mouse frame")
        self.frame_button.clicked.connect(self.enable_mouse_frame)
        self.clear_button = QPushButton("Clear frame")
        self.clear_button.clicked.connect(lambda: self._set_quad(None))
        self.menu_button = QPushButton("Open menu")
        self.menu_button.clicked.connect(self.open_menu)
        self.training_button = QPushButton("Train my gestures")
        self.training_button.clicked.connect(self.open_training)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        self.setup_button = QPushButton("Camera setup")
        self.setup_button.clicked.connect(self.open_setup)
        self.inspect_button = QPushButton("Inspect")
        self.inspect_button.clicked.connect(self.open_inspect)
        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.open_help)
        self.output_button = QPushButton("Start virtual camera")
        self.output_button.clicked.connect(self.toggle_output)
        self.output_status = QLabel("Virtual camera is off. Output is optional.")
        self.output_status.setWordWrap(True)
        themes = QComboBox()
        themes.setAccessibleName("Selected filter")
        themes.setMinimumWidth(100)
        for name in ("Red", "Yellow", "Blue"):
            themes.addItem(name, name.lower())
        themes.setCurrentIndex(themes.findData(self.theme))
        self.themes = themes
        themes.currentIndexChanged.connect(lambda: self._set_theme(themes.currentData()))
        themes.activated.connect(lambda: self._practice_observe(f"select:{themes.currentData()}"))
        for button, icon_name in ((self.start_button, 'camera'), (self.training_button, 'train'),
                                  (self.inspect_button, 'inspect')):
            try:
                verified_svg(f'icon-{icon_name}.svg')
                icon = QIcon(str(ROOT / f'icon-{icon_name}.svg'))
                if icon.pixmap(24, 24).isNull():
                    raise ValueError('Desktop icon is unreadable')
                button.setIcon(icon)
                button.setIconSize(QSize(24, 24))
            except (OSError, ValueError, RuntimeError):
                artwork_ready = False  # Keep readable text and recovery actions.
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        self.title_bar = TitleBar(self)
        layout.addWidget(self.title_bar)
        tools = QHBoxLayout()
        tools.setSpacing(8)
        for widget in (self.setup_button, self.training_button, self.inspect_button,
                       self.settings_button, self.output_button, self.help_button):
            tools.addWidget(widget)
        tools.addStretch()
        self.navigation_layout = tools
        layout.addLayout(tools)
        controls = QHBoxLayout()
        controls.setSpacing(8)
        for widget in (self.devices, self.start_button, self.stop_button, themes,
                       self.refresh_devices_button):
            controls.addWidget(widget)
        controls.addStretch()
        layout.addLayout(controls)
        framing = QHBoxLayout()
        framing.setSpacing(8)
        for widget in (self.menu_button, self.frame_button, self.clear_button, self.quality_button):
            framing.addWidget(widget)
        framing.addStretch()
        layout.addLayout(framing)
        self.practice_panel = QWidget()
        practice_layout = QHBoxLayout(self.practice_panel)
        self.practice_text = QLabel()
        self.practice_text.setWordWrap(True)
        self.practice_text.setAccessibleName("Guided control practice instructions")
        practice_layout.addWidget(self.practice_text, 1)
        leave_practice = QPushButton("Finish practice")
        leave_practice.clicked.connect(self.end_practice)
        practice_layout.addWidget(leave_practice)
        self.practice_panel.hide()
        layout.addWidget(self.practice_panel)
        layout.addWidget(self.preview, 1)
        self.asset_warning = QLabel('Some bundled graphics or the Carl-Cam font are unavailable. Desktop recovery controls remain usable; repair the installation for the finished artwork.')
        self.asset_warning.setWordWrap(True)
        self.asset_warning.setVisible(not (typography_ready and artwork_ready) or self.title_bar.artwork_missing)
        layout.addWidget(self.asset_warning)
        self.frame_instructions = QLabel("Mouse frame: drag a corner, or focus the preview and use 1–4 to choose a corner, then arrow keys. Escape clears it.")
        self.frame_instructions.setWordWrap(True)
        self.frame_instructions.hide()
        layout.addWidget(self.frame_instructions)
        layout.addWidget(self.status)
        layout.addWidget(self.segmentation_status)
        layout.addWidget(self.output_status)
        self.setCentralWidget(body)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(30)
        self._load_active_model()
        if not self._test_camera and not self.settings.setup_complete:
            QTimer.singleShot(0, self.open_setup)

    def refresh_devices(self) -> None:
        old = self.devices.currentData() if self.devices.count() else self.settings.camera_device
        self.devices.clear()
        devices = (CameraDevice(0, "Test camera"),) if self._test_camera else available_devices()
        self._camera_keys = {device.index: device.identity for device in devices}
        for device in devices:
            self.devices.addItem(device.name, device.index)
        if not devices:
            self.devices.addItem("No physical camera found — connect one and refresh", None)
        self.devices.setCurrentIndex(max(0, self.devices.findData(old)))

    def toggle_output(self) -> None:
        if self.output is not None and self.output.running:
            if self.output.stop():
                self.output = None
                self.output_button.setText("Start virtual camera")
                self.output_status.setText("Virtual camera is off")
            return
        if not self.camera or not self.camera.running or self.renderer is None:
            self.output_status.setText("Start a working camera before virtual output.")
            return
        from carl_cam.output import OutputWorker
        self.output = OutputWorker(self.renderer.results)
        self.output.start()
        self.output_button.setText("Stop virtual camera")

    def _load_active_model(self) -> None:
        self.model = None
        self._camera_recheck_needed = False
        self._diagnostic("model_invalid", False)
        if self.settings.active_model_kind == "personal" and self.settings.active_model_version:
            try:
                from carl_cam.model_store import ModelStore
                self.model = ModelStore(self.store).load_tested(self.settings.active_model_version)
                self.model_message = "Tested personal gestures ready"
                self.interaction.threshold, self.interaction.margin = self.model.threshold, self.model.margin
                if self._current_camera_key and not ModelStore(self.store).camera_verified(self.settings.active_model_version, self._current_camera_key):
                    self.model = None
                    self._camera_recheck_needed = True
                    self.model_message = "Camera changed — recheck saved gestures or use mouse/keyboard"
            except (OSError, ValueError, RuntimeError, KeyError, TypeError):
                self.model_message = "Saved gestures unavailable — retrain or use mouse/keyboard"
                self._diagnostic("model_invalid")
        else:
            self.model_message = "Mouse/keyboard controls — personal training available"
        self.training_button.setText("Recheck saved gestures" if self._camera_recheck_needed else "Train my gestures")

    def _set_theme(self, theme: str) -> None:
        if self.mode == "menu":
            self.dispatch_menu(theme)
            return
        self.theme = theme
        self.interaction.theme = theme
        self._publish_scene()
        self._practice_observe(f"select:{theme}")

    def _set_quad(self, quad) -> None:
        self._manual_frame = quad is not None
        self.frame_instructions.setVisible(self._manual_frame and self.mode in ("camera", "framed"))
        self.preview.quad = quad
        self.preview.update()
        self._publish_scene()
        self._practice_observe()

    def start_practice(self) -> None:
        if self.mode not in ("camera", "framed") or not self.camera or not self.camera.running:
            self.status.setText("Start a working camera and return to Camera before practicing.")
            return
        from carl_cam.practice import ControlPractice
        self.interaction.enter_mode("camera")
        self.mode = "camera"
        self._set_quad(None)
        self.practice = ControlPractice(self.settings.assignments, hands=self.model is not None)
        self.practice_panel.show()
        self.practice_text.setText(self.practice.instruction)
        self._publish_scene()

    def end_practice(self) -> None:
        self.practice = None
        self.practice_panel.hide()

    def _practice_observe(self, command=None) -> None:
        if self.practice is None:
            return
        status = self.camera.status.read() if self.camera else None
        if self.practice.observe(mode=self.mode, quad=self.preview.quad, theme=self.theme,
                                 command=command, manual=self._manual_frame,
                                 live=status is not None and status.state == "live"):
            self.practice_text.setText(self.practice.instruction)

    def _diagnostic(self, code, present=True):
        if not present:
            self._diagnostic_states.discard(code)
        elif code not in self._diagnostic_states:
            self._diagnostic_states.add(code)
            self.diagnostics.record(code)

    def _publish_scene(self) -> None:
        self.frame_instructions.setVisible(self._manual_frame and self.mode in ("camera", "framed"))
        self.menu_button.setText("Back to camera" if self.mode == "menu" else "Open menu")
        if self.themes.currentData() != self.theme:
            self.themes.blockSignals(True)
            self.themes.setCurrentIndex(self.themes.findData(self.theme))
            self.themes.blockSignals(False)
        if self.renderer:
            menu_layout = self.menu.layout if self.mode == "menu" else None
            focus = menu_layout.cards[self.preview.menu_focus_index % len(menu_layout.cards)][0] if menu_layout and menu_layout.cards else None
            pointer = (self._mouse_pointer or self.interaction.pointer) if menu_layout else None
            hover = menu_layout.hit(pointer) if menu_layout else None
            self.renderer.scene.publish(SceneState(self.theme, self.preview.quad, self.mode, menu_layout, pointer, hover, focus, self.settings.reduced_motion))

    def enable_mouse_frame(self) -> None:
        self._set_quad(((300.0, 150.0), (980.0, 150.0), (980.0, 570.0), (300.0, 570.0)))
        self.preview.setFocus()

    def start_camera(self) -> None:
        if self.camera and self.camera.running:
            return
        if self.camera is None:
            if self.devices.currentData() is None:
                self.status.setText("Connect a physical camera and press Refresh cameras. OBS Virtual Camera cannot be used as input.")
                return
            choice = QMessageBox.StandardButton.Yes if self.settings.privacy_accepted else QMessageBox.question(
                self, "Your camera stays on this computer",
                "Carl-Cam processes your camera live on this computer. It does not save or upload photos or video. "
                "When you train gestures later, it saves numerical hand measurements locally.\n\nStart the camera?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
            self.settings.privacy_accepted = True
            self._current_camera_key = self._camera_keys[self.devices.currentData()]
            self.settings = replace(self.settings, camera_device=self.devices.currentData(), camera_key=self._current_camera_key)
            try:
                self.store.save_settings(self.settings)
            except (OSError, ValueError):
                self.settings_warnings.append("Camera settings could not be saved; existing files were kept.")
            self.camera = CameraWorker(self.devices.currentData(), expected_identity=self._current_camera_key)
            self._load_active_model()
        self.performance.reset()
        self._quality_pending_request = None
        self._quality_capture_barrier = -1
        self.camera.start()
        if not self._test_camera or self._test_vision:
            self.vision = self._test_vision or VisionWorker(self.camera.frames)
            self.vision.start()
            if not self._test_camera:
                selected_backend = self.settings.region_backend
                self.regions = VisionWorker(self.camera.frames, regions=True,
                    factory=lambda: AdaptiveRegionEngine(selected_backend))
                self.regions.start()
                self.segmentation_status.setText('Masking: starting ' + selected_backend.upper())
                self.segmentation_status.show()
        self.renderer = RenderWorker(self.camera.frames,
                                     region_pairs=self.regions.completed if self.regions else None,
                                     region_status=self.regions.status if self.regions else None,
                                     perceptions=self.vision.results if self.vision else None)
        self.renderer.start()
        self._publish_scene()
        self.devices.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_camera(self) -> bool:
        if self.output and not self.output.stop():
            self._diagnostic("shutdown_delayed")
            self.output_status.setText("Still closing virtual output. Please try closing again.")
            return False
        self.output = None
        self.output_button.setText("Start virtual camera")
        self.output_status.setText("Virtual camera is off")
        if self.regions and not self.regions.stop():
            self._diagnostic("shutdown_delayed")
            self.status.setText("Still closing background masking. Please try closing again.")
            return False
        if self.vision and not self.vision.stop():
            self._diagnostic("shutdown_delayed")
            self.status.setText("Still closing local vision. Please try closing again.")
            return False
        if self.renderer and not self.renderer.stop():
            self._diagnostic("shutdown_delayed")
            self.status.setText("Still closing image processing. Please try closing again.")
            return False
        if self.camera and not self.camera.stop():
            self._diagnostic("shutdown_delayed")
            self.status.setText("Still closing the camera. Please try closing again.")
            return False
        self.renderer = None
        self.performance.reset()
        self._quality_pending_request = None
        self._quality_capture_barrier = -1
        self.quality_button.setEnabled(False)
        self._diagnostic("shutdown_delayed", False)
        self.end_practice()
        self.camera = None
        self.vision = None
        self.regions = None
        self.mode = "camera"
        self.interaction.enter_mode("camera")
        self.preview.menu_layout = None
        self._latest_input = None
        self._last_perception = -1
        self._manual_frame = False
        self.preview.quad = None
        self.devices.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.frame_instructions.hide()
        self.preview.set_rgb(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.status.setText("Camera stopped.")
        self.segmentation_status.setText('Masking: not started')
        self.segmentation_status.hide()
        return True

    def _request_input_quality(self, level):
        """Cancel old hand intent before the capture owner changes formats."""
        request = getattr(self.camera, 'request_quality_level', None)
        if request is None:
            return
        self._quality_pending_request = request(level)
        self.performance.changed(monotonic())
        self._latest_input = None
        self._last_perception = -1
        self.extractor = FeatureExtractor()
        self.corner_smoother.reset()
        self.pointer.reset()
        if self.mode == 'framed' and not self._manual_frame:
            self.mode = 'camera'
            self.preview.quad = None
        self.interaction.enter_mode(self.mode)
        self._publish_scene()

    def restore_input_quality(self):
        status = self.camera.status.read() if self.camera else None
        if (self.camera and self.camera.running and status and status.state == 'live'
                and self.mode in ('camera', 'framed', 'menu') and self._quality_pending_request is None):
            self._request_input_quality(0)

    def _observe_performance(self, result, camera_status):
        now = monotonic()
        capture = self.camera.frames.read() if self.camera else None
        if (self._quality_pending_request is not None and camera_status
                and camera_status.state == 'live'
                and camera_status.quality_request_id == self._quality_pending_request):
            self._quality_pending_request = None
            self._quality_capture_barrier = capture.frame_id if capture else -1
        enabled = bool(camera_status and camera_status.state == 'live' and capture
                       and 0 <= now - capture.capture_time_ns / 1e9 <= .200
                       and self.mode in ('camera', 'framed', 'menu')
                       and self._quality_pending_request is None
                       and hasattr(self.camera, 'request_quality_level'))
        reduced_input = bool(camera_status and (camera_status.quality_level > 0
                             or camera_status.source_size and (camera_status.source_size[0] < 1280
                                                               or camera_status.source_size[1] < 720)))
        self.quality_button.setEnabled(bool(enabled and reduced_input))
        level = camera_status.quality_level if camera_status else 0
        sequence = result.render_sequence if result and result.frame_id >= 0 else 0
        delay = ((result.render_time_ns - result.capture_time_ns) / 1_000_000
                 if result and result.frame_id >= 0 else None)
        change = self.performance.observe(now, sequence, delay, level, enabled=enabled)
        if change is not None:
            self._request_input_quality(change)

    def refresh(self) -> None:
        if self.regions:
            backend = self.regions.backend_status.read()
            if backend:
                self.segmentation_status.setText('Masking: ' + backend)
        if self.output:
            self.output_status.setText(self.output.status.read() or "Virtual camera is connecting…")
            self._diagnostic("output_unavailable", (self.output.status.read() or "").startswith("Output unavailable"))
            if not self.output.running:
                self.output_button.setText("Retry virtual camera")
        if not self.renderer:
            return
        self._update_live_controls()
        self._diagnostic("vision_unavailable", any(worker and (worker.status.read() or "").startswith("Vision unavailable") for worker in (self.vision, self.regions)))
        self._diagnostic("render_slow", (self.renderer.status.read() or "").startswith("Processing is too slow"))
        result = self.renderer.results.read()
        camera_status = self.camera.status.read() if self.camera else None
        self._observe_performance(result, camera_status)
        if result and monotonic_ns() - (result.capture_time_ns or result.render_time_ns) < 250_000_000:
            if result.frame_id != self._last_render:
                self.preview.set_rgb(result.preview_rgb)
                self._last_render = result.frame_id
        else:
            self.preview.set_rgb(np.zeros((720, 1280, 3), dtype=np.uint8))
        if camera_status:
            self._diagnostic("camera_unavailable", camera_status.state == "error")
            text = camera_status.message
            if camera_status.source_size:
                width, height = camera_status.source_size
                text += f" · {width}×{height} input · 1280×720 canvas · {camera_status.measured_fps:.0f} fps"
                if width < 1280 or height < 720:
                    text += " · Reduced input quality"
            if result and result.frame_id >= 0:
                text += f" · filter processing {result.render_ms:.0f} ms"
            if self.renderer.status.read():
                text += " · " + self.renderer.status.read()
            from carl_cam.vision_privacy import PRIVACY_HOLD
            if any(worker and worker.status.read() == 'Vision unavailable: ' + PRIVACY_HOLD
                   for worker in (self.vision, self.regions)):
                text += " · " + PRIVACY_HOLD
            if self.mode == "menu" and self.menu.layout.needs_space:
                text += " · Move farther from the camera to make space around your face. Use Selected filter or Back to camera meanwhile."
            self.status.setText(text)

    def open_training(self, *, force_collection=False) -> None:
        if not self.camera or not self.camera.running or self.vision is None:
            self.status.setText("Start a working camera first. Training also needs the local hand tracker.")
            return
        from carl_cam.vision_privacy import (
            PRIVACY_HOLD,
            VisionPrivacyError,
            require_verified_vision_runtime,
        )
        try:
            require_verified_vision_runtime()
        except VisionPrivacyError:
            self.status.setText(PRIVACY_HOLD)
            return
        from carl_cam.training_ui import TrainingDialog
        self.end_practice()
        self.mode = "training"
        self.interaction.enter_mode("training")
        self.preview.quad = None
        self.preview.menu_layout = None
        self._manual_frame = False
        self._publish_scene()

        recheck_version = self.settings.active_model_version if self._camera_recheck_needed and not force_collection else None
        dialog = TrainingDialog(self, recheck_version=recheck_version)
        dialog.exec()
        self.mode = "camera"
        self.interaction.enter_mode("camera")
        self.interaction.assignments = self.settings.assignments
        self._load_active_model()
        self._publish_scene()

    def open_settings(self) -> None:
        from carl_cam.settings_ui import SettingsDialog
        self.end_practice()
        self.mode = "settings"
        self.interaction.enter_mode("settings")
        self.preview.quad = None
        self.preview.menu_layout = None
        self._manual_frame = False
        self._publish_scene()
        dialog = SettingsDialog(self)
        dialog.exec()
        self.mode = "camera"
        self.interaction.enter_mode("camera")
        self._publish_scene()

    def open_setup(self) -> None:
        from carl_cam.setup_ui import SetupDialog
        self.end_practice()
        self.mode = "setup"
        self.interaction.enter_mode("setup")
        self.preview.quad = None
        self.preview.menu_layout = None
        self._manual_frame = False
        self._publish_scene()
        dialog = SetupDialog(self)
        dialog.exec()
        self.mode = "camera"
        self.interaction.enter_mode("camera")
        self._publish_scene()
        if dialog.practice_requested:
            self.start_practice()

    def open_menu(self) -> None:
        if self.mode == "menu":
            self.dispatch_menu("back")
            return
        if self.mode not in ("camera", "framed"):
            return
        self.interaction.dispatch("menu")
        self.mode = "menu"
        self.preview.quad = None
        self._manual_frame = False
        self.menu.reset()
        self.pointer.reset()
        self.preview.menu_layout = self.menu.layout
        self.preview.menu_focus_index = 0
        self.preview.setFocus()
        self._publish_scene()
        self._practice_observe("menu")

    def open_inspect(self) -> None:
        from carl_cam.inspection_ui import InspectionDialog
        self.end_practice()
        self.mode = "inspect"
        self.interaction.enter_mode("inspect")
        self.preview.menu_layout = None
        self._publish_scene()
        dialog = InspectionDialog(self)
        dialog.exec()
        self.mode = "camera"
        self.interaction.enter_mode("camera")
        if not self._manual_frame:
            self.preview.quad = None
        self._publish_scene()

    def open_help(self) -> None:
        from carl_cam.help_ui import HelpDialog
        dialog = HelpDialog(self)
        dialog.exec()
        if dialog.practice_requested:
            self.start_practice()

    def dispatch_menu(self, action: str) -> None:
        if self.mode != "menu":
            return
        self.interaction.dispatch(action)
        self.theme = self.interaction.theme
        self.mode = self.interaction.mode
        self.preview.menu_layout = None
        self._mouse_pointer = None
        self._publish_scene()
        self._practice_observe(action)

    def _update_live_controls(self) -> None:
        if self.mode in ("training", "settings", "setup", "inspect"):
            return
        now = monotonic()
        camera_status = self.camera.status.read() if self.camera else None
        if (self._quality_pending_request is not None
                or camera_status and camera_status.state != 'live'):
            self._latest_input = None
            return
        observation = self.vision.results.read() if self.vision else None
        fresh = (observation is not None and observation.frame_id >= self._quality_capture_barrier
                 and 0 <= now - observation.capture_time_ns / 1e9 <= 0.100)
        if fresh and observation.frame_id != self._last_perception:
            self._last_perception = observation.frame_id
            features = self.extractor.extract(observation.hands)
            quad = observed_quad(features)
            if quad:
                quad = tuple(tuple(float(value) for value in point) for point in self.corner_smoother.update(np.array(quad), observation.capture_time_ns / 1e9))
                if not validate_quad(quad, (1280, 720)):
                    quad = None
            elif not features.two_hand_stable:
                self.corner_smoother.reset()
            vector = features.vectors.get(self.model.width) if self.model else None
            scores = tuple(float(value) for value in self.model.scores(vector)[0]) if vector is not None else None
            pointer = self.pointer.update(features.hands, now) if self.mode == "menu" and self.model else None
            if pointer and pointer.owner_changed:
                self.interaction._cancel_pinch()
            self._latest_input = InputObservation(observation.capture_time_ns / 1e9, scores, quad,
                                                  pointer.position if pointer else None, pointer.pinch_ratio if pointer else None,
                                                  self.menu.layout.hit(pointer.position) if pointer else None,
                                                  features.two_hand_stable, released=not observation.hands)
        if self.mode == "menu":
            hand_position = self._latest_input.pointer if fresh and self._latest_input else None
            pointer = self._mouse_pointer or hand_position
            hover = self.menu.layout.hit(pointer) is not None
            head = observation.head_center if fresh else None
            face = Rect(*observation.head_box) if fresh and observation.head_box else None
            ratio = self._latest_input.pinch_ratio if fresh and self._latest_input else None
            self.menu.update(head, face, now, hover=hover, pinching=self.interaction.pinching or ratio is not None and ratio < 0.35)
            self.preview.menu_layout = self.menu.layout
        snapshot = self.interaction.update(self._latest_input if fresh else None, now)
        if snapshot.mode == "menu" and self.mode != "menu":
            self.mode = "menu"
            self.preview.quad = None
            self._manual_frame = False
            self.menu.reset()
            self.pointer.reset()
            if observation and observation.hands:
                eligible = []
                for hand in observation.hands:
                    single = FeatureExtractor().extract((hand,))
                    vector = single.vectors.get(self.model.width) if self.model else None
                    if vector is not None:
                        eligible.append((float(self.model.scores(vector)[0, 1]), hand))
                if eligible:
                    self.pointer.begin(max(eligible, key=lambda item: item[0])[1], now)
            self.preview.menu_layout = self.menu.layout
            self.preview.setFocus()
        elif self.mode == "menu" and snapshot.mode == "camera":
            self.mode = "camera"
            self.preview.menu_layout = None
            self._mouse_pointer = None
        elif not self._manual_frame:
            self.mode = snapshot.mode
            self.preview.quad = snapshot.quad
        self.theme = snapshot.theme
        self._publish_scene()
        self._practice_observe(snapshot.command)

    def closeEvent(self, event) -> None:
        if self.stop_camera():
            self.settings.selected_theme = self.theme
            self.settings.window_width = max(1024, self.width())
            self.settings.window_height = max(640, self.height())
            if not self._test_camera:
                try:
                    self.store.save_settings(self.settings)
                except (OSError, ValueError):
                    pass  # Closing must not erase incompatible settings or strand the user.
            event.accept()
        else:
            event.ignore()


def run() -> int:
    app = QApplication.instance() or QApplication([])
    window = CameraWindow()
    window.show()
    return app.exec()
