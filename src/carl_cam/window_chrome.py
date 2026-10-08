"""Accessible custom desktop chrome using the original Figma window icons."""

from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QWidget

from carl_cam.accessibility import CONTROL_RADIUS
from carl_cam.graphics import ROOT, verified_svg


class TitleBar(QWidget):
    def __init__(self, window, *, title="carl-cam", exit_window=None):
        super().__init__(window)
        self.window = window
        self.exit_window = exit_window
        self._exit_requested = False
        if exit_window is not None and hasattr(window, "finished"):
            window.finished.connect(lambda _result: self._finish_exit())
        self.setAccessibleName("Carl-Cam window title and controls")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(56)
        self.setStyleSheet(
            "TitleBar { background: #151823; }"
            "QLabel { color: #F1ECE2; font-weight: bold; }"
            "QPushButton { background: #10121B; border: 2px solid #4AC9E8; "
            "min-height: 0px; padding: 0px; "
            f"border-radius: {CONTROL_RADIUS}px; }}"
            "QPushButton:hover { background: #25162F; border-color: #F16AA5; }"
            "QPushButton:focus { border: 3px solid #F5C915; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        label = QLabel(title)
        label.setAccessibleName("Carl-Cam")
        layout.addWidget(label)
        layout.addStretch()
        self.buttons = {}
        self.artwork_missing = False
        for name, description, action in (
            ("minimize", "Minimize Carl-Cam", window.showMinimized),
            ("maximize", "Maximize Carl-Cam", self.toggle_maximize),
            ("close", "Close Carl-Cam", self.request_close),
        ):
            button = QPushButton()
            try:
                verified_svg(f'icon-{name}.svg')
                icon = QIcon(str(ROOT / f"icon-{name}.svg"))
                if icon.pixmap(24, 24).isNull():
                    raise ValueError('Window icon is unreadable')
                button.setIcon(icon)
            except (OSError, ValueError, RuntimeError):
                button.setText({'minimize': '_', 'maximize': '□', 'close': 'X'}[name])
                self.artwork_missing = True
            button.setIconSize(QSize(24, 24))
            button.setFixedSize(44, 44)
            button.setAccessibleName(description)
            button.setToolTip(description)
            button.clicked.connect(action)
            layout.addWidget(button)
            self.buttons[name] = button
        QApplication.instance().installEventFilter(self)

    def request_close(self):
        self._exit_requested = True
        self.window.close()
        self._finish_exit()

    def _finish_exit(self):
        if self._exit_requested and self.exit_window is not None and not self.window.isVisible():
            self._exit_requested = False
            self.exit_window.close()

    def toggle_maximize(self):
        if self.window.isMaximized():
            self.window.showNormal()
        else:
            self.window.showMaximized()
        description = "Restore Carl-Cam" if self.window.isMaximized() else "Maximize Carl-Cam"
        self.buttons["maximize"].setAccessibleName(description)
        self.buttons["maximize"].setToolTip(description)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.window.windowHandle():
            self.window.windowHandle().startSystemMove()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximize()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_F4
                and event.modifiers() & Qt.KeyboardModifier.AltModifier
                and isinstance(watched, QWidget) and watched.window() is self.window):
            self.request_close()
            return True
        if event.type() == QEvent.Type.Close and watched is self.window and self.exit_window is not None:
            # Native close requests carry exit intent. Qt can also issue an
            # internal non-spontaneous Close during reject/done; that is Back,
            # not authorization to exit the host. Explicit X/Alt+F4 set intent
            # before close, retaining it if a worker cannot stop immediately.
            if event.spontaneous():
                self._exit_requested = True
            QTimer.singleShot(0, self._finish_exit)
        if (event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and isinstance(watched, QWidget) and watched.window() is self.window
                and not self.window.isMaximized()):
            point = self.window.mapFromGlobal(event.globalPosition().toPoint())
            edges = Qt.Edge(0)
            if point.x() < 8:
                edges |= Qt.Edge.LeftEdge
            elif point.x() >= self.window.width() - 8:
                edges |= Qt.Edge.RightEdge
            if point.y() < 8:
                edges |= Qt.Edge.TopEdge
            elif point.y() >= self.window.height() - 8:
                edges |= Qt.Edge.BottomEdge
            handle = self.window.windowHandle()
            if edges and handle and handle.startSystemResize(edges):
                return True
        return super().eventFilter(watched, event)


def install_dialog_chrome(dialog, layout, host):
    """Back returns to Camera; the custom X closes the entire app orderly."""
    dialog.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    dialog.title_bar = TitleBar(dialog, title=dialog.windowTitle(), exit_window=host)
    layout.insertWidget(0, dialog.title_bar)
