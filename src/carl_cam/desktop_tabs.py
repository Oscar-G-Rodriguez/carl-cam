"""Original Figma tab appearance, with Qt's native navigation and accessibility."""

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QTabBar

from carl_cam.accessibility import (
    CONTROL_ACCENT,
    CONTROL_BORDER,
    CONTROL_CHECKBOX_RADIUS,
    CONTROL_FOCUS,
    CONTROL_FOCUS_STROKE,
    CONTROL_RADIUS,
    DESKTOP_CHROME,
    DESKTOP_FOREGROUND,
    DESKTOP_SURFACE,
)

TAB_HEIGHT = 44
TAB_HORIZONTAL_PADDING = 16
TAB_SELECTED_BAR = 3
TAB_FOCUS_INSET = 4


class SettingsTabBar(QTabBar):
    """Paint only: do not replace Qt's keys, focus, hit testing or tab selection."""

    def __init__(self, parent=None, *, high_contrast=False):
        super().__init__(parent)
        self.high_contrast = high_contrast
        self.setExpanding(False)
        self.setAccessibleName("Settings groups")

    def tabSizeHint(self, index):
        width = (self.fontMetrics().horizontalAdvance(self.tabText(index))
                 + 2 * (TAB_HORIZONTAL_PADDING + CONTROL_BORDER))
        return QSize(width, TAB_HEIGHT)

    def minimumTabSizeHint(self, index):
        return self.tabSizeHint(index)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index in range(self.count()):
            selected = index == self.currentIndex()
            bounds = QRectF(self.tabRect(index))
            shape = bounds.adjusted(CONTROL_BORDER / 2, CONTROL_BORDER / 2,
                                    -CONTROL_BORDER / 2, -CONTROL_BORDER / 2)
            radius = CONTROL_RADIUS - CONTROL_BORDER / 2
            path = QPainterPath()
            path.moveTo(shape.left(), shape.bottom())
            path.lineTo(shape.left(), shape.top() + radius)
            path.arcTo(QRectF(shape.left(), shape.top(), 2 * radius, 2 * radius), 180, -90)
            path.lineTo(shape.right() - radius, shape.top())
            path.arcTo(QRectF(shape.right() - 2 * radius, shape.top(),
                              2 * radius, 2 * radius), 90, -90)
            path.lineTo(shape.right(), shape.bottom())
            path.closeSubpath()
            surface = ("#000000" if self.high_contrast else
                       DESKTOP_CHROME if selected else DESKTOP_SURFACE)
            foreground = "#FFFFFF" if self.high_contrast else DESKTOP_FOREGROUND
            painter.setBrush(QColor(surface))
            painter.setPen(QPen(QColor(CONTROL_ACCENT), CONTROL_BORDER))
            painter.drawPath(path)
            label = QRectF(bounds)
            if selected:
                painter.fillRect(QRectF(bounds.left() + CONTROL_BORDER,
                                        bounds.bottom() - CONTROL_BORDER - TAB_SELECTED_BAR,
                                        bounds.width() - 2 * CONTROL_BORDER, TAB_SELECTED_BAR),
                                 QColor(CONTROL_ACCENT))
                label.setHeight(label.height() - TAB_SELECTED_BAR)
            painter.setPen(QColor(foreground))
            painter.drawText(label, Qt.AlignmentFlag.AlignCenter, self.tabText(index))
            if selected and self.hasFocus():
                inset = TAB_FOCUS_INSET + CONTROL_FOCUS_STROKE / 2
                focus = bounds.adjusted(inset, inset, -inset, -inset)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(CONTROL_FOCUS), CONTROL_FOCUS_STROKE))
                radius = CONTROL_CHECKBOX_RADIUS - CONTROL_FOCUS_STROKE / 2
                painter.drawRoundedRect(focus, radius, radius)
