"""Local Windows animation preference and keyboard-visible desktop styles."""

import sys

from carl_cam.graphics import ROOT, verified_svg

CONTROL_RADIUS = 8
CONTROL_DISABLED_TEXT = "#AEB3BD"
CONTROL_DISABLED_BORDER = "#6C7381"
CONTROL_PADDING_VERTICAL = 4
CONTROL_FOCUS_STROKE = 3
CONTROL_CHECKBOX_RADIUS = 4
CONTROL_BORDER = 2
CONTROL_ACCENT = "#4AC9E8"
CONTROL_FOCUS = "#F5C915"
DESKTOP_SURFACE = "#10121B"
DESKTOP_FOREGROUND = "#F1ECE2"
DESKTOP_CHROME = "#151823"


def os_reduced_motion() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        enabled = wintypes.BOOL()
        # SPI_GETCLIENTAREAANIMATION; read-only, no OS preference is changed.
        if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):
            return not bool(enabled.value)
    except (AttributeError, OSError):
        pass
    return False


def desktop_style(high_contrast: bool = False) -> str:
    surface = "#000000" if high_contrast else DESKTOP_SURFACE
    foreground = "#FFFFFF" if high_contrast else DESKTOP_FOREGROUND
    checkbox_indicator = ''
    try:
        verified_svg('icon-check.svg')
        checkbox_indicator = (
            'QCheckBox::indicator { width: 14px; height: 14px; '
            f'background: {surface}; border: 2px solid #4AC9E8; '
            f'border-radius: {CONTROL_CHECKBOX_RADIUS}px; }}'
            'QCheckBox::indicator:checked { '
            f'image: url("{(ROOT / "icon-check.svg").as_posix()}"); }}'
            'QCheckBox::indicator:disabled { '
            f'border-color: {CONTROL_DISABLED_BORDER}; }}'
        )
    except (OSError, ValueError):
        # Preserve Qt's native checked/unchecked indication during repair.
        pass
    return (
        f"QWidget {{ background-color: {surface}; color: {foreground}; }}"
        "QPushButton, QComboBox { min-height: 34px; "
        f"padding: {CONTROL_PADDING_VERTICAL}px 8px; border: 2px solid #4AC9E8; "
        f"border-radius: {CONTROL_RADIUS}px; }}"
        "QPushButton:hover, QComboBox:hover { border-color: #F16AA5; }"
        "QPushButton:focus, QComboBox:focus, QCheckBox:focus { "
        f"border: {CONTROL_FOCUS_STROKE}px solid #F5C915; }}"
        "QPushButton:disabled, QComboBox:disabled { "
        f"color: {CONTROL_DISABLED_TEXT}; border-color: {CONTROL_DISABLED_BORDER}; }}"
        "QLabel { background: transparent; } QProgressBar { border: 1px solid #4AC9E8; }"
        "QProgressBar::chunk { background: #4AC9E8; }"
        'QCheckBox { min-height: 44px; spacing: 8px; }'
        'QCheckBox:focus { min-height: 38px; }'
        f'QCheckBox:disabled {{ color: {CONTROL_DISABLED_TEXT}; }}'
        + checkbox_indicator
    )
