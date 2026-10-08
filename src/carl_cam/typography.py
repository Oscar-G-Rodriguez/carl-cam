"""Portable offline typography shared with the editable Carl-Cam design."""

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

FONT_FAMILY = "Arimo"


def configure_typography(*, strict: bool = True) -> bool:
    """Load our licensed font even when Qt's offscreen system database is empty."""
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("Typography requires the application to exist")
    if not app.property("carlCamFontLoaded"):
        path = Path(__file__).parent / "assets/fonts/Arimo.ttf"
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0 or FONT_FAMILY not in QFontDatabase.applicationFontFamilies(font_id):
            if strict:
                raise RuntimeError("The bundled Carl-Cam font is missing or unreadable")
            # Keep desktop recovery controls readable with Qt's system font.
            # Graphics requiring our licensed font still fail their strict
            # check and use the explicitly labeled recovery menu instead.
            return False
        font = QFont(FONT_FAMILY)
        font.setPixelSize(14)  # Match Figma's body/control style, independent of point rounding.
        app.setFont(font)
        app.setProperty("carlCamFontLoaded", True)
    return True
