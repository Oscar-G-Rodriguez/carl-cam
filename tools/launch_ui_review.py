"""Developer-only native window review with fresh, isolated local data.

Never uses a user's real settings/models and never starts capture or output.
The fresh fixture skips setup for window-control inspection only; privacy
consent stays false. It is not first-run or camera acceptance evidence.
"""

import argparse
from pathlib import Path

from PySide6.QtWidgets import QApplication

from carl_cam.storage import Settings, Store
from carl_cam.ui import CameraWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.data_root.absolute()
    if (target.exists() or not target.resolve().is_relative_to(root / 'build')
            or any(path.is_symlink() for path in (target, *target.parents))):
        raise ValueError('Use a new ordinary data directory inside Carl-Cam build/')
    store = Store(target)
    store.save_settings(Settings(setup_complete=True, privacy_accepted=False))
    app = QApplication([])
    window = CameraWindow(store=store)
    window.setWindowTitle('Carl-Cam — isolated interface review (camera off)')
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
