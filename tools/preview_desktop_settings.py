"""Render all Settings groups with an inert camera and isolated local data.

No setting is saved, model loaded or deletion confirmed. The review records
actual Qt geometry and file hashes rather than simulating a successful setup.
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QPushButton

from carl_cam.channels import LatestValue
from carl_cam.settings_ui import SettingsDialog
from carl_cam.storage import Store
from carl_cam.typography import configure_typography
from carl_cam.ui import CameraWindow


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'build' / ('desktop-settings-review-' + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    configure_typography(strict=True)
    camera = SimpleNamespace(running=False, frames=LatestValue(), status=LatestValue(),
                             stop=lambda: True)
    host = CameraWindow(test_camera=camera, store=Store(evidence / 'isolated-data'))
    host.timer.stop()
    dialog = SettingsDialog(host)
    renders = []
    try:
        for width, height in ((1024, 700), (1024, 640)):
            dialog.resize(width, height)
            for index in range(dialog.tabs.count()):
                dialog.tabs.setCurrentIndex(index)
                dialog.show()
                app.processEvents()
                output = evidence / f'settings-group-{index + 1}-{width}x{height}.png'
                if not dialog.grab().save(str(output)):
                    raise RuntimeError('Qt could not save a Settings review')

                def bounds(widget):
                    origin = widget.mapTo(dialog, QPoint())
                    return [origin.x(), origin.y(), widget.width(), widget.height()]

                page = dialog.pages[index]
                renders.append({'requested_size': [width, height],
                    'actual_size': [dialog.width(), dialog.height()],
                    'group': dialog.tabs.tabText(index), 'tab_bounds': bounds(dialog.tabs),
                    'tab_targets': [[r.x(), r.y(), r.width(), r.height()]
                                    for r in (dialog.tabs.tabBar().tabRect(i)
                                              for i in range(dialog.tabs.count()))],
                    'content_bounds': bounds(page.widget()),
                    'save_bounds': bounds(dialog.save_button), 'back_bounds': bounds(dialog.back),
                    'message_visible': dialog.message.isVisible(),
                    'controls': [{'text': b.text(), 'bounds': bounds(b)}
                                 for b in page.findChildren(QPushButton)],
                    'horizontal_scroll_maximum': page.horizontalScrollBar().maximum(),
                    'vertical_scroll_maximum': page.verticalScrollBar().maximum(),
                    'render': output.name,
                    'render_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
        assert not camera.running and host.vision is None and host.renderer is None
        assert host.output is None and host.model is None
    finally:
        dialog.reject()
        host.close()
    sources = ('src/carl_cam/settings_ui.py', 'src/carl_cam/desktop_tabs.py', 'src/carl_cam/ui.py',
               'src/carl_cam/accessibility.py', 'src/carl_cam/window_chrome.py',
               'src/carl_cam/typography.py', 'src/carl_cam/assets/graphics/manifest.json',
               'tools/preview_desktop_settings.py')
    record = {'created_utc': datetime.now(UTC).isoformat(), 'camera_started': False,
              'vision_initialized': False, 'render_worker_started': False,
              'virtual_output_started': False, 'personal_store_used': False,
              'settings_saved': False, 'model_selected': False, 'deletion_confirmed': False,
              'native_windows_verified': False, 'renders': renders,
              'source_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in sources}}
    (evidence / 'evidence.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(evidence)


if __name__ == '__main__':
    main()
