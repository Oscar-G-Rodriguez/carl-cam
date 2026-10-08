"""Render all four setup views with an inert camera and isolated settings.

Direct view selection is a review fixture, not evidence that onboarding gates
were passed. No capture, inference, model collection or virtual output starts.
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication

from carl_cam.channels import LatestValue
from carl_cam.setup_ui import SetupDialog
from carl_cam.storage import Store
from carl_cam.typography import configure_typography
from carl_cam.ui import CameraWindow

FIGMA_NODES = ('34:147', '34:148', '34:149', '34:150')


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'build' / ('desktop-setup-review-' + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    configure_typography(strict=True)
    camera = SimpleNamespace(running=False, frames=LatestValue(), status=LatestValue(),
                             stop=lambda: True)
    host = CameraWindow(test_camera=camera, store=Store(evidence / 'isolated-data'))
    host.timer.stop()
    dialog = SetupDialog(host)
    dialog.timer.stop()
    renders = []
    try:
        for width, height in ((1024, 700), (1024, 640)):
            dialog.resize(width, height)
            for step, node in enumerate(FIGMA_NODES):
                dialog.stack.setCurrentIndex(step)  # Fixture-only selection; no advance().
                dialog.show_step()
                dialog.show()
                dialog.stack.currentWidget().setFocus()
                app.processEvents()
                output = evidence / f'setup-{step + 1}-{width}x{height}.png'
                if not dialog.grab().save(str(output)):
                    raise RuntimeError('Qt could not save a setup review')
                scene = dialog.preview.scene_rect() if step == 1 else None
                renders.append({'step': step + 1, 'figma_node': node,
                                'requested_size': [width, height],
                                'actual_size': [dialog.width(), dialog.height()],
                                'camera_scene': [scene.x(), scene.y(), scene.width(), scene.height()]
                                if scene else None, 'render': output.name,
                                'render_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
        assert not camera.running and host.output is None and host.model is None
        assert not host.settings.setup_complete
    finally:
        dialog.reject()
        host.close()
    sources = ('src/carl_cam/setup_ui.py', 'src/carl_cam/ui.py',
               'src/carl_cam/accessibility.py', 'src/carl_cam/window_chrome.py',
               'src/carl_cam/typography.py', 'src/carl_cam/assets/graphics/manifest.json')
    record = {'created_utc': datetime.now(UTC).isoformat(), 'camera_started': False,
              'vision_initialized': False, 'virtual_output_started': False,
              'personal_store_used': False, 'native_windows_verified': False,
              'onboarding_completed': False, 'fixture_direct_view_selection': True,
              'figma_file': 'GOpu9lAhmkGMvEg8aUzi8q', 'renders': renders,
              'source_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in sources}}
    (evidence / 'evidence.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(evidence)


if __name__ == '__main__':
    main()
