"""Retain camera-off Qt renders without capture, SDK loading or personal data.

This is a visual layout check, not a Windows interaction or live-camera test.
The inert injected camera prevents device enumeration and first-run prompts.
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
from carl_cam.storage import Store
from carl_cam.typography import configure_typography
from carl_cam.ui import CameraWindow


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'build' / ('desktop-camera-review-' + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    configure_typography(strict=True)
    camera = SimpleNamespace(running=False, frames=LatestValue(), status=LatestValue(),
                             stop=lambda: True)
    window = CameraWindow(test_camera=camera, store=Store(evidence / 'isolated-data'))
    window.timer.stop()
    renders = []
    try:
        for width, height in ((1280, 800), (1024, 640)):
            window.resize(width, height)
            window.show()
            window.preview.setFocus()  # Review normal buttons; keyboard focus stays available.
            app.processEvents()
            image = window.grab()
            output = evidence / f'camera-off-{width}x{height}.png'
            if not image.save(str(output)):
                raise RuntimeError('Qt could not save the camera-off review')
            scene = window.preview.scene_rect()
            renders.append({'requested_size': [width, height],
                            'actual_size': [window.width(), window.height()],
                            'scene_rect': [scene.x(), scene.y(), scene.width(), scene.height()],
                            'body_font_pixels': app.font().pixelSize(),
                            'title_bar_height': window.title_bar.height(),
                            'render': output.name,
                            'render_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
    finally:
        window.close()
    sources = ('src/carl_cam/ui.py', 'src/carl_cam/accessibility.py',
               'src/carl_cam/window_chrome.py', 'src/carl_cam/typography.py',
               'src/carl_cam/assets/graphics/manifest.json')
    record = {'created_utc': datetime.now(UTC).isoformat(), 'camera_started': False,
              'vision_initialized': False, 'virtual_output_started': False,
              'personal_store_used': False, 'native_windows_verified': False,
              'figma_file': 'GOpu9lAhmkGMvEg8aUzi8q', 'figma_node': '26:48',
              'renders': renders,
              'source_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in sources}}
    (evidence / 'evidence.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(evidence)


if __name__ == '__main__':
    main()
