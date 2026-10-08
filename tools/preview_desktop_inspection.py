"""Render the six-panel desktop inspection layout without live diagnostics.

Every panel stays black. This verifies layout using isolated settings; it does
not capture pixels, load vision models, start a render worker or broadcast.
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
from PySide6.QtWidgets import QApplication

from carl_cam.channels import LatestValue
from carl_cam.inspection_ui import InspectionDialog
from carl_cam.storage import Store
from carl_cam.typography import configure_typography
from carl_cam.ui import CameraWindow


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'build' / ('desktop-inspection-review-' + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    configure_typography(strict=True)
    camera = SimpleNamespace(running=False, frames=LatestValue(), status=LatestValue(),
                             stop=lambda: True)
    host = CameraWindow(test_camera=camera, store=Store(evidence / 'isolated-data'))
    host.timer.stop()
    dialog = InspectionDialog(host)
    dialog.timer.stop()
    dialog.safe_refresh()
    renders = []
    try:
        for width, height in ((1200, 780), (1024, 640)):
            dialog.resize(width, height)
            dialog.show()
            app.processEvents()
            output = evidence / f'inspection-off-{width}x{height}.png'
            if not dialog.grab().save(str(output)):
                raise RuntimeError('Qt could not save an inspection review')
            scenes = [p.scene_rect() for p in dialog.cells]
            def bounds(widget):
                origin = widget.mapTo(dialog, QPoint())
                return [origin.x(), origin.y(), widget.width(), widget.height()]
            renders.append({'requested_size': [width, height],
                            'actual_size': [dialog.width(), dialog.height()],
                            'panel_scenes': [[r.x(), r.y(), r.width(), r.height()] for r in scenes],
                            'panel_names': [label.text() for label in dialog.panel_labels],
                            'panel_bounds': [bounds(p.parentWidget()) for p in dialog.cells],
                            'label_bounds': [bounds(label) for label in dialog.panel_labels],
                            'preview_bounds': [bounds(p) for p in dialog.cells],
                            'metrics_bounds': bounds(dialog.metrics),
                            'explanation_bounds': bounds(dialog.explanation),
                            'compare_bounds': bounds(dialog.compare),
                            'back_bounds': bounds(dialog.back),
                            'compare_has_focus': dialog.compare.hasFocus(),
                            'render': output.name,
                            'render_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
        assert not camera.running and host.vision is None and host.renderer is None
        assert host.output is None and host.model is None
    finally:
        dialog.reject()
        host.close()
    sources = ('src/carl_cam/inspection_ui.py', 'src/carl_cam/ui.py',
               'src/carl_cam/accessibility.py', 'src/carl_cam/window_chrome.py',
               'src/carl_cam/typography.py', 'src/carl_cam/assets/graphics/manifest.json',
               'tools/preview_desktop_inspection.py')
    record = {'created_utc': datetime.now(UTC).isoformat(), 'camera_started': False,
              'vision_initialized': False, 'render_worker_started': False,
              'virtual_output_started': False, 'personal_store_used': False,
              'native_windows_verified': False, 'renders': renders,
              'source_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in sources}}
    (evidence / 'evidence.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(evidence)


if __name__ == '__main__':
    main()
