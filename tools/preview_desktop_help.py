"""Render the real illustrated Help at both sizes with camera/output off."""

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
from carl_cam.help_ui import HelpDialog
from carl_cam.storage import Store
from carl_cam.training_ui import POSE_CROPS
from carl_cam.typography import configure_typography
from carl_cam.ui import CameraWindow


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'build' / ('desktop-help-review-' + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    configure_typography(strict=True)
    camera = SimpleNamespace(running=False, frames=LatestValue(), status=LatestValue(),
                             stop=lambda: True)
    host = CameraWindow(test_camera=camera, store=Store(evidence / 'isolated-data'))
    host.timer.stop()
    dialog = HelpDialog(host)
    renders = []
    guide_assets = {}
    try:
        for pose, image in dialog.guide_images.items():
            pixmap = image.pixmap()
            if pixmap is None or pixmap.isNull():
                raise RuntimeError('The owner-drawn Help guide is unavailable: ' + pose)
            target = evidence / ('guide-' + pose + '.png')
            if not pixmap.save(str(target)):
                raise RuntimeError('Qt could not export the unchanged Help crop: ' + pose)
            guide_assets[pose] = {'file': target.name,
                'size': [pixmap.width(), pixmap.height()], 'crop': list(POSE_CROPS[pose]),
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        for width, height in ((1024, 700), (1024, 640)):
            dialog.resize(width, height)
            dialog.show()
            app.processEvents()
            for position in ('instructions', 'illustrations'):
                bar = dialog.scroll.verticalScrollBar()
                bar.setValue(0 if position == 'instructions' else bar.maximum())
                app.processEvents()
                output = evidence / f'help-{position}-{width}x{height}.png'
                if not dialog.grab().save(str(output)):
                    raise RuntimeError('Qt could not save a Help review')

                def bounds(widget):
                    origin = widget.mapTo(dialog, QPoint())
                    return [origin.x(), origin.y(), widget.width(), widget.height()]

                renders.append({'requested_size': [width, height],
                    'actual_size': [dialog.width(), dialog.height()], 'position': position,
                    'runtime_notice_visible': dialog.runtime_notice.isVisible(),
                    'runtime_notice': dialog.runtime_notice.text(),
                    'horizontal_scroll_maximum': dialog.scroll.horizontalScrollBar().maximum(),
                    'practice_bounds': bounds(dialog.practice), 'back_bounds': bounds(dialog.back),
                    'notice_bounds': bounds(dialog.runtime_notice),
                    'scroll_bounds': bounds(dialog.scroll),
                    'body_bounds': bounds(dialog.scroll.widget()),
                    'vertical_scroll_value': bar.value(), 'vertical_scroll_maximum': bar.maximum(),
                    'sections': [{'title': heading.text(), 'heading_bounds': bounds(heading),
                        'paragraph': dialog.scroll.widget().layout().itemAt(2 * i + 1).widget().text(),
                        'paragraph_bounds': bounds(dialog.scroll.widget().layout().itemAt(2 * i + 1).widget())}
                        for i, heading in enumerate(dialog.headings)],
                    'guides': {pose: {'accessible_name': image.accessibleName(),
                                     'image_bounds': bounds(image),
                                     'cell_bounds': bounds(image.parentWidget()),
                                     'label_bounds': bounds(image.parentWidget().layout().itemAt(1).widget()),
                                     'size': [image.pixmap().width(), image.pixmap().height()]
                                     if image.pixmap() and not image.pixmap().isNull() else None}
                               for pose, image in dialog.guide_images.items()},
                    'render': output.name,
                    'render_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
        assert not camera.running and host.vision is None and host.renderer is None
        assert host.output is None and host.model is None and not dialog.practice_requested
    finally:
        dialog.reject()
        host.close()
    sources = ('src/carl_cam/help_ui.py', 'src/carl_cam/ui.py',
               'src/carl_cam/training_ui.py', 'src/carl_cam/accessibility.py',
               'src/carl_cam/window_chrome.py', 'src/carl_cam/typography.py',
               'src/carl_cam/vision_privacy.py', 'src/carl_cam/assets/guides/gesture-sheet.png',
               'src/carl_cam/assets/graphics/manifest.json', 'tools/preview_desktop_help.py')
    record = {'created_utc': datetime.now(UTC).isoformat(), 'camera_started': False,
              'vision_initialized': False, 'render_worker_started': False,
              'virtual_output_started': False, 'personal_store_used': False,
              'practice_requested': False, 'native_windows_verified': False, 'renders': renders,
              'guide_assets': guide_assets,
              'source_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in sources}}
    (evidence / 'evidence.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(evidence)


if __name__ == '__main__':
    main()
