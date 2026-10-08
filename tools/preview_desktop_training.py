"""Camera-free layout evidence for the personal-training desktop screen.

Collection views create an empty, isolated workflow after fixture consent.
Started/paused collection views allocate an empty round, without observations.
Later phases and fitting failures are explicit presentation fixtures, not trained
model evidence. It never captures a camera, fits a model or activates commands.
These renders verify layout, not recognition or successful personal training.
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
from carl_cam.evaluation import Exercise
from carl_cam.storage import Store
from carl_cam.training_ui import TrainingDialog
from carl_cam.typography import configure_typography
from carl_cam.ui import CameraWindow


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'build' / ('desktop-training-review-' + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    configure_typography(strict=True)
    camera = SimpleNamespace(running=False, frames=LatestValue(), status=LatestValue(),
                             stop=lambda: True)
    host = CameraWindow(test_camera=camera, store=Store(evidence / 'isolated-data'))
    host.timer.stop()
    renders = []
    for mode in ('assignment', 'collection', 'review', 'fitting', 'comparison',
                 'movement', 'sealed', 'needs_examples', 'ready',
                 'rehearsal', 'recheck', 'pose_ready', 'scoring', 'cancel_pending',
                 'collection_started', 'collection_paused', 'movement_paused', 'fitting_error'):
        dialog = TrainingDialog(host)
        dialog.timer.stop()
        try:
            if mode in ('collection', 'collection_started', 'collection_paused', 'fitting_error'):
                dialog.consent.setChecked(True)  # Fixture-only consent in isolated storage.
                dialog.handle_action()  # Opens prompts; never starts a recording round.
                assert dialog.flow.phase == 'collection' and dialog.flow.current is None
                assert dialog.flow.rows() == []
                if not dialog.drawing.pixmap().save(str(evidence / 'frame-guide.png')):
                    raise RuntimeError('Qt could not export the approved pose guide')
                if mode in ('collection_started', 'collection_paused'):
                    dialog.handle_action()  # Allocates only an empty isolated round.
                    assert dialog.flow.current is not None and dialog.flow.current.rows == []
                    if mode == 'collection_paused':
                        dialog.safe_refresh()  # Actual no-camera pause; accepts no rows.
                elif mode == 'fitting_error':
                    dialog.flow.phase = 'fitting'
                    error = LatestValue()
                    error.publish('Fixture-only training failure. Your previous model is kept; try training again.')
                    dialog.flow.worker = SimpleNamespace(result=LatestValue(), error=error,
                                                         progress=LatestValue(), stop=lambda **_: True)
                    dialog._show_phase(force=True)
                    dialog.safe_refresh()  # Exercises the real returned-Review error presentation.
                    assert dialog.flow.phase == 'review' and dialog.flow.rows() == []
            elif mode != 'assignment':
                # Only exercise existing presentation logic. Never fit, score,
                # reveal a test, or activate a model to manufacture a screenshot.
                phase = ('comparison' if mode in ('movement', 'movement_paused', 'pose_ready', 'scoring')
                         else 'rehearsal' if mode in ('rehearsal', 'recheck')
                         else 'fitting' if mode == 'cancel_pending' else mode)
                candidate = SimpleNamespace(width=112, threshold=0.8, margin=0.15)
                dialog.flow = SimpleNamespace(
                    phase=phase, rows=list, stop=lambda **_: True,
                    exercise=Exercise((candidate,), attempts_per_pose=10 if mode == 'rehearsal' else 5),
                    winner=SimpleNamespace(name='Fixture-only candidate — not trained'),
                    action_reports=None, sealed_report=None, rehearsal_report=None,
                )
                for widget in (dialog.recording_choice, dialog.mapping_widget,
                               dialog.consent, dialog.consent_info):
                    widget.hide()
                if mode in ('movement', 'movement_paused'):
                    dialog._trial_index = 25
                elif mode == 'recheck':
                    dialog.recheck_version = 'fixture-not-a-loaded-model'
                dialog._show_phase(force=True)
                if mode in ('pose_ready', 'scoring'):
                    dialog.handle_action()  # Explicit formation prompt, not a scored observation.
                    if mode == 'scoring':
                        dialog.handle_action()  # Empty four-second fixture; timer remains stopped.
                elif mode == 'cancel_pending':
                    dialog.flow.stop = lambda **_: False  # Presentation only, no running worker.
                    dialog.reject()
                elif mode == 'movement_paused':
                    # Empty deliberate-attempt markers satisfy the exercise's
                    # sequencing guard. No predictions or successes are added.
                    for pose in ('frame', 'peace', 'thumbs_up', 'open_palm', 'ily'):
                        for _ in range(5):
                            dialog.flow.exercise.start_trial(pose, 0.0)
                            dialog.flow.exercise.finish_trial()
                    dialog.handle_action()
                    dialog.safe_refresh()
                    assert dialog._negative_started and dialog.flow.exercise.negative_seconds == 0
            for width, height in ((1180, 780), (1024, 640)):
                dialog.resize(width, height)
                dialog.show()
                app.processEvents()
                output = evidence / f'training-{mode}-{width}x{height}.png'
                if not dialog.grab().save(str(output)):
                    raise RuntimeError('Qt could not save a training review')
                scene = dialog.preview.scene_rect()

                def bounds(widget, current_dialog=dialog):
                    origin = widget.mapTo(current_dialog, QPoint())
                    return [origin.x(), origin.y(), widget.width(), widget.height()]

                renders.append({'mode': mode, 'requested_size': [width, height],
                                'actual_size': [dialog.width(), dialog.height()],
                                'camera_scene': [scene.x(), scene.y(), scene.width(), scene.height()],
                                'presentation_fixture': mode not in ('assignment', 'collection'),
                                'heading': dialog.heading.text(),
                                'instructions': dialog.instructions.text(),
                                'detail': dialog.detail.text(),
                                'back_bounds': bounds(dialog.back),
                                'left_scroll_bounds': bounds(dialog.left_scroll),
                                'right_scroll_bounds': bounds(dialog.right_scroll),
                                'controls': {name: {'text': widget.text(),
                                    'visible': widget.isVisible(), 'enabled': widget.isEnabled(),
                                    'bounds': bounds(widget)}
                                    for name, widget in (('action', dialog.action),
                                        ('depth', dialog.depth), ('cuda', dialog.cuda),
                                        ('cuda_info', dialog.cuda_info),
                                        ('retry_depth', dialog.retry_depth), ('redo', dialog.redo),
                                        ('skip', dialog.skip))},
                                'progress_range': [dialog.progress.minimum(), dialog.progress.maximum()],
                                'progress_value': dialog.progress.value(),
                                'waiting_for_pose_marker': dialog._waiting_marker,
                                'attempt_timer_started': dialog._trial_until is not None,
                                'close_pending': dialog._pending_done is not None,
                                'empty_round_allocated': mode in ('collection_started', 'collection_paused'),
                                'negative_exercise_started': dialog._negative_started,
                                'render': output.name,
                                'render_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
            assert not camera.running and host.vision is None and host.output is None
            assert host.model is None and not host.settings.setup_complete
        finally:
            if mode == 'cancel_pending':
                dialog.flow.stop = lambda **_: True
            dialog.reject()
    host.close()
    sources = ('src/carl_cam/training_ui.py', 'src/carl_cam/training_flow.py',
               'src/carl_cam/ui.py', 'src/carl_cam/accessibility.py',
               'src/carl_cam/window_chrome.py', 'src/carl_cam/typography.py',
               'src/carl_cam/assets/guides/gesture-sheet.png',
               'src/carl_cam/assets/graphics/manifest.json',
               'tools/preview_desktop_training.py')
    record = {'created_utc': datetime.now(UTC).isoformat(),
              'camera_started': False, 'vision_initialized': False,
              'virtual_output_started': False, 'personal_store_used': False,
              'numerical_examples_collected': 0, 'model_fitted': False,
              'model_activated': False, 'fixture_consent': True,
              'empty_rounds_allocated_without_observations': True,
              'late_phases_are_presentation_fixtures': True,
              'native_windows_verified': False, 'renders': renders,
              'source_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in sources}}
    (evidence / 'evidence.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(evidence)


if __name__ == '__main__':
    main()
