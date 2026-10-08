"""Exercise the source Qt/GPU/filter pipeline without a camera or OBS.

A procedural seated figure supplies live, independently paced frames. The app's
actual GPU child, hand/head worker, renderer and Qt refresh run together. This is
functional and timing evidence, not real-person recognition or visual fidelity.
"""

import argparse
import cProfile
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from time import monotonic, monotonic_ns, perf_counter
from uuid import uuid4

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import numpy as np
from preview_themes import synthetic_scene
from PySide6.QtWidgets import QApplication

from carl_cam import render
from carl_cam.camera import CameraStatus
from carl_cam.channels import LatestValue
from carl_cam.gpu_process import AdaptiveRegionEngine, LaunchSpec, development_launch
from carl_cam.records import FramePacket
from carl_cam.render import RenderWorker
from carl_cam.storage import Settings, Store
from carl_cam.ui import CameraWindow
from carl_cam.vision import VisionWorker


class ProceduralCamera:
    def __init__(self):
        self.frames, self.status = LatestValue(), LatestValue()
        self.running = False
        self.halt = Event()
        self.thread = None
        image, _, self.quad = synthetic_scene()
        image.flags.writeable = False
        self.image = image

    def start(self):
        self.running = True
        self.halt.clear()
        self.status.publish(CameraStatus('live', 'Generated test input', (1280, 720), 30))

        def produce():
            identifier = 0
            deadline = monotonic()
            while not self.halt.is_set():
                self.frames.publish(FramePacket(identifier, monotonic_ns(), self.image))
                identifier += 1
                deadline += 1 / 30
                self.halt.wait(max(0, deadline - monotonic()))

        self.thread = Thread(target=produce, name='CarlCamGeneratedCamera', daemon=True)
        self.thread.start()

    def stop(self, timeout=3):
        self.halt.set()
        if self.thread:
            self.thread.join(timeout)
        self.running = False
        self.frames.clear()
        return self.thread is None or not self.thread.is_alive()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opencv-threads', type=int, choices=(1, 2, 4, 8))
    parser.add_argument('--gpu-opencv-threads', type=int, choices=(1, 2, 4, 8))
    parser.add_argument('--profile-render', action='store_true')
    parser.add_argument('--stage-timing', action='store_true')
    parser.add_argument('--disable-hand-head', action='store_true',
                        help='Diagnostic comparison only; omits required controls/head tracking')
    parser.add_argument('--python-switch-ms', type=float, choices=(.5, 1., 2., 5.),
                        help='Process-local thread handoff experiment; does not change app defaults')
    args = parser.parse_args()
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    if args.python_switch_ms is not None:
        sys.setswitchinterval(args.python_switch_ms / 1000)
    output = ROOT / 'build' / ('gpu-desktop-pipeline-' + uuid4().hex)
    output.mkdir(exist_ok=False)
    stage_times = {}
    if args.stage_timing:
        for name in ('regions_from_masks', 'signal_stages', 'subject_tones',
                     'compose_palette', 'draw_border', 'canvas_rgb'):
            stage_times[name] = []
            original_function = getattr(render, name)

            def timed_function(*arguments, _original=original_function, _name=name, **keywords):
                started = perf_counter()
                try:
                    return _original(*arguments, **keywords)
                finally:
                    stage_times[_name].append((perf_counter() - started) * 1000)

            setattr(render, name, timed_function)
    if args.profile_render:
        original_run = RenderWorker._run

        def profiled_run(worker):
            profile = cProfile.Profile()
            try:
                profile.runcall(original_run, worker)
            finally:
                profile.dump_stats(str(output / 'render-profile.pstats'))

        RenderWorker._run = profiled_run
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'physical_camera_used': False, 'virtual_output_started': False,
              'generated_pixels_only': True, 'source_offscreen_qt': True,
              'native_application_verified': False, 'real_person_quality_verified': False,
              'live_fps_gate_passed': False, 'passed': False, 'themes': {}, 'checks': {}}
    report['parent_opencv_threads'] = cv2.getNumThreads()
    report['python_thread_switch_ms'] = sys.getswitchinterval() * 1000
    report['gpu_opencv_threads_override'] = args.gpu_opencv_threads
    report['render_profile_enabled'] = args.profile_render
    report['hand_head_disabled_for_diagnostic'] = args.disable_hand_head
    report['full_functional_scope'] = not args.disable_hand_head
    app = QApplication.instance() or QApplication([])
    store = Store(output / 'controlled-settings')
    store.save_settings(Settings(region_backend='gpu', setup_complete=True))
    camera = ProceduralCamera()
    window = CameraWindow(test_camera=camera, store=store)
    # Keep the generated input, but exercise the physical-start worker path.
    # CameraWorker is never instantiated and no physical device is opened.
    window._test_camera = False

    def pump(seconds):
        deadline = monotonic() + seconds
        while monotonic() < deadline:
            app.processEvents()
            Event().wait(.003)

    try:
        window.show()
        window.start_camera()
        if args.disable_hand_head:
            if not window.vision.stop():
                raise RuntimeError('Hand/head worker did not close before the diagnostic')
            window.vision = None
            window.renderer.perceptions = None
        if args.gpu_opencv_threads is not None:
            if not window.regions.stop():
                raise RuntimeError('Original region worker did not close before the experiment')
            launch = development_launch()
            if launch is None:
                raise ValueError('Reviewed GPU interpreter is unavailable')
            command = (*launch.command[:4],
                       str(ROOT / 'tools/gpu_service_thread_experiment.py'),
                       '--opencv-threads', str(args.gpu_opencv_threads))
            selected = LaunchSpec(command, launch.candidate, launch.cwd)
            window.regions = VisionWorker(camera.frames, regions=True,
                factory=lambda: AdaptiveRegionEngine('gpu', launcher=lambda: selected))
            window.renderer.region_pairs = window.regions.completed
            window.renderer.region_status = window.regions.status
            window.regions.start()
        deadline = monotonic() + 45
        while monotonic() < deadline:
            pump(.02)
            status = window.regions.backend_status.read() or ''
            if status.startswith('CPU'):
                raise ValueError('Generated desktop check fell back to CPU')
            if status.startswith('GPU — full FP32'):
                break
        else:
            raise TimeoutError('GPU desktop startup did not finish')
        report['backend_status'] = window.regions.backend_status.read()
        report['checks']['gpu_label_visible'] = window.segmentation_status.isVisible()
        if not args.disable_hand_head:
            report['checks']['actual_hand_head_ready'] = window.vision.status.read() == 'Local vision ready'
        window._set_quad(camera.quad)
        for theme in ('red', 'yellow', 'blue'):
            window._set_theme(theme)
            pump(.4)
            for values in stage_times.values():
                values.clear()
            seen, delays, processing = set(), [], []
            deadline = monotonic() + 3
            while monotonic() < deadline:
                pump(.01)
                result = window.renderer.results.read()
                if result is None or result.frame_id < 0 or result.render_sequence in seen:
                    continue
                seen.add(result.render_sequence)
                delays.append((result.render_time_ns - result.capture_time_ns) / 1e6)
                processing.append(result.render_ms)
                if (result.preview_rgb.shape != (720, 1280, 3)
                        or result.output_rgb.shape != (720, 1280, 3)
                        or result.preview_rgb.flags.writeable or result.output_rgb.flags.writeable):
                    raise ValueError('Desktop renderer produced an invalid frame')
            report['themes'][theme] = {
                'distinct_completed_renders': len(seen), 'observation_seconds': 3,
                'completion_hz': len(seen) / 3,
                'median_capture_to_render_ms': float(np.median(delays)) if delays else None,
                'p95_capture_to_render_ms': float(np.percentile(delays, 95)) if delays else None,
                'median_filter_ms': float(np.median(processing)) if processing else None,
                'qt_preview_present': window.preview._image is not None,
                'render_status': window.renderer.status.read(),
                'backend_status': window.regions.backend_status.read(),
            }
            if stage_times:
                report['themes'][theme]['inclusive_stage_wall_ms'] = {
                    name: {'samples': len(values), 'median_ms': float(np.median(values)),
                           'p95_ms': float(np.percentile(values, 95))}
                    for name, values in stage_times.items() if values}
        report['checks']['all_filters_completed'] = all(
            value['distinct_completed_renders'] >= 10 and value['qt_preview_present']
            and str(value['backend_status']).startswith('GPU — full FP32')
            for value in report['themes'].values())
        report['checks']['no_output_started'] = window.output is None
        # Public appearance cannot be inferred from this schematic subject.
        report['sources'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in ('tools/check_gpu_desktop_pipeline.py',
                                          'src/carl_cam/ui.py', 'src/carl_cam/render.py',
                                          'src/carl_cam/gpu_process.py')}
    finally:
        report['checks']['finite_window_shutdown'] = window.stop_camera()
        window.close()
        pump(.05)
        report['checks']['generated_producer_stopped'] = not camera.running
        report['passed'] = all(report['checks'].values()) and len(report['themes']) == 3
        (output / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Generated desktop GPU evidence: ' + str(output), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
