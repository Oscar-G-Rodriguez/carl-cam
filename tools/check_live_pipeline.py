"""Explicitly consented, source-only live smoke test with numerical evidence.

Never writes camera pixels, masks, landmarks, device names or exception text.
This does not certify visual quality, neural commands, native UI or OBS output.
"""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

import numpy as np


def summary(values):
    if not values:
        return {'samples': 0}
    return {'samples': len(values), 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'max_ms': float(max(values))}


class TimingSamples:
    """Retain timings/counts only; never retain worker packets or observations."""

    def __init__(self):
        self.seen = {name: -1 for name in ('render', 'hands', 'regions')}
        self.timings = {name: [] for name in ('render', 'hands', 'regions', 'capture_to_render')}
        self.stale_completions = 0

    def consume(self, rendered, hands, regions):
        for name, value in (('render', rendered), ('hands', hands), ('regions', regions)):
            if value is None or value.frame_id < 0 or value.frame_id <= self.seen[name]:
                continue
            self.seen[name] = value.frame_id
            if name == 'render':
                age = (value.render_time_ns - value.capture_time_ns) / 1_000_000
                self.timings[name].append(float(value.render_ms))
                self.timings['capture_to_render'].append(float(age))
                self.stale_completions += age > 250
            else:
                self.timings[name].append(float(value.inference_ms))

    def evidence(self, seconds):
        return {'observation_seconds': seconds,
                'observed_distinct_render_fps': len(self.timings['render']) / seconds,
                'stale_render_completions': self.stale_completions,
                'stage_ms': {name: summary(values) for name, values in self.timings.items()},
                'raw_ms': self.timings}


def source_inventory(root):
    paths = list((root / 'src/carl_cam').rglob('*.py'))
    paths += [root / 'uv.lock', root / 'tools/check_live_pipeline.py',
              root / 'src/carl_cam/assets/models/manifest.json']
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def run(seconds_per_theme):
    # Imports and camera opening occur only after main has checked opt-in.
    from carl_cam.vision_privacy import require_verified_vision_runtime
    require_verified_vision_runtime()
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication

    from carl_cam.camera import CameraWorker, available_devices
    from carl_cam.render import RenderWorker, SceneState
    from carl_cam.vision import VisionWorker

    root = Path(__file__).resolve().parents[1]
    directory = root / 'build' / ('live-pipeline-' + uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    before = source_inventory(root)
    report = {'schema_version': 1, 'started_utc': datetime.now(UTC).isoformat(),
              'explicit_camera_consent': True, 'source_only': True,
              'pixels_masks_landmarks_retained': False, 'virtual_output_started': False,
              'visual_quality_evaluated': False, 'neural_commands_evaluated': False,
              'native_executable_evaluated': False, 'preview_presentation_evaluated': False,
              'manual_quad': True, 'sources': before, 'themes': {}, 'workers_stopped': {}}
    print(f'Numerical-only camera evidence: {directory}', flush=True)
    app = QApplication.instance() or QApplication([])
    devices = available_devices()
    if len(devices) != 1:
        raise ValueError('Select exactly one physical camera before this developer check')
    camera = CameraWorker(devices[0].index, expected_identity=devices[0].identity)
    hands = VisionWorker(camera.frames)
    regions = VisionWorker(camera.frames, regions=True)
    renderer = RenderWorker(camera.frames, region_pairs=regions.completed,
                            region_status=regions.status, perceptions=hands.results)
    workers = {'capture': camera, 'hands': hands, 'regions': regions, 'render': renderer}
    quad = ((265, 111), (1010, 149), (981, 619), (278, 589))
    try:
        for worker in workers.values():
            worker.start()
        # Bounded model/capture warmup; do not count construction as throughput.
        deadline = monotonic() + 15
        while monotonic() < deadline:
            app.processEvents()
            status = camera.status.read()
            if status and status.state == 'error':
                raise RuntimeError('camera_error')
            if camera.frames.read() is not None and hands.results.read() is not None and regions.results.read() is not None:
                break
            sleep(.01)
        else:
            raise RuntimeError('warmup_timeout')
        status = camera.status.read()
        report['input_size'] = status.source_size
        for theme in ('red', 'yellow', 'blue'):
            renderer.scene.publish(SceneState(theme, quad, mode='framed'))
            # Allow a result from the old scene to drain before counting.
            sleep(.3)
            samples = TimingSamples()
            for name, channel in (('render', renderer.results), ('hands', hands.results),
                                  ('regions', regions.results)):
                previous = channel.read()
                samples.seen[name] = previous.frame_id if previous is not None else -1
            previous = None  # Do not keep an old frame packet during measurement.
            start = monotonic()
            while monotonic() - start < seconds_per_theme:
                app.processEvents()
                status = camera.status.read()
                if status and status.state == 'error':
                    raise RuntimeError('camera_error')
                samples.consume(renderer.results.read(), hands.results.read(), regions.results.read())
                sleep(.003)
            evidence = samples.evidence(monotonic() - start)
            report['themes'][theme] = evidence
            if not evidence['stage_ms']['render']['samples']:
                raise RuntimeError('no_render_results')
            print(json.dumps({'theme': theme, 'stage_ms': evidence['stage_ms'],
                              'observed_distinct_render_fps': evidence['observed_distinct_render_fps']}), flush=True)
        report['source_pipeline_observed'] = True
    except Exception as error:  # noqa: BLE001 - privacy-safe developer boundary
        report['error_type'] = type(error).__name__
        report['source_pipeline_observed'] = False
    finally:
        # Stop production before consumers. Do not hide an unclosed worker.
        for name in ('capture', 'render', 'hands', 'regions'):
            report['workers_stopped'][name] = bool(workers[name].stop())
        report['source_matches_initial_inventory'] = source_inventory(root) == before
        report['finished_utc'] = datetime.now(UTC).isoformat()
        report['smoke_check_passed'] = (report.get('source_pipeline_observed', False)
                                        and all(report['workers_stopped'].values())
                                        and report['source_matches_initial_inventory'])
        report['full_build_or_performance_gate_passed'] = False
        (directory / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['smoke_check_passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-consent', action='store_true',
                        help='Explicitly authorize this bounded local camera test; no images are saved')
    parser.add_argument('--seconds-per-theme', type=int, choices=(5, 10, 15, 30), default=10)
    arguments = parser.parse_args()
    if not arguments.camera_consent:
        parser.error('Explicit --camera-consent is required; camera remains off')
    return run(arguments.seconds_per_theme)


if __name__ == '__main__':
    raise SystemExit(main())
