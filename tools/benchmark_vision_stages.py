"""Measure the unchanged vision adapters on generated pixels, never a webcam.

The timed wrapper delegates the exact native call without changing its inputs,
model, running mode or results. Schematic/noise data cannot establish usability.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns, perf_counter
from uuid import uuid4

import numpy as np
from benchmark_layer_blending import summary
from preview_themes import synthetic_scene

from carl_cam.records import FramePacket
from carl_cam.vision import MODEL_ROOT, HandHeadEngine, RegionEngine


class TimedSegmenter:
    def __init__(self, delegate):
        self.delegate = delegate
        self.elapsed_ms = 0.

    def segment_for_video(self, *arguments):
        started = perf_counter()
        result = self.delegate.segment_for_video(*arguments)
        self.elapsed_ms = (perf_counter() - started) * 1000
        return result

    def close(self):
        self.delegate.close()


def main():
    from carl_cam.vision_privacy import require_verified_vision_runtime
    require_verified_vision_runtime()
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'build' / ('synthetic-vision-' + uuid4().hex)
    evidence.mkdir(parents=True)
    image, _, _ = synthetic_scene()
    rng = np.random.default_rng(17)
    images = {'black': np.zeros_like(image), 'schematic': image,
              'noise': rng.integers(0, 256, image.shape, dtype=np.uint8)}
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'physical_camera_used': False, 'virtual_output_used': False,
              'generated_pixels_only': True, 'quality_evaluated': False,
              'native_executable_benchmarked': False, 'frames_per_fixture': 30,
              'warmup_frames': 5, 'sources': {}, 'models': {}, 'fixtures': {}}
    for relative in ('src/carl_cam/vision.py', 'src/carl_cam/camera.py',
                     'tools/benchmark_vision_stages.py', 'tools/preview_themes.py'):
        report['sources'][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    report['models'] = json.loads((MODEL_ROOT / 'manifest.json').read_text())['models']
    print(f'Unchanged synthetic vision evidence: {evidence}', flush=True)
    hand_engine = HandHeadEngine()
    region_engine = None
    try:
        region_engine = RegionEngine()
        timer = TimedSegmenter(region_engine.segmenter)
        region_engine.segmenter = timer
        index = 0
        for name, pixels in images.items():
            rows = {stage: [] for stage in ('hand_head', 'region_total', 'segmenter_call', 'region_adapter')}
            for frame in range(-5, 30):
                packet = FramePacket(index, monotonic_ns(), pixels)
                index += 1
                started = perf_counter()
                observation = hand_engine.process(packet)
                hand_ms = (perf_counter() - started) * 1000
                started = perf_counter()
                regions = region_engine.process(packet)
                region_ms = (perf_counter() - started) * 1000
                if (regions.confidence_masks.shape != (360, 640, 6)
                        or not np.isfinite(regions.confidence_masks).all()
                        or regions.frame_id != observation.frame_id):
                    raise ValueError('The measured adapter produced invalid results')
                if frame >= 0:
                    for stage, value in (('hand_head', hand_ms), ('region_total', region_ms),
                        ('segmenter_call', timer.elapsed_ms), ('region_adapter', region_ms - timer.elapsed_ms)):
                        rows[stage].append(value)
            report['fixtures'][name] = {'timings': {stage: summary(values) for stage, values in rows.items()},
                                        'raw_ms': rows}
            (evidence / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'fixture': name, 'timings': report['fixtures'][name]['timings']}), flush=True)
    finally:
        hand_engine.close()
        if region_engine is not None:
            region_engine.close()
        report['finished_utc'] = datetime.now(UTC).isoformat()
        (evidence / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
