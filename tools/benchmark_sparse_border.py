"""Compare sparse transparent-border blending with the current exact pixels.

Only nonzero-alpha pixels need source-over arithmetic. This developer experiment
gathers those pixels for narrow border overlays, retaining the same float32
native operations and conversion. It does not change the application defaults.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from preview_themes import synthetic_scene

from carl_cam import graphics
from carl_cam.dsp import premultiplied_rgb_over


def sparse_over(rgb, overlay):
    active = overlay[..., 3] != 0
    count = np.count_nonzero(active)
    if count > active.size * .4:
        return premultiplied_rgb_over(rgb, overlay)
    result = rgb.copy()
    if count:
        foreground = overlay[active].reshape(-1, 1, 4)
        background = rgb[active].reshape(-1, 1, 3)
        result[active] = premultiplied_rgb_over(background, foreground).reshape(-1, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=181)
    args = parser.parse_args()
    if not 181 <= args.frames <= 601:
        raise ValueError('Compare at least a full six-second animation cycle')
    output = ROOT / 'build' / ('sparse-border-' + uuid4().hex)
    output.mkdir(exist_ok=False)
    image, _, quad = synthetic_scene()
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'camera_used': False, 'generated_pixels_only': True,
              'application_default_changed': False, 'all_pixels_identical': False,
              'frames_per_theme': args.frames, 'warmup_frames': 12, 'themes': {},
              'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('tools/benchmark_sparse_border.py',
                                       'src/carl_cam/graphics.py', 'src/carl_cam/dsp.py')}}
    reference = graphics.premultiplied_border_over
    try:
        for theme in ('red', 'yellow', 'blue'):
            timings = {'reference': [], 'sparse': []}
            mismatches = 0
            maximum = 0
            for frame in range(-12, args.frames):
                variants = ('reference', 'sparse') if frame % 2 else ('sparse', 'reference')
                outputs = {}
                for variant in variants:
                    graphics.premultiplied_border_over = reference if variant == 'reference' else sparse_over
                    start = perf_counter()
                    outputs[variant] = graphics.draw_border(image, theme, quad, seconds=frame / 30)
                    if frame >= 0:
                        timings[variant].append((perf_counter() - start) * 1000)
                difference = np.abs(outputs['reference'].astype(np.int16)
                                    - outputs['sparse'].astype(np.int16))
                mismatches += int(np.count_nonzero(difference))
                maximum = max(maximum, int(difference.max()))
            report['themes'][theme] = {
                'differing_channels': mismatches, 'max_channel_difference': maximum,
                'timings': {variant: {'median_ms': float(np.median(values)),
                                     'p95_ms': float(np.percentile(values, 95))}
                            for variant, values in timings.items()}}
    finally:
        graphics.premultiplied_border_over = reference
        report['all_pixels_identical'] = len(report['themes']) == 3 and all(
            value['differing_channels'] == 0 for value in report['themes'].values())
        (output / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Paired sparse-border evidence: ' + str(output), flush=True)
    return 0 if report['all_pixels_identical'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
