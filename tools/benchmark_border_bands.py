"""Compare four contiguous border bands against full-box source-over.

The largest central zero-alpha rectangle found by two reductions needs no
arithmetic. Four nonoverlapping contiguous bands retain the exact original
source-over operations. The adopted implementation is checked against the
previous full-box blend; this tool does not alter saved user settings.
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
from carl_cam.dsp import premultiplied_border_over, premultiplied_rgb_over


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=181)
    args = parser.parse_args()
    if not 181 <= args.frames <= 601:
        raise ValueError('Compare at least a full animation cycle')
    output = ROOT / 'build' / ('border-bands-' + uuid4().hex)
    output.mkdir(exist_ok=False)
    image, _, base_quad = synthetic_scene()
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'camera_used': False, 'generated_pixels_only': True,
              'application_default_changed': True, 'all_pixels_identical': False,
              'geometry_changes_each_frame': True, 'frames_per_theme': args.frames,
              'warmup_frames': 12, 'themes': {},
              'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('tools/benchmark_border_bands.py',
                                       'src/carl_cam/graphics.py', 'src/carl_cam/dsp.py')}}
    current = graphics.premultiplied_border_over
    try:
        for theme in ('red', 'yellow', 'blue'):
            timings = {'reference': [], 'bands': []}
            mismatches = maximum = 0
            for frame in range(-12, args.frames):
                delta = 12 * np.sin(frame / 11)
                quad = tuple((x + delta * (-1 if index % 2 else 1), y + delta)
                             for index, (x, y) in enumerate(base_quad))
                variants = ('reference', 'bands') if frame % 2 else ('bands', 'reference')
                outputs = {}
                for variant in variants:
                    graphics.premultiplied_border_over = (premultiplied_rgb_over if variant == 'reference'
                                                         else premultiplied_border_over)
                    start = perf_counter()
                    outputs[variant] = graphics.draw_border(image, theme, quad, seconds=frame / 30)
                    if frame >= 0:
                        timings[variant].append((perf_counter() - start) * 1000)
                difference = np.abs(outputs['reference'].astype(np.int16)
                                    - outputs['bands'].astype(np.int16))
                mismatches += int(np.count_nonzero(difference))
                maximum = max(maximum, int(difference.max()))
            report['themes'][theme] = {
                'differing_channels': mismatches, 'max_channel_difference': maximum,
                'timings': {variant: {'median_ms': float(np.median(values)),
                                     'p95_ms': float(np.percentile(values, 95))}
                            for variant, values in timings.items()}}
    finally:
        graphics.premultiplied_border_over = current
        report['all_pixels_identical'] = len(report['themes']) == 3 and all(
            value['differing_channels'] == 0 for value in report['themes'].values())
        (output / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Paired band-border evidence: ' + str(output), flush=True)
    return 0 if report['all_pixels_identical'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
