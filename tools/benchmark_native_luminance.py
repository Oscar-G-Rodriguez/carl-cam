"""Test native LUT/channel arithmetic against every RGB8 color and timed frames.

The transfer table and weighted sum retain their original float32 operation
order. No approximate color conversion or different visual recipe is introduced.
This developer-only experiment leaves application code/defaults unchanged.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from carl_cam import dsp

WEIGHTED_LUT = np.stack([dsp._LINEAR_LUT * weight for weight in (.2126, .7152, .0722)],
                        axis=-1).reshape(256, 1, 3)


def weighted_luminance(rgb):
    return cv2.transform(cv2.LUT(rgb, WEIGHTED_LUT), np.ones((1, 3), np.float32))


def native_luminance(rgb):
    channels = cv2.split(cv2.LUT(rgb, dsp._LINEAR_LUT))
    red, green, blue = channels
    for plane, coefficient in zip(channels, (.2126, .7152, .0722), strict=True):
        cv2.multiply(plane, float(np.float32(coefficient)), dst=plane)
    cv2.add(red, green, dst=red)
    cv2.add(red, blue, dst=red)
    return red


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=('split', 'weighted-transform'), default='split')
    args = parser.parse_args()
    selected = native_luminance if args.implementation == 'split' else weighted_luminance
    output = ROOT / 'build' / ('native-luminance-' + uuid4().hex)
    output.mkdir(exist_ok=False)
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'generated_pixels_only': True, 'camera_used': False,
              'application_default_changed': False, 'all_rgb8_colors_checked': False,
              'all_luminance_bits_identical': False, 'timings': {}, 'passed': False,
              'implementation': args.implementation,
              'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('tools/benchmark_native_luminance.py',
                                       'src/carl_cam/dsp.py')}}
    red_slice = np.zeros((256, 256, 3), np.uint8)
    red_slice[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
    red_slice[..., 2] = np.arange(256, dtype=np.uint8)[None, :]
    different = 0
    max_error = 0.
    for red in range(256):
        red_slice[..., 0] = red
        reference = dsp.linear_luminance(red_slice, reference=True)
        candidate = selected(red_slice)
        different += int(np.count_nonzero(reference.view(np.uint32) != candidate.view(np.uint32)))
        max_error = max(max_error, float(np.max(np.abs(reference - candidate))))
    report.update(all_rgb8_colors_checked=True, checked_rgb8_colors=256**3,
                  differing_luminance_bits=different, maximum_absolute_error=max_error,
                  all_luminance_bits_identical=different == 0)
    samples = {'noise': np.random.default_rng(23).integers(0, 256, (360, 640, 3), np.uint8),
               'gradient': np.repeat(np.linspace(0, 255, 640, dtype=np.uint8)[None, :, None],
                                     360, axis=0).repeat(3, axis=2)}
    for name, image in samples.items():
        image.flags.writeable = False
        timings = {'reference': [], 'native': []}
        mismatched_frames = 0
        for index in range(-12, 181):
            order = ('reference', 'native') if index % 2 else ('native', 'reference')
            results = {}
            for variant in order:
                start = perf_counter()
                results[variant] = (dsp.linear_luminance(image, reference=True)
                                    if variant == 'reference' else selected(image))
                if index >= 0:
                    timings[variant].append((perf_counter() - start) * 1000)
            mismatched_frames += int(not np.array_equal(results['reference'], results['native']))
        report['timings'][name] = {
            'mismatched_frames': mismatched_frames,
            'variants': {variant: {'median_ms': float(np.median(values)),
                                   'p95_ms': float(np.percentile(values, 95))}
                         for variant, values in timings.items()}}
    report['passed'] = report['all_luminance_bits_identical'] and all(
        item['mismatched_frames'] == 0 for item in report['timings'].values())
    (output / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Exact-color native luminance evidence: ' + str(output), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
