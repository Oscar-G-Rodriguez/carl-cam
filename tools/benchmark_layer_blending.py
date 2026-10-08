"""Paired, camera-free timing and exact-pixel checks over full motion cycles.

Runs paired source variants with the selected blend/allocation comparison.
It never claims camera/segmentation, Qt presentation or OBS measurements.
Alternating order prevents one implementation always getting the first slot.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
from preview_themes import synthetic_scene

from carl_cam import graphics, render, themes
from carl_cam.dsp import AdaptiveRange, premultiplied_rgba_over
from carl_cam.render import SceneState, compose_palette


def reference_layers(base, layer):
    return layer + base * (1 - layer[..., 3:4])


def reference_polygon(local, path, color, alpha):
    layer = np.zeros_like(local)
    cv2.fillPoly(layer, [path], (*[c * alpha for c in color], alpha))
    local[:] = themes.premultiplied_rgba_over(local, layer)


def reference_palette(luminance, edges, shadow, midtone, highlight, *,
                      lower=.24, upper=.63, softness=.05, ink_strength=.70):
    """The previous allocation-heavy arithmetic, including uint8 truncation."""
    y = luminance.astype(np.float32)
    first = np.clip((y - lower) / softness + .5, 0, 1)
    second = np.clip((y - upper) / softness + .5, 0, 1)
    alpha = np.where(edges > .015, np.clip(.25 + edges * ink_strength, .25, .7), 0)
    output = np.empty((*y.shape, 3), np.uint8)
    for channel in range(3):
        tones = shadow[channel] + (midtone[channel] - shadow[channel]) * first
        tones += (highlight[channel] - tones) * second
        tones += (shadow[channel] - tones) * alpha
        output[..., channel] = np.clip(tones, 0, 255).astype(np.uint8)
    return output


def reference_regions(masks, shape):
    empty = np.zeros(shape, np.float32)
    if masks is None or masks.shape != (*shape, 6) or not np.isfinite(masks).all():
        return themes.Regions(empty, empty, empty, empty, False)
    masks = np.clip(masks, 0, 1)
    probability = np.clip(masks[..., 1:].sum(axis=-1), 0, 1)
    coverage = np.mean(np.maximum(probability, masks[..., 0]) >= .55)
    if not (coverage >= .90 and np.mean(probability >= .55) >= .01):
        return themes.Regions(empty, empty, empty, empty, False)
    winner = np.argmax(masks, axis=-1)
    def region(indices):
        return themes.feather((np.isin(winner, indices)
            & (masks[..., indices].sum(axis=-1) >= .55)).astype(np.float32))
    return themes.Regions(themes.feather((probability >= .55).astype(np.float32)),
                          region([1]), region([2, 3]), region([4]), True)


def reference_tones(stages, luminance, theme, regions):
    tones = reference_palette(luminance, stages.edge_magnitude, *themes.PALETTES[theme])
    if not regions.reliable:
        return tones
    for name, (palette, lower, upper) in themes.REGION_TONES[theme].items():
        mask = getattr(regions, name)
        ys, xs = np.where(mask > 0)
        if not len(ys):
            continue
        area = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        mapped = reference_palette(luminance[area], stages.edge_magnitude[area], *palette,
                             lower=lower, upper=upper)
        tones[area] = cv2.blendLinear(mapped, tones[area], mask[area], 1 - mask[area])
    return tones


def summary(values):
    return {'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
            'max_ms': float(np.max(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=181)
    parser.add_argument('--opencv-threads', type=int, choices=(1, 2, 4, 8, 16))
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--allocation-only', action='store_true',
                        help='Compare palette/NMS/polygon allocation against the preceding source')
    scope.add_argument('--signal-scratch-only', action='store_true',
                       help='Compare only owned magnitude/angle scratch reuse against original arithmetic')
    scope.add_argument('--mask-reduction-only', action='store_true',
                       help='Compare only six-label confidence/winner reductions')
    arguments = parser.parse_args()
    if not 181 <= arguments.frames <= 601:
        parser.error('Use 181–601 measured frames to cover a full six-second cycle')
    original_threads = cv2.getNumThreads()
    if arguments.opencv_threads is not None:
        cv2.setNumThreads(arguments.opencv_threads)
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'build' / ('paired-blend-' + uuid4().hex)
    evidence.mkdir(parents=True)
    image, masks, quad = synthetic_scene()
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'comparison_scope': (['six_label_reductions'] if arguments.mask_reduction_only else
                  ['signal_magnitude_angle_scratch'] if arguments.signal_scratch_only else
                  ['palette_scratch', 'nonmaximum_mask_copy', 'bounded_translucent_polygon']
                  if arguments.allocation_only else ['float_RGBA_layers', 'fine_region_copies',
                      'fine_region_bounds', 'palette_scratch', 'nonmaximum_mask_copy',
                      'bounded_translucent_polygon', 'signal_magnitude_angle_scratch']),
              'physical_camera_used': False, 'virtual_output_used': False,
              'capture_and_inference_included': False, 'Qt_presentation_included': False,
              'hand_hull_included': False, 'schematic_subject_exact_masks': True,
              'native_executable_benchmarked': False,
              'frames_per_variant_per_theme': arguments.frames, 'warmup_frames': 12,
              'simulated_animation_span_seconds': (arguments.frames - 1) / 30,
              'alternating_variant_order': True, 'opencv_threads': cv2.getNumThreads(),
              'numpy_version': np.__version__, 'opencv_version': cv2.__version__,
              'all_pixels_identical': False, 'themes': {}, 'source_sha256': {}}
    for relative in ('src/carl_cam/dsp.py', 'src/carl_cam/themes.py',
                     'src/carl_cam/graphics.py', 'src/carl_cam/render.py',
                     'tools/preview_themes.py', 'tools/benchmark_layer_blending.py'):
        report['source_sha256'][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    print(f'Paired synthetic evidence: {evidence}', flush=True)
    (evidence / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    native_regions, native_tones = themes.regions_from_masks, render.subject_tones
    native_palette, native_signal = themes.palette_map, render.signal_stages
    native_polygon = themes._translucent_polygon
    try:
        for name in ('red', 'yellow', 'blue'):
            adaptive = {'reference': AdaptiveRange(), 'native': AdaptiveRange()}
            timings = {variant: {phase: [] for phase in ('regions', 'palette_motifs', 'border', 'total')}
                       for variant in adaptive}
            graphics.asset_pixels('border-' + name)
            mismatched_pixels, max_channel_difference = 0, 0
            for frame in range(-12, arguments.frames):
                seconds = frame / 30
                outputs = {}
                order = ('native', 'reference') if frame % 2 else ('reference', 'native')
                for variant in order:
                    blend = (premultiplied_rgba_over
                             if variant == 'native' or arguments.allocation_only or arguments.signal_scratch_only or arguments.mask_reduction_only else reference_layers)
                    graphics.premultiplied_rgba_over = themes.premultiplied_rgba_over = blend
                    region_function = (native_regions
                        if variant == 'native' or arguments.allocation_only or arguments.signal_scratch_only else reference_regions)
                    if arguments.mask_reduction_only:
                        region_function = partial(native_regions,
                            reference_reductions=variant == 'reference')
                    render.subject_tones = (native_tones
                        if variant == 'native' or arguments.allocation_only or arguments.signal_scratch_only or arguments.mask_reduction_only else reference_tones)
                    themes.palette_map = native_palette if variant == 'native' or arguments.signal_scratch_only or arguments.mask_reduction_only else reference_palette
                    themes._translucent_polygon = native_polygon if variant == 'native' or arguments.signal_scratch_only or arguments.mask_reduction_only else reference_polygon
                    render.signal_stages = (native_signal if variant == 'native' or arguments.mask_reduction_only else
                        lambda pixels: native_signal(pixels,
                            reference_suppression=not arguments.signal_scratch_only,
                            reference_scratch=arguments.signal_scratch_only or not arguments.allocation_only))
                    started = perf_counter()
                    regions = region_function(masks, (360, 640))
                    prepared = perf_counter()
                    pixels, _ = compose_palette(image, SceneState(name, quad), regions=regions,
                                                adaptive=adaptive[variant], seconds=seconds)
                    composed = perf_counter()
                    pixels = graphics.draw_border(pixels, name, quad, seconds=seconds)
                    completed = perf_counter()
                    if frame >= 0:
                        for phase, duration in (('regions', prepared - started),
                            ('palette_motifs', composed - prepared), ('border', completed - composed),
                            ('total', completed - started)):
                            timings[variant][phase].append(duration * 1000)
                    outputs[variant] = pixels
                difference = np.abs(outputs['native'].astype(np.int16) - outputs['reference'].astype(np.int16))
                mismatched_pixels += int(np.count_nonzero(np.max(difference, axis=-1)))
                max_channel_difference = max(max_channel_difference, int(np.max(difference)))
            report['themes'][name] = {
                'timings': {variant: {phase: summary(values) for phase, values in phases.items()}
                            for variant, phases in timings.items()},
                'raw_ms': timings, 'mismatched_pixels': mismatched_pixels,
                'maximum_channel_difference': max_channel_difference}
            for variant, pixels in outputs.items():
                cv2.imwrite(str(evidence / f'{name}-{variant}.png'), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            (evidence / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'theme': name, 'mismatched_pixels': mismatched_pixels,
                'total': {variant: summary(phases['total']) for variant, phases in timings.items()}}), flush=True)
        report['all_pixels_identical'] = all(item['mismatched_pixels'] == 0 for item in report['themes'].values())
    finally:
        graphics.premultiplied_rgba_over = themes.premultiplied_rgba_over = premultiplied_rgba_over
        render.subject_tones = native_tones
        themes.palette_map, render.signal_stages = native_palette, native_signal
        themes._translucent_polygon = native_polygon
        cv2.setNumThreads(original_threads)
        report['finished_utc'] = datetime.now(UTC).isoformat()
        report['source_matches_initial_inventory'] = all(hashlib.sha256((root / relative).read_bytes()).hexdigest() == checksum
            for relative, checksum in report['source_sha256'].items())
        (evidence / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['all_pixels_identical'] and report['source_matches_initial_inventory'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
