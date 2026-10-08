import cv2
import numpy as np
import pytest

from carl_cam import graphics
from carl_cam.dsp import palette_map, premultiplied_rgba_over, signal_stages
from carl_cam.geometry import polygon_mask
from carl_cam.render import SceneState, compose_palette
from carl_cam.themes import regions_from_masks


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("crop", (np.s_[:, :], np.s_[::2, 1::2]))
def test_reused_palette_scratch_matches_original_arithmetic_and_preserves_inputs(dtype, crop):
    from carl_cam.themes import PALETTES, REGION_TONES

    rng = np.random.default_rng(17)
    light = rng.uniform(-.1, 1.1, (91, 123)).astype(dtype)
    edges = rng.uniform(-.1, 1.2, light.shape).astype(dtype)
    light[0, :7] = (.24, .63, .215, .265, .605, .655, 0)
    edges[0, :7] = (.015, .0150001, 0, 1, .7, .25, 0)
    light.flags.writeable = edges.flags.writeable = False
    before_light, before_edges = light.copy(), edges.copy()
    cases = [(palette, .24, .63) for palette in PALETTES.values()]
    cases += [recipe for variants in REGION_TONES.values() for recipe in variants.values()]
    for (shadow, midtone, highlight), lower, upper in cases:
        y = light[crop].astype(np.float32)
        first = np.clip((y - lower) / .05 + .5, 0, 1)
        second = np.clip((y - upper) / .05 + .5, 0, 1)
        alpha = np.where(edges[crop] > .015, np.clip(.25 + edges[crop] * .70, .25, .7), 0)
        expected = np.empty((*y.shape, 3), np.uint8)
        for channel in range(3):
            tones = shadow[channel] + (midtone[channel] - shadow[channel]) * first
            tones += (highlight[channel] - tones) * second
            tones += (shadow[channel] - tones) * alpha
            expected[..., channel] = np.clip(tones, 0, 255).astype(np.uint8)
        actual = palette_map(light[crop], edges[crop], shadow, midtone, highlight,
                             lower=lower, upper=upper)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(light, before_light)
    np.testing.assert_array_equal(edges, before_edges)


@pytest.mark.parametrize("shape", ((1, 1), (2, 5), (120, 200)))
def test_nonmaximum_mask_copy_is_exact_on_noise_steps_and_flat_fields(shape):
    rng = np.random.default_rng(17)
    noise = rng.integers(0, 256, (*shape, 3), np.uint8)
    step = np.zeros_like(noise)
    step[:, shape[1] // 2:] = 255
    for pixels in (noise, step, np.full_like(noise, 127)):
        actual = signal_stages(pixels)
        reference = signal_stages(pixels, reference_suppression=True)
        for name in ("luminance", "denoised", "edge_magnitude"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))


@pytest.mark.parametrize("shape", ((1, 1), (2, 5), (120, 200), (360, 640)))
@pytest.mark.parametrize("strided", (False, True))
def test_owned_signal_scratch_matches_original_arithmetic_and_keeps_inspection_inputs(shape, strided):
    rng = np.random.default_rng(29)
    noise = rng.integers(0, 256, (*shape, 3), np.uint8)
    step = np.zeros_like(noise)
    step[:, shape[1] // 2:] = 255
    for pixels in (noise, step, np.full_like(noise, 127)):
        if strided:
            pixels = pixels[:, ::-1]
        pixels.flags.writeable = False
        before = pixels.copy()
        actual = signal_stages(pixels)
        expected = signal_stages(pixels, reference_scratch=True)
        for name in ("luminance", "denoised", "edge_magnitude"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        np.testing.assert_array_equal(pixels, before)
        assert not np.shares_memory(actual.luminance, actual.denoised)
        assert not np.shares_memory(actual.denoised, actual.edge_magnitude)


@pytest.mark.parametrize("alpha", (.46, .62, .9))
@pytest.mark.parametrize("vertices", (
    ((0, 0), (65, 0), (41, 59), (0, 46)),
    ((40, 0), (99, 12), (99, 54), (19, 44)),
    ((13, 10), (80, 19), (38, 44)),
    ((-20, -10), (50, -3), (25, 42), (-4, 30)),
    ((70, 35), (123, 47), (110, 83), (60, 66)),
    ((-40, 5), (-30, 17), (-35, 25)),
    ((120, 80), (130, 83), (126, 92)),
    ((0, 0), (99, 0), (99, 59), (0, 59)),
))
def test_bounded_translucent_polygon_is_exact_with_clipping_and_overlaps(alpha, vertices):
    from carl_cam.themes import _translucent_polygon

    rng = np.random.default_rng(17)
    local = rng.random((60, 100, 4), dtype=np.float32)
    local[..., :3] *= local[..., 3:4] * 255
    path = np.asarray(vertices, np.int32)
    path.flags.writeable = False
    original_path = path.copy()
    color = (35, 85, 207)
    for _ in range(2):
        layer = np.zeros_like(local)
        cv2.fillPoly(layer, [path], (*[c * alpha for c in color], alpha))
        reference = premultiplied_rgba_over(local, layer)
        _translucent_polygon(local, path, color, alpha)
        np.testing.assert_array_equal(local, reference)
    np.testing.assert_array_equal(path, original_path)


def test_sparse_red_fragments_match_uncropped_layer_reference():
    image = np.full((720, 1280, 3), (80, 130, 170), np.uint8)
    quad = ((310, 150), (960, 180), (990, 590), (280, 550))
    transform = cv2.getPerspectiveTransform(graphics.ANCHORS, np.asarray(quad, np.float32))
    for seconds in (0., .37, 1., 3.):
        overlay = np.zeros((720, 1280, 4), np.float32)
        for source, (dx, dy) in zip(graphics._red_paper_layers(), graphics.paper_offsets(seconds), strict=True):
            offset = np.array(((1, 0, dx), (0, 1, dy), (0, 0, 1)), np.float64)
            layer = cv2.warpPerspective(source, offset @ transform, (1280, 720))
            overlay = layer + overlay * (1 - layer[..., 3:4])
        reference = graphics._over(image, overlay)
        optimized = graphics.draw_border(image, 'red', quad, seconds=seconds)
        # OpenCV's fixed interpolation bins can differ at a few translated
        # boundaries; they cannot change solid colors or introduce a new layer.
        difference = np.abs(reference.astype(np.int16) - optimized.astype(np.int16))
        assert np.mean(difference) < .01
        assert np.count_nonzero(np.max(difference, axis=-1) > 1) < 150
        np.testing.assert_array_equal(optimized[250:450, 400:850], image[250:450, 400:850])


def test_prepared_regions_are_identical_and_native_blend_preserves_exterior():
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (720, 1280, 3), np.uint8)
    masks = np.zeros((360, 640, 6), np.float32)
    masks[..., 0] = 1
    masks[80:310, 180:400, 0] = 0
    masks[80:310, 180:400, 3] = 1
    state = SceneState('yellow', ((310, 150), (960, 180), (990, 590), (280, 550)))
    first, _ = compose_palette(image, state, masks=masks)
    prepared, _ = compose_palette(image, state, masks=masks, regions=regions_from_masks(masks, (360, 640)))
    np.testing.assert_array_equal(first, prepared)
    alpha = polygon_mask(state.quad, (1280, 720))
    np.testing.assert_array_equal(first[alpha == 0], image[alpha == 0])
    front = rng.integers(0, 256, (100, 150, 3), np.uint8)
    back = rng.integers(0, 256, front.shape, np.uint8)
    weights = rng.random(front.shape[:2], dtype=np.float32)
    old = cv2.convertScaleAbs(front * weights[..., None] + back * (1 - weights[..., None]))
    new = cv2.blendLinear(front, back, weights, 1 - weights)
    assert np.max(np.abs(old.astype(np.int16) - new.astype(np.int16))) <= 1


def test_native_gaussian_sobel_matches_written_reference_at_boundaries():
    rng = np.random.default_rng(17)
    pixels = rng.integers(0, 256, (120, 200, 3), np.uint8)
    native = signal_stages(pixels)
    reference = signal_stages(pixels, reference_convolution=True)
    np.testing.assert_array_equal(native.luminance, reference.luminance)
    np.testing.assert_allclose(native.denoised, reference.denoised, rtol=0, atol=2e-7)
    # Both paths keep the explicit Sobel kernels and one-pixel ridge policy.
    # Different float addition order may move a tie at a ridge boundary.
    assert np.mean(np.abs(native.edge_magnitude - reference.edge_magnitude)) < 1e-5


def test_native_float_layers_are_exact_and_do_not_mutate_strided_cached_inputs():
    rng = np.random.default_rng(17)
    for alpha in (None, 0., 1.):
        base = rng.random((80, 100, 4), dtype=np.float32)
        layer = rng.random(base.shape, dtype=np.float32)
        if alpha is not None:
            layer[..., 3] = alpha
        for source in (base, layer):
            source[..., :3] *= 255 * source[..., 3:4]
            source.flags.writeable = False
        for crop in (np.s_[:, :], np.s_[7:65, 8:73], np.s_[::2, ::2]):
            before_base, before_layer = base[crop].copy(), layer[crop].copy()
            reference = layer[crop] + base[crop] * (1 - layer[crop][..., 3:4])
            result = premultiplied_rgba_over(base[crop], layer[crop])
            np.testing.assert_array_equal(result, reference)
            np.testing.assert_array_equal(base[crop], before_base)
            np.testing.assert_array_equal(layer[crop], before_layer)


def test_all_theme_motifs_match_reference_float_composition(monkeypatch):
    from carl_cam import themes

    quad = ((155, 62), (480, 90), (495, 295), (140, 275))
    native = themes.premultiplied_rgba_over
    for name in ('red', 'yellow', 'blue'):
        for seconds in (0., .37, 1., 3., 6.):
            optimized = themes.motif_overlay(name, quad, (640, 360), seconds)
            monkeypatch.setattr(themes, 'premultiplied_rgba_over',
                lambda base, layer: layer + base * (1 - layer[..., 3:4]))
            reference = themes.motif_overlay(name, quad, (640, 360), seconds)
            monkeypatch.setattr(themes, 'premultiplied_rgba_over', native)
            np.testing.assert_array_equal(optimized, reference)


def test_fine_region_fast_path_is_exact_on_probabilities_ties_and_uncertainty():
    from carl_cam.themes import feather

    rng = np.random.default_rng(17)
    for random in (False, True):
        masks = rng.random((360, 640, 6), dtype=np.float32) if random else np.zeros((360, 640, 6), np.float32)
        if not random:
            masks[..., 0] = 1
        masks /= np.maximum(1e-6, masks.sum(axis=-1, keepdims=True))
        # Confident patches, exact ties and finer-region uncertainty coexist.
        for index in range(6):
            masks[30 + index * 30:60 + index * 30, 100:500] = 0
            masks[30 + index * 30:60 + index * 30, 100:500, index] = 1
        masks[250:270, 100:500] = (0., .2, .2, .2, .2, .2)
        masks[270:300, 100:500] = (0., .45, .55, 0., 0., 0.)
        result = regions_from_masks(masks, (360, 640))
        assert result.reliable
        winner = np.argmax(masks, axis=-1)
        for field, indices in (('hair', [1]), ('skin', [2, 3]), ('clothes', [4])):
            reference = feather((np.isin(winner, indices) & (masks[..., indices].sum(axis=-1) >= .55)).astype(np.float32))
            np.testing.assert_array_equal(getattr(result, field), reference)


@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.float16, np.uint8, np.int32))
@pytest.mark.parametrize('shape', ((1, 1), (9, 23), (360, 640)))
def test_category_reductions_preserve_sum_winner_ties_and_readonly_inputs(dtype, shape):
    from carl_cam.themes import _category_winner, _foreground_channels

    rng = np.random.default_rng(20261002)
    masks = rng.uniform(-.2, 1.2, (*shape, 6)).astype(dtype)
    masks = np.clip(masks, 0, 1)
    masks[0, 0] = .55 if np.issubdtype(dtype, np.floating) else 1
    for incoming in (masks, masks[:, ::-1]):
        before = incoming.copy()
        incoming.flags.writeable = False
        foreground, channels = _foreground_channels(incoming)
        expected_foreground = incoming[..., 1:].sum(axis=-1)
        np.testing.assert_array_equal(foreground, expected_foreground)
        assert foreground.dtype == expected_foreground.dtype
        winner = _category_winner(incoming, channels)
        np.testing.assert_array_equal(winner, incoming.argmax(axis=-1))
        np.testing.assert_array_equal(incoming, before)


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
@pytest.mark.parametrize('distribution', ('clipped', 'normalized', 'background', 'uncertain'))
def test_all_region_masks_match_preceding_reductions_exactly(dtype, distribution):
    rng = np.random.default_rng(20261002)
    masks = rng.uniform(-.2, 1.2, (180, 320, 6)).astype(dtype)
    if distribution == 'normalized':
        masks = np.clip(masks, 0, 1)
        masks /= np.maximum(1e-6, masks.sum(axis=-1, keepdims=True))
    elif distribution == 'background':
        masks[:] = (1, 0, 0, 0, 0, 0)
    elif distribution == 'uncertain':
        masks[:] = (.5, .1, .1, .1, .1, .1)
    # Place exact ties and adjacent float thresholds in otherwise varied input.
    masks[0, 0] = (0, .55, .55, 0, 0, 0)
    masks[0, 1] = (0, 0, .275, .275, 0, 0)
    masks[0, 2] = (0, .11, .11, .11, .11, .11)
    masks[0, 3] = (0, np.nextafter(dtype(.55), dtype(0)), 0, 0, 0, 0)
    masks[0, 4] = (0, np.nextafter(dtype(.55), dtype(1)), 0, 0, 0, 0)
    for incoming in (masks, masks[::2, ::-2]):
        incoming.flags.writeable = False
        before = incoming.copy()
        actual = regions_from_masks(incoming, incoming.shape[:2])
        reference = regions_from_masks(incoming, incoming.shape[:2], reference_reductions=True)
        assert actual.reliable == reference.reliable
        for field in ('person', 'hair', 'skin', 'clothes'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field))
        np.testing.assert_array_equal(incoming, before)


def test_unreliable_regions_do_not_compute_category_winners(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Unreliable segmentation should not compute winners')
    monkeypatch.setattr('carl_cam.themes._category_winner', forbidden)
    assert not regions_from_masks(np.full((9, 23, 6), .1, np.float32), (9, 23)).reliable
