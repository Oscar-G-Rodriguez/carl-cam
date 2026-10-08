"""Exact border composition, including dense/tilted fallback and occlusion."""

import numpy as np
import pytest

from carl_cam import graphics
from carl_cam.dsp import premultiplied_border_over, premultiplied_rgb_over


@pytest.mark.parametrize('shape', [(1, 1), (1, 40), (40, 1), (7, 7), (8, 8), (65, 129)])
@pytest.mark.parametrize('coverage', ['empty', 'ring', 'dense', 'center', 'offcenter'])
def test_bands_preserve_every_channel_and_never_modify_inputs(shape, coverage):
    rng = np.random.default_rng(19)
    rgb = rng.integers(0, 256, (*shape, 3), np.uint8)
    alpha = np.zeros(shape, np.float32)
    if coverage == 'dense':
        alpha[:] = rng.random(shape, dtype=np.float32)
    elif coverage == 'ring':
        alpha[:2] = alpha[-2:] = alpha[:, :2] = alpha[:, -2:] = .75
    elif coverage == 'center':
        alpha[shape[0] // 2, shape[1] // 2] = 1
    elif coverage == 'offcenter':
        alpha[:, :max(1, shape[1] // 5)] = .3
    overlay = np.empty((*shape, 4), np.float32)
    overlay[..., :3] = rng.random((*shape, 3), dtype=np.float32) * 255 * alpha[..., None]
    overlay[..., 3] = alpha
    rgb.flags.writeable = overlay.flags.writeable = False
    expected = premultiplied_rgb_over(rgb, overlay)
    actual = premultiplied_border_over(rgb, overlay)
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, rgb)
    assert not np.shares_memory(actual, overlay)


@pytest.mark.parametrize('theme', ['red', 'yellow', 'blue'])
@pytest.mark.parametrize('quad', [((310, 150), (960, 180), (990, 590), (280, 550)),
                                ((80, 50), (1150, 120), (1100, 640), (110, 620)),
                                ((20, 100), (700, 30), (900, 580), (110, 620))])
@pytest.mark.parametrize('reduced_motion', [False, True])
def test_actual_vectors_preserve_exact_pixels_with_moving_layers_and_hands(
        theme, quad, reduced_motion, monkeypatch):
    rng = np.random.default_rng(39)
    rgb = rng.integers(0, 256, (720, 1280, 3), np.uint8)
    hands = np.zeros((720, 1280), np.float32)
    hands[80:210, 170:360] = .6
    rgb.flags.writeable = hands.flags.writeable = False
    monkeypatch.setattr(graphics, 'premultiplied_border_over', premultiplied_rgb_over)
    expected = graphics.draw_border(rgb, theme, quad, hands, seconds=1.1,
                                    reduced_motion=reduced_motion)
    monkeypatch.setattr(graphics, 'premultiplied_border_over', premultiplied_border_over)
    actual = graphics.draw_border(rgb, theme, quad, hands, seconds=1.1,
                                  reduced_motion=reduced_motion)
    np.testing.assert_array_equal(actual, expected)
