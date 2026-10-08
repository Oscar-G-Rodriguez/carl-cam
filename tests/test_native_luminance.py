"""Guard the original linear-light values and all downstream signal arrays."""

import numpy as np
import pytest

from carl_cam import dsp


def test_native_weighted_lookup_preserves_every_rgb8_color_bit_for_bit():
    image = np.zeros((256, 256, 3), np.uint8)
    image[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
    image[..., 2] = np.arange(256, dtype=np.uint8)[None, :]
    for red in range(256):
        image[..., 0] = red
        expected = dsp.linear_luminance(image, reference=True)
        result = dsp.linear_luminance(image)
        np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('view', ['normal', 'reversed', 'subsampled', 'one_pixel', 'one_row'])
def test_native_luminance_accepts_owned_and_noncontiguous_rgb_views_without_writing(view):
    image = np.random.default_rng(29).integers(0, 256, (29, 39, 3), np.uint8)
    selected = {'normal': image, 'reversed': image[:, ::-1],
                'subsampled': image[::2, ::3], 'one_pixel': image[:1, :1],
                'one_row': image[:1]}[view]
    before = image.copy()
    selected.flags.writeable = False
    actual = dsp.linear_luminance(selected)
    expected = dsp.linear_luminance(selected, reference=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)
    assert not np.shares_memory(actual, image) and actual.dtype == np.float32
    assert actual.shape == selected.shape[:2]


@pytest.mark.parametrize('shape', [(0, 12, 3), (12, 0, 3), (0, 0, 3)])
def test_empty_rgb_preserves_reference_shape(shape):
    image = np.empty(shape, np.uint8)
    assert dsp.linear_luminance(image).shape == shape[:2]
    assert dsp.linear_luminance(image, reference=True).shape == shape[:2]


@pytest.mark.parametrize('image', [None, np.zeros((3, 4), np.uint8),
                                    np.zeros((3, 4, 4), np.uint8),
                                    np.zeros((3, 4, 3), np.float32)])
def test_invalid_rgb_is_rejected(image):
    with pytest.raises(ValueError):
        dsp.linear_luminance(image)


def test_complete_signal_stages_remain_bit_identical_to_luminance_oracle(monkeypatch):
    function = dsp.linear_luminance
    image = np.random.default_rng(61).integers(0, 256, (180, 320, 3), np.uint8)
    actual = dsp.signal_stages(image)
    monkeypatch.setattr(dsp, 'linear_luminance', lambda rgb: function(rgb, reference=True))
    expected = dsp.signal_stages(image)
    for name in ('luminance', 'denoised', 'edge_magnitude'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
