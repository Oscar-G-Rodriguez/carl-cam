import numpy as np
import pytest

from carl_cam.dsp import GAUSSIAN_3, convolve3, linear_luminance, palette_map, signal_stages


def test_luminance_black_white_and_channels() -> None:
    rgb = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0]]], dtype=np.uint8)
    values = linear_luminance(rgb)[0]
    assert values[0] == pytest.approx(0)
    assert values[1] == pytest.approx(1)
    assert values[2] == pytest.approx(0.2126, abs=0.001)


def test_gaussian_preserves_constant_image() -> None:
    image = np.full((8, 9), 0.4, dtype=np.float32)
    assert np.allclose(convolve3(image, GAUSSIAN_3), image)


def test_sobel_emphasizes_step_edge() -> None:
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[:, 12:] = 255
    stages = signal_stages(rgb)
    assert stages.edge_magnitude[:, 11:13].mean() > 0.2
    assert stages.edge_magnitude[:, :7].max() == pytest.approx(0)


def test_palette_output_is_rgb8_and_dark_edges() -> None:
    light = np.full((5, 5), 0.8, dtype=np.float32)
    edges = np.zeros((5, 5), dtype=np.float32)
    edges[2, 2] = 1
    output = palette_map(light, edges, (0, 0, 0), (100, 10, 10), (250, 250, 240))
    assert output.shape == (5, 5, 3) and output.dtype == np.uint8
    assert output[2, 2, 0] < output[0, 0, 0]
