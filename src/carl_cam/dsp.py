"""Inspectable image-processing primitives used by all three graphic themes.

The implementation keeps the intermediate luminance, denoised luminance, and
edge strength available to the Inspection screen. Sobel convolution is written
out here so the operation can be studied and checked against SciPy/OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import cv2
import numpy as np

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T
GAUSSIAN_3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
_SRGB_VALUES = np.arange(256, dtype=np.float32) / 255
_LINEAR_LUT = np.where(_SRGB_VALUES <= .04045, _SRGB_VALUES / 12.92,
                       ((_SRGB_VALUES + .055) / 1.055) ** 2.4).astype(np.float32)
_WEIGHTED_LINEAR_LUT = np.stack(
    [_LINEAR_LUT * weight for weight in (.2126, .7152, .0722)], axis=-1).reshape(256, 1, 3)
_SUM_LINEAR_CHANNELS = np.ones((1, 3), np.float32)
_WEIGHTED_LINEAR_LUT.flags.writeable = _SUM_LINEAR_CHANNELS.flags.writeable = False


@dataclass(frozen=True, slots=True)
class SignalStages:
    luminance: np.ndarray
    denoised: np.ndarray
    edge_magnitude: np.ndarray


def premultiplied_rgba_over(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Combine premultiplied float RGBA without strided broadcast temporaries.

    Both colors use RGB255 and alpha [0, 1]. Explicit native multiply then add
    keeps the same float operation order as layer + base * (1 - layer.alpha).
    The inputs are never modified, including read-only cached Figma fragments.
    """
    if base.shape != layer.shape or base.ndim != 3 or base.shape[-1] != 4:
        raise ValueError('Premultiplied RGBA layers must have the same shape')
    if base.dtype != np.float32 or layer.dtype != np.float32:
        raise ValueError('Premultiplied RGBA layers must be float32')
    opacity = 1 - layer[..., 3]
    weights = cv2.merge((opacity, opacity, opacity, opacity))
    cv2.multiply(base, weights, dst=weights)
    cv2.add(layer, weights, dst=weights)
    return weights


def premultiplied_rgb_over(rgb: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Native source-over with premultiplied RGB255 and alpha in [0, 1].

    Separate native multiply/add preserve the reference float operation order.
    Explicit channel packing avoids NumPy's repeated strided RGB broadcasting
    across a large transparent border or decoration plane.
    """
    opacity = 1 - overlay[..., 3]
    weights = cv2.merge((opacity, opacity, opacity))
    cv2.multiply(rgb, weights, dst=weights, dtype=cv2.CV_32F)
    colors = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)
    cv2.add(weights, colors, dst=weights)
    return cv2.convertScaleAbs(weights)


def _central_zero_run(active: np.ndarray, center: int):
    if active[center]:
        return None
    before = np.flatnonzero(active[:center])
    after = np.flatnonzero(active[center:])
    return (int(before[-1]) + 1 if before.size else 0,
            center + int(after[0]) if after.size else len(active))


def premultiplied_border_over(rgb: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Blend a premultiplied border while leaving a proven clear center alone.

    The input comes from interpolated premultiplied Figma vectors, so zero
    alpha also means zero source RGB. Two reductions find a rectangular clear
    area; four nonoverlapping contiguous bands use the ordinary exact blend.
    Dense or severely tilted borders fall back to the full-box operation.
    No geometry, interpolation, animation or occlusion recipe changes.
    """
    height, width = rgb.shape[:2]
    if min(height, width) < 8:
        return premultiplied_rgb_over(rgb, overlay)
    active = overlay[..., 3] != 0
    horizontal = _central_zero_run(
        np.any(active[height // 4:3 * height // 4], axis=0), width // 2)
    if horizontal is None:
        return premultiplied_rgb_over(rgb, overlay)
    left, right = horizontal
    vertical = _central_zero_run(np.any(active[:, left:right], axis=1), height // 2)
    if vertical is None:
        return premultiplied_rgb_over(rgb, overlay)
    top, bottom = vertical
    if (right - left) * (bottom - top) < height * width // 4:
        return premultiplied_rgb_over(rgb, overlay)
    result = rgb.copy()
    for area in (np.s_[:top, :], np.s_[bottom:, :],
                 np.s_[top:bottom, :left], np.s_[top:bottom, right:]):
        if rgb[area].size:
            result[area] = premultiplied_rgb_over(rgb[area], overlay[area])
    return result


def linear_luminance(rgb: np.ndarray, *, reference: bool = False) -> np.ndarray:
    """Convert sRGB8 to relative linear-light luminance in [0, 1]."""
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3
            or rgb.shape[-1] != 3 or rgb.dtype != np.uint8):
        raise ValueError("Expected H×W×3 uint8 RGB image")
    if not rgb.size:
        return np.empty(rgb.shape[:2], np.float32)
    if not reference:
        # Weight each of the 256 linear-light values once, using the original
        # float32 multiplication. LUT selects the R/G/B-specific table, then
        # transform sums the already weighted channels in the original order.
        # Exhaustive RGB8 tests guard the exact values, not just visual closeness.
        return cv2.transform(cv2.LUT(rgb, _WEIGHTED_LINEAR_LUT), _SUM_LINEAR_CHANNELS)
    # Retain the explicit NumPy recipe as the teaching/numerical oracle.
    return (
        0.2126 * _LINEAR_LUT[rgb[..., 0]]
        + 0.7152 * _LINEAR_LUT[rgb[..., 1]]
        + 0.0722 * _LINEAR_LUT[rgb[..., 2]]
    ).astype(np.float32)


def convolve3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3×3 edge-replicated convolution without hiding the filter kernel."""
    if image.ndim != 2 or kernel.shape != (3, 3):
        raise ValueError("Expected 2D image and 3×3 kernel")
    padded = np.pad(image.astype(np.float32), 1, mode="edge")
    output = np.zeros(image.shape, dtype=np.float32)
    for row in range(3):
        for col in range(3):
            output += kernel[row, col] * padded[
                row : row + image.shape[0], col : col + image.shape[1]
            ]
    return output


def signal_stages(rgb: np.ndarray, *, reference_convolution: bool = False,
                  reference_suppression: bool = False,
                  reference_scratch: bool = False) -> SignalStages:
    luminance = linear_luminance(rgb)
    # Keep the written-out convolution as an inspectable numerical oracle.
    # Production uses the same explicit kernels/boundary rule in OpenCV's
    # native implementation, avoiding 27 large temporary array operations.
    kernel_filter = convolve3 if reference_convolution else lambda image, kernel: cv2.filter2D(
        image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE)
    denoised = kernel_filter(luminance, GAUSSIAN_3)
    gradient_x = kernel_filter(denoised, SOBEL_X)
    gradient_y = kernel_filter(denoised, SOBEL_Y)
    magnitude = np.hypot(gradient_x, gradient_y)
    # A step from black to white yields roughly four units with this kernel.
    if reference_scratch:
        magnitude = np.clip(magnitude / 4, 0, 1).astype(np.float32)
    else:
        # hypot owns this float32 array; retain its division/clip order while
        # avoiding three full working-canvas allocations. No stage input is
        # reused as scratch, so Inspection retains the original luminance.
        np.divide(magnitude, 4, out=magnitude)
        np.clip(magnitude, 0, 1, out=magnitude)
    # Nonmaximum suppression selects a one-pixel ridge across the gradient.
    # At the half-resolution working canvas this is a two-pixel contour.
    angle = np.arctan2(gradient_y, gradient_x)
    if reference_scratch:
        angle = (np.rad2deg(angle) + 180) % 180
    else:
        np.rad2deg(angle, out=angle)
        np.add(angle, 180, out=angle)
        np.remainder(angle, 180, out=angle)
    padded = np.pad(magnitude, 1, mode="edge")
    thin = np.zeros_like(magnitude)
    for lower, upper, first, second in (
        (0, 22.5, padded[1:-1, :-2], padded[1:-1, 2:]),
        (22.5, 67.5, padded[:-2, :-2], padded[2:, 2:]),
        (67.5, 112.5, padded[:-2, 1:-1], padded[2:, 1:-1]),
        (112.5, 157.5, padded[:-2, 2:], padded[2:, :-2]),
        (157.5, 180, padded[1:-1, :-2], padded[1:-1, 2:]),
    ):
        ridge = (angle >= lower) & (angle < upper) & (magnitude >= first) & (magnitude > second)
        if reference_suppression:
            thin[ridge] = magnitude[ridge]
        else:
            # Copy under the same mask without gathering a temporary vector.
            # Keep the indexed version as a numerical/performance oracle.
            np.copyto(thin, magnitude, where=ridge)
    return SignalStages(luminance, denoised, thin)


class AdaptiveRange:
    """Time-based percentile EMA; lighting changes cannot remap a face instantly.

    Percentiles 10/90 and a 750 ms time constant are implementation starting
    values, exposed in diagnostics rather than advertised as measured quality.
    A minimum 0.20 range avoids amplifying a nearly constant dim scene.
    """

    def __init__(self) -> None:
        self.bounds: tuple[float, float] | None = None
        self.last_time: float | None = None

    def reset(self) -> None:
        self.bounds = self.last_time = None

    def normalize(self, luminance: np.ndarray, timestamp: float,
                  subject: np.ndarray | None = None) -> np.ndarray:
        values = luminance[subject > .55] if subject is not None and np.count_nonzero(subject > .55) >= 64 else luminance.ravel()
        low, high = (float(value) for value in np.percentile(values, (10, 90)))
        if high - low < .20:
            midpoint = (low + high) / 2
            low, high = max(0., midpoint - .10), min(1., midpoint + .10)
            if high - low < .20:
                low, high = (0., .20) if low == 0 else (.80, 1.)
        if self.bounds is None:
            self.bounds = low, high
        else:
            dt = max(0., min(.1, timestamp - (self.last_time or timestamp)))
            amount = 1 - exp(-dt / .75)
            self.bounds = tuple(old + amount * (new - old) for old, new in zip(self.bounds, (low, high), strict=True))
        self.last_time = timestamp
        return np.clip((luminance - self.bounds[0]) / (self.bounds[1] - self.bounds[0]), 0, 1).astype(np.float32)


def palette_map(
    luminance: np.ndarray,
    edges: np.ndarray,
    shadow: tuple[int, int, int],
    midtone: tuple[int, int, int],
    highlight: tuple[int, int, int],
    *,
    lower: float = 0.24,
    upper: float = 0.63,
    softness: float = 0.05,
    ink_strength: float = 0.70,
) -> np.ndarray:
    """Map lightness to three flat tones with soft threshold bands and edge ink."""
    if luminance.shape != edges.shape or luminance.ndim != 2:
        raise ValueError("Luminance and edges must share a 2D shape")
    if not (0 <= lower < upper <= 1 and 0 < softness <= 0.25):
        raise ValueError("Invalid tone thresholds")
    y = luminance.astype(np.float32, copy=False)
    # Each band owns one scratch array. Reuse it through the original
    # subtract/divide/add/clip order; never write into the live luminance.
    first = y - lower
    np.divide(first, softness, out=first)
    np.add(first, .5, out=first)
    np.clip(first, 0, 1, out=first)
    second = y - upper
    np.divide(second, softness, out=second)
    np.add(second, .5, out=second)
    np.clip(second, 0, 1, out=second)
    alpha = np.where(edges > .015, np.clip(.25 + edges * ink_strength, .25, .7), 0)
    output = np.empty((*y.shape, 3), np.uint8)
    adjustment = np.empty_like(y)
    for channel in range(3):
        tones = shadow[channel] + (midtone[channel] - shadow[channel]) * first
        np.subtract(highlight[channel], tones, out=adjustment)
        np.multiply(adjustment, second, out=adjustment)
        np.add(tones, adjustment, out=tones)
        np.subtract(shadow[channel], tones, out=adjustment)
        # Preserve NumPy's wider intermediate when a caller supplies
        # float64 edges; forcing that product into float32 changes rounding.
        if alpha.dtype == adjustment.dtype:
            np.multiply(adjustment, alpha, out=adjustment)
            np.add(tones, adjustment, out=tones)
        else:
            tones += adjustment * alpha
        np.clip(tones, 0, 255, out=tones)
        output[..., channel] = tones.astype(np.uint8)
    return output
