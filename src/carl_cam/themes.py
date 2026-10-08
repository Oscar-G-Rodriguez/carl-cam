"""Original, deterministic live-pixel recipes, independent of Qt and model APIs.

Only decoration is projected onto the quad. Subject pixels remain in their
camera coordinates. Category order follows MediaPipe SelfieMulticlass:
background, hair, body skin, face skin, clothes, accessories.
"""

from dataclasses import dataclass
from math import cos, pi, sin

import cv2
import numpy as np

from carl_cam.dsp import SignalStages, palette_map, premultiplied_rgba_over

PALETTES = {
    "red": ((17, 16, 20), (194, 30, 41), (241, 236, 226)),
    "yellow": ((37, 22, 47), (245, 201, 21), (255, 244, 200)),
    "blue": ((10, 24, 63), (35, 85, 207), (216, 248, 255)),
}

# Both panels complete the approved six-second cycle. The cyan panel's 10%
# smaller travel gives it a slightly lower speed, within the 5 px limit.
BLUE_PANEL_MOTION = ((5., 6., 0.), (4.5, 6., 2.))

# All colors belong to the approved finite theme palettes. Fine labels alter
# tone grouping, not camera geometry; uncertain labels retain the broad recipe.
REGION_TONES = {
    "red": {
        "hair": (((17, 16, 20), (17, 16, 20), (241, 236, 226)), .32, .68),
        "skin": (PALETTES["red"], .24, .63),
        "clothes": (PALETTES["red"], .30, .70),
    },
    "yellow": {
        "hair": (((37, 22, 47), (37, 22, 47), (255, 244, 200)), .32, .70),
        "skin": (PALETTES["yellow"], .24, .63),
        "clothes": (PALETTES["yellow"], .30, .70),
    },
    "blue": {
        "hair": (PALETTES["blue"], .32, .72),
        "skin": (((10, 24, 63), (66, 216, 238), (216, 248, 255)), .24, .63),
        "clothes": (((10, 24, 63), (35, 85, 207), (66, 216, 238)), .30, .70),
    },
}


@dataclass(frozen=True, slots=True)
class Regions:
    person: np.ndarray
    hair: np.ndarray
    skin: np.ndarray
    clothes: np.ndarray
    reliable: bool


def feather(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Two working pixels form a four-canvas-pixel softened edge."""
    return cv2.GaussianBlur(mask.astype(np.float32), (radius * 2 + 1,) * 2, 0)


def _foreground_channels(masks):
    """Keep the five-label sum's order; split float planes for native selection.

    The model supplies float32, and float64 remains supported for reference
    checks. Other dtypes retain NumPy's accumulation/promotion behavior.
    """
    if masks.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        return masks[..., 1:].sum(axis=-1), None
    channels = cv2.split(masks)
    probability = channels[1] + channels[2]
    for channel in channels[3:]:
        probability += channel
    return probability, channels


def _category_winner(masks, channels):
    """Strict greater-than retains argmax's first label on an exact tie."""
    if channels is None:
        return np.argmax(masks, axis=-1)
    best = channels[0].copy()
    winner = np.zeros(best.shape, np.uint8)
    labels = np.empty_like(winner)
    for label, channel in enumerate(channels[1:], 1):
        changed = cv2.compare(channel, best, cv2.CMP_GT)
        labels.fill(label)
        cv2.copyTo(labels, changed, winner)
        cv2.max(best, channel, dst=best)
    return winner


def regions_from_masks(masks: np.ndarray | None, shape: tuple[int, int], *,
                       reference_reductions: bool = False) -> Regions:
    empty = np.zeros(shape, np.float32)
    if masks is None or masks.shape != (*shape, 6) or not np.isfinite(masks).all():
        return Regions(empty, empty, empty, empty, False)
    masks = np.clip(masks, 0, 1)
    if reference_reductions:
        foreground, channels = masks[..., 1:].sum(axis=-1), None
    else:
        foreground, channels = _foreground_channels(masks)
    probability = np.clip(foreground, 0, 1)
    # Reject uncertain whole-scene outputs; individual fine labels can still
    # abstain while a confident foreground/background union remains usable.
    coverage = np.mean(np.maximum(probability, masks[..., 0]) >= .55)
    reliable = bool(coverage >= .90 and np.mean(probability >= .55) >= .01)
    if not reliable:
        return Regions(empty, empty, empty, empty, False)
    person = feather((probability >= .55).astype(np.float32))
    winner = _category_winner(masks, channels)
    def region(indices):
        # Direct comparisons avoid advanced-index channel copies for each
        # fine region. The two skin channels retain the original addition.
        selected = winner == indices[0]
        confidence = channels[indices[0]] if channels is not None else masks[..., indices[0]]
        for index in indices[1:]:
            selected |= winner == index
            confidence = confidence + (channels[index] if channels is not None else masks[..., index])
        return feather((selected & (confidence >= .55)).astype(np.float32))
    return Regions(person, region([1]), region([2, 3]), region([4]), True)


def subject_tones(stages: SignalStages, luminance: np.ndarray, theme: str,
                  regions: Regions) -> np.ndarray:
    tones = palette_map(luminance, stages.edge_magnitude, *PALETTES[theme])
    if not regions.reliable:
        return tones
    for name, (palette, lower, upper) in REGION_TONES[theme].items():
        mask = getattr(regions, name)
        x, y, width, height = cv2.boundingRect((mask > 0).view(np.uint8))
        if width == 0 or height == 0:
            continue
        area = np.s_[y:y + height, x:x + width]
        mapped = palette_map(luminance[area], stages.edge_magnitude[area], *palette,
                             lower=lower, upper=upper)
        alpha = mask[area]
        tones[area] = cv2.blendLinear(mapped, tones[area], alpha, 1 - alpha)
    return tones


def _translucent_polygon(local: np.ndarray, path: np.ndarray, color, alpha: float) -> None:
    """Blend a finite polygon only where its integer raster can be nonzero.

    Opaque polygons already write directly into the design plane. For Blue's
    translucent panels/shards, avoid allocating and combining an empty full
    plane around every shape. Integer translation retains fillPoly's raster;
    untouched exterior pixels retain the preceding premultiplied layer.
    """
    x, y, width, height = cv2.boundingRect(path)
    left, top = max(0, x), max(0, y)
    right, bottom = min(local.shape[1], x + width), min(local.shape[0], y + height)
    if right <= left or bottom <= top:
        return
    layer = np.zeros((bottom - top, right - left, 4), np.float32)
    translated = path - np.array((left, top), np.int32)
    cv2.fillPoly(layer, [translated], (*[c * alpha for c in color], alpha))
    area = np.s_[top:bottom, left:right]
    local[area] = premultiplied_rgba_over(local[area], layer)


def motif_overlay(theme: str, quad, size: tuple[int, int], seconds: float,
                  reduced_motion: bool = False) -> np.ndarray:
    """Premultiplied RGBA decoration; fixed placements, no random per-frame noise.

    Small motions are specified in *screen pixels*, converted through quad edge
    lengths into the local design plane. Blue alpha reveals styled pixels only.
    """
    width, height = size
    local = np.zeros((360, 640, 4), np.float32)
    points = np.asarray(quad, np.float32)
    edge_x = max(1., min(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
    edge_y = max(1., min(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
    def poly(coords, color, alpha=1., offset=(0., 0.)):
        path = np.asarray([(u * 639 + offset[0], v * 359 + offset[1]) for u, v in coords], np.float32).round().astype(np.int32)
        if alpha == 1:
            cv2.fillPoly(local, [path], (*color, 1.))
            return
        _translucent_polygon(local, path, color, alpha)
    def motion(amplitude, period, phase=0.):
        if reduced_motion:
            return 0., 0.
        wave = sin(2 * pi * seconds / period + phase)
        amplitude *= width / 1280  # Approved motion is measured at the 720p canvas.
        return amplitude * wave * 639 / edge_x / 2**.5, amplitude * wave * 359 / edge_y / 2**.5
    if theme == "red":
        poly(((0, .08), (.61, 0), (.43, .62), (0, .83)), (141, 7, 21), offset=motion(4, 4))
        poly(((.44, .06), (1, .21), (1, .72), (.31, .58)), (17, 16, 20), offset=motion(4, 4, 1.7))
        poly(((.06, .70), (.54, .44), (.90, 1), (0, 1)), (241, 236, 226), offset=motion(4, 4, 3.3))
        # One fixed halftone patch. No printing texture touches the subject.
        for y in range(34, 180, 12):
            for x in range(440, 622, 12):
                cv2.circle(local, (x + (6 if y % 24 else 0), y), 3, (241., 236., 226., 1.), -1)
    elif theme == "yellow":
        for center, axes, color, phase in (((.16, .25), (.29, .39), (74, 201, 232), 0.), ((.85, .80), (.32, .36), (241, 106, 165), 2.)):
            dx, dy = motion(6, 5, phase)
            cv2.ellipse(local, (round(center[0] * 639 + dx), round(center[1] * 359 + dy)),
                        (round(axes[0] * 639), round(axes[1] * 359)), 18, 0, 360, (*color, 1.), -1)
        scale = 1. if reduced_motion else 1. + .04 * sin(2 * pi * seconds / 5)
        for u, v, radius in ((.14, .83, .12), (.85, .18, .10)):
            flower = []
            for angle in np.linspace(0, 2 * pi, 80, endpoint=False):
                radial = radius * scale * (.72 + .28 * cos(5 * angle))
                flower.append((u + radial * cos(angle), v + radial * 1.7 * sin(angle)))
            poly(flower, (255, 244, 200))
    elif theme == "blue":
        poly(((0, .08), (.67, 0), (.42, 1), (0, .77)), (35, 85, 207), .62, motion(*BLUE_PANEL_MOTION[0]))
        poly(((.40, 0), (1, .20), (1, .91), (.19, .74)), (66, 216, 238), .46, motion(*BLUE_PANEL_MOTION[1]))
        poly(((0, .68), (.28, .23), (.14, .77)), (216, 248, 255), .9)
        poly(((.68, .07), (1, .48), (.79, .35)), (10, 24, 63), .9)
    else:
        raise ValueError("Unknown filter theme")
    transform = cv2.getPerspectiveTransform(np.array(((0, 0), (639, 0), (639, 359), (0, 359)), np.float32), points)
    return cv2.warpPerspective(local, transform, (width, height), flags=cv2.INTER_LINEAR)
