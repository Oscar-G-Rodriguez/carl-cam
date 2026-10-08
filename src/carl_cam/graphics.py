"""Offline Figma vectors mapped onto live frame geometry.

The SVGs supply original borders and menu surfaces, never fixed filter pixels.
Premultiplied colors are interpolated with alpha so transparent edges cannot
leave dark fringes when a tilted frame is projected into the live camera.
"""

import hashlib
import json
import re
from functools import lru_cache
from math import pi, sin
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
from PySide6.QtCore import QByteArray, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtSvg import QSvgRenderer

from carl_cam.dsp import (
    premultiplied_border_over,
    premultiplied_rgb_over,
    premultiplied_rgba_over,
)
from carl_cam.geometry import Quad, validate_quad

ROOT = Path(__file__).parent / "assets/graphics"
ANCHORS = np.array(((32, 32), (608, 32), (608, 328), (32, 328)), dtype=np.float32)
ACCENTS = {"red": "#C21E29", "yellow": "#F5C915", "blue": "#4AC9E8"}
MENU_STATES = ("default", "hover", "focused", "selected")


def _pixels(image: QImage) -> np.ndarray:
    rows = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )
    rgba = rows[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()
    rgba.flags.writeable = False
    return rgba


def svg_pixels(source: bytes, width: int, height: int) -> np.ndarray:
    renderer = QSvgRenderer(QByteArray(source))
    if not renderer.isValid():
        raise ValueError("The bundled graphic is unreadable")
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(0)
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, width, height))
    painter.end()
    return _pixels(image)


@lru_cache(maxsize=32)
def asset_pixels(name: str) -> np.ndarray:
    if name not in {
        "border-red", "border-yellow", "border-blue", "pointer", "menu-back",
        "icon-close", "icon-minimize", "icon-maximize", "icon-camera", "icon-train",
        "icon-inspect",
    } | {f"menu-{theme}-{state}" for theme in ACCENTS for state in MENU_STATES}:
        raise ValueError("Unknown bundled graphic")
    source = verified_svg(f"{name}.svg")
    renderer = QSvgRenderer(QByteArray(source))
    size = renderer.defaultSize()
    return svg_pixels(source, size.width(), size.height())


def verified_svg(filename: str) -> bytes:
    """Detect corrupted offline exports before Qt parses their vector content."""
    manifest_path = ROOT / "manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 32 * 1024:
        raise ValueError("Invalid graphics manifest. Repair the installation.")
    manifest = json.loads(manifest_path.read_bytes())
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("assets"), dict)):
        raise ValueError("Unsupported graphics manifest. Repair the installation.")
    metadata = manifest.get("assets", {}).get(filename)
    path = ROOT / filename
    if (not isinstance(metadata, dict) or path.name != filename or path.is_symlink()
            or not path.is_file() or not 0 < path.stat().st_size <= 256 * 1024):
        raise ValueError("Missing graphic. Repair the installation.")
    source = path.read_bytes()
    if len(source) != metadata.get("bytes") or hashlib.sha256(source).hexdigest() != metadata.get("sha256"):
        raise ValueError("Graphic checksum failed. Repair the installation.")
    return source


def _premultiplied(rgba: np.ndarray) -> np.ndarray:
    result = rgba.astype(np.float32)
    result[..., 3] /= 255
    result[..., :3] *= result[..., 3:4]
    return result


def _over(rgb: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    # Premultiplied source-over is nonnegative. OpenCV's saturated, rounded
    # conversion avoids separate full RGB rint/clip/astype allocations.
    return premultiplied_rgb_over(rgb, overlay)


@lru_cache(maxsize=12)
def _asset_premultiplied(name: str) -> np.ndarray:
    pixels = _premultiplied(asset_pixels(name))
    pixels.flags.writeable = False
    return pixels


@lru_cache(maxsize=1)
def _red_paper_layers():
    """Split the genuine export's two strokes, retaining fixed corner anchors.

    No asset is rewritten or replaced: screen-space motion is a runtime
    transform of the editable Figma layers. Stroke colors identify this small,
    audited export; an incompatible replacement fails instead of being guessed.
    """
    source = ElementTree.fromstring(verified_svg("border-red.svg"))
    paths = list(source)
    if len(paths) != 4 or paths[1].get("stroke") != "#F1ECE2" or paths[2].get("stroke") != "#C21E29":
        raise ValueError("Red border layer structure changed. Repair the installation.")
    layers = []
    for path in paths:
        tree = ElementTree.Element(source.tag, source.attrib)
        tree.append(path)
        pixels = _premultiplied(svg_pixels(ElementTree.tostring(tree), 640, 360))
        pixels.flags.writeable = False
        layers.append(pixels)
    return tuple(layers)


@lru_cache(maxsize=1)
def _red_paper_fragments():
    """Crop the genuine SVG's disconnected strokes/corners before projecting.

    The connected black outline stays one layer. Only independent M subpaths
    are separated; their original style, order and screen-space motion remain
    unchanged. Small projected strips avoid processing the empty whole frame
    once for every stroke. The exported SVG bytes are never modified.
    """
    source = ElementTree.fromstring(verified_svg('border-red.svg'))
    paths = list(source)
    _red_paper_layers()  # Validate this exact export's approved layer contract.
    fragments = []
    for index, path in enumerate(paths):
        pieces = re.findall(r'M[^M]+', path.get('d', '')) if index else [path.get('d', '')]
        if not pieces or ''.join(pieces) != path.get('d'):
            raise ValueError('Red border path structure is incompatible')
        for piece in pieces:
            attributes = dict(path.attrib, d=piece)
            tree = ElementTree.Element(source.tag, source.attrib)
            tree.append(ElementTree.Element(path.tag, attributes))
            pixels = _premultiplied(svg_pixels(ElementTree.tostring(tree), 640, 360))
            ys, xs = np.where(pixels[..., 3] > 0)
            if not len(ys):
                continue
            x, y = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
            right, bottom = min(640, int(xs.max()) + 3), min(360, int(ys.max()) + 3)
            crop = np.ascontiguousarray(pixels[y:bottom, x:right])
            crop.flags.writeable = False
            fragments.append((index, x, y, crop))
    return tuple(fragments)


def paper_offsets(seconds: float, *, reduced_motion=False):
    if reduced_motion:
        return ((0., 0.),) * 4
    distance = 4 / 2**.5
    wave = sin(2 * pi * seconds / 4)
    return ((0., 0.), (distance * wave, distance * wave),
            (-distance * wave, distance * wave), (0., 0.))


def draw_border(rgb: np.ndarray, theme: str, quad: Quad | None,
                hand_mask: np.ndarray | None = None, *, seconds: float = 0.,
                reduced_motion: bool = False) -> np.ndarray:
    height, width = rgb.shape[:2]
    if quad is None or not validate_quad(quad, (width, height)):
        return rgb.copy()
    transform = cv2.getPerspectiveTransform(ANCHORS, np.asarray(quad, dtype=np.float32))
    source = asset_pixels(f"border-{theme}")
    corners = np.array((((0, 0), (source.shape[1], 0),
                         (source.shape[1], source.shape[0]), (0, source.shape[0])),), np.float32)
    projected = cv2.perspectiveTransform(corners, transform)[0]
    padding = 6 if theme == "red" and not reduced_motion else 2
    left, top = np.maximum(0, np.floor(projected.min(axis=0) - padding)).astype(int)
    right, bottom = np.minimum((width, height), np.ceil(projected.max(axis=0) + padding + 1)).astype(int)
    result = rgb.copy()
    if right <= left or bottom <= top:
        return result
    translate = np.array(((1, 0, -left), (0, 1, -top), (0, 0, 1)), np.float64)
    if theme == "red" and not reduced_motion:
        overlay = np.zeros((bottom - top, right - left, 4), np.float32)
        offsets = paper_offsets(seconds)
        for index, x, y, fragment in _red_paper_fragments():
            dx, dy = offsets[index]
            offset = np.array(((1, 0, dx), (0, 1, dy), (0, 0, 1)), np.float64)
            origin = np.array(((1, 0, x), (0, 1, y), (0, 0, 1)), np.float64)
            matrix = translate @ offset @ transform @ origin
            h, w = fragment.shape[:2]
            corners = np.array((((0, 0), (w, 0), (w, h), (0, h)),), np.float32)
            bounds = cv2.perspectiveTransform(corners, matrix)[0]
            a, b = np.maximum(0, np.floor(bounds.min(axis=0) - 2)).astype(int)
            c, d = np.minimum((right - left, bottom - top), np.ceil(bounds.max(axis=0) + 3)).astype(int)
            if c <= a or d <= b:
                continue
            crop_origin = np.array(((1, 0, -a), (0, 1, -b), (0, 0, 1)), np.float64)
            projected_layer = cv2.warpPerspective(fragment, crop_origin @ matrix,
                                                  (c - a, d - b), flags=cv2.INTER_LINEAR)
            area = np.s_[b:d, a:c]
            overlay[area] = premultiplied_rgba_over(overlay[area], projected_layer)
    else:
        overlay = cv2.warpPerspective(_asset_premultiplied(f"border-{theme}"), translate @ transform,
                                      (right - left, bottom - top), flags=cv2.INTER_LINEAR)
    area = np.s_[top:bottom, left:right]
    if hand_mask is not None:
        if hand_mask.shape != (height, width) or not np.isfinite(hand_mask).all():
            raise ValueError("Hand occlusion must match the current camera canvas")
        # The base already contains stylized hands inside and plain hands outside
        # the quad. Suppressing border alpha retains exactly that base image.
        overlay *= (1 - np.clip(hand_mask[area], 0, 1))[..., None]
    result[area] = premultiplied_border_over(rgb[area], overlay)
    return result


@lru_cache(maxsize=32)
def menu_card(theme: str, selected: bool = False, *, interaction: str = "default") -> np.ndarray:
    """Use the exact exported theme/state, with outlined upright Figma labels.

    A selected card keeps its original current-filter dot while an overlapping
    hover/focus uses the design's interaction border. Each view positions this
    complete card after coordinate conversion; text is never camera-mirrored.
    """
    if interaction not in ("default", "hover", "focused"):
        raise ValueError("Unknown menu interaction")
    if theme == "back":
        return asset_pixels("menu-back")
    if theme not in ACCENTS:
        raise ValueError("Unknown menu choice")
    state = "selected" if selected else interaction
    pixels = asset_pixels(f"menu-{theme}-{state}")
    if not selected or interaction == "default":
        return pixels
    image = QImage(pixels.data, 300, 100, 1200, QImage.Format.Format_RGBA8888).copy()
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(ACCENTS["blue"] if interaction == "hover" else ACCENTS["yellow"]), 3))
    # The Figma 292x92 inside stroke starts at (4,4); centerline is inset 1.5.
    painter.drawRect(QRectF(5.5, 5.5, 289, 89))
    painter.end()
    return _pixels(image)


def place_asset(rgb: np.ndarray, rgba: np.ndarray, x: float, y: float,
                width: float | None = None, height: float | None = None) -> np.ndarray:
    result = rgb.copy()
    w = max(1, round(width if width is not None else rgba.shape[1]))
    h = max(1, round(height if height is not None else rgba.shape[0]))
    overlay = cv2.resize(_premultiplied(rgba), (w, h), interpolation=cv2.INTER_LINEAR)
    x, y = round(x), round(y)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(result.shape[1], x + w), min(result.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return result
    result[y0:y1, x0:x1] = _over(result[y0:y1, x0:x1], overlay[y0-y:y1-y, x0-x:x1-x])
    return result
