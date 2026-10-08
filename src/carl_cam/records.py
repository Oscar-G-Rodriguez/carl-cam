"""Immutable records exchanged between camera, perception, interaction, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ThemeName = Literal["red", "yellow", "blue"]
GestureName = Literal["frame", "peace", "thumbs_up", "open_palm", "ily", "other"]


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: int
    capture_time_ns: int
    rgb: np.ndarray

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


@dataclass(frozen=True, slots=True)
class HandObservation:
    landmarks: tuple[tuple[float, float, float], ...]
    handedness: str
    score: float


@dataclass(frozen=True, slots=True)
class PerceptionPacket:
    frame_id: int
    capture_time_ns: int
    hands: tuple[HandObservation, ...]
    person_mask: np.ndarray | None = None
    head_center: tuple[float, float] | None = None
    inference_ms: float = 0.0
    head_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class GesturePacket:
    frame_id: int
    capture_time_ns: int
    feature_schema: str
    features: np.ndarray
    scores: tuple[float, float, float, float, float, float]
    quality_ok: bool


@dataclass(frozen=True, slots=True)
class InteractionSnapshot:
    mode: str
    theme: ThemeName
    quad: tuple[tuple[float, float], ...] | None
    pointer: tuple[float, float] | None = None
    menu_anchor: tuple[float, float] | None = None
    command: str | None = None


@dataclass(frozen=True, slots=True)
class RenderPacket:
    frame_id: int
    preview_rgb: np.ndarray
    output_rgb: np.ndarray
    render_time_ns: int
    render_ms: float
    capture_time_ns: int = 0
    render_sequence: int = 0
