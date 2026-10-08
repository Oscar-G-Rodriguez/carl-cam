"""Offline vision, with independent hand/head and segmentation workers.

Every result keeps its source frame ID. Hand/head commands never wait for the
slower region model; the renderer can match a mask to its original RGB frame.
Coordinates are canonical mirrored 1280×720 pixels; z uses canvas-width units.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import monotonic_ns, perf_counter
from typing import TYPE_CHECKING

import cv2
import numpy as np

from carl_cam.camera import CANVAS_SIZE, canvas_rgb
from carl_cam.channels import LatestValue
from carl_cam.records import FramePacket, HandObservation, PerceptionPacket
from carl_cam.vision_privacy import (
    PRIVACY_HOLD,
    VisionPrivacyError,
    require_verified_vision_runtime,
)

MODEL_ROOT = Path(__file__).resolve().parent / "assets/models"

if TYPE_CHECKING:
    from carl_cam.dsp import SignalStages
    from carl_cam.themes import Regions


def model_path(name: str, root: Path = MODEL_ROOT) -> Path:
    with (root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or name not in manifest.get("models", {}):
        raise ValueError("Missing vision-model manifest. Repair the installation.")
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Missing or invalid vision asset. Repair the installation.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["models"][name]["sha256"]:
        raise ValueError("Vision asset checksum failed. Repair the installation.")
    return path


def mirrored_canvas(packet: FramePacket) -> np.ndarray:
    return np.ascontiguousarray(canvas_rgb(packet.rgb)[:, ::-1])


def convert_hands(result, size: tuple[int, int] = CANVAS_SIZE) -> tuple[HandObservation, ...]:
    width, height = size
    observations = []
    for landmarks, categories in zip(result.hand_landmarks, result.handedness, strict=True):
        if len(landmarks) != 21 or not categories:
            continue
        points = tuple((float(point.x * width), float(point.y * height), float(point.z)) for point in landmarks)
        if not np.isfinite(np.array(points)[:, :2]).all():
            continue
        # This score is handedness confidence, not landmark visibility. The
        # model's separate detection/presence/tracking gates are each 0.5.
        observations.append(HandObservation(points, categories[0].category_name, float(categories[0].score)))
    return tuple(observations)


@dataclass(frozen=True, slots=True)
class RegionPacket:
    frame_id: int
    capture_time_ns: int
    confidence_masks: np.ndarray  # 360×640×6, mirrored canvas.
    inference_ms: float
    prepared_regions: Regions | None = None
    prepared_signals: SignalStages | None = None


class HandHeadEngine:
    def __init__(self) -> None:
        require_verified_vision_runtime()
        import mediapipe as mp
        self._mp = mp
        base, vision = mp.tasks.BaseOptions, mp.tasks.vision
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(model_path("hand_landmarker.task"))),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
        ))
        try:
            self.face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                base_options=base(model_asset_path=str(model_path("blaze_face_short_range.tflite"))),
                running_mode=vision.RunningMode.VIDEO, min_detection_confidence=0.5,
            ))
        except Exception:
            self.hand.close()
            raise
        self._timestamp_ms = -1

    def process(self, packet: FramePacket) -> PerceptionPacket:
        started = perf_counter()
        mp = self._mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mirrored_canvas(packet))
        timestamp = max(self._timestamp_ms + 1, packet.capture_time_ns // 1_000_000)
        self._timestamp_ms = timestamp
        hands = self.hand.detect_for_video(image, timestamp)
        faces = self.face.detect_for_video(image, timestamp)
        head = None
        head_box = None
        if faces.detections:
            face = max(faces.detections, key=lambda detection: detection.bounding_box.width * detection.bounding_box.height)
            box = face.bounding_box
            head = (float(box.origin_x + box.width / 2), float(box.origin_y + box.height / 2))
            head_box = (float(box.origin_x), float(box.origin_y), float(box.width), float(box.height))
        return PerceptionPacket(packet.frame_id, packet.capture_time_ns, convert_hands(hands),
                                head_center=head, head_box=head_box, inference_ms=(perf_counter() - started) * 1000)

    def close(self) -> None:
        self.hand.close()
        self.face.close()


class RegionEngine:
    def __init__(self) -> None:
        require_verified_vision_runtime()
        import mediapipe as mp
        self._mp = mp
        vision = mp.tasks.vision
        self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path("selfie_multiclass_256x256.tflite"))),
            running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True, output_category_mask=False,
        ))
        self._timestamp_ms = -1

    def process(self, packet: FramePacket) -> RegionPacket:
        started = perf_counter()
        mp = self._mp
        rgb = cv2.resize(mirrored_canvas(packet), (640, 360), interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(self._timestamp_ms + 1, packet.capture_time_ns // 1_000_000)
        self._timestamp_ms = timestamp
        result = self.segmenter.segment_for_video(image, timestamp)
        if not result.confidence_masks or len(result.confidence_masks) != 6:
            raise ValueError("Region model has incompatible categories")
        channels = []
        for mask in result.confidence_masks:
            channel = mask.numpy_view().copy()
            if channel.ndim == 3 and channel.shape[-1] == 1:
                channel = channel[..., 0]
            if channel.ndim != 2:
                raise ValueError("Region model produced an incompatible mask shape")
            channels.append(cv2.resize(channel, (640, 360), interpolation=cv2.INTER_LINEAR))
        masks = np.stack(channels, axis=-1)
        if not np.isfinite(masks).all():
            raise ValueError("Region model produced invalid masks")
        masks = np.clip(masks, 0, 1).astype(np.float32)
        masks.flags.writeable = False
        return RegionPacket(packet.frame_id, packet.capture_time_ns, masks, (perf_counter() - started) * 1000)

    def close(self) -> None:
        self.segmenter.close()


class VisionWorker:
    def __init__(self, frames: LatestValue[FramePacket], *, regions: bool = False, factory=None) -> None:
        self.frames = frames
        self.results = LatestValue()
        self.source = LatestValue[FramePacket]()
        self.completed = LatestValue()
        self.status = LatestValue[str]()
        self.backend_status = LatestValue[str]()
        self._factory = factory or (RegionEngine if regions else HandHeadEngine)
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="CarlCamVision", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        self.results.clear()
        self.source.clear()
        self.completed.clear()
        self.backend_status.clear()
        return self._thread is None or not self._thread.is_alive()

    def _run(self) -> None:
        engine = None
        try:
            self.status.publish("Starting local vision models…")
            engine = self._factory()
            initialize = getattr(engine, 'initialize', None)
            if initialize is not None:
                initialize(self._stop)
            if self._stop.is_set():
                return
            self.backend_status.publish(getattr(engine, 'backend_status', 'CPU'))
            self.status.publish("Local vision ready")
            last_id = -1
            while not self._stop.is_set():
                packet = self.frames.read()
                if packet is None or monotonic_ns() - packet.capture_time_ns > 200_000_000:
                    self.results.clear()
                    self.source.clear()
                    self.completed.clear()
                    self._stop.wait(0.01)
                    continue
                if packet.frame_id == last_id:
                    self._stop.wait(0.003)
                    continue
                result = engine.process(packet)
                self.backend_status.publish(getattr(engine, 'backend_status', 'CPU'))
                if self._stop.is_set() or monotonic_ns() - packet.capture_time_ns > 200_000_000:
                    self.results.clear()
                    self.source.clear()
                    self.completed.clear()
                else:
                    self.source.publish(packet)
                    self.results.publish(result)
                    # One atomic pair prevents readers joining a new result to
                    # the preceding source during independent channel updates.
                    self.completed.publish((packet, result))
                last_id = packet.frame_id
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, cv2.error) as error:
            self.results.clear()
            self.source.clear()
            self.completed.clear()
            message = PRIVACY_HOLD if isinstance(error, VisionPrivacyError) else f"{type(error).__name__}. Mouse and keyboard still work."
            self.status.publish(f"Vision unavailable: {message}")
        finally:
            if engine is not None:
                engine.close()
