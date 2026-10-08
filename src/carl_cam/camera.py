"""Camera lifecycle and full-image letterboxing, independently of the Qt UI."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Event, Thread
from time import monotonic, monotonic_ns

import cv2
import numpy as np

from carl_cam.channels import LatestValue
from carl_cam.geometry import letterbox_transform
from carl_cam.records import FramePacket

CANVAS_SIZE = (1280, 720)
CAPTURE_SIZES = ((1280, 720), (960, 540), (640, 360))


@dataclass(frozen=True, slots=True)
class CameraDevice:
    index: int
    name: str
    identity: str = ""


def available_devices(*, enumerator=None) -> tuple[CameraDevice, ...]:
    """Read DirectShow names without opening a stream; exclude OBS feedback."""
    if enumerator is None:
        from cv2_enumerate_cameras import enumerate_cameras
        enumerator = enumerate_cameras
    try:
        devices = enumerator(cv2.CAP_DSHOW)
    except (OSError, RuntimeError, ValueError, NotImplementedError):
        return ()
    return tuple(CameraDevice(int(item.index), str(item.name), hashlib.sha256(
                 str(getattr(item, "path", None) or f"{item.name}:{item.index}").encode()).hexdigest()) for item in devices
                 if "obs" not in str(item.name).casefold() and 0 <= int(item.index) <= 99)


@dataclass(frozen=True, slots=True)
class CameraStatus:
    state: str
    message: str
    source_size: tuple[int, int] | None = None
    measured_fps: float = 0.0
    quality_level: int = 0  # Requested tier; actual dimensions remain separate.
    quality_request_id: int = 0


def canvas_rgb(rgb: np.ndarray, size: tuple[int, int] = CANVAS_SIZE) -> np.ndarray:
    """Resize uniformly into dark bars, preserving every source pixel's field of view."""
    source_height, source_width = rgb.shape[:2]
    scale, offset_x, offset_y = letterbox_transform((source_width, source_height), size)
    resized = cv2.resize(
        rgb,
        (round(source_width * scale), round(source_height * scale)),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    x, y = round(offset_x), round(offset_y)
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


class CameraWorker:
    """Own capture on one thread; publish fresh RGB packets or clear on loss.

    Construction does not open the camera. The UI starts this worker only after
    the user accepts the local-processing explanation and chooses a device.
    """

    def __init__(self, device: int = 0, *, expected_identity: str | None = None) -> None:
        self.device = device
        self.expected_identity = expected_identity
        self.frames: LatestValue[FramePacket] = LatestValue()
        self.status: LatestValue[CameraStatus] = LatestValue()
        self._stop = Event()
        self._thread: Thread | None = None
        self._quality_requests = LatestValue[tuple[int, int]]()
        self._quality_requests.publish((0, 0))
        self._quality_serial = 0

    def request_quality_level(self, level: int) -> int:
        """Queue format changes for the capture owner, never touch its device here."""
        if type(level) is not int or not 0 <= level < len(CAPTURE_SIZES):
            raise ValueError('Unsupported camera quality level')
        self._quality_serial += 1
        self._quality_requests.publish((self._quality_serial, level))
        return self._quality_serial

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.status.publish(CameraStatus("connecting", "Connecting to your camera…"))
        self._thread = Thread(target=self._run, name="CarlCamCapture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        stopped = self._thread is None or not self._thread.is_alive()
        if stopped:
            self.frames.clear()
        return stopped

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        capture = None
        try:
            selected = next((item for item in available_devices() if item.index == self.device), None)
            if selected is None or self.expected_identity is not None and selected.identity != self.expected_identity:
                self.status.publish(CameraStatus("error", "This input is unavailable or is OBS Virtual Camera. Refresh cameras and choose a physical camera."))
                return
            # DirectShow supplies common USB camera controls on supported Windows.
            capture = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
            if not capture.isOpened():
                self.status.publish(CameraStatus("error", "Camera unavailable. Choose another device or check Windows camera access."))
                return
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            first_bgr = None
            for quality_level, (width, height) in enumerate(CAPTURE_SIZES):
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                capture.set(cv2.CAP_PROP_FPS, 30)
                ok, image = capture.read()
                if ok and image is not None:
                    first_bgr = image
                    break
            if first_bgr is None:
                self.status.publish(CameraStatus("error", "Camera connected but did not send an image. Retry or choose another device."))
                return
            applied_request = self._quality_requests.read()[0]
            frame_id, count = 0, 0
            epoch = monotonic()
            bgr = first_bgr
            while not self._stop.is_set():
                request_id, requested = self._quality_requests.read()
                if request_id != applied_request:
                    self.frames.clear()
                    self.status.publish(CameraStatus('reconfiguring', 'Changing input quality…', quality_level=requested))
                    bgr = None
                    for quality_level in range(requested, len(CAPTURE_SIZES)):
                        width, height = CAPTURE_SIZES[quality_level]
                        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        capture.set(cv2.CAP_PROP_FPS, 30)
                        ok, image = capture.read()
                        if ok and image is not None:
                            bgr = image
                            break
                    if bgr is None:
                        self.status.publish(CameraStatus('error', 'Camera could not change input quality. Stop and retry the camera.'))
                        return
                    applied_request = request_id
                    count, epoch = 0, monotonic()
                captured = monotonic_ns()
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                rgb.flags.writeable = False
                self.frames.publish(FramePacket(frame_id, captured, rgb))
                frame_id += 1
                count += 1
                elapsed = monotonic() - epoch
                source_size = (bgr.shape[1], bgr.shape[0])
                message = 'Reduced input quality requested' if quality_level > 0 else 'Camera live'
                self.status.publish(CameraStatus("live", message, source_size, count / max(elapsed, 0.001), quality_level, applied_request))
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    self.frames.clear()
                    self.status.publish(CameraStatus("error", "Camera stopped sending images. Reconnect it and press Retry."))
                    return
        except (cv2.error, OSError, RuntimeError, ValueError) as error:
            self.frames.clear()
            self.status.publish(CameraStatus("error", f"Camera could not start: {error}"))
        finally:
            if capture is not None:
                capture.release()
            self.frames.clear()
            if self._stop.is_set():
                self.status.publish(CameraStatus("stopped", "Camera stopped"))
