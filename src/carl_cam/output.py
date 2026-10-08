"""Explicit, launch-local OBS output with a capture-age safety slate.

This module never installs drivers, opens an input camera, or starts OBS.
The independent 30 Hz sender can duplicate only a fresh completed frame.
"""

from threading import Event, Thread
from time import monotonic, monotonic_ns

import numpy as np
import pyvirtualcam

from carl_cam.channels import LatestValue
from carl_cam.records import RenderPacket


def safe_output(packet: RenderPacket | None, now_ns: int) -> np.ndarray:
    if packet is not None and packet.frame_id >= 0 and 0 <= now_ns - packet.capture_time_ns <= 250_000_000:
        image = packet.output_rgb
        if image.shape == (720, 1280, 3) and image.dtype == np.uint8:
            return np.ascontiguousarray(image)
    return np.zeros((720, 1280, 3), np.uint8)


class OutputWorker:
    def __init__(self, frames: LatestValue[RenderPacket], *, factory=None) -> None:
        self.frames = frames
        self.factory = factory or pyvirtualcam.Camera
        self.status = LatestValue[str]()
        self.status.publish("Virtual camera is off")
        self._stop = Event()
        self._thread: Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.status.publish("Connecting to OBS Virtual Camera…")
        self._thread = Thread(target=self._run, name="CarlCamOutput", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        return not self.running

    def _run(self) -> None:
        sender = None
        try:
            sender = self.factory(width=1280, height=720, fps=30, fmt=pyvirtualcam.PixelFormat.RGB,
                                  backend="obs", device="OBS Virtual Camera", print_fps=False)
            # A known black initial frame is required even when a live result is
            # already available. Output consent is separate from camera consent.
            sender.send(np.zeros((720, 1280, 3), np.uint8))
            self.status.publish("Sending to OBS Virtual Camera — choose it in your other app")
            deadline = monotonic() + 1 / 30
            while not self._stop.is_set():
                if self._stop.wait(max(0., deadline - monotonic())):
                    break
                sender.send(safe_output(self.frames.read(), monotonic_ns()))
                deadline += 1 / 30
                if deadline < monotonic() - 1 / 30:
                    deadline = monotonic() + 1 / 30
        except (RuntimeError, OSError, ValueError, TypeError):
            self.status.publish("Output unavailable. Install OBS with its Virtual Camera, close other virtual-camera senders, then retry. Your preview still works.")
        finally:
            if sender is not None:
                try:
                    sender.close()
                except (RuntimeError, OSError):
                    self.status.publish("Output stopped, but its driver did not close cleanly. Restart the receiving app before retrying.")
            if self._stop.is_set():
                self.status.publish("Virtual camera is off")
