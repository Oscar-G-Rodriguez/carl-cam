"""Managed GPU child process and an explicit, visible CPU fallback.

The current source-development launcher uses the separately reviewed GPU
environment. Frozen deployment needs its own verified helper bundle; it never
borrows a developer interpreter or silently imports its packages into the app.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic, perf_counter
from uuid import uuid4

import numpy as np

from carl_cam.camera import canvas_rgb
from carl_cam.dsp import SignalStages
from carl_cam.gpu_bundle import HELPER_DIRECTORY, verified_helper
from carl_cam.gpu_ipc import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    SharedFrames,
    decode_message,
    encode_message,
)
from carl_cam.themes import Regions
from carl_cam.vision import RegionEngine, RegionPacket

DEVELOPMENT_PYTHON_SHA256 = '3b5cc4150dc98800ee46b09034187bb2d8ba3c8ea3368d3a46e9bb42876b58db'
DEVELOPMENT_CANDIDATE = 'gpu-segmentation-a86408c64bbd45cf8661e9d6b7fdfa60'


class GpuProcessError(RuntimeError):
    """A safe error code; original child exception text is never propagated."""


@dataclass(frozen=True)
class LaunchSpec:
    command: tuple[str, ...]
    candidate: Path
    cwd: Path


def development_launch() -> LaunchSpec | None:
    if '__compiled__' in globals():
        return None
    root = Path(__file__).absolute().parents[2]
    interpreter = root / 'build/gpu-conversion-env/Scripts/python.exe'
    script = root / 'src/carl_cam/gpu_service.py'
    candidate = root / 'build' / DEVELOPMENT_CANDIDATE
    for path in (interpreter, script, candidate):
        if (not path.exists() or any(part.is_symlink() or part.is_junction()
                                     for part in (path, *path.parents))):
            return None
    if (not interpreter.is_file() or interpreter.stat().st_size > 16 * 1024**2
            or hashlib.sha256(interpreter.read_bytes()).hexdigest() != DEVELOPMENT_PYTHON_SHA256):
        return None
    return LaunchSpec((str(interpreter), '-I', '-B', '-u', str(script)), candidate, root)


def runtime_launch() -> LaunchSpec | None:
    """Frozen apps use only their verified sibling helper, never development Python."""
    if '__compiled__' not in globals():
        return development_launch()
    bundle = Path(__file__).absolute().parents[1] / HELPER_DIRECTORY
    try:
        executable, candidate = verified_helper(bundle)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return LaunchSpec((str(executable),), candidate, bundle)


class GpuProcessRegionEngine:
    def __init__(self, launch: LaunchSpec):
        self.launch = launch
        self.frames = None
        self.process_handle = None
        self.reader = None
        self.responses = Queue(maxsize=2)
        self.closing = Event()
        self.backend_status = 'GPU — starting'

    def initialize(self, cancel: Event):
        try:
            self.frames = SharedFrames('CarlCamGpu_' + uuid4().hex, create=True)
            command = [*self.launch.command, '--buffer', self.frames.memory.name,
                       '--candidate', str(self.launch.candidate)]
            self.process_handle = subprocess.Popen(
                command, cwd=self.launch.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            self.reader = Thread(target=self._read, name='CarlCamGpuReplies', daemon=True)
            self.reader.start()
            ready = self._reply(30, cancel)
            if ready != {'schema_version': PROTOCOL_VERSION, 'kind': 'ready', 'backend': 'cuda_fp32'}:
                raise GpuProcessError('gpu_initialization_failed')
            self.backend_status = 'GPU — full FP32 (experimental)'
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            while not self.closing.is_set():
                payload = self.process_handle.stdout.readline(MAX_MESSAGE_BYTES + 1)
                value = decode_message(payload) if payload else None
                try:
                    self.responses.put(value, timeout=.05)
                except Full:
                    return
                if value is None:
                    return
        except (OSError, ValueError):
            try:
                self.responses.put(None, timeout=.05)
            except Full:
                pass

    def _reply(self, timeout: float, cancel: Event | None = None):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if self.closing.is_set() or (cancel is not None and cancel.is_set()):
                raise GpuProcessError('gpu_cancelled')
            try:
                response = self.responses.get(timeout=.02)
            except Empty:
                continue
            if response is None or response.get('kind') == 'error':
                raise GpuProcessError('gpu_service_unavailable')
            return response
        raise GpuProcessError('gpu_timeout')

    def _send(self, message):
        if self.process_handle is None or self.process_handle.poll() is not None:
            raise GpuProcessError('gpu_service_unavailable')
        try:
            self.process_handle.stdin.write(encode_message(message))
            self.process_handle.stdin.flush()
        except (OSError, ValueError) as error:
            raise GpuProcessError('gpu_service_unavailable') from error

    def process(self, packet) -> RegionPacket:
        started = perf_counter()
        np.copyto(self.frames.input, canvas_rgb(packet.rgb))
        self._send({'schema_version': PROTOCOL_VERSION, 'kind': 'frame', 'frame_id': packet.frame_id,
                    'capture_time_ns': packet.capture_time_ns})
        response = self._reply(.15)
        expected = {'schema_version', 'kind', 'frame_id', 'capture_time_ns', 'inference_ms',
                    'regions_reliable'}
        duration = response.get('inference_ms')
        if (set(response) != expected or response.get('kind') != 'result'
                or type(response.get('frame_id')) is not int
                or response['frame_id'] != packet.frame_id
                or type(response.get('capture_time_ns')) is not int
                or response['capture_time_ns'] != packet.capture_time_ns
                or type(duration) not in (int, float) or not math.isfinite(duration)
                or type(response.get('regions_reliable')) is not bool
                or not 0 <= duration <= 30_000):
            raise GpuProcessError('gpu_frame_mismatch')
        masks = self.frames.output.copy()
        if (not np.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1
                or not np.allclose(masks.sum(axis=-1), 1., rtol=0., atol=2e-5)):
            raise GpuProcessError('gpu_invalid_masks')
        masks.flags.writeable = False
        region_planes = self.frames.prepared_regions.copy()
        signal_planes = self.frames.prepared_signals.copy()
        for planes in (region_planes, signal_planes):
            if not np.isfinite(planes).all() or planes.min() < 0 or planes.max() > 1:
                raise GpuProcessError('gpu_invalid_preparation')
            planes.flags.writeable = False
        if not response['regions_reliable'] and np.count_nonzero(region_planes):
            raise GpuProcessError('gpu_invalid_preparation')
        regions = Regions(*region_planes, response['regions_reliable'])
        stages = SignalStages(*signal_planes)
        return RegionPacket(packet.frame_id, packet.capture_time_ns, masks,
                            (perf_counter() - started) * 1000, regions, stages)

    def close(self):
        if self.process_handle is not None:
            try:
                if self.process_handle.poll() is None:
                    self._send({'schema_version': PROTOCOL_VERSION, 'kind': 'stop'})
                    self.process_handle.wait(timeout=.5)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                if self.process_handle.poll() is None:
                    self.process_handle.terminate()
                    try:
                        self.process_handle.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        self.process_handle.kill()
                        self.process_handle.wait(timeout=1)
            self.closing.set()
            if self.reader is not None:
                self.reader.join(timeout=.2)
            for stream in (self.process_handle.stdin, self.process_handle.stdout):
                stream.close()
            self.process_handle = None
        self.closing.set()
        if self.frames is not None:
            self.frames.close()
            self.frames = None


class AdaptiveRegionEngine:
    """Choose GPU only on request; any GPU failure visibly restores CPU masks."""

    def __init__(self, preference='cpu', *, launcher=runtime_launch,
                 gpu_factory=GpuProcessRegionEngine, cpu_factory=RegionEngine):
        if preference not in ('cpu', 'gpu'):
            raise ValueError('Unknown segmentation backend')
        self.preference = preference
        self.launcher, self.gpu_factory, self.cpu_factory = launcher, gpu_factory, cpu_factory
        self.engine = None
        self.backend_status = 'CPU'
        self.cancel = None

    def initialize(self, cancel: Event):
        self.cancel = cancel
        launch = self.launcher() if self.preference == 'gpu' else None
        if launch is not None and not cancel.is_set():
            try:
                self.engine = self.gpu_factory(launch)
                self.engine.initialize(cancel)
                self.backend_status = self.engine.backend_status
                return
            except (OSError, ValueError, RuntimeError):
                if self.engine is not None:
                    self.engine.close()
                self.engine = None
        if cancel.is_set():
            raise GpuProcessError('gpu_cancelled')
        self.engine = self.cpu_factory()
        self.backend_status = 'CPU — GPU unavailable' if self.preference == 'gpu' else 'CPU'

    def process(self, packet):
        try:
            return self.engine.process(packet)
        except (OSError, ValueError, RuntimeError):
            if self.preference != 'gpu' or self.backend_status.startswith('CPU'):
                raise
            self.engine.close()
            self.engine = None
            if self.cancel is not None and self.cancel.is_set():
                raise GpuProcessError('gpu_cancelled')
            self.engine = self.cpu_factory()
            self.backend_status = 'CPU — GPU stopped; restart camera to retry'
            return self.engine.process(packet)

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None
