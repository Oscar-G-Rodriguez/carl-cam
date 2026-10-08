"""Optional, unpromoted six-class ONNX/CUDA segmentation adapter.

Python owns orchestration; an approved native runtime would execute the graph.
No runtime is approved yet. Construction fails before importing ONNX Runtime
until the exact candidate and installed package inventory have been reviewed.
The existing CPU engine and application default are intentionally unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from carl_cam.gpu_dependencies import inspect_nvidia_dependencies, runtime_distribution
from carl_cam.vision import RegionPacket, mirrored_canvas
from carl_cam.vision_privacy import VisionPrivacyError

SOURCE_MODEL_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
GPU_PRIVACY_HOLD = "GPU segmentation runtime is not verified. The CPU default is unchanged."
# Tuples of (exact preparation record SHA-256, combined ORT/NVIDIA inventory SHA-256).
# Production promotion still requires actual loading, parity and performance tests.
APPROVED_GPU_CONFIGURATIONS: frozenset[tuple[str, str]] = frozenset()
# Developer experiments permit only this structurally checked FP32 export and
# exact ORT/NVIDIA inventory. The October 3 policy change removes the network
# gate, not integrity/parity checks or the distinction from app promotion.
EXPERIMENTAL_GPU_CONFIGURATIONS: frozenset[tuple[str, str]] = frozenset({(
    'ab07f9774b2d1e64106886be7f3aed1ebfb9d4440a29d4d67f1897ef62e126f8',
    'a838900a566bba0124e38c086a63aa59e3a2a292e0b67d642ab6e5f8c3507d02',
)})


@dataclass(frozen=True, slots=True)
class GpuCandidate:
    model: Path
    input_name: str
    output_name: str
    evidence_sha256: str


def _plain_file(path: Path, maximum: int, *, allow_empty_python: bool = False) -> bytes:
    minimum = 0 if allow_empty_python and path.suffix == '.py' else 1
    if (any(item.is_symlink() or item.is_junction() for item in (path, *path.parents))
            or not path.is_file() or not minimum <= path.stat().st_size <= maximum):
        raise ValueError("GPU candidate is missing or unsafe")
    return path.read_bytes()


def inspect_candidate(directory: Path) -> GpuCandidate:
    """Validate public-model provenance and shapes; no runtime import/inference."""
    evidence = _plain_file(directory / "evidence.json", 256 * 1024)
    report = json.loads(evidence)
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("source_model_sha256") != SOURCE_MODEL_SHA256
            or report.get("conversion_verified") is not True
            or not isinstance(report.get("onnx_contract"), dict)
            or report.get("source_model_unchanged") is not True):
        raise ValueError("GPU candidate conversion has not been structurally verified")
    source = report.get("source_contract", {})
    if (not isinstance(source, dict)
            or source.get("normalization") != {"mean": [127.5], "std": [127.5]}
            or source.get("activation") != "SOFTMAX"
            or source.get("activation_explicit") is not True
            or not isinstance(source.get("associated_files"), dict)
            or source.get("associated_files", {}).get("labels.txt") !=
                "background\nhair\nbody-skin\nface-skin\nclothes\nothers\n"):
        raise ValueError("GPU candidate has incompatible category or processing metadata")
    interface = report.get("onnx_contract", {})
    names = []
    for role, shape in (("inputs", [1, 256, 256, 3]), ("outputs", [1, 256, 256, 6])):
        entries = interface.get(role)
        if (not isinstance(entries, list) or len(entries) != 1
                or not isinstance(entries[0], dict)
                or entries[0].get("shape") != shape or entries[0].get("dtype") != "FLOAT32"
                or not isinstance(entries[0].get("name"), str) or not entries[0]["name"]):
            raise ValueError("GPU candidate has an incompatible tensor interface")
        names.append(entries[0]["name"])
    model = directory / "selfie_multiclass_fp32.onnx"
    payload = _plain_file(model, 32 * 1024 * 1024)
    if (hashlib.sha256(payload).hexdigest() != report.get("onnx_sha256")
            or len(payload) != report.get("onnx_bytes")):
        raise ValueError("GPU candidate checksum failed")
    return GpuCandidate(model, *names, hashlib.sha256(evidence).hexdigest())


def runtime_inventory() -> str:
    """Hash installed Python/native ORT and pinned NVIDIA wheels before import.

This inventories package payloads, not every dynamically loaded Windows library.
Promotion additionally requires native loading, parity and packaged-app checks.
"""
    distribution = runtime_distribution("onnxruntime-gpu")
    if distribution.version != "1.30.0":
        raise VisionPrivacyError(GPU_PRIVACY_HOLD)
    files = []
    for relative in distribution.files or ():
        if str(relative).startswith("onnxruntime/") and str(relative).endswith(
                (".py", ".dll", ".pyd")):
            path = Path(distribution.locate_file(relative))
            payload = _plain_file(path, 512 * 1024 * 1024, allow_empty_python=True)
            files.append((str(relative), hashlib.sha256(payload).hexdigest()))
    if not files or not any(name.endswith(".pyd") for name, _ in files):
        raise VisionPrivacyError(GPU_PRIVACY_HOLD)
    dependencies = inspect_nvidia_dependencies()
    record = {"version": distribution.version, "files": sorted(files),
              "nvidia_fingerprint": dependencies.fingerprint}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def require_verified_gpu_candidate(directory: Path, *, experiment: bool = False) -> GpuCandidate:
    configurations = EXPERIMENTAL_GPU_CONFIGURATIONS if experiment else APPROVED_GPU_CONFIGURATIONS
    if not configurations:
        raise VisionPrivacyError(GPU_PRIVACY_HOLD)
    try:
        candidate = inspect_candidate(directory)
        identity = runtime_inventory()
        if (candidate.evidence_sha256, identity) not in configurations:
            raise VisionPrivacyError(GPU_PRIVACY_HOLD)
        return candidate
    except (OSError, ValueError, TypeError, KeyError,
            importlib.metadata.PackageNotFoundError) as error:
        raise VisionPrivacyError(GPU_PRIVACY_HOLD) from error


def _load_cuda_dependencies(candidate: GpuCandidate, *, experiment: bool = False) -> tuple:
    """Load only a promoted package set, by absolute Windows paths.

ORT's convenience preloader can fall back to bare DLL names after a failure.
Here we reject collisions and propagate errors instead. Package promotion still
requires actual loaded-dependency, parity and installed-app verification.
"""
    configurations = EXPERIMENTAL_GPU_CONFIGURATIONS if experiment else APPROVED_GPU_CONFIGURATIONS
    if not configurations:
        raise VisionPrivacyError(GPU_PRIVACY_HOLD)
    try:
        if (candidate.evidence_sha256, runtime_inventory()) not in configurations:
            raise VisionPrivacyError(GPU_PRIVACY_HOLD)
        dependencies = inspect_nvidia_dependencies()
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        raise VisionPrivacyError(GPU_PRIVACY_HOLD) from error
    if os.name != "nt":
        raise RuntimeError("The experimental GPU loader requires Windows")
    import ctypes
    from contextlib import ExitStack
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    module_handle = kernel.GetModuleHandleW
    module_handle.argtypes = [wintypes.LPCWSTR]
    module_handle.restype = wintypes.HMODULE
    module_filename = kernel.GetModuleFileNameW
    module_filename.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    module_filename.restype = wintypes.DWORD

    def check_origin(path: Path, handle) -> None:
        buffer = ctypes.create_unicode_buffer(32768)
        length = module_filename(handle, buffer, len(buffer))
        if (not length or length >= len(buffer)
                or os.path.normcase(os.path.abspath(buffer.value)) != os.path.normcase(str(path))):
            raise RuntimeError(f"GPU DLL was loaded from an unexpected location: {path.name}")

    libraries = []
    scope = ExitStack()
    try:
        for directory in sorted({path.parent for path in dependencies.preload_paths}):
            scope.enter_context(os.add_dll_directory(str(directory)))
        for path in dependencies.preload_paths:
            existing = module_handle(path.name)
            if existing:
                check_origin(path, existing)
            library = ctypes.WinDLL(str(path), winmode=0x1100)
            check_origin(path, library._handle)
            libraries.append(library)
        # Recheck all after dependency resolution, including indirect early loads.
        for path, library in zip(dependencies.preload_paths, libraries, strict=True):
            check_origin(path, library._handle)
    except BaseException:
        scope.close()
        raise
    # Keep process-local search directories alive for delayed cuDNN loads. The
    # lease closes with the session; it does not edit PATH or Windows settings.
    return (scope, *libraries)


def prepare_tensor(rgb: np.ndarray) -> np.ndarray:
    """Full-ROI OpenCV projection matching the pinned CPU preprocessing recipe.

The ROI corners map to 0..256, rather than cv2.resize's half-pixel mapping.
Native numerical parity still requires the retained paired-model test.
"""
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
            or rgb.shape != (360, 640, 3)):
        raise ValueError("Expected the canonical RGB region image")
    corners = cv2.boxPoints(((320., 180.), (640., 360.), 0.))
    destination = np.array([[0., 256.], [0., 0.], [256., 0.], [256., 256.]], np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(rgb, transform, (256, 256), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    normalized = cv2.addWeighted(warped, 2. / 255., warped, 0., -1., dtype=cv2.CV_32F)
    return np.ascontiguousarray(normalized[None])


def confidence_masks(logits: np.ndarray) -> np.ndarray:
    """Apply metadata softmax before independent per-class bilinear resizing."""
    if (not isinstance(logits, np.ndarray) or logits.dtype != np.float32
            or logits.shape != (1, 256, 256, 6)
            or not np.isfinite(logits).all()):
        raise ValueError("GPU model produced incompatible logits")
    values = logits[0] - np.max(logits[0], axis=-1, keepdims=True)
    np.exp(values, out=values)
    values /= np.sum(values, axis=-1, keepdims=True)
    masks = np.stack([cv2.resize(values[..., index], (640, 360),
                                 interpolation=cv2.INTER_LINEAR) for index in range(6)], axis=-1)
    np.clip(masks, 0., 1., out=masks)
    masks.flags.writeable = False
    return masks


class GpuRegionEngine:
    """Experimental factory for VisionWorker; never selected automatically."""

    def __init__(self, candidate_directory: Path, *, experiment: bool = False) -> None:
        self.candidate = require_verified_gpu_candidate(candidate_directory, experiment=experiment)
        self._cuda_libraries = ()
        self.session = None
        try:
            self._cuda_libraries = _load_cuda_dependencies(self.candidate, experiment=experiment)
            import onnxruntime as ort
            ort.disable_telemetry_events()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("The CUDA inference provider is unavailable")
            options = ort.SessionOptions()
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(self.candidate.model), sess_options=options,
                providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})])
            self.session.disable_fallback()
            if "CUDAExecutionProvider" not in self.session.get_providers():
                raise RuntimeError("GPU initialization fell back to CPU")
        except BaseException:
            self.close()
            raise
        self.last_stage_ms: dict[str, float] = {}

    def process(self, packet) -> RegionPacket:
        started = perf_counter()
        rgb = cv2.resize(mirrored_canvas(packet), (640, 360), interpolation=cv2.INTER_AREA)
        tensor = prepare_tensor(rgb)
        before_inference = perf_counter()
        outputs = self.session.run([self.candidate.output_name],
                                   {self.candidate.input_name: tensor})
        after_inference = perf_counter()
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise ValueError("GPU model produced incompatible outputs")
        masks = confidence_masks(outputs[0])
        ended = perf_counter()
        self.last_stage_ms = {"preprocess": (before_inference - started) * 1000,
                              "inference": (after_inference - before_inference) * 1000,
                              "postprocess": (ended - after_inference) * 1000}
        return RegionPacket(packet.frame_id, packet.capture_time_ns, masks,
                            (ended - started) * 1000)

    def close(self) -> None:
        self.session = None
        if self._cuda_libraries:
            self._cuda_libraries[0].close()
        self._cuda_libraries = ()
