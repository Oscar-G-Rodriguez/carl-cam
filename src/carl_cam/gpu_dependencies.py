"""Static NVIDIA wheel inventory. Importing this module loads no GPU libraries.

Wheel RECORD checks establish consistency with installed package metadata, not
publisher trust, privacy, successful loading, or numerical correctness. The
separate pip receipt preserves download origins and wheel hashes for review.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

NVIDIA_PACKAGE_VERSIONS = {
    "nvidia-cuda-runtime": "13.0.96",
    "nvidia-cuda-nvrtc": "13.0.88",
    "nvidia-cublas": "13.0.2.14",
    "nvidia-cufft": "12.0.0.61",
    "nvidia-curand": "10.4.0.35",
    "nvidia-nvjitlink": "13.0.88",
    "nvidia-cudnn-cu13": "9.13.1.26",
}
GPU_PACKAGE_VERSIONS = {**NVIDIA_PACKAGE_VERSIONS, "onnxruntime-gpu": "1.30.0"}
CUDA_DIRECTORY = PurePosixPath("nvidia/cu13/bin/x86_64")
CUDNN_DIRECTORY = PurePosixPath("nvidia/cudnn/bin")
# Dependency order for explicit, absolute-path preloading. Auxiliary DLLs are
# still inventoried but unused alternate NVRTC/BLAS/FFT entry points are not loaded.
PRELOAD_DLL_NAMES = (
    "nvJitLink_130_0.dll", "nvrtc-builtins64_130.dll", "nvrtc64_130_0.dll",
    "curand64_10.dll", "cublasLt64_13.dll", "cublas64_13.dll", "cufft64_12.dll",
    "cudart64_13.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_adv64_9.dll",
    "cudnn_heuristic64_9.dll", "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll", "cudnn_graph64_9.dll", "cudnn64_9.dll",
)


@dataclass(frozen=True, slots=True)
class DependencyFile:
    package: str
    relative: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class NvidiaDependencies:
    fingerprint: str
    files: tuple[DependencyFile, ...]
    preload_paths: tuple[Path, ...]


def runtime_distribution(package: str) -> importlib.metadata.Distribution:
    """Resolve only the fixed GPU wheel metadata adjacent to a compiled helper.

The helper has no site-packages search path. Original wheel metadata and payloads
are staged next to it, where their exact inventory is checked before CUDA loads.
Source execution retains the ordinary private-environment metadata lookup.
"""
    if package not in GPU_PACKAGE_VERSIONS:
        raise ValueError("Unknown GPU runtime package")
    if "__compiled__" not in globals():
        return importlib.metadata.distribution(package)
    root = Path(__file__).absolute().parents[1]
    metadata = root / (package.replace("-", "_") + "-"
                       + GPU_PACKAGE_VERSIONS[package] + ".dist-info")
    if (not metadata.is_dir()
            or any(item.is_symlink() or item.is_junction() for item in (metadata, *metadata.parents))):
        raise importlib.metadata.PackageNotFoundError(package)
    return importlib.metadata.PathDistribution(metadata)


def file_sha256(path: Path) -> str:
    """Bounded-memory hashing; reject links and reparse-point parent aliases."""
    if (not path.is_absolute() or not path.is_file()
            or any(item.is_symlink() or item.is_junction() for item in (path, *path.parents))
            or not 0 < path.stat().st_size <= 2 * 1024**3):
        raise ValueError("Missing or unsafe GPU dependency file")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_nvidia_dependencies() -> NvidiaDependencies:
    """Check pinned versions, recorded hashes, exact DLL origins, and completeness.

No NVIDIA/ORT package import, DLL loading, camera, or inference occurs here.
Absolute locations are retained for loading; the fingerprint is portable across
equivalent environment roots and covers versions, metadata, licenses and payloads.
"""
    inventory = []
    dll_paths: dict[str, Path] = {}
    bin_directories = set()
    for package, version in NVIDIA_PACKAGE_VERSIONS.items():
        distribution = runtime_distribution(package)
        if distribution.version != version:
            raise ValueError(f"GPU dependency version differs: {package}")
        if not distribution.files:
            raise ValueError(f"GPU dependency inventory is incomplete: {package}")
        record_text = distribution.read_text("RECORD")
        if not isinstance(record_text, str) or not 0 < len(record_text) <= 5 * 1024**2:
            raise ValueError(f"GPU dependency RECORD is missing or unsafe: {package}")
        selected = []
        recorded_paths = set()
        # Distribution.files filters nonexistent entries in recent Python
        # releases. Parse raw RECORD so missing/unsafe entries remain visible.
        for row in csv.reader(io.StringIO(record_text)):
            if len(row) != 3:
                raise ValueError("GPU wheel has an invalid RECORD row")
            name, recorded_hash, recorded_size = row
            normalized = PurePosixPath(name)
            if (not name or normalized.is_absolute() or ".." in normalized.parts
                    or "\\" in name or ":" in name or "\0" in name):
                raise ValueError("GPU wheel contains an unsafe recorded path")
            if name in recorded_paths:
                raise ValueError("GPU wheel has duplicate RECORD paths")
            recorded_paths.add(name)
            if (normalized.suffix.lower() not in (".dll", ".pyd", ".py")
                    and normalized.name not in ("METADATA", "WHEEL", "RECORD")
                    and "licenses" not in normalized.parts):
                continue
            path = Path(distribution.locate_file(normalized)).absolute()
            digest = file_sha256(path)
            size = path.stat().st_size
            if normalized.name != "RECORD":
                encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
                if recorded_hash != "sha256=" + encoded or recorded_size != str(size):
                    raise ValueError(f"GPU dependency RECORD checksum differs: {package}/{name}")
            selected.append(DependencyFile(package, name, path, size, digest))
            if normalized.suffix.lower() in (".dll", ".pyd"):
                expected = CUDNN_DIRECTORY if package == "nvidia-cudnn-cu13" else CUDA_DIRECTORY
                if normalized.suffix.lower() != ".dll" or normalized.parent != expected:
                    raise ValueError("GPU DLL is outside its expected NVIDIA package directory")
                key = normalized.name.lower()
                if key in dll_paths:
                    raise ValueError("GPU packages record an ambiguous DLL name")
                dll_paths[key] = path
                bin_directories.add(path.parent)
        if (not any(item.relative.endswith(".dll") for item in selected)
                or not {"METADATA", "WHEEL", "RECORD"}.issubset(
                    {PurePosixPath(item.relative).name for item in selected})):
            raise ValueError(f"GPU dependency inventory is incomplete: {package}")
        inventory.extend(selected)
    # Avoid loading unrecorded native neighbors as transitive dependencies.
    actual = {path.absolute() for directory in bin_directories for path in directory.iterdir()
              if path.suffix.lower() in (".dll", ".pyd")}
    if actual != set(dll_paths.values()):
        raise ValueError("NVIDIA directories contain missing or unrecorded native files")
    if not all(name.lower() in dll_paths for name in PRELOAD_DLL_NAMES):
        raise ValueError("Required CUDA/cuDNN DLL is missing")
    files = tuple(sorted(inventory, key=lambda item: (item.package, item.relative)))
    record = {"schema_version": 1, "versions": NVIDIA_PACKAGE_VERSIONS,
              "files": [(item.package, item.relative, item.size, item.sha256) for item in files],
              "preload_order": PRELOAD_DLL_NAMES}
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return NvidiaDependencies(identity, files,
                              tuple(dll_paths[name.lower()] for name in PRELOAD_DLL_NAMES))
