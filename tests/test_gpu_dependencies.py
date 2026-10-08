"""Wheel and loader-policy fixtures; no NVIDIA, ORT, or DLL initialization."""

import base64
import builtins
import csv
import hashlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from carl_cam import gpu_dependencies, gpu_regions
from carl_cam.vision_privacy import VisionPrivacyError

DLL_OWNERS = {
    "nvidia-cuda-runtime": ("cudart64_13.dll",),
    "nvidia-cuda-nvrtc": ("nvrtc-builtins64_130.dll", "nvrtc64_130_0.dll"),
    "nvidia-cublas": ("cublasLt64_13.dll", "cublas64_13.dll"),
    "nvidia-cufft": ("cufft64_12.dll",),
    "nvidia-curand": ("curand64_10.dll",),
    "nvidia-nvjitlink": ("nvJitLink_130_0.dll",),
    "nvidia-cudnn-cu13": tuple(name for name in gpu_dependencies.PRELOAD_DLL_NAMES
                               if name.startswith("cudnn")),
}


@pytest.fixture
def wheels(tmp_path, monkeypatch):
    distributions = {}
    for package, version in gpu_dependencies.NVIDIA_PACKAGE_VERSIONS.items():
        info = tmp_path / f"{package.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Name: {package}\nVersion: {version}\n")
        (info / "WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-win_amd64\n")
        directory = gpu_dependencies.CUDNN_DIRECTORY if package == "nvidia-cudnn-cu13" \
            else gpu_dependencies.CUDA_DIRECTORY
        payloads = [info / "METADATA", info / "WHEEL"]
        for name in DLL_OWNERS[package]:
            path = tmp_path.joinpath(*directory.parts, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"private DLL fixture " + name.encode())
            payloads.append(path)
        rows = []
        for path in payloads:
            payload = path.read_bytes()
            encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
            rows.append((path.relative_to(tmp_path).as_posix(), "sha256=" + encoded, len(payload)))
        rows.append(((info / "RECORD").relative_to(tmp_path).as_posix(), "", ""))
        with (info / "RECORD").open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        distributions[package] = importlib.metadata.PathDistribution(info)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distributions[name])
    return tmp_path, distributions


def test_inventory_is_static_complete_and_deterministic(wheels, monkeypatch):
    original = builtins.__import__

    def no_runtime(name, *arguments, **keywords):
        assert not name.startswith(("onnxruntime", "nvidia", "tensorflow", "mediapipe"))
        return original(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", no_runtime)
    snapshot = gpu_dependencies.inspect_nvidia_dependencies()
    assert len(snapshot.fingerprint) == 64
    assert [path.name for path in snapshot.preload_paths] == list(gpu_dependencies.PRELOAD_DLL_NAMES)
    assert len(snapshot.files) == 37  # sixteen payloads + three metadata files per wheel
    assert gpu_dependencies.inspect_nvidia_dependencies().fingerprint == snapshot.fingerprint


def test_changed_dll_bytes_cannot_use_stale_record(wheels):
    root, _ = wheels
    path = root.joinpath(*gpu_dependencies.CUDA_DIRECTORY.parts, "cudart64_13.dll")
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        gpu_dependencies.inspect_nvidia_dependencies()


def test_missing_dll_is_rejected(wheels):
    root, _ = wheels
    root.joinpath(*gpu_dependencies.CUDA_DIRECTORY.parts, "cudart64_13.dll").unlink()
    with pytest.raises(ValueError, match="Missing"):
        gpu_dependencies.inspect_nvidia_dependencies()


def test_unrecorded_transitive_native_neighbor_is_rejected(wheels):
    root, _ = wheels
    root.joinpath(*gpu_dependencies.CUDA_DIRECTORY.parts, "unexpected.dll").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unrecorded"):
        gpu_dependencies.inspect_nvidia_dependencies()


def test_wrong_version_is_rejected_before_reading_native_bytes(wheels):
    _, distributions = wheels
    distributions["nvidia-cuda-runtime"] = SimpleNamespace(version="13.4.92")
    with pytest.raises(ValueError, match="version differs"):
        gpu_dependencies.inspect_nvidia_dependencies()


def test_missing_wheel_is_rejected(wheels, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        gpu_dependencies.inspect_nvidia_dependencies()


def test_empty_wheel_inventory_is_rejected(wheels):
    _, distributions = wheels
    distributions["nvidia-cuda-runtime"] = SimpleNamespace(version="13.0.96", files=None)
    with pytest.raises(ValueError, match="incomplete"):
        gpu_dependencies.inspect_nvidia_dependencies()


def test_metadata_change_is_detected(wheels):
    root, _ = wheels
    info = root / "nvidia_cuda_runtime-13.0.96.dist-info" / "METADATA"
    info.write_text("Name: nvidia-cuda-runtime\nVersion: 13.0.96\nchanged\n")
    with pytest.raises(ValueError, match="checksum"):
        gpu_dependencies.inspect_nvidia_dependencies()


def test_unsafe_record_path_is_rejected(wheels):
    root, _ = wheels
    record = root / "nvidia_cuda_runtime-13.0.96.dist-info" / "RECORD"
    with record.open("a", newline="") as stream:
        csv.writer(stream).writerow(("../outside.dll", "", ""))
    with pytest.raises(ValueError, match="unsafe recorded path"):
        gpu_dependencies.inspect_nvidia_dependencies()


def test_relative_dependency_path_is_rejected():
    with pytest.raises(ValueError, match="unsafe"):
        gpu_dependencies.file_sha256(Path("relative.dll"))


def test_compiled_inventory_resolves_only_adjacent_wheel_metadata(wheels, monkeypatch):
    root, _ = wheels
    monkeypatch.setattr(gpu_dependencies, '__compiled__', True, raising=False)
    monkeypatch.setattr(gpu_dependencies, '__file__', str(root / 'carl_cam/gpu_dependencies.py'))
    monkeypatch.setattr(importlib.metadata, 'distribution',
                        lambda name: pytest.fail('Compiled helper must not search host packages'))
    result = gpu_dependencies.inspect_nvidia_dependencies()
    assert len(result.files) == 37
    assert all(path.is_relative_to(root) for path in result.preload_paths)


def test_compiled_inventory_does_not_borrow_missing_host_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_dependencies, '__compiled__', True, raising=False)
    monkeypatch.setattr(gpu_dependencies, '__file__', str(tmp_path / 'carl_cam/gpu_dependencies.py'))
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        gpu_dependencies.runtime_distribution('onnxruntime-gpu')


def test_runtime_metadata_rejects_unknown_package():
    with pytest.raises(ValueError, match='Unknown'):
        gpu_dependencies.runtime_distribution('../unexpected')


def test_loading_remains_held_before_inventory_or_native_api(monkeypatch):
    def forbidden():
        pytest.fail("Unapproved loader must not inspect or load native dependencies")

    monkeypatch.setattr(gpu_regions, "runtime_inventory", forbidden)
    monkeypatch.setattr(gpu_regions, "inspect_nvidia_dependencies", forbidden)
    with pytest.raises(VisionPrivacyError, match="not verified"):
        gpu_regions._load_cuda_dependencies(None)


def test_changed_combined_inventory_cannot_load(monkeypatch):
    monkeypatch.setattr(gpu_regions, "APPROVED_GPU_CONFIGURATIONS",
                        frozenset({("candidate", "approved-old-inventory")}))
    monkeypatch.setattr(gpu_regions, "runtime_inventory", lambda: "changed-inventory")
    with pytest.raises(VisionPrivacyError, match="not verified"):
        gpu_regions._load_cuda_dependencies(SimpleNamespace(evidence_sha256="candidate"))


@pytest.fixture
def native_loader_fixture(wheels, monkeypatch):
    """Only fake Windows API objects; no DLL constructor is actually called."""
    import ctypes
    from contextlib import contextmanager

    dependencies = gpu_dependencies.inspect_nvidia_dependencies()
    candidate = SimpleNamespace(evidence_sha256="candidate")
    monkeypatch.setattr(gpu_regions, "APPROVED_GPU_CONFIGURATIONS",
                        frozenset({("candidate", "fixture-inventory")}))
    monkeypatch.setattr(gpu_regions, "runtime_inventory", lambda: "fixture-inventory")
    monkeypatch.setattr(gpu_regions, "inspect_nvidia_dependencies", lambda: dependencies)
    state = {"handles": {}, "loads": [], "open_directories": set(), "wrong_origin": False,
             "fail_load": False}

    @contextmanager
    def add_directory(path):
        state["open_directories"].add(path)
        try:
            yield
        finally:
            state["open_directories"].remove(path)

    def module_handle(name):
        if state["wrong_origin"]:
            state["handles"][999] = str(wheels[0] / "unexpected" / name)
            return 999
        return next((handle for handle, path in state["handles"].items()
                     if Path(path).name == name), 0)

    def module_filename(handle, buffer, maximum):
        buffer.value = state["handles"][handle]
        return len(buffer.value)

    kernel = SimpleNamespace(GetModuleHandleW=module_handle, GetModuleFileNameW=module_filename)

    def load(path, **options):
        if path == "kernel32":
            assert options == {"use_last_error": True}
            return kernel
        assert Path(path).is_absolute()
        assert options == {"winmode": 0x1100}
        assert state["open_directories"] == {str(path.parent) for path in dependencies.preload_paths}
        if state["fail_load"]:
            raise OSError("private loader fixture failure")
        handle = len(state["loads"]) + 1
        state["handles"][handle] = path
        state["loads"].append(path)
        return SimpleNamespace(_handle=handle)

    monkeypatch.setattr(ctypes, "WinDLL", load)
    monkeypatch.setattr(gpu_regions.os, "add_dll_directory", add_directory)
    return candidate, dependencies, state


def test_loader_uses_absolute_ordered_paths_and_keeps_directory_lease(native_loader_fixture):
    candidate, dependencies, state = native_loader_fixture
    lease = gpu_regions._load_cuda_dependencies(candidate)
    assert state["loads"] == [str(path) for path in dependencies.preload_paths]
    assert len(lease) == len(dependencies.preload_paths) + 1
    assert state["open_directories"]
    lease[0].close()
    lease[0].close()
    assert not state["open_directories"]


def test_loader_rejects_existing_wrong_origin_without_default_search(native_loader_fixture):
    candidate, _, state = native_loader_fixture
    state["wrong_origin"] = True
    with pytest.raises(RuntimeError, match="unexpected location"):
        gpu_regions._load_cuda_dependencies(candidate)
    assert not state["loads"] and not state["open_directories"]


def test_loader_propagates_native_failure_and_releases_search_directories(native_loader_fixture):
    candidate, _, state = native_loader_fixture
    state["fail_load"] = True
    with pytest.raises(OSError, match="fixture failure"):
        gpu_regions._load_cuda_dependencies(candidate)
    assert not state["loads"] and not state["open_directories"]


@pytest.mark.parametrize("name", ["../outside.dll", "C:/outside.dll", "nested\\outside.dll"])
def test_raw_unsafe_missing_entries_are_not_filtered(wheels, name):
    root, _ = wheels
    record = root / "nvidia_cuda_runtime-13.0.96.dist-info" / "RECORD"
    with record.open("a", newline="") as stream:
        csv.writer(stream).writerow((name, "", ""))
    with pytest.raises(ValueError, match="unsafe recorded path"):
        gpu_dependencies.inspect_nvidia_dependencies()


def test_benchmark_reaches_hold_without_qt_or_runtime_imports(monkeypatch, tmp_path, capsys):
    import importlib.util
    import sys

    original = builtins.__import__

    def no_ui_or_runtime(name, *arguments, **keywords):
        assert not name.startswith(("PySide6", "onnxruntime", "nvidia", "mediapipe",
                                    "tensorflow", "benchmark_layer_blending"))
        return original(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", no_ui_or_runtime)
    tool = Path(__file__).resolve().parents[1] / "tools" / "benchmark_gpu_regions.py"
    spec = importlib.util.spec_from_file_location("gpu_benchmark_fixture", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(tool), "--candidate", str(tmp_path / "absent")])
    assert module.main() == 2
    assert "Required GPU benchmark held" in capsys.readouterr().out
