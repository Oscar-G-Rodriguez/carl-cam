"""A transfer kit must remain static, private and unapproved."""

import builtins
import ctypes
import importlib.util
import json
import socket
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def preparer(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("vm_preparer", tools / "prepare_vision_vm_candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def transfer_fixture(preparer, tmp_path):
    root = tmp_path / "repo"
    build = root / "build"
    upstream = build / "vendor/mediapipe-f8ef212d"
    attempt_dir = build / "attempt"
    attempt_dir.mkdir(parents=True)
    installed = root / ".venv/Lib/site-packages/mediapipe/tasks/python"
    source_inputs = {}
    for name in ("WORKSPACE", "mediapipe/tasks/c/BUILD"):
        source = upstream / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fixture source: " + name)
        source_inputs[name] = preparer.identity(source)
        retained = attempt_dir / "source-inputs" / name
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_bytes(source.read_bytes())
    attempt = {"action": "build", "compiler_returncode": 0, "inputs_unchanged": True,
               "finished_utc": "2026-10-02T13:28:39Z", "source_commit": preparer.SOURCE_COMMIT,
               "input_hashes": {name: item["sha256"] for name, item in source_inputs.items()}}
    attempt_path = attempt_dir / "attempt.json"
    attempt_path.write_text(json.dumps(attempt))
    diff = attempt_dir / "source.diff"
    diff.write_text("fixture compatibility patch")
    wrappers = {}
    for name in preparer.WRAPPERS:
        source = upstream / "mediapipe/tasks/python" / name
        current = installed / name
        for path in (source, current):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"raise RuntimeError('audit file must never execute')\n")
        wrappers[name] = {"installed": preparer.identity(current), "source": preparer.identity(source),
                          "text_matches_after_newline_normalization": True}
    libraries = {}
    for name in preparer.LIBRARIES:
        path = build / "native" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture binary " + name.encode())
        libraries[name] = {"identity": preparer.identity(path), "imports": [
            {"name": "KERNEL32.dll", "kind": "windows_system",
             "identity": {"path": "NOT COPIED FROM HOST"}},
            {"name": "api-ms-win-crt-heap-l1-1-0.dll", "kind": "windows_api_set"}]}
    libraries["mediapipe_source.dll"]["imports"].append({
        "name": "opencv_core3411.dll", "kind": "source_build",
        "identity": libraries["opencv_core3411.dll"]["identity"]})
    record = {"schema_version": 1, "source_commit": preparer.SOURCE_COMMIT,
              "static_export_check_passed": True, "inspection_inputs_unchanged": True,
              "missing_exports": [], "attempt": preparer.identity(attempt_path),
              "source_diff": preparer.identity(diff), "source_inputs": source_inputs,
              "candidate": libraries["mediapipe_source.dll"]["identity"],
              "libraries": libraries, "wrappers": wrappers, **dict.fromkeys(preparer.UNVERIFIED, False)}
    static = build / "static.json"
    static.write_text(json.dumps(record))
    models = root / "src/carl_cam/assets/models"
    models.mkdir(parents=True)
    model_records = {}
    for name in preparer.MODELS:
        path = models / name
        path.write_bytes(b"public perception fixture " + name.encode())
        value = preparer.identity(path)
        model_records[name] = {"sha256": value["sha256"], "bytes": value["size"]}
    (models / "manifest.json").write_text(json.dumps({"schema_version": 1, "models": model_records}))
    (models / "Apache-2.0.txt").write_text("fixture model notice")
    (upstream / "LICENSE").write_text("fixture MediaPipe notice")
    opencv_license = build / "opencv-LICENSE"
    opencv_license.write_text("fixture OpenCV notice")
    (root / "uv.lock").write_text("fixture locked environment reference")
    (root / "pyproject.toml").write_text("fixture project reference")
    # Neither an unrelated file in the model directory nor any user-data tree
    # may expand the explicitly enumerated transfer set.
    (models / "creator-private.pt").write_text("private-marker-83b7")
    (root / "personal-data").mkdir()
    (root / "personal-data/secret.txt").write_text("private-marker-83b7")
    return root, static, opencv_license, build / "vision-vm-candidate-fixture"


def test_transfer_is_exact_and_never_loads_installs_traces_or_promotes(preparer, transfer_fixture, monkeypatch):
    imported = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        assert not name.startswith(("mediapipe", "carl_cam.vision")), "SDK import forbidden"
        return imported(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime, process or socket operation forbidden")

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    report = preparer.prepare(*transfer_fixture)
    destination = transfer_fixture[-1]
    assert report["preparation_verified"] and report["copy_hashes_verified"]
    assert all(report[name] is False for name in preparer.UNVERIFIED)
    assert all(report[name] is False for name in (
        "vm_started", "physical_camera_used", "personal_data_included",
        "host_runtime_environment_modified", "is_app_installer", "is_self_contained_runner",
        "public_redistribution_approved"))
    assert len([entry for entry in report["files"] if entry["role"] == "source_built_library"]) == 3
    assert len([entry for entry in report["files"] if entry["role"] == "public_perception_model"]) == 4
    assert not list(destination.rglob("libmediapipe.dll"))
    assert not list(destination.rglob("KERNEL32.dll"))
    assert not list(destination.rglob("*.pt"))
    for entry in report["files"]:
        current = preparer.identity(destination / entry["relative"])
        assert (current["size"], current["sha256"]) == (entry["size"], entry["sha256"])
        assert b"private-marker-83b7" not in (destination / entry["relative"]).read_bytes()


@pytest.mark.parametrize("field,value", (
    ("static_export_check_passed", False), ("static_export_check_passed", 1),
    ("inspection_inputs_unchanged", False), ("missing_exports", ["MpImageFree"]),
    ("source_commit", "unverified"), ("runtime_loaded", True),
    ("abi_verified", True), ("privacy_verified", True), ("numerical_verified", True), ("promoted", True),
))
def test_unverified_or_not_static_record_creates_no_destination(preparer, transfer_fixture, field, value):
    root, static, notice, destination = transfer_fixture
    record = json.loads(static.read_text())
    record[field] = value
    static.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        preparer.prepare(root, static, notice, destination)
    assert not destination.exists()


@pytest.mark.parametrize("role", ("candidate", "dependency", "wrapper", "source", "snapshot", "model"))
def test_changed_inputs_block_before_any_copy(preparer, transfer_fixture, role):
    root, static, notice, destination = transfer_fixture
    record = json.loads(static.read_text())
    choices = {
        "candidate": Path(record["candidate"]["path"]),
        "dependency": Path(record["libraries"]["opencv_core3411.dll"]["identity"]["path"]),
        "wrapper": Path(record["wrappers"][preparer.WRAPPERS[0]]["installed"]["path"]),
        "source": Path(record["source_inputs"]["WORKSPACE"]["path"]),
        "snapshot": root / "build/attempt/source-inputs/WORKSPACE",
        "model": root / "src/carl_cam/assets/models/hand_landmarker.task",
    }
    choices[role].write_bytes(b"changed fixture")
    with pytest.raises(ValueError):
        preparer.prepare(root, static, notice, destination)
    assert not destination.exists()


@pytest.mark.parametrize("name", ("../secret", "folder/../secret", "/absolute", "C:/secret",
                                 "C:secret", "folder\\secret", "folder//secret", "./secret"))
def test_audit_paths_cannot_escape_copy_root(preparer, name):
    with pytest.raises(ValueError, match="relative audit"):
        preparer.safe_relative(name)


def test_existing_or_broad_destinations_are_preserved(preparer, transfer_fixture):
    root, static, notice, destination = transfer_fixture
    destination.mkdir()
    marker = destination / "user-file.txt"
    marker.write_text("preserve")
    for target in (root, root / "build", root / "build/nested/vision-vm-candidate-fixture", destination):
        with pytest.raises(ValueError, match="fresh candidate"):
            preparer.prepare(root, static, notice, target)
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("change", ("source", "copy"))
def test_mid_copy_changes_retain_failure_without_passing_manifest(preparer, transfer_fixture, monkeypatch, change):
    root, static, notice, destination = transfer_fixture
    original = preparer.shutil.copyfileobj
    calls = 0

    def copy_and_change(reader, writer, *args, **kwargs):
        nonlocal calls
        original(reader, writer, *args, **kwargs)
        calls += 1
        if calls == 2:
            victim = static if change == "source" else destination / "evidence/static-manifest.json"
            victim.write_bytes(b"changed during preparation")

    monkeypatch.setattr(preparer.shutil, "copyfileobj", copy_and_change)
    with pytest.raises(ValueError):
        preparer.prepare(root, static, notice, destination)
    failure = json.loads((destination / "failure.json").read_text())
    assert failure["partial_files_retained"] and not failure["preparation_verified"]
    assert not (destination / "manifest.json").exists()
    assert all(failure[name] is False for name in preparer.UNVERIFIED)
