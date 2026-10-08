"""Offline environment preparation must never initialize vision or networking."""

import base64
import builtins
import ctypes
import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def guest_tools():
    folder = Path(__file__).resolve().parents[1] / "tools"
    modules = []
    for name in ("prepare_vision_guest_environment", "verify_vision_guest_environment"):
        spec = importlib.util.spec_from_file_location(name, folder / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


@pytest.fixture
def guest_fixture(guest_tools, tmp_path, monkeypatch):
    preparer, _verifier = guest_tools
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "tools").mkdir()
    source = Path(__file__).resolve().parents[1] / "tools/verify_vision_guest_environment.py"
    (root / "tools/verify_vision_guest_environment.py").write_bytes(source.read_bytes())
    gui = Path(__file__).resolve().parents[1] / "tools/vision_guest_checks.py"
    (root / "tools/vision_guest_checks.py").write_bytes(gui.read_bytes())
    (root / "uv.lock").write_text(
        '[[package]]\nname = "mediapipe"\nversion = "0.10.35"\n'
        'source = { registry = "https://pypi.org/simple" }\n')
    runtime = tmp_path / "cpython-3.12.14-windows-x86_64-none"
    runtime.mkdir()
    for name in ("python.exe", "pythonw.exe", "python3.dll", "python312.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt", "BUILD"):
        (runtime / name).write_bytes(b"fixture runtime: " + name.encode())
    for name in ("Lib", "DLLs", "tcl"):
        (runtime / name).mkdir()
        (runtime / name / "fixture.txt").write_text("stdlib fixture")
    (runtime / "Lib/site-packages").mkdir()
    (runtime / "Lib/site-packages/private.py").write_text("DO NOT COPY")
    site = tmp_path / "site"
    info = site / "mediapipe-0.10.35.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: mediapipe\nVersion: 0.10.35\n")
    (site / "mediapipe").mkdir()
    (site / "mediapipe/__init__.py").write_text("raise AssertionError('never import')\n")
    held = site / preparer.HELD_DLL
    held.parent.mkdir(parents=True)
    held.write_bytes(b"held telemetry binary fixture")
    record_lines = []
    for path in (info / "METADATA", site / "mediapipe/__init__.py", held):
        digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=")
        record_lines.append(f"{path.relative_to(site).as_posix()},sha256={digest},{path.stat().st_size}\n")
    record_lines.append("mediapipe-0.10.35.dist-info/RECORD,,\n")
    (info / "RECORD").write_text("".join(record_lines))
    kit = root / "build/candidate"
    (kit / "libs").mkdir(parents=True)
    entries = []
    for name in ("mediapipe_source.dll", "opencv_core3411.dll", "opencv_imgproc3411.dll"):
        path = kit / "libs" / name
        path.write_bytes(b"source-built fixture: " + name.encode())
        entries.append({"relative": "libs/" + name, **preparer.identity(path)})
    monkeypatch.setattr(preparer, "CANDIDATE_HASH", preparer.identity(kit / "libs/mediapipe_source.dll")["sha256"])
    record = {"schema_version": 1, "source_commit": preparer.SOURCE_COMMIT,
              "preparation_verified": True, "source_inputs_rechecked": True,
              "copy_hashes_verified": True, "files": entries,
              **dict.fromkeys(("runtime_loaded", "abi_verified", "privacy_verified",
                               "numerical_verified", "promoted", "personal_data_included",
                               "is_app_installer"), False)}
    (kit / "manifest.json").write_text(json.dumps(record))
    destination = root / "build/vision-guest-environment-fixture"
    return root, runtime, site, kit, destination


def test_actual_copy_and_verifier_never_load_or_network(guest_tools, guest_fixture, monkeypatch):
    preparer, verifier = guest_tools
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith(("mediapipe", "numpy", "cv2", "carl_cam", "torch")):
            raise AssertionError("Native SDK import forbidden")
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Process, socket or native load forbidden")

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    result = preparer.prepare(*guest_fixture)
    destination = guest_fixture[-1]
    checksum = verifier.file_hash(destination / "environment.json")
    verified = verifier.verify(destination, checksum)
    assert verified["files_verified"] == len(result["files"])
    assert all(result[k] is False for k in preparer.UNVERIFIED)
    staged = destination / "python/packages/mediapipe/tasks/c/libmediapipe.dll"
    assert staged.read_bytes() == (guest_fixture[3] / "libs/mediapipe_source.dll").read_bytes()
    assert (guest_fixture[2] / preparer.HELD_DLL).read_bytes() == b"held telemetry binary fixture"
    assert (destination / "python/python312._pth").read_text() == preparer.ISOLATED_PATHS
    assert "import site" not in preparer.ISOLATED_PATHS
    assert not (destination / "python/Lib/site-packages").exists()
    assert len([e for e in result["excluded"] if e["reason"] == "telemetry_held_wheel_dll"]) == 1


@pytest.mark.parametrize("path", (
    "../escape", "/absolute", "C:/host", "C:host", "a\\b", "a//b", "a/./b",
    "a/../b", "a. ", "a.", "CON", "nul.txt", "COM1.dll", "LPT9", "a:stream",
    "a?b", 'a"b', "a\x00b", "a|b",
))
def test_windows_transfer_aliases_rejected(guest_tools, path):
    with pytest.raises(ValueError, match="Unsafe transfer"):
        guest_tools[0].relative(path)


@pytest.mark.parametrize("change", ("version", "record_hash", "missing_hash", "extra_kit",
                                   "candidate", "flag", "startup_hook", "collision"))
def test_changed_inputs_fail_before_creating_destination(guest_tools, guest_fixture, monkeypatch, change):
    preparer, _ = guest_tools
    root, runtime, site, kit, destination = guest_fixture
    if change == "version":
        (root / "uv.lock").write_text((root / "uv.lock").read_text().replace("0.10.35", "0.10.34"))
    elif change == "record_hash":
        (site / "mediapipe/__init__.py").write_text("changed")
    elif change == "missing_hash":
        record = site / "mediapipe-0.10.35.dist-info/RECORD"
        record.write_text(record.read_text() + "mediapipe/unknown.py,,\n")
        (site / "mediapipe/unknown.py").write_text("unknown")
    elif change == "extra_kit":
        (kit / "secret.pt").write_text("private")
    elif change == "candidate":
        (kit / "libs/mediapipe_source.dll").write_bytes(b"changed native bytes")
    elif change == "flag":
        record = json.loads((kit / "manifest.json").read_text())
        record["privacy_verified"] = True
        (kit / "manifest.json").write_text(json.dumps(record))
    elif change == "startup_hook":
        (runtime / "Lib/sitecustomize.py").write_text("startup hook")
    else:
        original = preparer.runtime_plan

        def collision_plan(path):
            entries = original(path)
            name, source, expected, role = entries[0]
            return [*entries, (name.upper(), source, expected, role)]

        monkeypatch.setattr(preparer, "runtime_plan", collision_plan)
    with pytest.raises(ValueError):
        preparer.prepare(*guest_fixture)
    assert not destination.exists()


@pytest.mark.parametrize("change", ("byte", "extra", "missing", "manifest", "unsafe", "claim"))
def test_transferred_changes_are_not_privacy_passes(guest_tools, guest_fixture, change):
    preparer, verifier = guest_tools
    preparer.prepare(*guest_fixture)
    destination = guest_fixture[-1]
    manifest = destination / "environment.json"
    checksum = verifier.file_hash(manifest)
    if change == "byte":
        (destination / "python/python.exe").write_bytes(b"changed")
    elif change == "extra":
        (destination / "personal.pt").write_text("unexpected")
    elif change == "missing":
        (destination / "python/python.exe").unlink()
    elif change == "manifest":
        manifest.write_text(manifest.read_text() + " ")
    else:
        record = json.loads(manifest.read_text())
        if change == "unsafe":
            record["files"][0]["relative"] = "../outside"
        else:
            record["privacy_verified"] = True
        manifest.write_text(json.dumps(record))
        checksum = verifier.file_hash(manifest)
    with pytest.raises((ValueError, OSError)):
        verifier.verify(destination, checksum)


def test_existing_destination_is_preserved(guest_tools, guest_fixture):
    destination = guest_fixture[-1]
    destination.mkdir()
    marker = destination / "user.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="fresh guest"):
        guest_tools[0].prepare(*guest_fixture)
    assert marker.read_text() == "preserve"


def test_unreviewed_marker_cannot_silently_change_dependency_set(guest_tools):
    lock = {"package": [{"name": "mediapipe", "version": "0.10.35",
                         "source": {"registry": "https://pypi.org/simple"},
                         "dependencies": [{"name": "other", "marker": "sys_platform == 'linux'"}]}]}
    with pytest.raises(ValueError, match="Unreviewed guest dependency marker"):
        guest_tools[0].closure(lock)


@pytest.mark.parametrize("entry", ("../../private.txt", "../../Scripts/../../private.txt",
                                   "../../share/man/man1/ttx.1"))
def test_unknown_external_record_entry_cannot_expand_copy_set(guest_tools, guest_fixture, entry):
    site = guest_fixture[2]
    record = site / "mediapipe-0.10.35.dist-info/RECORD"
    record.write_text(record.read_text() + entry + ",,\n")
    with pytest.raises(ValueError):
        guest_tools[0].prepare(*guest_fixture)
    assert not guest_fixture[-1].exists()


def test_interrupted_copy_retains_failure(guest_tools, guest_fixture, monkeypatch):
    preparer, _ = guest_tools

    def failed_copy(*args, **kwargs):
        raise OSError("fixture interrupted copy")

    monkeypatch.setattr(preparer.shutil, "copyfileobj", failed_copy)
    with pytest.raises(OSError):
        preparer.prepare(*guest_fixture)
    failure = json.loads((guest_fixture[-1] / "failure.json").read_text())
    assert failure["partial_files_retained"] and not failure["preparation_verified"]
    assert not (guest_fixture[-1] / "environment.json").exists()


@pytest.fixture
def graphical_tools(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    modules = []
    for name in ("prepare_vision_guest_disc", "vision_guest_checks"):
        spec = importlib.util.spec_from_file_location(name, folder / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def isolated_runtime(environment):
    python = environment / "python"
    return SimpleNamespace(
        executable=str(python / "pythonw.exe"), version_info=(3, 12, 14),
        flags=SimpleNamespace(isolated=1, ignore_environment=1, no_site=1),
        path=[str(python / name) for name in ("Lib", "DLLs", "packages", ".")],
        dont_write_bytecode=True, modules={"json": object()})


def test_disc_payload_and_launcher_are_closed_and_unexecuted(
        guest_tools, graphical_tools, guest_fixture, monkeypatch):
    preparer, verifier = guest_tools
    preparer.prepare(*guest_fixture)
    root, _, _, _, environment = guest_fixture
    checksum = verifier.file_hash(environment / "environment.json")
    disc = root / "build/vision-guest-disc-fixture"

    def forbidden(*args, **kwargs):
        raise AssertionError("Disc staging must not launch, load or connect")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    result = graphical_tools[0].prepare(root, environment, checksum, disc)
    assert result["payload_verified"]
    assert all(result[k] is False for k in ("iso_created", "disc_mounted", "guest_launched",
                                           "vision_runtime_loaded", "privacy_verified", "gpu_benchmarked"))
    verifier.verify(disc / "payload/environment", checksum)
    launch = (disc / "payload/Open Carl-Cam VM Checks.vbs").read_text()
    assert '" -B "' in launch and "pythonw.exe" in launch and checksum in launch
    for forbidden_text in ("cmd.exe", "powershell", "mshta", "http", "--native", "mediapipe"):
        assert forbidden_text not in launch.lower()
    assert "shell.Run command, 1, False" in launch


@pytest.mark.parametrize("change", ("host_python", "version", "environment", "site",
                                   "path", "duplicate_path", "bytecode", "sdk"))
def test_python_isolation_requires_actual_runtime_conditions(graphical_tools, tmp_path, change):
    runtime = isolated_runtime(tmp_path)
    if change == "host_python":
        runtime.executable = str(tmp_path / "other/pythonw.exe")
    elif change == "version":
        runtime.version_info = (3, 12, 10)
    elif change == "environment":
        runtime.flags.ignore_environment = 0
    elif change == "site":
        runtime.flags.no_site = 0
    elif change == "path":
        runtime.path.append(str(tmp_path / "host-site"))
    elif change == "duplicate_path":
        runtime.path.append(runtime.path[0])
    elif change == "bytecode":
        runtime.dont_write_bytecode = False
    else:
        runtime.modules["mediapipe.tasks"] = object()
    with pytest.raises(ValueError):
        graphical_tools[1].python_isolation(tmp_path, runtime=runtime)


def test_python_prerequisite_is_not_a_vision_or_privacy_pass(graphical_tools, tmp_path):
    result = graphical_tools[1].python_isolation(tmp_path, runtime=isolated_runtime(tmp_path))
    assert result["python_isolation_verified"]
    assert not result["vision_runtime_loaded"] and not result["privacy_verified"]


def test_gui_functions_do_not_import_tk_or_native_sdk(graphical_tools, guest_tools, guest_fixture):
    assert "tkinter" not in sys.modules
    preparer, verifier = guest_tools
    preparer.prepare(*guest_fixture)
    environment = guest_fixture[-1]
    checksum = verifier.file_hash(environment / "environment.json")
    passed = graphical_tools[1].run_check(environment, checksum, "files")
    assert passed["check_passed"]
    assert not passed["vision_runtime_loaded"] and not passed["privacy_verified"]
    (environment / "python/python.exe").write_bytes(b"changed")
    failed = graphical_tools[1].run_check(environment, checksum, "files")
    assert not failed["check_passed"] and failed["error"]
    assert not failed["vision_runtime_loaded"]


def test_report_saving_never_overwrites(graphical_tools, tmp_path):
    path = tmp_path / "check.json"
    report = {"check_passed": True, "privacy_verified": False}
    graphical_tools[1].save_report(path, report)
    assert json.loads(path.read_text()) == report
    with pytest.raises(ValueError, match="new JSON"):
        graphical_tools[1].save_report(path, {"privacy_verified": True})
    assert json.loads(path.read_text()) == report


@pytest.mark.parametrize("target", ("check.txt", "absent/check.json"))
def test_report_destination_is_explicit_and_existing(graphical_tools, tmp_path, target):
    with pytest.raises(ValueError):
        graphical_tools[1].save_report(tmp_path / target, {})


def test_disc_source_change_retains_failed_attempt(
        guest_tools, graphical_tools, guest_fixture, monkeypatch):
    preparer, verifier = guest_tools
    preparer.prepare(*guest_fixture)
    root, _, _, _, environment = guest_fixture
    checksum = verifier.file_hash(environment / "environment.json")
    disc = root / "build/vision-guest-disc-interrupted"
    original = graphical_tools[0].shutil.copyfileobj

    def change_after_copy(reader, writer, *args, **kwargs):
        original(reader, writer, *args, **kwargs)
        (environment / "python/python.exe").write_bytes(b"changed source")

    monkeypatch.setattr(graphical_tools[0].shutil, "copyfileobj", change_after_copy)
    with pytest.raises(ValueError):
        graphical_tools[0].prepare(root, environment, checksum, disc)
    failed = json.loads((disc / "failure.json").read_text())
    assert failed["partial_files_retained"] and not failed["payload_verified"]
    assert not (disc / "disc-preparation.json").exists()
