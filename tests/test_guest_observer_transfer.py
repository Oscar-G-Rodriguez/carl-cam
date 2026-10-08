"""Staging fixtures never execute the fake binaries or launch Windows UI."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def transfer(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("observer_transfer_test", folder / "prepare_guest_observer_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def disc_tools(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("observer_disc_test", folder / "prepare_vision_guest_disc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(transfer, tmp_path):
    root = tmp_path / "repo"
    parent = root / "build" / ("vision-guest-environment-" + "a" * 32)
    native = root / "build" / ("guest-observer-schema-" + "b" * 32)
    destination = root / "build" / ("vision-guest-environment-" + "c" * 32)
    parent.mkdir(parents=True)
    native.mkdir()
    (root / "tools/native").mkdir(parents=True)
    for name in ("afd_observer.cpp", "afd_schema.h"):
        (root / "tools/native" / name).write_bytes(b"fixture source: " + name.encode())
        (native / name).write_bytes((root / "tools/native" / name).read_bytes())
    for name in ("guest_socket_events.py", "guest_observer_checks.py"):
        (root / "tools" / name).write_bytes(b"# fixture source, not executed\n")
    for name in ("pythonw.exe", "python312._pth", "vision_guest_checks.py", "verify_vision_guest_environment.py"):
        path = parent / "python" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture prerequisite: " + name.encode())
    (native / "carl-cam-afd-control.exe").write_bytes(b"fixture, not a runnable binary")
    (native / "compiler.log").write_text("fixture compile receipt")
    helper = transfer.identity(native / "carl-cam-afd-control.exe")
    receipt = {"schema_version": 1, "compiler_returncode": 0, "binary_kind": "private_vm_positive_control",
               "source_unchanged": True, "source_sha256": transfer.identity(native / "afd_observer.cpp")["sha256"],
               "header_sha256": transfer.identity(native / "afd_schema.h")["sha256"],
               "executable_bytes": helper["size"], "executable_sha256": helper["sha256"],
               **dict.fromkeys(("executable_executed", "trace_started", "privacy_verified", "gpu_benchmarked"), False)}
    (native / "build-report.json").write_text(json.dumps(receipt))
    manifest = {"schema_version": 1, "preparation_verified": True,
                **dict.fromkeys(("runtime_loaded", "abi_verified", "privacy_verified", "numerical_verified",
                                 "promoted", "gpu_benchmarked", "guest_environment_executed", "network_enabled",
                                 "physical_camera_used", "host_runtime_modified"), False),
                "files": [{"relative": path.relative_to(parent).as_posix(), "role": "fixture",
                           **transfer.identity(path)} for path in sorted(parent.rglob("*")) if path.is_file()]}
    (parent / "environment.json").write_text(json.dumps(manifest))
    return root, parent, transfer.identity(parent / "environment.json")["sha256"], native, helper["sha256"], destination


def test_clone_preserves_parent_and_closes_new_inventory(transfer, bundle):
    _root, parent, checksum, _native, helper, destination = bundle
    before = {path.relative_to(parent).as_posix(): path.read_bytes()
              for path in parent.rglob("*") if path.is_file()}
    report = transfer.prepare(*bundle)
    assert report["files_verified"] == 12
    assert report["observer_executed"] is False and report["privacy_verified"] is False
    assert report["helper_sha256"] == helper
    transfer.verify(destination, report["manifest_sha256"])
    assert before == {path.relative_to(parent).as_posix(): path.read_bytes()
                      for path in parent.rglob("*") if path.is_file()}
    manifest = json.loads((destination / "environment.json").read_text())
    assert manifest["parent_environment_manifest_sha256"] == checksum
    assert (destination / "reference/observer/parent-environment.json").read_bytes() == before["environment.json"]
    assert manifest["observer_preparation_verified"] is True
    assert all(manifest[name] is False for name in ("runtime_loaded", "privacy_verified", "gpu_benchmarked",
                                                  "observer_executed", "observer_positive_control_passed"))


@pytest.mark.parametrize("field,value", [("schema_version", True), ("compiler_returncode", False),
                                        ("compiler_returncode", 1), ("binary_kind", "read_only_schema"),
                                        ("source_unchanged", False), ("executable_executed", True),
                                        ("trace_started", True), ("privacy_verified", True),
                                        ("gpu_benchmarked", True), ("executable_bytes", "30"),
                                        ("executable_sha256", "0" * 64), ("source_sha256", "0" * 64)])
def test_unreviewed_native_receipt_refused_before_copy(transfer, bundle, field, value):
    path = bundle[3] / "build-report.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        transfer.prepare(*bundle)
    assert not bundle[-1].exists()


@pytest.mark.parametrize("case", ["helper", "snapshot", "source", "parent", "extra_parent",
                                  "existing", "broad_target", "missing_gui", "hash"])
def test_changed_or_unsafe_transfer_is_refused(transfer, bundle, case):
    args = list(bundle)
    root, parent, _checksum, native, _helper, destination = args
    if case == "helper":
        (native / "carl-cam-afd-control.exe").write_bytes(b"changed")
    elif case == "snapshot":
        (native / "afd_schema.h").write_bytes(b"changed")
    elif case == "source":
        (root / "tools/native/afd_observer.cpp").write_bytes(b"changed")
    elif case == "parent":
        (parent / "python/pythonw.exe").write_bytes(b"changed")
    elif case == "extra_parent":
        (parent / "extra.bin").write_bytes(b"unexpected")
    elif case == "existing":
        destination.mkdir()
    elif case == "broad_target":
        args[-1] = root / "build"
    elif case == "missing_gui":
        (root / "tools/guest_observer_checks.py").unlink()
    else:
        args[4] = "NOT_A_HASH"
    with pytest.raises((ValueError, FileNotFoundError)):
        transfer.prepare(*args)


def test_observer_disc_keeps_prerequisites_and_adds_explicit_launchers(transfer, disc_tools, bundle):
    prepared = transfer.prepare(*bundle)
    disc = bundle[0] / "build" / ("vision-guest-disc-" + "d" * 32)
    result = disc_tools.prepare(bundle[0], bundle[-1], prepared["manifest_sha256"], disc,
                                observer_hash=bundle[4])
    assert result["observer_executed"] is False
    assert result["observer_positive_control_passed"] is False
    assert result["disc_mounted"] is False and result["guest_launched"] is False
    payload = disc / "payload"
    plain = (payload / "Open Carl-Cam VM Checks.vbs").read_text()
    observer = (payload / "Open Carl-Cam Observer.vbs").read_text()
    admin = (payload / "Open Carl-Cam Observer as Admin.vbs").read_text()
    assert "vision_guest_checks.py" in plain and "runas" not in plain
    assert "guest_observer_checks.py" in observer and "runas" not in observer
    assert " --helper-sha256 " + bundle[4] in observer
    assert "ShellExecute interpreter, Mid(command, Len(interpreter) + 3)" in admin
    assert '"runas"' in admin and "MsgBox" in admin
    for source in (plain, observer, admin):
        assert "cmd.exe" not in source and "powershell" not in source and "http" not in source
    transfer.verify(payload / "environment", prepared["manifest_sha256"])


@pytest.mark.parametrize("field,value", [("observer_preparation_verified", False), ("observer_executed", True),
                                        ("observer_positive_control_passed", True), ("observer_helper_sha256", "0" * 64)])
def test_disc_refuses_false_observer_claim_or_identity(transfer, disc_tools, bundle, field, value):
    transfer.prepare(*bundle)
    manifest = bundle[-1] / "environment.json"
    record = json.loads(manifest.read_text())
    record[field] = value
    manifest.write_text(json.dumps(record))
    disc = bundle[0] / "build" / ("vision-guest-disc-" + "d" * 32)
    with pytest.raises(ValueError):
        disc_tools.prepare(bundle[0], bundle[-1], transfer.identity(manifest)["sha256"], disc,
                           observer_hash=bundle[4])
    assert not disc.exists()


@pytest.mark.parametrize("value", [None, "", "g" * 64, "a" * 63, "A" * 64])
def test_launch_hashes_cannot_inject_an_unreviewed_command(disc_tools, value):
    with pytest.raises(ValueError):
        disc_tools.observer_launcher("a" * 64, value)
    with pytest.raises(ValueError):
        disc_tools.observer_launcher(value, "a" * 64)
