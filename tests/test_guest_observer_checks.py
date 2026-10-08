"""Controller fixtures: no real tracing, sockets, firmware read or SDK load."""

import builtins
import hashlib
import importlib.util
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def checker(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("guest_observer_checks_test", tools / "guest_observer_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def document():
    events, controls = [], []
    for index, (label, kind, socket_type, protocol) in enumerate(
            (("tcp4", 4, 1, 6), ("tcp6", 5, 1, 6), ("udp4", 21, 2, 17), ("udp6", 22, 2, 17))):
        owner = 1000 + index
        sequence = 2 * index
        base = {"provider": "e53c6823-7bb8-44bb-90dc-3f86090d48a6", "version": 0,
                "header_pid": 4, "process_key": 100 + index, "endpoint_key": 200 + index}
        address = "::1" if label.endswith("6") else "127.0.0.1"
        events.extend([base | {"sequence": sequence, "qpc": 100 + sequence, "event_id": 1,
                              "owner_pid": owner, "socket_type": socket_type, "protocol": protocol,
                              "remote_address": None, "remote_port": None},
                       base | {"sequence": sequence + 1, "qpc": 101 + sequence, "event_id": kind,
                              "owner_pid": None, "socket_type": None, "protocol": None,
                              "remote_address": address, "remote_port": 42000}])
        controls.append({"label": label, "child_pid": owner, "begin_qpc": 100 + sequence,
                         "end_qpc": 101 + sequence, "call_completed": True, "child_exited": True,
                         "remote_address": address, "remote_port": 42000})
    return {"events": events, "controls": controls,
            "capture": {"vm_uuid": "2cb2eaf1-068d-47a3-8539-8b0d9084b32e", "trace_started": True,
                        "trace_stopped": True, "decoder_ok": True, "events_lost": 0,
                        "buffers_lost": 0, "qpc_frequency": 100, "begin_qpc": 90, "end_qpc": 200,
                        "packet_content_captured": False, "event_limit_reached": False}}


@pytest.fixture
def producer(checker, document, monkeypatch, tmp_path):
    calls = []
    helper = tmp_path / "python/carl-cam-afd-control.exe"
    monkeypatch.setattr(checker, "verify", lambda *args: calls.append("files") or {"files_verified": 10})
    monkeypatch.setattr(checker, "python_isolation", lambda *args: calls.append("python") or {"python_isolation_verified": True})
    monkeypatch.setattr(checker, "machine_identity", lambda: calls.append("machine") or document["capture"]["vm_uuid"])
    monkeypatch.setattr(checker, "helper_path", lambda *args: calls.append("helper") or helper)

    class FakeProcess:
        returncode = 0
        pid = 999

        def communicate(self, *, timeout):
            calls.append(("communicate", timeout))
            return json.dumps(document).encode(), b""

    process = FakeProcess()

    def create(command, **kwargs):
        calls.append(("create", command, kwargs))
        return process

    monkeypatch.setattr(subprocess, "Popen", create)
    return process, calls, helper


def test_json_control_validation_never_authenticates_native_execution(checker, document):
    result = checker.control_report(document)
    assert result["observer_control_evidence_matches"]
    assert result["native_observer_execution_verified"] is False
    assert result["privacy_verified"] is False


@pytest.mark.parametrize("report,expected", [
    ({"cleanup_required": True, "observer_positive_control_passed": True}, "Cleanup is uncertain"),
    ({"cleanup_required": False, "observer_positive_control_passed": True}, "Observer control passed"),
    ({"cleanup_required": False, "observer_positive_control_passed": False}, "Observer control did not pass"),
    ({}, "Cleanup is uncertain"),
])
def test_cleanup_instruction_takes_precedence(checker, report, expected):
    assert checker.status_message(report).startswith(expected)


@pytest.mark.parametrize("case", ["extra", "missing", "fixture", "error", "list"])
def test_control_report_refuses_unreviewed_outer_format(checker, document, case):
    value = deepcopy(document)
    if case == "extra":
        value["assumed_passed"] = True
    elif case == "missing":
        value.pop("capture")
    elif case == "fixture":
        value["fixture_only"] = True
    elif case == "error":
        value = {"error": "failed"}
    else:
        value = []
    with pytest.raises(ValueError):
        checker.control_report(value)


def test_controller_fixture_checks_both_sides_of_exact_command(checker, producer, tmp_path):
    _process, calls, helper = producer
    result = checker.run_control(tmp_path, "a" * 64, "b" * 64)
    assert result["observer_positive_control_passed"] is True
    assert result["cleanup_required"] is False
    # These are simulated controller paths, not real native observer evidence.
    assert result["native_observer_execution_verified"] is True
    assert result["observer"]["native_observer_execution_verified"] is False
    assert calls[:4] == ["files", "python", "machine", "helper"]
    create = calls[4]
    assert create[1] == [str(helper), "--positive-control-only"]
    assert create[2]["shell"] is False and create[2]["close_fds"] is True
    assert create[2]["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert calls[5] == ("communicate", 45)
    assert calls[6:] == ["helper", "files", "python", "machine"]
    assert all(result[name] is False for name in ("privacy_verified", "vision_runtime_loaded",
                                                 "abi_verified", "numerical_verified", "gpu_benchmarked"))
    serialized = json.dumps(result)
    assert "endpoint_key" not in serialized and "process_key" not in serialized
    assert "\"events\"" not in serialized and "Buffer" not in serialized


@pytest.mark.parametrize("name", ["verify", "python_isolation", "machine_identity", "helper_path"])
def test_failed_prerequisite_never_creates_a_process(checker, producer, monkeypatch, tmp_path, name):
    def fail(*args):
        raise ValueError("prerequisite failed")
    monkeypatch.setattr(checker, name, fail)
    result = checker.run_control(tmp_path, "a" * 64, "b" * 64)
    assert "error" in result and not result["observer_positive_control_passed"]
    assert not result["cleanup_required"]
    assert not any(isinstance(call, tuple) and call[0] == "create" for call in producer[1])


@pytest.mark.parametrize("name,value", [("events_lost", 1), ("decoder_ok", False),
                                        ("trace_stopped", False), ("packet_content_captured", True)])
def test_invalid_completed_capture_cannot_pass(checker, producer, document, tmp_path, name, value):
    document["capture"][name] = value
    result = checker.run_control(tmp_path, "a" * 64, "b" * 64)
    assert not result["observer_positive_control_passed"]
    assert not result["native_observer_execution_verified"]
    assert result["cleanup_required"] is (name == "trace_stopped")


def test_missed_control_is_failed_even_with_clean_trace(checker, producer, document, tmp_path):
    document["events"][1]["remote_port"] += 1
    result = checker.run_control(tmp_path, "a" * 64, "b" * 64)
    assert not result["observer_positive_control_passed"]
    assert not result["cleanup_required"]


@pytest.mark.parametrize("case", ["timeout", "malformed", "oversized", "error_clean", "error_dirty"])
def test_failure_retains_cleanup_uncertainty(checker, producer, monkeypatch, tmp_path, case):
    process = producer[0]

    def communicate(*, timeout):
        if case == "timeout":
            raise subprocess.TimeoutExpired("owned-helper", timeout)
        if case == "malformed":
            return b"not json", b""
        if case == "oversized":
            return b"x" * 100, b""
        process.returncode = 1
        return json.dumps({"error": "failed", "cleanup_required": case == "error_dirty"}).encode(), b""

    monkeypatch.setattr(process, "communicate", communicate)
    monkeypatch.setattr(checker, "MAX_OUTPUT", 50)
    result = checker.run_control(tmp_path, "a" * 64, "b" * 64)
    assert not result["observer_positive_control_passed"]
    assert result["cleanup_required"] is (case != "error_clean")
    if case == "timeout":
        assert checker._pending_helpers == [process]
        assert result["helper_pid"] == process.pid


def test_actual_hash_check_without_executing_helper(checker, tmp_path):
    path = tmp_path / "python/carl-cam-afd-control.exe"
    path.parent.mkdir()
    path.write_bytes(b"fixture-not-an-executable")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    assert checker.helper_path(tmp_path, checksum) == path
    with pytest.raises(ValueError):
        checker.helper_path(tmp_path, "0" * 64)


@pytest.mark.parametrize("checksum", [None, "", "A" * 64, "g" * 64, "0" * 63])
def test_invalid_independent_helper_hash_refused(checker, tmp_path, checksum):
    with pytest.raises(ValueError):
        checker.helper_path(tmp_path, checksum)


def test_import_does_not_activate_ui_firmware_process_or_vision(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    original = builtins.__import__
    forbidden = {"ctypes", "subprocess", "tkinter", "socket", "mediapipe", "cv2", "numpy", "torch", "onnxruntime"}

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError("Import must not run a prerequisite, trace, GUI or SDK")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("guest_observer_no_activation", tools / "guest_observer_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
