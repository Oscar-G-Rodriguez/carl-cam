"""Fixture policy checks do not execute a socket or start a tracing provider."""

import builtins
import importlib.util
import struct
from copy import deepcopy
from pathlib import Path
from uuid import UUID

import pytest


@pytest.fixture
def policy():
    path = Path(__file__).resolve().parents[1] / "tools/guest_socket_events.py"
    spec = importlib.util.spec_from_file_location("guest_socket_events_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event(policy, sequence, event_id, **changes):
    value = {"sequence": sequence, "qpc": 100 + sequence, "provider": policy.AFD_PROVIDER,
             "event_id": event_id, "version": 0, "header_pid": 4,
             "process_key": 99, "endpoint_key": 80, "owner_pid": None,
             "socket_type": None, "protocol": None, "remote_address": None, "remote_port": None}
    if event_id == 1:
        value.update(owner_pid=1234, socket_type=1, protocol=6)
    if event_id in policy.ATTEMPTS:
        value.update(remote_address="127.0.0.1" if policy.ATTEMPTS[event_id][1] == 4 else "::1",
                     remote_port=42000)
    return value | changes


@pytest.fixture
def evidence(policy):
    events, controls = [], []
    for index, (label, (event_id, socket_type, protocol)) in enumerate(policy.CONTROL_KINDS.items()):
        pid, endpoint = 1234 + index, 80 + index
        events.extend([event(policy, 2 * index, 1, owner_pid=pid, endpoint_key=endpoint,
                             socket_type=socket_type, protocol=protocol),
                       event(policy, 2 * index + 1, event_id, endpoint_key=endpoint)])
        controls.append({"label": label, "child_pid": pid, "begin_qpc": 100 + 2 * index,
                         "end_qpc": 101 + 2 * index, "call_completed": True, "child_exited": True,
                         "remote_address": events[-1]["remote_address"], "remote_port": 42000})
    capture = {"vm_uuid": policy.PRIVATE_VM_UUID, "trace_started": True, "trace_stopped": True,
               "decoder_ok": True, "events_lost": 0, "buffers_lost": 0, "qpc_frequency": 10,
               "begin_qpc": 90, "end_qpc": 200, "packet_content_captured": False,
               "event_limit_reached": False}
    return events, controls, capture


def raw_smbios(identity, *, major=3, minor=2):
    system = bytes((1, 24, 0, 0, 0, 0, 0, 0)) + UUID(identity).bytes_le + b"\0\0"
    table = system + bytes((127, 4, 1, 0, 0, 0))
    return bytes((0, major, minor, 0)) + struct.pack("<I", len(table)) + table


def test_exact_smbios_identity(policy):
    assert policy.private_smbios_uuid(raw_smbios(policy.PRIVATE_VM_UUID)) == policy.PRIVATE_VM_UUID


@pytest.mark.parametrize("case", ["other", "short", "size", "old", "missing_end", "extra",
                                  "duplicate", "no_strings", "no_uuid", "bad_length"])
def test_smbios_refuses_other_host_and_malformed_tables(policy, case):
    raw = raw_smbios(policy.PRIVATE_VM_UUID)
    if case == "other":
        raw = raw_smbios("11111111-2222-3333-4444-555555555555")
    elif case == "short":
        raw = raw[:6]
    elif case == "size":
        raw = raw[:4] + struct.pack("<I", 1) + raw[8:]
    elif case == "old":
        raw = raw_smbios(policy.PRIVATE_VM_UUID, major=2, minor=5)
    else:
        table = raw[8:]
        if case == "missing_end":
            table = table[:-6]
        elif case == "extra":
            table += b"\0"
        elif case == "duplicate":
            table = table[:-6] + table
        elif case == "no_strings":
            table = table[:24] + b"x"
        elif case == "no_uuid":
            table = bytes((1, 4, 0, 0, 0, 0, 127, 4, 1, 0, 0, 0))
        elif case == "bad_length":
            table = bytes((1, 255)) + table[2:]
        raw = raw[:4] + struct.pack("<I", len(table)) + table
    with pytest.raises(ValueError):
        policy.private_smbios_uuid(raw)


def test_attribution_never_uses_kernel_header_pid(policy):
    events = [event(policy, 0, 1), event(policy, 1, 4)]
    assert policy.correlate(events)[0]["owner_pid"] == 1234
    events[0]["header_pid"] = 888
    events[1]["header_pid"] = 999
    assert policy.correlate(events)[0]["owner_pid"] == 1234


def test_timestamp_order_does_not_depend_on_list_delivery_order(policy):
    events = [event(policy, 0, 1), event(policy, 1, 4)]
    assert policy.correlate(events[::-1]) == policy.correlate(events)


def test_close_ends_owner_and_reuse_requires_new_creation(policy):
    events = [event(policy, 0, 1), event(policy, 1, 13), event(policy, 2, 4),
              event(policy, 3, 1, owner_pid=5678), event(policy, 4, 4)]
    attempts = policy.correlate(events)
    assert attempts[0]["owner_pid"] is None
    assert attempts[1]["owner_pid"] == 5678


def test_eprocess_endpoint_pair_prevents_cross_process_attribution(policy):
    assert policy.correlate([event(policy, 0, 1), event(policy, 1, 4, process_key=100)])[0]["owner_pid"] is None


@pytest.mark.parametrize("changes", [{"provider": "other"}, {"version": 1}, {"event_id": 99},
                                      {"owner_pid": True}, {"qpc": 0}, {"endpoint_key": False},
                                      {"process_key": "99"}, {"sequence": -1}, {"header_pid": -1}])
def test_changed_event_identity_or_coerced_types_are_refused(policy, changes):
    with pytest.raises(ValueError):
        policy.correlate([event(policy, 0, 1) | changes])


@pytest.mark.parametrize("case", ["duplicate_sequence", "missing_sequence", "raw_payload",
                                  "address_family", "not_canonical", "missing_address",
                                  "fake_owner", "ambiguous_lifetime", "too_many"])
def test_incomplete_or_unsafe_event_inventory_refused(policy, case, monkeypatch):
    events = [event(policy, 0, 1), event(policy, 1, 4)]
    if case == "duplicate_sequence":
        events[1]["sequence"] = 0
    elif case == "missing_sequence":
        events[1]["sequence"] = 2
    elif case == "raw_payload":
        events[0]["payload"] = "unwanted bytes"
    elif case == "address_family":
        events[1]["remote_address"] = "::1"
    elif case == "not_canonical":
        events[1] = event(policy, 1, 5, remote_address="0:0:0:0:0:0:0:1")
    elif case == "missing_address":
        events[1]["remote_address"] = None
    elif case == "fake_owner":
        events[1]["owner_pid"] = 1234
    elif case == "ambiguous_lifetime":
        events[1] = event(policy, 1, 1)
    elif case == "too_many":
        monkeypatch.setattr(policy, "MAX_EVENTS", 1)
    with pytest.raises(ValueError):
        policy.correlate(events)


def test_four_control_matches_cannot_approve_native_or_privacy(policy, evidence):
    report = policy.evaluate_positive_control(*evidence)
    assert report["observer_control_evidence_matches"]
    assert report["unattributed_attempts"] == 0
    for name in ("native_observer_execution_verified", "vision_runtime_loaded", "privacy_verified",
                 "abi_verified", "numerical_verified", "gpu_benchmarked"):
        assert report[name] is False
    assert all(control["matching_attempts"] == 1 for control in report["controls"])


@pytest.mark.parametrize("field,value", [("trace_started", False), ("trace_stopped", False),
                                        ("decoder_ok", False), ("trace_started", 1),
                                        ("events_lost", 1), ("buffers_lost", 1),
                                        ("packet_content_captured", True), ("event_limit_reached", True),
                                        ("qpc_frequency", 0), ("end_qpc", 1000),
                                        ("vm_uuid", "other")])
def test_capture_loss_or_unverified_scope_refused(policy, evidence, field, value):
    events, controls, capture = deepcopy(evidence)
    capture[field] = value
    with pytest.raises(ValueError):
        policy.evaluate_positive_control(events, controls, capture)


@pytest.mark.parametrize("case", ["missing", "duplicate", "pid_reuse", "not_completed",
                                  "not_exited", "external_address", "window", "extra_field"])
def test_invalid_control_execution_metadata_refused(policy, evidence, case):
    events, controls, capture = deepcopy(evidence)
    if case == "missing":
        controls.pop()
    elif case == "duplicate":
        controls[1] = deepcopy(controls[0])
    elif case == "pid_reuse":
        controls[1]["child_pid"] = controls[0]["child_pid"]
    elif case == "not_completed":
        controls[0]["call_completed"] = False
    elif case == "not_exited":
        controls[0]["child_exited"] = False
    elif case == "external_address":
        controls[0]["remote_address"] = "192.0.2.1"
    elif case == "window":
        controls[0]["begin_qpc"] = 1
    elif case == "extra_field":
        controls[0]["assumed_passed"] = True
    with pytest.raises(ValueError):
        policy.evaluate_positive_control(events, controls, capture)


@pytest.mark.parametrize("case", ["port", "time", "type", "protocol", "pid", "missing_create"])
def test_false_positive_controls_do_not_match(policy, evidence, case):
    events, controls, capture = deepcopy(evidence)
    if case == "port":
        events[1]["remote_port"] += 1
    elif case == "time":
        events[1]["qpc"] += 10
    elif case == "type":
        events[0]["socket_type"] = 2
    elif case == "protocol":
        events[0]["protocol"] = 17
    elif case == "pid":
        events[0]["owner_pid"] = 4321
    elif case == "missing_create":
        events.pop(0)
        for index, item in enumerate(events):
            item["sequence"] = index
    assert not policy.evaluate_positive_control(events, controls, capture)["observer_control_evidence_matches"]


def test_import_does_not_load_windows_sdk_network_or_vision(monkeypatch):
    original = builtins.__import__
    forbidden = {"ctypes", "socket", "subprocess", "mediapipe", "numpy", "cv2", "torch", "onnxruntime"}

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError("Policy import must not activate a native/network dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    path = Path(__file__).resolve().parents[1] / "tools/guest_socket_events.py"
    spec = importlib.util.spec_from_file_location("guarded_guest_socket_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
