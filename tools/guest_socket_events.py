"""Pure policy for a guest-only Winsock observer; never starts tracing.

AFD's header PID can name a kernel worker. Attribution therefore follows the
EPROCESS/endpoint pair from socket creation, with close/cleanup ending its
lifetime. Inputs come from a future reviewed native adapter, not a Python hook
around socket.connect (which would miss SDK calls). Fixture success here cannot
approve a runtime, certify ETW decoding or pass the real positive control.
"""

import ipaddress
import struct
from uuid import UUID

PRIVATE_VM_UUID = "2cb2eaf1-068d-47a3-8539-8b0d9084b32e"
AFD_PROVIDER = "e53c6823-7bb8-44bb-90dc-3f86090d48a6"
MAX_EVENTS = 50_000
SOCKET_CREATE = 1
CLOSE_EVENTS = frozenset((13, 14))
ATTEMPTS = {4: ("connect", 4), 5: ("connect", 6),
            21: ("sendto", 4), 22: ("sendto", 6)}
ALLOWED_EVENTS = frozenset((SOCKET_CREATE, *CLOSE_EVENTS, *ATTEMPTS))
CONTROL_KINDS = {"tcp4": (4, 1, 6), "tcp6": (5, 1, 6),
                 "udp4": (21, 2, 17), "udp6": (22, 2, 17)}


def integer(value, name, maximum, *, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}")
    return value


def private_smbios_uuid(raw):
    """Validate Microsoft's RawSMBIOSData and the exact type-1 machine UUID.

    This parses bytes only. The native adapter must obtain these bytes from
    GetSystemFirmwareTable('RSMB', 0); a textbox or environment variable is not
    evidence of the machine on which a trace would run.
    """
    if not isinstance(raw, bytes) or len(raw) < 8 or len(raw) > 1024 * 1024:
        raise ValueError("Invalid raw SMBIOS data")
    major, minor = raw[1:3]
    if (major, minor) < (2, 6) or struct.unpack_from("<I", raw, 4)[0] != len(raw) - 8:
        raise ValueError("Unsupported or truncated SMBIOS data")
    position, identities, ended = 8, [], False
    while position < len(raw):
        if len(raw) - position < 4:
            raise ValueError("Truncated SMBIOS structure")
        kind, length = raw[position:position + 2]
        if length < 4 or position + length > len(raw):
            raise ValueError("Invalid SMBIOS structure length")
        strings_end = raw.find(b"\0\0", position + length)
        if strings_end < 0:
            raise ValueError("Unterminated SMBIOS string area")
        if kind == 1:
            if length < 24:
                raise ValueError("SMBIOS system UUID is missing")
            identities.append(str(UUID(bytes_le=raw[position + 8:position + 24])))
        position = strings_end + 2
        if kind == 127:
            ended = True
            break
    if not ended or position != len(raw) or identities != [PRIVATE_VM_UUID]:
        raise ValueError("This is not the exact reviewed private VM")
    return PRIVATE_VM_UUID


def validate_event(event):
    """Accept only selected metadata, never arbitrary ETW or packet payloads."""
    required = {"sequence", "qpc", "provider", "event_id", "version", "header_pid",
                "process_key", "endpoint_key", "owner_pid", "socket_type", "protocol",
                "remote_address", "remote_port"}
    if not isinstance(event, dict) or event.keys() != required:
        raise ValueError("Unexpected event fields; raw payloads are not accepted")
    integer(event["sequence"], "event sequence", MAX_EVENTS - 1)
    integer(event["qpc"], "QPC timestamp", 2**63 - 1, minimum=1)
    integer(event["header_pid"], "header PID", 2**32 - 1)
    integer(event["process_key"], "process key", 2**64 - 1, minimum=1)
    integer(event["endpoint_key"], "endpoint key", 2**64 - 1, minimum=1)
    integer(event["event_id"], "event ID", 65535)
    integer(event["version"], "event version", 255)
    if (event["provider"] != AFD_PROVIDER or event["version"] != 0
            or event["event_id"] not in ALLOWED_EVENTS):
        raise ValueError("Unreviewed event provider, version or ID")
    if event["event_id"] == SOCKET_CREATE:
        integer(event["owner_pid"], "socket owner PID", 2**32 - 1, minimum=1)
        integer(event["socket_type"], "socket type", 65535, minimum=1)
        integer(event["protocol"], "socket protocol", 65535)
    elif any(event[name] is not None for name in ("owner_pid", "socket_type", "protocol")):
        raise ValueError("Only socket creation supplies ownership/type metadata")
    if event["event_id"] in ATTEMPTS:
        family = ATTEMPTS[event["event_id"]][1]
        if not isinstance(event["remote_address"], str):
            raise ValueError("Missing remote address")
        try:
            address = ipaddress.ip_address(event["remote_address"])
        except ValueError as error:
            raise ValueError("Invalid remote address") from error
        if address.version != family or str(address) != event["remote_address"]:
            raise ValueError("Address must use its canonical event-family format")
        integer(event["remote_port"], "remote port", 65535, minimum=1)
    elif event["remote_address"] is not None or event["remote_port"] is not None:
        raise ValueError("Unexpected address metadata")


def correlate(events):
    """Return attempt attribution and uncertainty, without a privacy verdict."""
    if not isinstance(events, list) or len(events) > MAX_EVENTS:
        raise ValueError("Event inventory exceeds the bounded observer limit")
    for event in events:
        validate_event(event)
    if sorted(event["sequence"] for event in events) != list(range(len(events))):
        raise ValueError("Missing or duplicate event sequence")
    ordered = sorted(events, key=lambda event: (event["qpc"], event["sequence"]))
    owners, attempts = {}, []
    for event in ordered:
        key = event["process_key"], event["endpoint_key"]
        kind = event["event_id"]
        if kind == SOCKET_CREATE:
            if key in owners:
                raise ValueError("Ambiguous socket lifetime without close/cleanup")
            owners[key] = (event["owner_pid"], event["socket_type"], event["protocol"])
        elif kind in CLOSE_EVENTS:
            owners.pop(key, None)
        else:
            owner = owners.get(key)
            attempts.append({"sequence": event["sequence"], "qpc": event["qpc"],
                             "event_id": kind, "owner_pid": owner[0] if owner else None,
                             "socket_type": owner[1] if owner else None,
                             "protocol": owner[2] if owner else None,
                             "remote_address": event["remote_address"],
                             "remote_port": event["remote_port"]})
    return attempts


def evaluate_positive_control(events, controls, capture):
    """Require all four independently identified child/control observations.

    The native producer must attest completed calls and child lifetimes; events
    must independently corroborate each PID, family, operation and destination.
    This function validates supplied evidence. It does not authenticate a JSON
    producer or establish that supplied records came from a real ETW session.
    """
    capture_fields = {"vm_uuid", "trace_started", "trace_stopped", "decoder_ok",
                      "events_lost", "buffers_lost", "qpc_frequency", "begin_qpc",
                      "end_qpc", "packet_content_captured", "event_limit_reached"}
    if not isinstance(capture, dict) or capture.keys() != capture_fields:
        raise ValueError("Invalid capture metadata")
    if capture["vm_uuid"] != PRIVATE_VM_UUID:
        raise ValueError("Capture is not identified with the exact private VM")
    for name in ("trace_started", "trace_stopped", "decoder_ok"):
        if capture[name] is not True:
            raise ValueError("The capture did not complete cleanly")
    for name in ("packet_content_captured", "event_limit_reached"):
        if capture[name] is not False:
            raise ValueError("Capture contents or bounds are not acceptable")
    for name in ("events_lost", "buffers_lost"):
        if integer(capture[name], name, 2**32 - 1) != 0:
            raise ValueError("Lost events cannot support observer validation")
    frequency = integer(capture["qpc_frequency"], "QPC frequency", 10**12, minimum=1)
    begin = integer(capture["begin_qpc"], "capture start", 2**63 - 1, minimum=1)
    end = integer(capture["end_qpc"], "capture end", 2**63 - 1, minimum=begin + 1)
    if end - begin > 30 * frequency:
        raise ValueError("Positive-control capture exceeds thirty seconds")
    attempts = correlate(events)
    if any(not begin <= event["qpc"] <= end for event in events):
        raise ValueError("Event outside capture interval")
    if not isinstance(controls, list) or len(controls) != len(CONTROL_KINDS):
        raise ValueError("All four socket controls are required")
    seen, pids, checks = set(), set(), []
    for control in controls:
        fields = {"label", "child_pid", "begin_qpc", "end_qpc", "call_completed",
                  "child_exited", "remote_address", "remote_port"}
        if not isinstance(control, dict) or control.keys() != fields:
            raise ValueError("Invalid control metadata")
        label = control["label"]
        if not isinstance(label, str) or label not in CONTROL_KINDS or label in seen:
            raise ValueError("Unknown or duplicate control kind")
        seen.add(label)
        pid = integer(control["child_pid"], "control child PID", 2**32 - 1, minimum=1)
        if pid in pids:
            raise ValueError("Each control requires its own child process")
        pids.add(pid)
        start = integer(control["begin_qpc"], "control start", end, minimum=begin)
        stop = integer(control["end_qpc"], "control end", end, minimum=start)
        if control["call_completed"] is not True or control["child_exited"] is not True:
            raise ValueError("Control did not complete and exit")
        event_id, socket_type, protocol = CONTROL_KINDS[label]
        address = "127.0.0.1" if ATTEMPTS[event_id][1] == 4 else "::1"
        port = integer(control["remote_port"], "control destination port", 65535, minimum=1)
        if control["remote_address"] != address:
            raise ValueError("Positive controls must stay on guest loopback")
        matched = [event for event in attempts
                   if event["owner_pid"] == pid and start <= event["qpc"] <= stop
                   and event["event_id"] == event_id and event["socket_type"] == socket_type
                   and event["protocol"] == protocol and event["remote_address"] == address
                   and event["remote_port"] == port]
        # A dedicated control child should do only its declared operation. Any
        # conflicting attributed attempt is a failure, never silently dropped.
        conflicting = [event for event in attempts if event["owner_pid"] == pid
                       and event not in matched]
        checks.append({"label": label, "child_pid": pid,
                       "matching_attempts": len(matched), "conflicting_attempts": len(conflicting),
                       "passed": bool(matched) and not conflicting})
    return {"schema_version": 1, "observer_control_evidence_matches": all(c["passed"] for c in checks),
            "controls": checks, "events_received": len(events),
            "unattributed_attempts": sum(event["owner_pid"] is None for event in attempts),
            "native_observer_execution_verified": False, "vision_runtime_loaded": False,
            "abi_verified": False, "numerical_verified": False, "privacy_verified": False,
            "gpu_benchmarked": False}
