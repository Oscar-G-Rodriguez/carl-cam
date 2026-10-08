"""Exercise the actual PowerShell disc-policy prefix without creating media."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows disc-policy reader")


@pytest.fixture
def iso_policy(tmp_path):
    root = tmp_path / "repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    disc = root / "build" / ("vision-guest-disc-" + "a" * 32)
    payload = disc / "payload"
    payload.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "tools/create_vision_guest_iso.ps1"
    text = source.read_text(encoding="utf-8")
    boundary = "\nTest-Payload\n$isoPath"
    assert text.count(boundary) == 1
    # The actual target/schema/flag/path/identity functions are executed. Only
    # the following ISO/COM/write phase is omitted; no policy is stubbed.
    policy = tools / "policy-prefix.ps1"
    policy.write_text(text.split(boundary)[0] + "\nTest-Payload\nWrite-Output 'PAYLOAD_CHECK_PASSED'\n")
    item = payload / "fixture.txt"
    item.write_bytes(b"private fixture payload")
    record = {"schema_version": 1, "payload_verified": True,
              **dict.fromkeys(("iso_created", "disc_mounted", "guest_launched",
                               "vision_runtime_loaded", "privacy_verified", "gpu_benchmarked"), False),
              "files": [{"relative": item.name, "size": item.stat().st_size,
                         "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}]}
    record_path = disc / "disc-preparation.json"
    record_path.write_text(json.dumps(record))
    return policy, disc, item, record_path, record


def invoke_policy(fixture, target=None):
    program = shutil.which("pwsh")
    assert program, "The actual Windows PowerShell policy tests require pwsh"
    policy, disc, *_ = fixture
    return subprocess.run([program, "-NoLogo", "-NoProfile", "-File", str(policy),
                           "-DiscDirectory", str(target or disc)],
                          capture_output=True, text=True, timeout=30, check=False)


def test_valid_payload_policy_is_read_only(iso_policy):
    result = invoke_policy(iso_policy)
    assert result.returncode == 0, result.stderr
    assert "PAYLOAD_CHECK_PASSED" in result.stdout
    assert not list(iso_policy[1].glob("*.iso"))
    assert not list(iso_policy[1].glob("iso-attempt-*"))


@pytest.mark.parametrize("change", ("extra", "missing", "bytes", "duplicate", "escape",
                                   "size_string", "hash", "schema_float", "flag_string",
                                   "privacy_claim", "runtime_claim", "missing_flag"))
def test_changed_payload_or_claim_fails_before_iso_phase(iso_policy, change):
    _, disc, item, record_path, record = iso_policy
    if change == "extra":
        (item.parent / "unexpected.txt").write_text("do not include")
    elif change == "missing":
        item.unlink()
    elif change == "bytes":
        item.write_bytes(b"changed fixture")
    elif change == "duplicate":
        record["files"].append(dict(record["files"][0]))
    elif change == "escape":
        record["files"][0]["relative"] = "../outside.txt"
    elif change == "size_string":
        record["files"][0]["size"] = str(record["files"][0]["size"])
    elif change == "hash":
        record["files"][0]["sha256"] = "0" * 64
    elif change == "schema_float":
        record["schema_version"] = 1.0
    elif change == "flag_string":
        record["vision_runtime_loaded"] = "False"
    elif change == "privacy_claim":
        record["privacy_verified"] = True
    elif change == "runtime_claim":
        record["vision_runtime_loaded"] = True
    else:
        del record["gpu_benchmarked"]
    record_path.write_text(json.dumps(record))
    result = invoke_policy(iso_policy)
    assert result.returncode != 0
    assert "PAYLOAD_CHECK_PASSED" not in result.stdout
    assert not list(disc.glob("*.iso"))
    assert not list(disc.glob("iso-attempt-*"))


def test_broad_target_cannot_reach_com_or_iso_creation(iso_policy):
    result = invoke_policy(iso_policy, iso_policy[1].parent)
    assert result.returncode != 0
    assert "generated private disc directory" in result.stderr
    assert not list(iso_policy[1].glob("*.iso"))
