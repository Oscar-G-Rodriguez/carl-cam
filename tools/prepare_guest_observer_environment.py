"""Clone a verified private environment and add the pinned observer tools.

The earlier environment remains unchanged. No executable, model, trace or
guest is started. The new manifest retains the parent inventory and native
build receipt so subsequent transfer checks can identify every added byte.
"""

import argparse
import json
import re
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from prepare_vision_guest_environment import MAX_TOTAL, identity, ordinary, relative
from verify_vision_guest_environment import verify


def checksum(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected an independently recorded SHA-256")
    return value


def native_plan(root, build, helper_hash):
    root = ordinary(root, directory=True)
    build = ordinary(build, directory=True)
    if build.parent != root / "build" or re.fullmatch(r"guest-observer-schema-[0-9a-f]{32}", build.name) is None:
        raise ValueError("Use the exact native observer build inside this repository")
    helper_hash = checksum(helper_hash)
    receipt_path = build / "build-report.json"
    receipt_identity = identity(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
            or type(receipt.get("compiler_returncode")) is not int or receipt["compiler_returncode"] != 0
            or receipt.get("binary_kind") != "private_vm_positive_control"
            or receipt.get("source_unchanged") is not True
            or any(receipt.get(name) is not False for name in
                   ("executable_executed", "trace_started", "privacy_verified", "gpu_benchmarked"))):
        raise ValueError("Expected a successful, unchanged and unexecuted native observer build")
    executable = build / "carl-cam-afd-control.exe"
    helper_identity = identity(executable)
    if (type(receipt.get("executable_bytes")) is not int
            or helper_identity != {"size": receipt["executable_bytes"], "sha256": helper_hash}
            or receipt.get("executable_sha256") != helper_hash):
        raise ValueError("Native observer differs from its independent hash or build receipt")
    entries = [("python/carl-cam-afd-control.exe", executable, helper_identity, "guest_positive_control")]
    for name, field in (("afd_observer.cpp", "source_sha256"), ("afd_schema.h", "header_sha256")):
        current = identity(root / "tools/native" / name)
        snapshot = identity(build / name)
        if current != snapshot or current["sha256"] != checksum(receipt.get(field)):
            raise ValueError("Native source, snapshot or build receipt changed")
        entries.append(("reference/observer/" + name, build / name, snapshot, "observer_source_audit"))
    entries.extend(("reference/observer/" + name, build / name, identity(build / name), "observer_build_audit")
                   for name in ("build-report.json", "compiler.log"))
    if identity(receipt_path) != receipt_identity:
        raise ValueError("Native build receipt changed during inspection")
    return entries


def prepare(root, parent, parent_hash, native_build, helper_hash, destination):
    root = ordinary(root, directory=True)
    parent = ordinary(parent, directory=True)
    build = ordinary(root / "build", directory=True)
    destination = Path(destination).absolute()
    if (parent.parent != build or re.fullmatch(r"vision-guest-environment-[0-9a-f]{32}", parent.name) is None
            or destination.parent != build
            or re.fullmatch(r"vision-guest-environment-[0-9a-f]{32}", destination.name) is None
            or destination.exists() or destination.is_symlink() or destination.is_junction()):
        raise ValueError("Use a reviewed parent and a fresh private environment inside build")
    checksum(parent_hash)
    verify(parent, parent_hash)
    parent_manifest = parent / "environment.json"
    record = json.loads(parent_manifest.read_text(encoding="utf-8"))
    needed = {"python/pythonw.exe", "python/python312._pth", "python/vision_guest_checks.py",
              "python/verify_vision_guest_environment.py"}
    if not needed <= {item["relative"] for item in record["files"]}:
        raise ValueError("Parent lacks the verified Python prerequisites")
    additions = native_plan(root, native_build, helper_hash)
    additions.append(("reference/observer/parent-environment.json", parent_manifest,
                      identity(parent_manifest), "parent_environment_audit"))
    for name in ("guest_socket_events.py", "guest_observer_checks.py"):
        source = root / "tools" / name
        additions.append(("python/" + name, source, identity(source), "guest_observer_controller"))
    planned = [(item["relative"], parent / item["relative"],
                {"size": item["size"], "sha256": item["sha256"]}, item["role"])
               for item in record["files"]] + additions
    seen = set()
    for name, _origin, _expected, _role in planned:
        key = relative(name).as_posix().casefold()
        if key in seen:
            raise ValueError("Observer addition collides with the parent inventory")
        seen.add(key)
    if sum(item[2]["size"] for item in planned) > MAX_TOTAL:
        raise ValueError("Observer environment exceeds the private transfer limit")
    destination.mkdir(exist_ok=False)
    report = deepcopy(record)
    report.update(recorded_utc=datetime.now(UTC).isoformat(), preparation_verified=False, files=[],
                  parent_environment_manifest_sha256=parent_hash, observer_helper_sha256=helper_hash,
                  observer_preparation_verified=False, observer_executed=False,
                  observer_positive_control_passed=False)
    try:
        for name, source, expected, role in planned:
            if identity(source) != expected:
                raise ValueError("Source changed before observer transfer")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            if identity(target) != expected:
                raise ValueError("Copied observer environment differs from source")
            report["files"].append({"relative": name, "role": role, "origin_path": str(source), **expected})
        verify(parent, parent_hash)
        native_plan(root, native_build, helper_hash)
        for name, source, expected, _role in planned:
            if identity(source) != expected or identity(destination / name) != expected:
                raise ValueError("Observer source or copy changed during preparation")
        report.update(preparation_verified=True, observer_preparation_verified=True)
        with (destination / "environment.json").open("x", encoding="utf-8") as writer:
            json.dump(report, writer, indent=2)
        manifest_identity = identity(destination / "environment.json")
        verify(destination, manifest_identity["sha256"])
        return {"environment": str(destination), "manifest_sha256": manifest_identity["sha256"],
                "files_verified": len(report["files"]), "helper_sha256": helper_hash,
                "observer_executed": False, "vision_runtime_loaded": False,
                "privacy_verified": False, "gpu_benchmarked": False}
    except (OSError, ValueError, KeyError, TypeError):
        with (destination / "failure.json").open("x", encoding="utf-8") as writer:
            json.dump({"preparation_verified": False, "partial_files_retained": True,
                       "observer_executed": False, "privacy_verified": False}, writer, indent=2)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--helper-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "build" / ("vision-guest-environment-" + uuid4().hex)
    print(json.dumps(prepare(root, args.parent, args.parent_manifest_sha256,
                            args.native_build, args.helper_sha256, destination), indent=2))


if __name__ == "__main__":
    main()
