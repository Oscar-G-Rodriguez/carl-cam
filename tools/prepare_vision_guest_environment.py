"""Stage an offline guest Python environment without importing any vision SDK.

Only the existing CPython runtime, the locked MediaPipe dependency closure and
the verified private candidate kit are copied. Installed wheel RECORD hashes
are checked; the telemetry-held wheel DLL is excluded, not executed/replaced
on the host. This preparation cannot pass privacy, ABI or numerical gates.
"""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import re
import shutil
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

PREFIX = "vision-guest-environment-"
SOURCE_COMMIT = "f8ef212d5c962c0e853db7e59d217056b187084b"
CANDIDATE_HASH = "f9de3fadf4cff7358288895ad2e33afcc9cff4246c76b4e4b39769191795968d"
HELD_DLL = "mediapipe/tasks/c/libmediapipe.dll"
UNVERIFIED = ("runtime_loaded", "abi_verified", "privacy_verified", "numerical_verified",
              "promoted", "gpu_benchmarked", "guest_environment_executed")
ISOLATED_PATHS = "Lib\nDLLs\npackages\n.\n"
MAX_FILE = 128 * 1024 * 1024
MAX_TOTAL = 768 * 1024 * 1024


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    if (any(p.is_symlink() or p.is_junction() for p in (path, *path.parents))
            or not (path.is_dir() if directory else path.is_file())):
        raise ValueError("Expected an ordinary " + ("directory" if directory else "file"))
    return path


def identity(path):
    path = ordinary(path)
    size = path.stat().st_size
    if not 0 <= size <= MAX_FILE:
        raise ValueError("File exceeds transfer limit")
    with path.open("rb") as reader:
        digest = hashlib.file_digest(reader, "sha256").hexdigest()
    return {"size": size, "sha256": digest}


def relative(name):
    """Reject host paths, Windows aliases and alternate data streams."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Unsafe transfer path")
    parts = name.split("/")
    if any(p in ("", ".", "..") or p.endswith((" ", ".")) or re.search(
            r'[<>"|?*\x00-\x1f]', p) or p.split(".")[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))} for p in parts):
        raise ValueError("Unsafe transfer path")
    return PurePosixPath(name)


def closure(lock):
    """Resolve this lock's CPython dependency graph without invoking pip."""
    records = lock.get("package", [])
    packages = {}
    for record in records:
        name = record["name"]
        if name in packages:
            raise ValueError("Ambiguous package identity in lock")
        packages[name] = record
    needed, pending = {}, ["mediapipe"]
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        package = packages[name]
        if package.get("source") != {"registry": "https://pypi.org/simple"}:
            raise ValueError("Guest dependency must have the locked PyPI origin")
        needed[name] = package["version"]
        for dep in package.get("dependencies", []):
            if dep.get("marker") not in (None, "implementation_name != 'PyPy'"):
                raise ValueError("Unreviewed guest dependency marker")
            pending.append(dep["name"])
    return needed


def candidate_plan(kit):
    kit = ordinary(kit, directory=True)
    manifest_path = kit / "manifest.json"
    before = identity(manifest_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("source_commit") != SOURCE_COMMIT
            or any(record.get(k) is not True for k in
                   ("preparation_verified", "source_inputs_rechecked", "copy_hashes_verified"))
            or any(record.get(k) is not False for k in
                   ("runtime_loaded", "abi_verified", "privacy_verified", "numerical_verified",
                    "promoted", "personal_data_included", "is_app_installer"))):
        raise ValueError("Expected an unapproved static candidate transfer")
    entries = [("candidate/manifest.json", manifest_path, before, "candidate_audit")]
    names = set()
    for item in record["files"]:
        name = relative(item["relative"]).as_posix()
        if name.casefold() in names or name.casefold() == "manifest.json":
            raise ValueError("Duplicate candidate transfer path")
        names.add(name.casefold())
        current = identity(kit / name)
        if (type(item.get("size")) is not int or current !=
                {"size": item["size"], "sha256": item["sha256"]}):
            raise ValueError("Candidate transfer bytes changed")
        entries.append(("candidate/" + name, kit / name, current, "candidate_transfer"))
    actual = {p.relative_to(kit).as_posix().casefold() for p in kit.rglob("*") if p.is_file()}
    if actual != names | {"manifest.json"}:
        raise ValueError("Unexpected file in private candidate kit")
    expected_libs = {"libs/mediapipe_source.dll", "libs/opencv_core3411.dll",
                     "libs/opencv_imgproc3411.dll"}
    if {name for name in names if name.startswith("libs/")} != expected_libs:
        raise ValueError("Unexpected candidate library set")
    if identity(kit / "libs/mediapipe_source.dll")["sha256"] != CANDIDATE_HASH:
        raise ValueError("Unreviewed source candidate bytes")
    if identity(manifest_path) != before:
        raise ValueError("Candidate manifest changed during inspection")
    for name in sorted(expected_libs):
        target = "libmediapipe.dll" if name.endswith("mediapipe_source.dll") else Path(name).name
        entries.append(("python/packages/mediapipe/tasks/c/" + target, kit / name,
                        identity(kit / name), "source_candidate_placement"))
    return entries


def dependency_plan(site, pins):
    """Read only a closed wheel inventory; exclude wheel executable scripts."""
    site = ordinary(site, directory=True)
    distributions = {}
    for dist in importlib.metadata.distributions(path=[str(site)]):
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name in pins:
            if name in distributions:
                raise ValueError("Duplicate installed guest dependency")
            distributions[name] = dist
    if set(distributions) != set(pins):
        raise ValueError("Missing locked guest dependency")
    entries, excluded = [], []
    for name in sorted(pins):
        dist = distributions[name]
        record_text = dist.read_text("RECORD")
        if dist.version != pins[name] or not record_text:
            raise ValueError("Installed dependency differs from lock or lacks RECORD")
        saw_held = False
        # Read every raw RECORD row. Distribution.files may suppress missing
        # entries, which would hide an unknown external path from inspection.
        for row in csv.reader(io.StringIO(record_text)):
            if len(row) != 3:
                raise ValueError("Malformed wheel RECORD")
            raw, recorded_hash, recorded_size = row
            raw = raw.replace("\\", "/")
            # RECORD may name launchers outside site-packages. They are never
            # copied, resolved or run. Guest tests require no pip/console tools.
            if raw.startswith("../"):
                if name == "fonttools" and raw == "../../share/man/man1/ttx.1":
                    excluded.append({"package": name, "relative": raw, "reason": "wheel_manual"})
                    continue
                if not raw.startswith("../../Scripts/"):
                    raise ValueError("Unexpected external wheel RECORD entry")
                relative(raw.removeprefix("../../Scripts/"))
                excluded.append({"package": name, "relative": raw, "reason": "wheel_launcher"})
                continue
            rel = relative(raw).as_posix()
            if rel.endswith(".pyc") and "__pycache__" in PurePosixPath(rel).parts:
                excluded.append({"package": name, "relative": rel, "reason": "local_bytecode"})
                continue
            if (rel.endswith(".pth") or Path(rel).name in ("sitecustomize.py", "usercustomize.py")):
                raise ValueError("Guest dependencies may not contain startup hooks")
            path = site / rel
            current = identity(path)
            if not recorded_hash:
                if not rel.endswith(".dist-info/RECORD") or recorded_size:
                    raise ValueError("Wheel file lacks recorded hash")
            else:
                encoded = base64.urlsafe_b64encode(bytes.fromhex(current["sha256"])).decode().rstrip("=")
                if (recorded_hash != "sha256=" + encoded or not recorded_size.isdecimal()
                        or int(recorded_size) != current["size"]):
                    raise ValueError("Installed wheel RECORD mismatch")
            if rel.casefold() == HELD_DLL:
                if name != "mediapipe":
                    raise ValueError("Unexpected ownership of held native DLL")
                saw_held = True
                excluded.append({"package": name, "relative": rel,
                                 "reason": "telemetry_held_wheel_dll", **current})
                continue
            entries.append(("python/packages/" + rel, path, current, "locked_dependency:" + name))
        if name == "mediapipe" and not saw_held:
            raise ValueError("Cannot identify the held native wheel DLL")
    return entries, excluded


def runtime_plan(runtime):
    runtime = ordinary(runtime, directory=True)
    if runtime.name != "cpython-3.12.14-windows-x86_64-none":
        raise ValueError("Use the existing reviewed CPython 3.12.14 runtime")
    fixed = ("python.exe", "pythonw.exe", "python3.dll", "python312.dll",
             "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt", "BUILD")
    planned = [("python/" + name, runtime / name, identity(runtime / name), "cpython_runtime")
               for name in fixed]
    for folder in ("Lib", "DLLs", "tcl"):
        base = ordinary(runtime / folder, directory=True)
        for path in sorted(base.rglob("*")):
            # Check directories as well, including junctions skipped by rglob.
            if path.is_symlink() or path.is_junction():
                raise ValueError("Runtime contains linked files or directories")
            if path.is_dir():
                continue
            rel = relative(path.relative_to(runtime).as_posix()).as_posix()
            parts = PurePosixPath(rel).parts
            if ("site-packages" in parts or "__pycache__" in parts or path.suffix == ".pyc"):
                continue
            if path.name in ("sitecustomize.py", "usercustomize.py") or path.suffix == ".pth":
                raise ValueError("Runtime contains a startup hook")
            planned.append(("python/" + rel, path, identity(path), "cpython_runtime"))
    return planned


def prepare(root, runtime, site, kit, destination):
    root = ordinary(root, directory=True)
    destination = Path(destination).absolute()
    build = ordinary(root / "build", directory=True)
    if (destination.parent != build or not destination.name.startswith(PREFIX)
            or destination.exists() or destination.is_symlink() or destination.is_junction()):
        raise ValueError("Use a fresh guest environment directly inside build")
    lock_path = root / "uv.lock"
    lock_id = identity(lock_path)
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    pins = closure(lock)
    dependencies, excluded = dependency_plan(site, pins)
    planned = [*runtime_plan(runtime), *dependencies, *candidate_plan(kit),
               ("reference/uv.lock", lock_path, lock_id, "environment_reference")]
    verifier = root / "tools/verify_vision_guest_environment.py"
    planned.append(("verify_vision_guest_environment.py", verifier, identity(verifier), "verifier"))
    planned.append(("python/verify_vision_guest_environment.py", verifier, identity(verifier), "verifier"))
    gui = root / "tools/vision_guest_checks.py"
    planned.append(("python/vision_guest_checks.py", gui, identity(gui), "guest_graphical_checker"))
    seen = set()
    for name, _, _, _ in planned:
        key = relative(name).as_posix().casefold()
        if key in seen:
            raise ValueError("Colliding guest environment path")
        seen.add(key)
    if sum(item[2]["size"] for item in planned) > MAX_TOTAL:
        raise ValueError("Guest environment exceeds transfer limit")
    destination.mkdir(exist_ok=False)
    report = {"schema_version": 1, "recorded_utc": datetime.now(UTC).isoformat(),
              "preparation_verified": False, "package_pins": pins, "excluded": excluded,
              "files": [], "is_app_installer": False, "is_self_contained_test_runner": False,
              "public_redistribution_approved": False, "host_runtime_modified": False,
              "network_enabled": False, "physical_camera_used": False,
              **dict.fromkeys(UNVERIFIED, False)}
    try:
        for name, source, expected, role in planned:
            if identity(source) != expected:
                raise ValueError("Source changed before copy")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            if identity(target) != expected:
                raise ValueError("Transferred environment bytes differ")
            report["files"].append({"relative": name, "role": role,
                                    "origin_path": str(source), **expected})
        # ._pth ignores PYTHONPATH/registry/user site and does not enable site
        # initialization. The staged SDK is not imported here.
        path_config = destination / "python/python312._pth"
        with path_config.open("x", encoding="utf-8", newline="\n") as writer:
            writer.write(ISOLATED_PATHS)
        report["files"].append({"relative": "python/python312._pth", "role": "isolated_path_config",
                                **identity(path_config)})
        for name, source, expected, _ in planned:
            if identity(source) != expected or identity(destination / name) != expected:
                raise ValueError("Source or transfer changed during preparation")
        report["preparation_verified"] = True
        with (destination / "environment.json").open("x", encoding="utf-8") as writer:
            json.dump(report, writer, indent=2)
        return report
    except (OSError, ValueError, KeyError, TypeError):
        report["preparation_verified"] = False
        report["partial_files_retained"] = True
        with (destination / "failure.json").open("x", encoding="utf-8") as writer:
            json.dump(report, writer, indent=2)
        raise


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-kit", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 12, 14):
        parser.error("Prepare with the existing CPython 3.12.14 environment")
    destination = root / "build" / (PREFIX + uuid4().hex)
    report = prepare(root, args.runtime_root, args.site_packages, args.candidate_kit, destination)
    print(f"Offline guest environment: {destination}", flush=True)
    print(f"Verified {len(report['files'])} files; no native load or guest execution.", flush=True)


if __name__ == "__main__":
    main()
