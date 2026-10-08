"""Stage a private VM candidate without loading vision or installing anything.

This is a byte-transfer kit, not an installer, runnable test or privacy verdict.
Only the three source-built DLLs, public model assets and retained audit inputs
are copied. Windows system libraries, personal data and the held wheel DLL are
excluded. All sources and copies are rechecked before a preparation result.
"""

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from build_vision_runtime import SOURCE_COMMIT
from inspect_vision_candidate import WRAPPERS, identity, require_terminal_attempt

LIBRARIES = frozenset(("mediapipe_source.dll", "opencv_core3411.dll", "opencv_imgproc3411.dll"))
MODELS = frozenset(("hand_landmarker.task", "gesture_recognizer.task",
                    "blaze_face_short_range.tflite", "selfie_multiclass_256x256.tflite"))
UNVERIFIED = ("runtime_loaded", "abi_verified", "privacy_verified", "numerical_verified", "promoted")
GUIDANCE = """# Carl-Cam private vision candidate transfer

This directory is test preparation only. It does not approve vision, install
anything, run inference, prove a VM exists or provide a self-contained runner.
Do not use it to replace the host environment or the Carl-Cam privacy allowlist.
No Windows system DLL, prebuilt MediaPipe wheel DLL, personal settings, camera
image, landmark recording or personal gesture model is included.

1. A separately approved isolated Windows guest is required. Do not start a
   host-wide trace. Keep host personal folders, camera and microphone absent
   from the guest; transfer only this private test directory.
2. Recheck every transferred size/hash in manifest.json before any load. The
   lock/config files are reference inputs, not a copied Python environment.
   Prepare a separate compatible guest environment and resolve its own Windows
   system/runtime dependencies. This kit does not authorize Windows feature,
   hypervisor, firewall or security-policy changes.
3. The candidate retains its original mediapipe_source.dll name here. The
   audited Python wrapper expects mediapipe/tasks/c/libmediapipe.dll. Any guest
   test placement must identify and verify those exact candidate bytes BEFORE
   importing MediaPipe; never let the held wheel DLL load by default. Test
   preparation on the host does not authorize host SDK initialization.
4. Validate the guest socket observer with a positive control. Record the
   observer, guest-only boundary, process identities and exact dependency
   bytes. No packet-content capture or personal-data source is needed. A
   missing/unvalidated observer cannot establish no-network behavior.
5. ABI calls, each public model, six-category outputs, reviewed numerical
   comparisons and frozen loading all need actual evidence. Static exports,
   marker absence or a successful copy establish none of those. Retain
   failures and timing/output summaries; do not manufacture passing results.
6. Only after the required review may the exact runtime hash be approved and
   the current app rebuilt/installed/tested. Keep creator-demo and public
   release approval separate. Included notices are attribution evidence,
   not a complete native-dependency licensing or redistribution certificate.

evidence/ preserves unchanged build/static inputs; wrappers/ holds the exact
installed and upstream files used by that static audit, for comparison rather
than direct execution. models/ contains only the four public perception assets,
not a trained Carl-Cam gesture classifier. libs/ contains the source-built
candidate and its two source-built OpenCV dependencies. Windows import names
are listed in manifest.json but no host System32 file is copied.
"""


def read_json(path):
    if identity(path)["size"] > 512 * 1024:
        raise ValueError("Audit JSON exceeds the preparation limit")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Expected an audit object")
    return value


def checked_identity(expected):
    if (not isinstance(expected, dict) or not isinstance(expected.get("path"), str)
            or not Path(expected["path"]).is_absolute()
            or type(expected.get("size")) is not int or expected["size"] <= 0
            or not isinstance(expected.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None):
        raise ValueError("Invalid retained file identity")
    actual = identity(Path(expected["path"]))
    if actual != expected:
        raise ValueError("Retained source identity changed")
    return actual


def safe_relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe relative audit path")
    return Path(name)


def plan_inputs(root, static_path, opencv_license):
    """Resolve a closed, verified copy set before creating any destination."""
    report_identity = identity(static_path)
    record = read_json(static_path)
    if (record.get("schema_version") != 1 or record.get("source_commit") != SOURCE_COMMIT
            or record.get("static_export_check_passed") is not True
            or record.get("inspection_inputs_unchanged") is not True
            or record.get("missing_exports") != []
            or any(record.get(name) is not False for name in UNVERIFIED)):
        raise ValueError("Expected an unchanged, static-only source candidate")
    attempt_id = checked_identity(record.get("attempt"))
    attempt = read_json(attempt_id["path"])
    require_terminal_attempt(attempt)
    diff_id = checked_identity(record.get("source_diff"))
    candidate = checked_identity(record.get("candidate"))
    if Path(candidate["path"]).name != "mediapipe_source.dll":
        raise ValueError("Unexpected candidate library name")
    libraries = record.get("libraries")
    if not isinstance(libraries, dict) or set(libraries) != LIBRARIES:
        raise ValueError("Unexpected source-built library set")
    planned = []
    seen = set()

    def add(relative, source_identity, role):
        target = safe_relative(relative).as_posix()
        if target.casefold() in seen:
            raise ValueError("Colliding transfer destination")
        seen.add(target.casefold())
        origin = checked_identity(source_identity)
        if origin["size"] > 128 * 1024 * 1024:
            raise ValueError("Transfer input exceeds the per-file limit")
        planned.append({"relative": target, "origin": origin, "role": role})

    add("evidence/static-manifest.json", report_identity, "static_audit")
    add("evidence/build-attempt.json", attempt_id, "build_audit")
    add("evidence/source.diff", diff_id, "build_audit")
    windows = set()
    for name in sorted(LIBRARIES):
        entry = libraries[name]
        library = checked_identity(entry.get("identity"))
        if Path(library["path"]).name.casefold() != name:
            raise ValueError("Library identity/name mismatch")
        if name == "mediapipe_source.dll" and library != candidate:
            raise ValueError("Candidate does not match its import inventory")
        add("libs/" + Path(library["path"]).name, library, "source_built_library")
        for dependency in entry.get("imports", []):
            dependency_name, kind = dependency.get("name"), dependency.get("kind")
            if (not isinstance(dependency_name, str)
                    or re.fullmatch(r"[A-Za-z0-9_.-]+\.dll", dependency_name, re.IGNORECASE) is None):
                raise ValueError("Invalid import name")
            if kind == "source_build":
                target = libraries.get(dependency_name.casefold())
                if target is None or dependency.get("identity") != target["identity"]:
                    raise ValueError("Unresolved source-built import")
            elif kind in ("windows_system", "windows_api_set"):
                windows.add((dependency_name, kind))
            else:
                raise ValueError("Unrecognized import category")
    inputs = record.get("source_inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(attempt.get("input_hashes", {})):
        raise ValueError("Build source inventory mismatch")
    upstream = root / "build/vendor/mediapipe-f8ef212d"
    for name, source_id in inputs.items():
        relative = safe_relative(name)
        current = identity(upstream / relative)
        retained = identity(Path(attempt_id["path"]).parent / "source-inputs" / relative)
        if (current != source_id or current["sha256"] != attempt["input_hashes"][name]
                or retained["sha256"] != current["sha256"]):
            raise ValueError("Built source input changed")
        add("evidence/source-inputs/" + relative.as_posix(), source_id, "build_audit")
    wrappers = record.get("wrappers")
    if not isinstance(wrappers, dict) or set(wrappers) != set(WRAPPERS):
        raise ValueError("Wrapper inventory mismatch")
    installed = root / ".venv/Lib/site-packages/mediapipe/tasks/python"
    for name in WRAPPERS:
        pair = wrappers[name]
        installed_id, source_id = identity(installed / name), identity(upstream / "mediapipe/tasks/python" / name)
        if (pair.get("installed") != installed_id or pair.get("source") != source_id
                or pair.get("text_matches_after_newline_normalization") is not True
                or Path(installed_id["path"]).read_text(encoding="utf-8")
                != Path(source_id["path"]).read_text(encoding="utf-8")):
            raise ValueError("Audited wrappers changed")
        add("wrappers/installed/" + name, installed_id, "wrapper_audit")
        add("wrappers/source/" + name, source_id, "wrapper_audit")
    models = root / "src/carl_cam/assets/models"
    model_record = read_json(models / "manifest.json")
    if model_record.get("schema_version") != 1 or set(model_record.get("models", {})) != MODELS:
        raise ValueError("Public perception model inventory mismatch")
    for name in sorted(MODELS):
        metadata = model_record["models"][name]
        model = identity(models / name)
        if (type(metadata.get("bytes")) is not int or model["size"] != metadata["bytes"]
                or not 0 < model["size"] <= 64 * 1024 * 1024
                or model["sha256"] != metadata.get("sha256")):
            raise ValueError("Public perception model identity changed")
        add("models/" + name, model, "public_perception_model")
    for relative, source, role in (
        ("models/manifest.json", models / "manifest.json", "public_model_audit"),
        ("notices/model-Apache-2.0.txt", models / "Apache-2.0.txt", "notice"),
        ("notices/mediapipe-LICENSE", upstream / "LICENSE", "notice"),
        ("notices/opencv-LICENSE", opencv_license, "notice"),
        ("reference/uv.lock", root / "uv.lock", "environment_reference"),
        ("reference/pyproject.toml", root / "pyproject.toml", "environment_reference"),
    ):
        add(relative, identity(source), role)
    if sum(item["origin"]["size"] for item in planned) > 512 * 1024 * 1024:
        raise ValueError("Transfer exceeds the total-size limit")
    return planned, [{"name": name, "kind": kind} for name, kind in sorted(windows)]


def prepare(root, static_path, opencv_license, destination):
    root = Path(root).absolute()
    destination = Path(destination).absolute()
    build = root / "build"
    if (any(path.is_symlink() for path in (build, *build.parents))
            or destination.parent.resolve() != build.resolve()
            or not destination.name.startswith("vision-vm-candidate-")
            or destination.exists() or destination.is_symlink()):
        raise ValueError("Use a fresh candidate directory directly inside the repository build folder")
    planned, windows = plan_inputs(root, Path(static_path), Path(opencv_license))
    destination.mkdir(exist_ok=False)
    try:
        copied = []
        for entry in planned:
            source = Path(entry["origin"]["path"])
            checked_identity(entry["origin"])
            target = destination / entry["relative"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            result = identity(target)
            if (result["size"], result["sha256"]) != (
                    entry["origin"]["size"], entry["origin"]["sha256"]):
                raise ValueError("Transferred bytes differ")
            copied.append({**entry, "size": result["size"], "sha256": result["sha256"]})
        for entry in planned:
            checked_identity(entry["origin"])
            transferred = identity(destination / entry["relative"])
            if (transferred["size"], transferred["sha256"]) != (
                    entry["origin"]["size"], entry["origin"]["sha256"]):
                raise ValueError("Transferred bytes changed during preparation")
        (destination / "README.md").write_text(GUIDANCE, encoding="utf-8")
        guide = identity(destination / "README.md")
        copied.append({"relative": "README.md", "role": "test_guidance",
                       "size": guide["size"], "sha256": guide["sha256"]})
        report = {"schema_version": 1, "recorded_utc": datetime.now(UTC).isoformat(),
                  "source_commit": SOURCE_COMMIT, "preparation_verified": True,
                  "source_inputs_rechecked": True, "copy_hashes_verified": True,
                  "files": copied, "windows_imports_not_copied": windows,
                  "vm_started": False, "physical_camera_used": False,
                  "personal_data_included": False, "host_runtime_environment_modified": False,
                  "is_app_installer": False, "is_self_contained_runner": False,
                  "public_redistribution_approved": False, **dict.fromkeys(UNVERIFIED, False)}
        (destination / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except (OSError, ValueError, TypeError, KeyError):
        (destination / "failure.json").write_text(json.dumps({
            "preparation_verified": False, "partial_files_retained": True,
            "vm_started": False, **dict.fromkeys(UNVERIFIED, False)}, indent=2), encoding="utf-8")
        raise


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-manifest", type=Path, required=True)
    parser.add_argument("--opencv-license", type=Path, required=True)
    args = parser.parse_args()
    destination = root / "build" / ("vision-vm-candidate-" + uuid4().hex)
    report = prepare(root, args.static_manifest, args.opencv_license, destination)
    print(f"Private VM candidate transfer: {destination}", flush=True)
    print(f"Verified {len(report['files'])} files; no runtime load, VM start or approval.", flush=True)


if __name__ == "__main__":
    main()
