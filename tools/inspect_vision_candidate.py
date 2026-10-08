"""Retain static candidate evidence without loading the held vision SDK.

Export names are an ABI prerequisite, not proof of ABI compatibility. Import
names and marker scans likewise do not establish actual no-network behavior.
The report deliberately cannot authorize inference, installation or release.
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from build_vision_runtime import SOURCE_COMMIT

WRAPPERS = (
    "core/mediapipe_c_bindings.py",
    "core/mediapipe_c_utils.py",
    "vision/core/image.py",
    "vision/hand_landmarker.py",
    "vision/face_detector.py",
    "vision/image_segmenter.py",
    "vision/gesture_recognizer.py",
)
MARKERS = ("portable_clearcut_uploader", "play.googleapis.com", "clearcut")


def identity(path):
    """Hash ordinary exact files, including their absolute read-only origin."""
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"Expected an ordinary file: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest}


def parse_exports(output):
    names = set(re.findall(r"^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)",
                           output, re.MULTILINE))
    if not names:
        raise ValueError("No PE export table could be read")
    return names


def parse_dependencies(output):
    # /dependents includes normal imports and any delay-load dependencies.
    # Only isolated DLL-name rows count; paths, headings and summary text do not.
    return sorted(set(re.findall(r"^\s+([\w.-]+\.dll)\s*$", output,
                                re.MULTILINE | re.IGNORECASE)), key=str.casefold)


def wrapper_symbols(text):
    """Read literal C signatures without importing/executing wrapper code."""
    names = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if function not in {"CFunction", "_CFunction", "CStatusFunction"}:
            continue
        arg = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "func_name"), None)
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            raise TypeError("A required C signature is not a literal name")
        if not arg.value.startswith("Mp"):
            raise ValueError("Unexpected C signature namespace")
        names.add(arg.value)
    return names


def require_terminal_attempt(record):
    if (record.get("action") != "build" or type(record.get("compiler_returncode")) is not int
            or record["compiler_returncode"] != 0 or record.get("inputs_unchanged") is not True
            or not record.get("finished_utc") or record.get("source_commit") != SOURCE_COMMIT):
        raise ValueError("A successful unchanged pinned source build is required")


def marker_scan(path):
    data = path.read_bytes().lower()
    return {marker: {"ascii": marker.encode() in data,
                     "utf16le": marker.encode("utf-16le") in data} for marker in MARKERS}


def inspect(root, attempt, candidate, dumpbin, opencv, system, evidence):
    """Inspect exact files; explicit directories replace implicit PATH lookup."""
    attempt_file = attempt / "attempt.json"
    attempt_identity = identity(attempt_file)
    record = json.loads(attempt_file.read_text(encoding="utf-8"))
    require_terminal_attempt(record)
    source = root / "build/vendor/mediapipe-f8ef212d"
    diff_path = attempt / "source.diff"
    diff_identity = identity(diff_path)
    actual_diff = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=source, text=True)
    if actual_diff != diff_path.read_text(encoding="utf-8"):
        raise ValueError("Source patch changed after the successful build")
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual_commit != SOURCE_COMMIT:
        raise ValueError("Pinned source commit changed")
    inputs = {}
    for name, digest in record["input_hashes"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid source input path")
        current = identity(source / relative)
        retained = identity(attempt / "source-inputs" / relative)
        if current["sha256"] != digest or retained["sha256"] != digest:
            raise ValueError(f"Source input changed: {name}")
        inputs[name] = current
    installed = root / ".venv/Lib/site-packages/mediapipe/tasks/python"
    wrapper_records, required = {}, {"MpErrorFree"}
    for name in WRAPPERS:
        wrapper = installed / name
        symbols = wrapper_symbols(wrapper.read_text(encoding="utf-8"))
        upstream = source / "mediapipe/tasks/python" / name
        wrapper_records[name] = {"installed": identity(wrapper), "source": identity(upstream),
                                 "required_symbols": sorted(symbols),
                                 "text_matches_after_newline_normalization":
                                 wrapper.read_text(encoding="utf-8") == upstream.read_text(encoding="utf-8")}
        # Git's Windows checkout changes LF to CRLF. Retain both exact hashes
        # and compare decoded source after universal-newline normalization only.
        if not wrapper_records[name]["text_matches_after_newline_normalization"]:
            raise ValueError(f"Installed and source wrapper differ: {name}")
        required.update(symbols)
    tool = identity(dumpbin)
    libraries, observed = {}, {}

    def run(mode, path, stem):
        result = subprocess.run([str(dumpbin), "/nologo", mode, str(path)],
                                text=True, capture_output=True, check=False)
        (evidence / f"{stem}-{mode[1:]}.txt").write_text(
            result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise ValueError(f"Static inspection failed: {path.name} {mode}")
        return result.stdout

    def visit(path):
        key = path.name.casefold()
        if key in libraries:
            return
        original = identity(path)
        observed[str(path)] = original
        dependencies = parse_dependencies(run("/dependents", path, key))
        entry = {"identity": original, "imports": [], "marker_scan": marker_scan(path)}
        libraries[key] = entry
        for name in dependencies:
            if name.lower().startswith(("api-ms-", "ext-ms-")):
                entry["imports"].append({"name": name, "kind": "windows_api_set",
                                         "actual_loader_resolution_verified": False})
                continue
            choices = [folder / name for folder in (candidate.parent, opencv, system)
                       if (folder / name).is_file()]
            if not choices:
                raise ValueError(f"Unresolved import: {name}")
            resolved = choices[0]
            selected = identity(resolved)
            # Record the real exact system files, without recursively traversing
            # Windows's entire graph or pretending this proves loader resolution.
            kind = "windows_system" if resolved.parent == system else "source_build"
            entry["imports"].append({"name": name, "kind": kind, "identity": selected,
                                     "actual_loader_resolution_verified": False})
            observed[str(resolved)] = selected
            if kind == "source_build":
                visit(resolved)

    visit(candidate)
    exports = parse_exports(run("/exports", candidate, "candidate"))
    missing = sorted(required - exports)
    unchanged = all(identity(Path(path)) == item for path, item in observed.items())
    unchanged = unchanged and identity(dumpbin) == tool and identity(attempt_file) == attempt_identity
    unchanged = unchanged and identity(diff_path) == diff_identity
    unchanged = unchanged and all(identity(source / name) == item for name, item in inputs.items())
    unchanged = unchanged and all(identity(installed / name) == item["installed"]
                                  and identity(source / "mediapipe/tasks/python" / name) == item["source"]
                                  for name, item in wrapper_records.items())
    report = {"schema_version": 1, "recorded_utc": datetime.now(UTC).isoformat(),
              "attempt": attempt_identity, "source_commit": SOURCE_COMMIT,
              "source_diff": diff_identity, "source_inputs": inputs, "tool": tool,
              "candidate": identity(candidate), "libraries": libraries,
              "wrappers": wrapper_records, "export_names": sorted(exports),
              "required_export_names": sorted(required), "missing_exports": missing,
              "static_export_check_passed": not missing,
              "inspection_inputs_unchanged": unchanged,
              "runtime_loaded": False, "abi_verified": False,
              "privacy_verified": False, "numerical_verified": False, "promoted": False,
              "limits": ["Static export names do not prove structure layouts or correct calls.",
                         "Candidate bytes are recorded now, not retrospectively certified at build completion.",
                         "Marker absence does not prove absence of telemetry or networking.",
                         "Import inventory does not establish dynamic loading or Windows API-set resolution.",
                         "Dependency identities do not establish redistribution licensing."]}
    (evidence / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if missing or not unchanged:
        raise ValueError("Candidate static requirements failed; retained report identifies the failure")
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dumpbin", type=Path, required=True)
    parser.add_argument("--opencv-directory", type=Path, required=True)
    parser.add_argument("--system-directory", type=Path, default=Path(r"C:\Windows\System32"))
    args = parser.parse_args()
    evidence = root / "build" / ("vision-candidate-static-" + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    print(f"Static candidate evidence: {evidence}", flush=True)
    try:
        report = inspect(root, args.attempt, args.candidate, args.dumpbin,
                         args.opencv_directory, args.system_directory, evidence)
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        (evidence / "failure.json").write_text(json.dumps({
            "error": str(error), "runtime_loaded": False, "privacy_verified": False,
            "abi_verified": False, "promoted": False}, indent=2), encoding="utf-8")
        raise
    print(f"Matched {len(report['required_export_names'])} wrapper symbols; "
          f"inventoried {len(report['libraries'])} candidate/build libraries. "
          "Runtime, privacy and numerical checks remain unverified.")


if __name__ == "__main__":
    main()
