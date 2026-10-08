"""Build the pinned public MediaPipe C API without loading the held SDK.

Query/build logs and input identities are retained separately for every attempt.
A compiler success is only a candidate: this tool never promotes its DLL into
the app's privacy allowlist or replaces an installed wheel.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

SOURCE_COMMIT = "f8ef212d5c962c0e853db7e59d217056b187084b"
BAZEL_SHA256 = "4a76eddf6c5115e1d93355fd11db5ac2fc20e58f197f5d65d3f21da92aa0925b"
SOURCE_TARGET = "//mediapipe/tasks/c:mediapipe_source"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root, output_root, action, *, output_base=None):
    """Resolve the approved source/tool; reject precompiled substitutes."""
    source = root / "build/vendor/mediapipe-f8ef212d"
    bazel = root / "build/tools/bazel/bazel-7.4.1-windows-x86_64.exe"
    if source.is_symlink() or bazel.is_symlink() or checksum(bazel) != BAZEL_SHA256:
        raise ValueError("Pinned source/tool identity could not be verified")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != SOURCE_COMMIT or (source / ".bazelversion").read_text().strip() != "7.4.1":
        raise ValueError("The vision checkout is not the approved source version")
    changes = subprocess.check_output(
        ["git", "diff", "HEAD", "--"], cwd=source, text=True)
    # Development compatibility patches may be necessary, but must be retained
    # verbatim. They never establish privacy/ABI parity by themselves.
    inputs = ["WORKSPACE", ".bazelrc", ".bazelversion", "third_party/BUILD",
              "mediapipe/tasks/c/BUILD",
              "mediapipe/gpu/gpu_service.cc", "mediapipe/gpu/gpu_service.h",
              "mediapipe/framework/graph_service.h",
              "mediapipe/framework/legacy_calculator_support.cc",
              "mediapipe/framework/legacy_calculator_support.h",
              "mediapipe/framework/api3/calculator_context.h",
              "mediapipe/framework/api3/graph.h",
              "mediapipe/framework/api3/subgraph_context.h",
              "mediapipe/tasks/cc/core/logging/factory/logging_factory.cc",
              "mediapipe/tasks/cc/core/logging/factory/BUILD",
              "mediapipe/tasks/cc/core/logging/tasks_dummy_logger.h"]
    for name in ("third_party/carl_cam_llvm_windows_copy.diff",
                 "third_party/carl_cam_opencv_msvc_hash.diff"):
        compatibility_patch = source / name
        if compatibility_patch.is_file():
            inputs.append(name)
    inventory = {name: checksum(source / name) for name in inputs}
    environment = dict(os.environ)
    environment.update({
        "BAZEL_VC": r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC",
        "BAZEL_SH": r"C:\Program Files\Git\usr\bin\bash.exe",
        "PYTHON_BIN_PATH": sys.executable,
        "HERMETIC_PYTHON_VERSION": "3.12",
        "PATH": r"C:\Program Files\Git\usr\bin;" + environment["PATH"],
    })
    for name in ("BAZEL_VC", "BAZEL_SH"):
        if not Path(environment[name]).exists():
            raise ValueError(f"Missing build prerequisite: {name}")
    command = [str(bazel), "--batch", f"--output_user_root={output_root}"]
    if output_base is not None:
        if output_base.is_symlink() or not output_base.is_absolute():
            raise ValueError("Build output base must be an absolute ordinary directory")
        command.append(f"--output_base={output_base}")
    if action == "query":
        # An unconfigured `query` follows every select branch, including a
        # disabled converter's private @odml repository. `cquery` records the
        # actual Windows/third-party build graph without changing upstream code.
        command += ["cquery", f"deps({SOURCE_TARGET})", "--output=label"]
    else:
        command += ["build", SOURCE_TARGET]
    command += ["-c", "opt", "--jobs=4", "--local_resources=memory=8192",
                "--define=MEDIAPIPE_DISABLE_GPU=1", "--define=MEDIAPIPE_DISABLE_OPENCV=1",
                "--define=BUILD_FOR_3P=1", "--define=OPENCV=source"]
    # pthreadpool's pinned BUILD file supplies GCC's -std=c11 even on MSVC.
    # Enable the equivalent C language/atomic features for that dependency
    # only; do not alter C++ dialects or substitute inference/threading code.
    command.append("--per_file_copt=external/pthreadpool/.*@/std:c11,/experimental:c11atomics")
    # MediaPipe's status macros use C++20 __VA_OPT__. MSVC's traditional
    # preprocessor is not enabled by /std:c++20 alone, including tool builds.
    command += ["--cxxopt=/Zc:preprocessor", "--host_cxxopt=/Zc:preprocessor"]
    command += ["--repo_env=HERMETIC_PYTHON_VERSION=3.12",
                f"--repo_env=PYTHON_BIN_PATH={sys.executable}", "--color=no", "--curses=no"]
    return source, command, environment, inventory, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--output-base", type=Path, default=Path(r"C:\CarlCamBuild"),
                        help="Dedicated short Windows compiler directory; no existing cache is deleted")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # A short, dedicated temp root avoids unnecessarily long compiler paths.
    # No existing cache is purged and no system path/registry value is changed.
    output_root = Path(os.environ["TEMP"]) / "cc-bzl-741"
    action = "query" if arguments.query else "build"
    source, command, environment, inventory, changes = prepare(
        root, output_root, action, output_base=arguments.output_base)
    if arguments.inspect_only:
        print(json.dumps({"command": command, "source_commit": SOURCE_COMMIT,
                          "input_hashes": inventory, "source_diff": changes}, indent=2))
        return 0
    evidence = root / "build" / ("vision-source-" + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / "source.diff").write_text(changes, encoding="utf-8")
    for name in inventory:
        target = evidence / "source-inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    record = {"started_utc": datetime.now(UTC).isoformat(), "action": action,
              "command": command, "source_commit": SOURCE_COMMIT,
              "bazel_sha256": BAZEL_SHA256, "input_hashes": inventory,
              "build_environment": {key: environment[key] for key in
                                    ("BAZEL_VC", "BAZEL_SH", "PYTHON_BIN_PATH",
                                     "HERMETIC_PYTHON_VERSION")},
              "compiler_returncode": None, "privacy_verified": False,
              "abi_verified": False, "promoted": False}
    manifest = evidence / "attempt.json"
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Vision source {action} evidence: {evidence}", flush=True)
    process = subprocess.Popen(command, cwd=source, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    with (evidence / "bazel.log").open("w", encoding="utf-8") as log:
        graph_labels = 0
        for line in process.stdout:
            log.write(line)
            log.flush()
            if action == "query" and line.startswith(("//", "@")):
                graph_labels += 1
            else:
                print(line, end="", flush=True)
    if action == "query":
        record["configured_graph_labels"] = graph_labels
        print(f"Retained {graph_labels} configured dependency labels in bazel.log", flush=True)
    record["compiler_returncode"] = process.wait()
    record["finished_utc"] = datetime.now(UTC).isoformat()
    record["inputs_unchanged"] = all(checksum(source / name) == digest
                                     for name, digest in inventory.items())
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if not record["inputs_unchanged"]:
        print("Source inputs changed during this attempt; candidate remains unverified.")
        return 3
    return record["compiler_returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
