"""Build the schema inspector or private-VM positive-control helper.

Retain exact source, compiler command, output and hashes in a new build folder.
The builder never runs the resulting executable or imports a vision runtime.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

VC = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207")
SDK = Path(r"C:\Program Files (x86)\Windows Kits\10")
SDK_VERSION = "10.0.26100.0"


def digest(path):
    with Path(path).open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def build(root, *, positive_control=False):
    root = Path(root).resolve()
    stem = "afd_observer" if positive_control else "afd_schema"
    source = root / "tools/native" / (stem + ".cpp")
    header = root / "tools/native/afd_schema.h"
    compiler = VC / "bin/Hostx64/x64/cl.exe"
    if os.name != "nt" or not source.is_file() or not header.is_file() or not compiler.is_file():
        raise ValueError("Expected the reviewed Windows source and installed MSVC compiler")
    for path in (root, root / "build", source, header, *source.parents):
        if path.is_symlink() or path.is_junction():
            raise ValueError("Build paths must not traverse a filesystem link")
    output = root / "build" / ("guest-observer-schema-" + uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / source.name
    shutil.copy2(source, snapshot)
    shutil.copy2(header, output / header.name)
    executable = output / ("carl-cam-afd-control.exe" if positive_control else "carl-cam-afd-schema.exe")
    environment = dict(os.environ)
    environment["INCLUDE"] = ";".join(str(path) for path in (
        VC / "include", SDK / "Include" / SDK_VERSION / "ucrt",
        SDK / "Include" / SDK_VERSION / "shared", SDK / "Include" / SDK_VERSION / "um"))
    environment["LIB"] = ";".join(str(path) for path in (
        VC / "lib/x64", SDK / "Lib" / SDK_VERSION / "ucrt/x64",
        SDK / "Lib" / SDK_VERSION / "um/x64"))
    command = [str(compiler), "/nologo", "/std:c++17", "/EHsc", "/W4", "/WX", "/O2",
               "/MT", "/utf-8", "/DUNICODE", "/D_UNICODE", str(snapshot),
               "/Fe:" + str(executable), "/Fo:" + str(output / (stem + ".obj")),
               "/link", "/INCREMENTAL:NO", "tdh.lib", "kernel32.lib"]
    if positive_control:
        command.extend(("advapi32.lib", "ws2_32.lib", "iphlpapi.lib", "bcrypt.lib", "uuid.lib"))
    manifest = {"schema_version": 1, "started_utc": datetime.now(UTC).isoformat(),
                "source_sha256": digest(source), "compiler_sha256": digest(compiler),
                "header_sha256": digest(header),
                "binary_kind": "private_vm_positive_control" if positive_control else "read_only_schema",
                "command": command, "compiler_returncode": None, "executable_executed": False,
                "trace_started": False, "privacy_verified": False, "gpu_benchmarked": False}
    manifest_path = output / "build-report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = subprocess.run(command, cwd=output, env=environment, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
    (output / "compiler.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    manifest["compiler_returncode"] = result.returncode
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    manifest["source_unchanged"] = (
        digest(source) == manifest["source_sha256"] == digest(snapshot)
        and digest(header) == manifest["header_sha256"] == digest(output / header.name))
    if result.returncode == 0 and executable.is_file() and manifest["source_unchanged"]:
        manifest["executable_sha256"] = digest(executable)
        manifest["executable_bytes"] = executable.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **manifest}, indent=2))
    if "executable_sha256" not in manifest:
        print(result.stdout + result.stderr)
        return result.returncode or 2
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-positive-control", action="store_true",
                        help="Compile, but never execute, the guest-only tracing helper")
    arguments = parser.parse_args()
    raise SystemExit(build(Path(__file__).resolve().parents[1],
                          positive_control=arguments.build_positive_control))
