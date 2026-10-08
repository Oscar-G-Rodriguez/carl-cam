"""Repeat Qt's configured Nuitka build without purging useful intermediates.

pyside6-deploy unconditionally deletes deployment/, even in dry-run mode. This
adapter uses its own Nuitka command builder, records the exact inputs/command,
and runs that command with the existing compiler cache. It does not substitute
another freezer or label a failed compiler invocation as a completed app.
"""

import argparse
import configparser
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import PySide6


def input_inventory(root):
    """Record all shipped source/assets and deployment settings without links."""
    paths = sorted((root / "src").rglob("*")) + [root / name for name in (
        "Carl-Cam.py", "pysidedeploy.spec", "uv.lock", "pyproject.toml",
        "packaging/carl-cam.nuitka-package.config.yml")]
    inventory = []
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError("Build inputs must not use symbolic links")
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        inventory.append({"path": path.relative_to(root).as_posix(),
                          "sha256": checksum, "bytes": path.stat().st_size})
    return inventory


def changed_inputs(before, after):
    """Include additions/deletions as well as edits during compilation."""
    old = {item["path"]: (item["sha256"], item["bytes"]) for item in before}
    new = {item["path"]: (item["sha256"], item["bytes"]) for item in after}
    return sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = configparser.ConfigParser()
    config.read(root / "pysidedeploy.spec")
    if config["nuitka"]["mode"] != "standalone" or config["app"]["input_file"] != "Carl-Cam.py":
        raise ValueError("This adapter supports only the approved Carl-Cam directory build")
    for path in (root / "deployment", root / "build", root / "dist"):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("A build target leaves the Carl-Cam repository")
    sys.path.insert(0, str(Path(PySide6.__file__).parent / "scripts"))
    from deploy_lib.config import DesktopConfig
    from deploy_lib.nuitka_helper import Nuitka
    captured = []
    def capture(command, dry_run, fetch_output=False):
        captured.extend(str(part) for part in command)
        return subprocess.list2cmdline(captured), None
    with patch("deploy_lib.nuitka_helper.run_command", capture):
        Nuitka([sys.executable, "-m", "nuitka"]).create_executable(
            source_file=root / "Carl-Cam.py", extra_args=config["nuitka"]["extra_args"],
            qml_files=[], qt_plugins=config["qt"]["plugins"].split(","),
            excluded_qml_plugins=[], icon=str(root / config["app"]["icon"]),
            dry_run=True, permissions=[], mode=DesktopConfig.NuitkaMode.STANDALONE)
    if arguments.inspect_only:
        print(json.dumps({"Qt_generated_command": captured, "existing_intermediates_will_not_be_purged": True}))
        return 0
    evidence = root / "build" / ("native-input-" + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    inventory = input_inventory(root)
    for item in inventory:
        relative = Path(item["path"])
        path = root / relative
        if path.suffix in (".py", ".spec", ".toml", ".yml", ".json", ".lock"):
            target = evidence / "snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    previous = root / "build/nuitka-report.xml"
    if previous.is_file():
        shutil.copy2(previous, evidence / "previous-compilation-report.xml")
    manifest = {"schema_version": 1, "started_utc": datetime.now(UTC).isoformat(),
                "command": captured, "inputs": inventory, "compiler_returncode": None,
                "bundle_selftest_passed": False, "installer_built": False}
    manifest_path = evidence / "inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Qt-generated command and source snapshot retained at {evidence}", flush=True)
    environment = dict(os.environ)
    environment["NUITKA_CACHE_DIR"] = str(root / "build/nuitka-cache")
    process = subprocess.Popen(captured, cwd=root, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    with (evidence / "compiler.log").open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
    result = process.wait()
    manifest["compiler_returncode"] = result
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    manifest["changed_inputs_during_build"] = changed_inputs(inventory, input_inventory(root))
    manifest["source_matches_initial_inventory"] = not manifest["changed_inputs_during_build"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if result == 0:
        executable = root / "deployment/Carl-Cam.dist/Carl-Cam.exe"
        if not executable.is_file():
            raise ValueError("Compiler returned success but its executable is missing")
        if manifest["changed_inputs_during_build"]:
            print("Native output is preliminary: source/assets changed during compilation. Rebuild before bundle verification.")
            return 3
        print("Native executable created. Run its bundle check before compiling an installer.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
