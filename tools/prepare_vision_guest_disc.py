"""Create a closed read-only-disc payload; never mount or execute it.

The VBS launcher opens the staged Python GUI directly, with no terminal, Run
dialog, network access, registry changes, installer or SDK execution. Actual
launch in Windows requires the owner's action-time confirmation.
"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from prepare_vision_guest_environment import identity, ordinary
from verify_vision_guest_environment import verify


def launcher(manifest_hash):
    if (not isinstance(manifest_hash, str) or len(manifest_hash) != 64
            or any(c not in "0123456789abcdef" for c in manifest_hash)):
        raise ValueError("Expected the independently recorded environment hash")
    return '''Option Explicit
Dim fileSystem, shell, discRoot, environmentRoot, command
Set fileSystem = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
discRoot = fileSystem.GetParentFolderName(WScript.ScriptFullName)
environmentRoot = fileSystem.BuildPath(discRoot, "environment")
command = Chr(34) & fileSystem.BuildPath(environmentRoot, "python\\pythonw.exe") & Chr(34) _
    & " -B " & Chr(34) & fileSystem.BuildPath(environmentRoot, "python\\vision_guest_checks.py") & Chr(34) _
    & " --environment " & Chr(34) & environmentRoot & Chr(34) _
    & " --manifest-sha256 ''' + manifest_hash + '''"
shell.Run command, 1, False
'''


def observer_launcher(manifest_hash, helper_hash, *, elevated=False):
    if (not isinstance(helper_hash, str) or len(helper_hash) != 64
            or any(c not in "0123456789abcdef" for c in helper_hash)):
        raise ValueError("Expected the independently recorded observer hash")
    source = launcher(manifest_hash).replace("vision_guest_checks.py", "guest_observer_checks.py")
    source = source.replace(" --manifest-sha256 " + manifest_hash,
                            " --manifest-sha256 " + manifest_hash + " --helper-sha256 " + helper_hash)
    if elevated:
        source = source.replace("Dim fileSystem, shell, discRoot, environmentRoot, command",
                                "Dim fileSystem, shell, discRoot, environmentRoot, command, adminShell, interpreter")
        source = source.replace("shell.Run command, 1, False", '''If MsgBox("This is for the isolated Carl-Cam VM only. Request administrator permission for the observer test? No camera or vision model will run.", vbYesNo + vbQuestion, "Carl-Cam private guest test") <> vbYes Then WScript.Quit
interpreter = fileSystem.BuildPath(environmentRoot, "python\\pythonw.exe")
Set adminShell = CreateObject("Shell.Application")
adminShell.ShellExecute interpreter, Mid(command, Len(interpreter) + 3), "", "runas", 1''')
    return source


def prepare(root, environment, manifest_hash, destination, *, observer_hash=None):
    root = ordinary(root, directory=True)
    environment = ordinary(environment, directory=True)
    build = ordinary(root / "build", directory=True)
    destination = Path(destination).absolute()
    if (environment.parent != build or not environment.name.startswith("vision-guest-environment-")
            or destination.parent != build or not destination.name.startswith("vision-guest-disc-")
            or destination.exists() or destination.is_symlink() or destination.is_junction()):
        raise ValueError("Use the selected build environment and a fresh disc directory")
    verified = verify(environment, manifest_hash)
    manifest = environment / "environment.json"
    source = json.loads(manifest.read_text(encoding="utf-8"))
    required = {"python/vision_guest_checks.py", "python/verify_vision_guest_environment.py",
                "python/pythonw.exe", "python/python312._pth"}
    if not required <= {entry["relative"] for entry in source["files"]}:
        raise ValueError("Selected environment lacks the graphical checker")
    if observer_hash is not None:
        observer_launcher(manifest_hash, observer_hash)
        required_observer = {"python/guest_observer_checks.py", "python/guest_socket_events.py",
                             "python/carl-cam-afd-control.exe"}
        if (source.get("observer_preparation_verified") is not True
                or source.get("observer_executed") is not False
                or source.get("observer_positive_control_passed") is not False
                or source.get("observer_helper_sha256") != observer_hash
                or not required_observer <= {entry["relative"] for entry in source["files"]}
                or identity(environment / "python/carl-cam-afd-control.exe")["sha256"] != observer_hash):
            raise ValueError("Selected environment lacks the unexecuted pinned observer")
    destination.mkdir(exist_ok=False)
    payload = destination / "payload"
    payload.mkdir()
    copied = payload / "environment"
    copied.mkdir()
    result = {"schema_version": 1, "recorded_utc": datetime.now(UTC).isoformat(),
              "environment_manifest_sha256": manifest_hash, "payload_verified": False,
              "files": [], "iso_created": False, "disc_mounted": False, "guest_launched": False,
              "vision_runtime_loaded": False, "privacy_verified": False, "gpu_benchmarked": False}
    try:
        for entry in [*source["files"], {"relative": "environment.json", **identity(manifest)}]:
            name = entry["relative"]
            origin = environment / name
            expected = {"size": entry["size"], "sha256": entry["sha256"]}
            if identity(origin) != expected:
                raise ValueError("Source environment changed before disc preparation")
            target = copied / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with origin.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            if identity(target) != expected:
                raise ValueError("Disc copy differs from source environment")
        verify(copied, manifest_hash)
        verify(environment, manifest_hash)
        with (payload / "Open Carl-Cam VM Checks.vbs").open("x", encoding="ascii", newline="\r\n") as writer:
            writer.write(launcher(manifest_hash))
        if observer_hash is not None:
            for name, elevated in (("Open Carl-Cam Observer.vbs", False),
                                   ("Open Carl-Cam Observer as Admin.vbs", True)):
                with (payload / name).open("x", encoding="ascii", newline="\r\n") as writer:
                    writer.write(observer_launcher(manifest_hash, observer_hash, elevated=elevated))
            result.update(observer_helper_sha256=observer_hash, observer_executed=False,
                          observer_positive_control_passed=False)
        with (payload / "READ ME FIRST.txt").open("x", encoding="utf-8") as writer:
            writer.write("Carl-Cam private VM prerequisites\n\n"
                         "Only after approval, open 'Open Carl-Cam VM Checks.vbs' in the private VM.\n"
                         "The window offers file verification and Python isolation checks.\n"
                         "It does not run vision models, use a camera, enable networking or trace sockets.\n"
                         "Save a new JSON report with the Save report button. Reports stay in the guest.\n"
                         "Do not bypass any Windows security warning; report it for review.\n\n"
                         "Environment manifest SHA-256 (also retained separately on the host):\n"
                         + manifest_hash + "\n")
            if observer_hash is not None:
                writer.write("\nObserver test (only in the isolated private VM):\n"
                             "Open 'Open Carl-Cam Observer.vbs' after first-launch approval.\n"
                             "Click Run observer control, confirm, and wait for the result.\n"
                             "This separate test observes guest socket metadata and performs four\n"
                             "guest-loopback calls. No vision model, camera or internet is used.\n"
                             "If Windows denies trace access, close that window and manually open\n"
                             "'Open Carl-Cam Observer as Admin.vbs'. It requests guest administrator\n"
                             "permission; you must handle Windows' prompt yourself. No permission\n"
                             "group, firewall, networking, host sharing or security setting is changed.\n"
                             "If cleanup is uncertain, save the report and stop the private VM\n"
                             "before another test. Do not run the vision candidate yet.\n"
                             "Observer positive-control success is not a vision privacy result.\n"
                             "Native helper SHA-256: " + observer_hash + "\n")
        result["files"] = [{"relative": p.relative_to(payload).as_posix(), **identity(p)}
                           for p in sorted(payload.rglob("*")) if p.is_file()]
        result["payload_verified"] = True
        result["verified_environment"] = verified
        with (destination / "disc-preparation.json").open("x", encoding="utf-8") as writer:
            json.dump(result, writer, indent=2)
        return result
    except (OSError, ValueError, KeyError, TypeError):
        result["payload_verified"] = False
        result["partial_files_retained"] = True
        with (destination / "failure.json").open("x", encoding="utf-8") as writer:
            json.dump(result, writer, indent=2)
        raise


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--observer-helper-sha256", default=None)
    args = parser.parse_args()
    destination = root / "build" / ("vision-guest-disc-" + uuid4().hex)
    record = prepare(root, args.environment, args.manifest_sha256, destination,
                     observer_hash=args.observer_helper_sha256)
    print(f"Read-only disc payload: {destination}", flush=True)
    print(f"Verified {len(record['files'])} files; not mounted or executed.", flush=True)


if __name__ == "__main__":
    main()
