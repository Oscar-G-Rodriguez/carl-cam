"""Graphical, camera-free checks for the private Windows guest environment.

Only transferred file identity and Python path isolation are checked here.
Model execution/network observation are deliberately absent until the observer
has been prepared and independently validated. No console/terminal is needed.
"""

import argparse
import json
import queue
import struct
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from verify_vision_guest_environment import verify

NATIVE_MODULES = ("mediapipe", "numpy", "cv2", "torch", "onnxruntime", "tensorflow")


def python_isolation(root, *, runtime=sys):
    """Check the executing interpreter, not just its configuration file."""
    python = (Path(root).absolute() / "python").resolve()
    executable = Path(runtime.executable).resolve()
    if executable.parent != python or executable.name.lower() not in ("python.exe", "pythonw.exe"):
        raise ValueError("This check must use the Python interpreter inside the transferred bundle.")
    if tuple(runtime.version_info[:3]) != (3, 12, 14) or struct.calcsize("P") != 8:
        raise ValueError("Expected the staged 64-bit CPython 3.12.14 interpreter.")
    flags = runtime.flags
    if not (flags.isolated == 1 and flags.ignore_environment == 1 and flags.no_site == 1):
        raise ValueError("Python environment isolation is not active.")
    expected = {str((python / name).resolve()).casefold() for name in ("Lib", "DLLs", "packages", ".")}
    actual = [str(Path(name).resolve()).casefold() for name in runtime.path]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Python is searching outside the four staged directories.")
    if not runtime.dont_write_bytecode:
        raise ValueError("Launch with bytecode writes disabled to preserve the transfer inventory.")
    if any(name == prefix or name.startswith(prefix + ".")
           for name in runtime.modules for prefix in NATIVE_MODULES):
        raise ValueError("A held native SDK was imported before the guest checks.")
    return {"python_isolation_verified": True, "python_version": "3.12.14",
            "pointer_bits": 64, "python_executable": str(executable),
            "python_search_paths": list(runtime.path), "bytecode_writes_disabled": True,
            "vision_runtime_loaded": False, "privacy_verified": False, "gpu_benchmarked": False}


def run_check(root, manifest_hash, kind):
    """Recheck all transferred files before reporting either prerequisite."""
    if kind not in ("files", "python"):
        raise ValueError("Unsupported guest check")
    report = {"schema_version": 1, "recorded_utc": datetime.now(UTC).isoformat(),
              "check": kind, "manifest_sha256": manifest_hash, "check_passed": False,
              "vision_runtime_loaded": False, "privacy_verified": False,
              "abi_verified": False, "numerical_verified": False, "gpu_benchmarked": False}
    try:
        report["file_check"] = verify(root, manifest_hash)
        if kind == "python":
            report["python_check"] = python_isolation(root)
        report["check_passed"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["error"] = str(error)
    return report


def save_report(path, report):
    """Save only a new, explicitly selected diagnostic file; never overwrite."""
    path = Path(path).absolute()
    if (path.suffix.lower() != ".json" or path.exists()
            or any(p.is_symlink() or p.is_junction() for p in (path, *path.parents))
            or not path.parent.is_dir()):
        raise ValueError("Choose a new JSON filename in an ordinary existing folder.")
    with path.open("x", encoding="utf-8") as writer:
        json.dump(report, writer, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    # Tk and its standard-library native component are UI prerequisites, not
    # the held vision SDK. Import only when showing the window.
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Carl-Cam — Private VM Checks")
    root.geometry("740x580")
    root.minsize(600, 480)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Carl-Cam private VM checks", font=("Segoe UI", 18, "bold")).pack(anchor="w")
    ttk.Label(frame, text="No camera, model execution or network tracing is started here.",
              wraplength=660).pack(anchor="w", pady=(10, 4))
    ttk.Label(frame, text="1. Check transferred files.  2. Check Python isolation.  3. Save the report.",
              wraplength=660).pack(anchor="w", pady=(0, 12))
    actions = ttk.Frame(frame)
    actions.pack(fill="x")
    status = tk.StringVar(value="Ready. Native vision and GPU verification are still held.")
    ttk.Label(frame, textvariable=status, wraplength=660).pack(anchor="w", pady=12)
    output = tk.Text(frame, wrap="word", height=18, font=("Consolas", 10), state="disabled")
    output.pack(fill="both", expand=True)
    messages = queue.Queue()
    state = {"busy": False, "report": None}

    def show(report):
        state["report"] = report
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("end", json.dumps(report, indent=2))
        output.configure(state="disabled")
        status.set("Check passed. Vision/privacy/GPU checks remain unpassed." if report["check_passed"]
                   else "Check failed. Save the report; do not proceed to native model tests.")

    def start(kind):
        if state["busy"]:
            return
        state["busy"] = True
        for button in (files_button, python_button, save_button):
            button.configure(state="disabled")
        status.set("Checking transferred files. Please leave this window open…")

        def worker():
            messages.put(run_check(args.environment, args.manifest_sha256, kind))

        threading.Thread(target=worker, daemon=True).start()

    def poll():
        try:
            report = messages.get_nowait()
        except queue.Empty:
            pass
        else:
            state["busy"] = False
            show(report)
            for button in (files_button, python_button, save_button):
                button.configure(state="normal")
        root.after(100, poll)

    def save():
        if state["report"] is None or state["busy"]:
            return
        path = filedialog.asksaveasfilename(
            parent=root, title="Save the private VM check report", defaultextension=".json",
            initialfile="carl-cam-vm-check-" + uuid4().hex[:8] + ".json",
            filetypes=[("JSON check report", "*.json")])
        if not path:
            return
        try:
            save_report(path, state["report"])
        except (OSError, ValueError) as error:
            messagebox.showerror("Report was not saved", str(error), parent=root)
        else:
            status.set("Report saved locally. No report was uploaded or copied to the host.")

    files_button = ttk.Button(actions, text="Check transferred files", command=lambda: start("files"))
    files_button.pack(side="left", padx=(0, 8))
    python_button = ttk.Button(actions, text="Check Python isolation", command=lambda: start("python"))
    python_button.pack(side="left", padx=(0, 8))
    save_button = ttk.Button(actions, text="Save report…", command=save, state="disabled")
    save_button.pack(side="left")
    ttk.Label(frame, text="This utility does not activate Carl-Cam’s candidate runtime. "
              "Keep VM networking and host integrations disabled.", wraplength=660).pack(anchor="w", pady=(12, 0))
    root.after(100, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
