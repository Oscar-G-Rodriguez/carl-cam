"""Guest-only observer positive control, with no camera or vision execution.

The first checker established file and Python prerequisites. This separate
window rechecks them before executing a pinned native helper. The helper must
independently enforce private VM identity, adapter isolation and bounded ETW
cleanup. A positive control verifies this observer, never the vision runtime's
privacy or the host GPU. Tk/ctypes/subprocess are imported only on use.
"""

import argparse
import json
import queue
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from guest_socket_events import evaluate_positive_control, private_smbios_uuid
from verify_vision_guest_environment import file_hash, verify
from vision_guest_checks import python_isolation, save_report

MAX_OUTPUT = 24 * 1024 * 1024
_pending_helpers = []


def machine_identity():
    if sys.platform != "win32":
        raise ValueError("The observer is available only in the exact private Windows VM")
    import ctypes

    function = ctypes.WinDLL("kernel32", use_last_error=True).GetSystemFirmwareTable
    function.argtypes = (ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
    function.restype = ctypes.c_uint32
    count = function(0x52534D42, 0, None, 0)
    if not 8 <= count <= 1024 * 1024:
        raise ValueError("Cannot read bounded private VM firmware identity")
    buffer = ctypes.create_string_buffer(count)
    if function(0x52534D42, 0, buffer, count) != count:
        raise ValueError("Firmware identity changed while reading it")
    return private_smbios_uuid(buffer.raw)


def helper_path(root, helper_hash):
    if (not isinstance(helper_hash, str) or len(helper_hash) != 64
            or any(char not in "0123456789abcdef" for char in helper_hash)):
        raise ValueError("Expected the independently recorded native helper hash")
    path = Path(root).absolute() / "python/carl-cam-afd-control.exe"
    if file_hash(path) != helper_hash:
        raise ValueError("Native observer bytes do not match the independently recorded helper hash")
    return path


def control_report(document):
    """Validate a completed producer's metadata; do not authenticate JSON alone."""
    if not isinstance(document, dict) or document.keys() != {"events", "controls", "capture"}:
        raise ValueError("Native helper did not return the complete reviewed evidence format")
    return evaluate_positive_control(document["events"], document["controls"], document["capture"])


def status_message(report):
    if report.get("cleanup_required") is not False:
        return "Cleanup is uncertain. Save this report and stop the private VM before another test."
    if report.get("observer_positive_control_passed") is True:
        return "Observer control passed. Vision privacy and GPU tests are still pending."
    return "Observer control did not pass. Save the report; do not run the vision candidate."


def run_control(root, manifest_hash, helper_hash):
    report = {"schema_version": 1, "recorded_utc": datetime.now(UTC).isoformat(),
              "observer_positive_control_passed": False, "native_observer_execution_verified": False,
              "vision_runtime_loaded": False, "privacy_verified": False, "abi_verified": False,
              "numerical_verified": False, "gpu_benchmarked": False, "cleanup_required": False}
    try:
        report["file_check"] = verify(root, manifest_hash)
        report["python_check"] = python_isolation(root)
        report["vm_uuid"] = machine_identity()
        helper = helper_path(root, helper_hash)
        import subprocess

        # No shell, automatic elevation, terminal, downloaded executable or
        # user-selected command. Both parent and native helper enforce the VM.
        process = subprocess.Popen([str(helper), "--positive-control-only"], cwd=helper.parent,
                                   shell=False, close_fds=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        report["cleanup_required"] = True
        try:
            output, errors = process.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            # Killing a trace controller could orphan its ETW session. Retain
            # the owned process/pipes and explicitly require private-VM cleanup.
            _pending_helpers.append(process)
            report.update(cleanup_required=True, helper_pid=process.pid)
            raise ValueError("Observer did not finish. Stop the private VM before another test; "
                             "the helper was not forcibly killed and cleanup is unverified.") from None
        if len(output) > MAX_OUTPUT or len(errors) > 4096:
            raise ValueError("Native helper returned an oversized report")
        if errors or process.returncode != 0:
            try:
                failure = json.loads(output.decode("utf-8"))
            except (ValueError, UnicodeError):
                failure = {}
            detail = failure.get("error") if isinstance(failure, dict) else None
            if isinstance(failure, dict) and failure.get("cleanup_required") is False:
                report["cleanup_required"] = False
            raise ValueError(detail if isinstance(detail, str) and len(detail) <= 512
                             else "Native observer failed; no test passed")
        document = json.loads(output.decode("utf-8"))
        if (isinstance(document, dict) and isinstance(document.get("capture"), dict)
                and document["capture"].get("trace_stopped") is True):
            report["cleanup_required"] = False
        detail = control_report(document)
        # Recheck bytes and prerequisites after the observed local execution.
        helper_path(root, helper_hash)
        verify(root, manifest_hash)
        python_isolation(root)
        machine_identity()
        report["observer"] = detail
        report["helper_sha256"] = helper_hash
        report["native_observer_execution_verified"] = True
        report["observer_positive_control_passed"] = detail["observer_control_evidence_matches"]
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        report["error"] = str(error)
    # Raw records, kernel endpoint keys and buffer pointers are not saved or
    # displayed. Only the attribution/control summary leaves this function.
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--helper-sha256", required=True)
    args = parser.parse_args()
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Carl-Cam — Private Observer Control")
    root.geometry("780x620")
    root.minsize(640, 500)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Private VM network observer test", font=("Segoe UI", 18, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Run only in the isolated Carl-Cam VM. This test observes guest socket metadata "
              "and makes four guest-only loopback calls. No camera, vision model, packet contents, "
              "internet connection or host trace is used.", wraplength=720).pack(anchor="w", pady=12)
    actions = ttk.Frame(frame)
    actions.pack(fill="x")
    status = tk.StringVar(value="Ready. Passing this test does not verify vision privacy or GPU speed.")
    ttk.Label(frame, textvariable=status, wraplength=720).pack(anchor="w", pady=12)
    output = tk.Text(frame, wrap="word", height=22, font=("Consolas", 10), state="disabled")
    output.pack(fill="both", expand=True)
    messages = queue.Queue()
    state = {"busy": False, "report": None, "cleanup": False}

    def start():
        if state["busy"] or state["cleanup"]:
            return
        if not messagebox.askyesno("Confirm private guest test",
                "Start a bounded socket-metadata test inside this isolated VM? "
                "Keep VM networking disabled. No model or camera will be opened.", parent=root):
            return
        state["busy"] = True
        run_button.configure(state="disabled")
        save_button.configure(state="disabled")
        status.set("Checking transferred files and private VM identity, then running the observer control…")

        def worker():
            messages.put(run_control(args.environment, args.manifest_sha256, args.helper_sha256))

        threading.Thread(target=worker, daemon=True).start()

    def poll():
        try:
            report = messages.get_nowait()
        except queue.Empty:
            pass
        else:
            state.update(busy=False, report=report, cleanup=report["cleanup_required"])
            output.configure(state="normal")
            output.delete("1.0", "end")
            output.insert("end", json.dumps(report, indent=2))
            output.configure(state="disabled")
            status.set(status_message(report))
            save_button.configure(state="normal")
            if not state["cleanup"]:
                run_button.configure(state="normal")
        root.after(100, poll)

    def save():
        if state["busy"] or state["report"] is None:
            return
        path = filedialog.asksaveasfilename(parent=root, title="Save observer control report",
                defaultextension=".json", initialfile="carl-cam-observer-control.json",
                filetypes=[("JSON check report", "*.json")])
        if path:
            try:
                save_report(path, state["report"])
            except (OSError, ValueError) as error:
                messagebox.showerror("Report was not saved", str(error), parent=root)
            else:
                status.set("Report saved only inside the guest. No upload or host transfer occurred.")

    def close():
        if state["busy"]:
            messagebox.showinfo("Test is running", "Wait for the bounded test to finish before closing.", parent=root)
        else:
            root.destroy()

    run_button = ttk.Button(actions, text="Run observer control", command=start)
    run_button.pack(side="left", padx=(0, 8))
    save_button = ttk.Button(actions, text="Save report…", command=save, state="disabled")
    save_button.pack(side="left")
    root.protocol("WM_DELETE_WINDOW", close)
    poll()
    root.mainloop()


if __name__ == "__main__":
    main()
