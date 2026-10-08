"""Retain a static NVIDIA wheel inventory without importing/loading GPU runtimes."""

import argparse
import importlib.metadata
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carl_cam.gpu_dependencies import (
    NVIDIA_PACKAGE_VERSIONS,
    file_sha256,
    inspect_nvidia_dependencies,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-report", type=Path, required=True)
    arguments = parser.parse_args()
    receipt_path = arguments.install_report.absolute()
    receipt_hash = file_sha256(receipt_path)
    if receipt_path.stat().st_size > 5 * 1024**2:
        raise ValueError("Installation receipt is unexpectedly large")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {**NVIDIA_PACKAGE_VERSIONS, "opencv-contrib-python": "5.0.0.93"}
    sources = {}
    for item in receipt.get("install", []):
        metadata = item["metadata"]
        name = metadata["name"].lower().replace("_", "-")
        # A fresh environment receipt also lists converter dependencies. Retain
        # that full receipt but validate this step's GPU wheels specifically.
        if name not in expected:
            if name.startswith(("nvidia-", "opencv-")):
                raise ValueError("Unexpected GPU dependency in installation receipt")
            continue
        if name in sources or metadata["version"] != expected[name]:
            raise ValueError("Unexpected or duplicate installed GPU dependency")
        download = item["download_info"]
        url = urlparse(download["url"])
        digest = download["archive_info"]["hashes"]["sha256"]
        if (url.scheme != "https" or url.hostname != "files.pythonhosted.org"
                or not url.path.endswith("-win_amd64.whl")
                or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Unverified GPU wheel origin or archive hash")
        if importlib.metadata.version(name) != expected[name]:
            raise ValueError("Installed version differs from the wheel receipt")
        sources[name] = {"version": expected[name], "url": download["url"], "sha256": digest}
    if set(sources) != set(expected):
        raise ValueError("GPU installation receipt is incomplete")
    dependencies = inspect_nvidia_dependencies()
    files = [{**asdict(item), "path": str(item.path)} for item in dependencies.files]
    if file_sha256(receipt_path) != receipt_hash:
        raise ValueError("Installation receipt changed during inspection")
    if any(file_sha256(item.path) != item.sha256 for item in dependencies.files):
        raise ValueError("NVIDIA dependency changed during inspection")
    if any(name.startswith(("onnxruntime", "tensorflow", "mediapipe", "nvidia"))
           for name in sys.modules):
        raise RuntimeError("A held runtime was unexpectedly imported")
    output = ROOT / "build" / ("gpu-dependencies-audit-" + uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
              "install_report": str(receipt_path), "install_report_sha256": receipt_hash,
              "wheel_sources": sources, "nvidia_fingerprint": dependencies.fingerprint,
              "files": files, "preload_paths": [str(path) for path in dependencies.preload_paths],
              "requirements_sha256": file_sha256(ROOT / "tools/requirements-gpu-experiment.txt"),
              "loader_sha256": file_sha256(ROOT / "src/carl_cam/gpu_regions.py"),
              "inventory_source_sha256": file_sha256(ROOT / "src/carl_cam/gpu_dependencies.py"),
              "tool_sha256": file_sha256(Path(__file__)), "python": sys.version,
              "installed_versions_verified": True, "wheel_records_verified": True,
              "static_dependency_inventory_verified": True,
              "runtime_imported": False, "native_loading_attempted": False,
              "loaded_dependencies_verified": False, "privacy_verified": False,
              "numerical_parity_verified": False, "inference_run": False,
              "gpu_benchmarked": False, "application_default_changed": False}
    (output / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Static NVIDIA dependency audit: {output}")
    print(f"Verified {len(files)} recorded files; {len(dependencies.preload_paths)} preload DLLs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
