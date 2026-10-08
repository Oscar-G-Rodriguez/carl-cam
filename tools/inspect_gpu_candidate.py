"""Inventory an exported GPU candidate and installed ORT bytes, without loading ORT.

Run in the private conversion environment. Structural validity and file hashes
do not establish native loading, numerical equivalence, GPU speed or privacy.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from prepare_gpu_segmentation import MODEL_SHA256, check_onnx, digest, ordinary_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from carl_cam.gpu_dependencies import inspect_nvidia_dependencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = arguments.candidate.absolute()
    record_path = ordinary_file(directory / "evidence.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record.get("source_model_sha256") != MODEL_SHA256
            or record.get("conversion_verified") is not True
            or record.get("source_model_unchanged") is not True):
        raise ValueError("GPU export preparation was not successful")
    model = ordinary_file(directory / "selfie_multiclass_fp32.onnx")
    if digest(model) != record.get("onnx_sha256") or model.stat().st_size != record.get("onnx_bytes"):
        raise ValueError("GPU candidate bytes differ from the export record")
    contract = check_onnx(model)
    if contract != record.get("onnx_contract"):
        raise ValueError("GPU candidate graph differs from the export contract")
    distribution = importlib.metadata.distribution("onnxruntime-gpu")
    if distribution.version != "1.30.0":
        raise ValueError("GPU runtime package differs from the staged version")
    inventory = []
    for relative in distribution.files or ():
        if str(relative).startswith("onnxruntime/") and str(relative).endswith(
                (".py", ".dll", ".pyd")):
            path = ordinary_file(Path(distribution.locate_file(relative)))
            inventory.append({"relative": str(relative), "origin": str(path),
                              "bytes": path.stat().st_size, "sha256": digest(path)})
    if not any(item["relative"].endswith(".pyd") for item in inventory):
        raise ValueError("GPU native runtime is missing")
    dependencies = inspect_nvidia_dependencies()
    identity_record = {"version": distribution.version,
                       "files": sorted((item["relative"], item["sha256"]) for item in inventory),
                       "nvidia_fingerprint": dependencies.fingerprint}
    identity = hashlib.sha256(json.dumps(identity_record, sort_keys=True).encode()).hexdigest()
    output = root / "build" / ("gpu-candidate-audit-" + uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
              "export_record": str(record_path), "export_record_sha256": digest(record_path),
              "onnx_sha256": digest(model), "onnx_contract": contract,
              "runtime_version": distribution.version, "runtime_files": inventory,
              "nvidia_fingerprint": dependencies.fingerprint,
              "combined_runtime_inventory_sha256": identity,
              "static_cuda_dependency_inventory_verified": True,
              "tool_sha256": digest(Path(__file__)), "runtime_imported": False,
              "inference_run": False, "gpu_benchmarked": False,
              "cuda_dependencies_verified": False, "loaded_dependencies_verified": False,
              "numerical_parity_verified": False, "privacy_verified": False,
              "application_default_changed": False}
    if any(digest(Path(item["origin"])) != item["sha256"] for item in inventory):
        raise ValueError("GPU runtime changed during inventory")
    if digest(model) != record["onnx_sha256"]:
        raise ValueError("GPU model changed during inventory")
    (output / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Static GPU candidate/runtime audit: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
