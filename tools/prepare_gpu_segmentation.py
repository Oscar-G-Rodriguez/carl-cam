"""Inspect/export the exact six-class segmenter without running inference.

Run in the isolated conversion environment. The default only inspects public
model bytes using generated metadata readers. --convert exports FP32 ONNX for
later runtime validation; it never installs/promotes an app backend or model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

MODEL_NAME = "selfie_multiclass_256x256.tflite"
MODEL_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
PINS = {"tf2onnx": "1.17.0", "tensorflow": "2.20.0", "onnx": "1.19.1",
        "numpy": "2.2.6", "tflite": "2.18.0"}
SCHEMA_HASHES = {
    "metadata_schema.fbs": "115cea931c40c8bc8d8271d39840f4951916888a10effc82d4ceab1db1239e2d",
    "image_segmenter_metadata_schema.fbs":
        "8cce27d0bbe5f2f6d16eefaeed34d634e2bdd260a0265b26bc44b03796f6605d",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ordinary_file(path):
    if (any(item.is_symlink() or item.is_junction() for item in (path, *path.parents))
            or not path.is_file()):
        raise ValueError(f"Expected ordinary file: {path.name}")
    return path


def reader(path):
    specification = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def inspect_model(path, metadata_reader, segmenter_reader):
    """Parse serialized public-model data, never allocate an interpreter."""
    import tflite
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MODEL_SHA256:
        raise ValueError("The segmenter is not the approved source model")
    if payload[4:8] != b"TFL3":
        raise ValueError("Invalid TFLite identifier")
    model = tflite.Model.GetRootAsModel(payload, 0)
    if model.SubgraphsLength() != 1:
        raise ValueError("Only the approved single-subgraph segmenter is supported")
    graph = model.Subgraphs(0)
    inputs, outputs = [], []
    tensor_types = Counter()
    type_names = {value: name for name, value in vars(tflite.TensorType).items()
                  if isinstance(value, int)}
    op_names = {value: name for name, value in vars(tflite.BuiltinOperator).items()
                if isinstance(value, int)}
    for index in range(graph.TensorsLength()):
        tensor = graph.Tensors(index)
        tensor_types[type_names[tensor.Type()]] += 1
    for indices, destination in ((graph.InputsAsNumpy(), inputs),
                                 (graph.OutputsAsNumpy(), outputs)):
        for index in indices:
            tensor = graph.Tensors(int(index))
            destination.append({"name": tensor.Name().decode(),
                                "shape": tensor.ShapeAsNumpy().tolist(),
                                "dtype": type_names[tensor.Type()]})
    if ([item["shape"] for item in inputs] != [[1, 256, 256, 3]]
            or [item["shape"] for item in outputs] != [[1, 256, 256, 6]]
            or any(item["dtype"] != "FLOAT32" for item in inputs + outputs)):
        raise ValueError("Unexpected six-class model interface")
    operations = Counter()
    for index in range(graph.OperatorsLength()):
        code = model.OperatorCodes(graph.Operators(index).OpcodeIndex())
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        name = code.CustomCode().decode() if builtin == 32 else op_names[builtin]
        operations[name] += 1
    metadata = None
    for index in range(model.MetadataLength()):
        item = model.Metadata(index)
        if item.Name() == b"TFLITE_METADATA":
            data = model.Buffers(item.Buffer()).DataAsNumpy().tobytes()
            metadata = metadata_reader.ModelMetadata.GetRootAsModelMetadata(data, 0)
    if metadata is None or metadata.SubgraphMetadataLength() != 1:
        raise ValueError("Missing model preprocessing metadata")
    meta_graph = metadata.SubgraphMetadata(0)
    input_meta = meta_graph.InputTensorMetadata(0)
    normalization = None
    for index in range(input_meta.ProcessUnitsLength()):
        unit = input_meta.ProcessUnits(index)
        if unit.OptionsType() == metadata_reader.ProcessUnitOptions.NormalizationOptions:
            options = unit.Options()
            parsed = metadata_reader.NormalizationOptions()
            parsed.Init(options.Bytes, options.Pos)
            normalization = {"mean": parsed.MeanAsNumpy().tolist(),
                             "std": parsed.StdAsNumpy().tolist()}
    activation = "NONE"  # Upstream ImageSegmenter's default when metadata is absent.
    activation_explicit = False
    for index in range(meta_graph.CustomMetadataLength()):
        custom = meta_graph.CustomMetadata(index)
        if custom.Name() == b"SEGMENTER_METADATA":
            options = segmenter_reader.ImageSegmenterOptions.GetRootAsImageSegmenterOptions(
                custom.DataAsNumpy().tobytes(), 0)
            activation = {0: "NONE", 1: "SIGMOID", 2: "SOFTMAX"}[options.Activation()]
            activation_explicit = True
    associated = {}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                info = archive.getinfo(name)
                if info.file_size > 32 * 1024:
                    raise ValueError("Unexpected associated-file size")
                associated[name] = archive.read(name).decode("utf-8")
    return {"inputs": inputs, "outputs": outputs, "operations": dict(operations),
            "tensor_types": dict(tensor_types), "normalization": normalization,
            "activation": activation, "activation_explicit": activation_explicit,
            "associated_files": associated}


def check_onnx(path):
    import onnx
    model = onnx.load(path, load_external_data=False)
    if (any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
            or model.functions or any(n.domain not in ("", "ai.onnx") for n in model.graph.node)):
        raise ValueError("Export has external weights or unsupported custom operations")
    onnx.checker.check_model(model, full_check=True)
    interface = {}
    for role, values, shape in (("inputs", model.graph.input, [1, 256, 256, 3]),
                                ("outputs", model.graph.output, [1, 256, 256, 6])):
        if len(values) != 1:
            raise ValueError("Unexpected ONNX interface")
        tensor = values[0].type.tensor_type
        actual = [dim.dim_value for dim in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or actual != shape:
            raise ValueError("Export changed tensor type or shape")
        interface[role] = [{"name": values[0].name, "shape": actual, "dtype": "FLOAT32"}]
    return {**interface, "operations": dict(Counter(n.op_type for n in model.graph.node)),
            "opsets": {item.domain: item.version for item in model.opset_import}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--flatc", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = ordinary_file(root / "src/carl_cam/assets/models" / MODEL_NAME)
    flatc = ordinary_file(arguments.flatc.absolute())
    if digest(model) != MODEL_SHA256:
        raise ValueError("Source model checksum failed")
    versions = {name: importlib.metadata.version(name) for name in PINS}
    if versions != PINS:
        raise ValueError("Use the pinned isolated conversion environment")
    output = root / "build" / ("gpu-segmentation-" + uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
              "source_model_sha256": MODEL_SHA256, "packages": versions,
              "tool_sha256": digest(Path(__file__)), "flatc_sha256": digest(flatc),
              "environment_packages": {item.metadata["Name"]: item.version
                                       for item in importlib.metadata.distributions()},
              "conversion_requested": arguments.convert, "conversion_verified": False,
              "inference_run": False, "gpu_benchmarked": False,
              "numerical_parity_verified": False, "privacy_verified": False,
              "application_default_changed": False, "physical_camera_used": False}
    try:
        source = root / "build/vendor/mediapipe-f8ef212d"
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source,
                                         text=True).strip()
        if commit != "f8ef212d5c962c0e853db7e59d217056b187084b":
            raise ValueError("Metadata source version changed")
        schemas = []
        report["schema_source_commit"] = commit
        report["schemas"] = {}
        for name in ("metadata_schema.fbs", "image_segmenter_metadata_schema.fbs"):
            schema = ordinary_file(source / "mediapipe/tasks/metadata" / name)
            if digest(schema) != SCHEMA_HASHES[name]:
                raise ValueError("Metadata schema differs from the pinned source")
            report["schemas"][name] = digest(schema)
            copied = output / name
            shutil.copyfile(schema, copied)
            schemas.append(str(copied))
        generated = subprocess.run([str(flatc), "--python", "--gen-onefile", "-o",
                                    str(output), *schemas], capture_output=True, text=True,
                                   check=True)
        (output / "schema-generation.log").write_text(
            generated.stdout + generated.stderr, encoding="utf-8")
        report["source_contract"] = inspect_model(
            model, reader(output / "metadata_schema_generated.py"),
            reader(output / "image_segmenter_metadata_schema_generated.py"))
        if arguments.convert:
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1",
                               TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2")
            candidate = output / "selfie_multiclass_fp32.onnx"
            command = [sys.executable, "-m", "tf2onnx.convert", "--tflite", str(model),
                       "--opset", "17", "--output", str(candidate)]
            report["conversion_command"] = command
            with (output / "conversion.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=600, check=False)
            report["conversion_exit_code"] = result.returncode
            if result.returncode:
                raise ValueError("Conversion failed; see retained conversion.log")
            report["onnx_contract"] = check_onnx(candidate)
            report["onnx_sha256"] = digest(candidate)
            report["onnx_bytes"] = candidate.stat().st_size
            report["conversion_verified"] = True
        report["source_model_unchanged"] = digest(model) == MODEL_SHA256
        if not report["source_model_unchanged"]:
            raise ValueError("Source model changed during inspection/export")
        return_code = 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError,
            subprocess.SubprocessError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return_code = 1
    finally:
        (output / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"GPU candidate preparation evidence: {output}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
