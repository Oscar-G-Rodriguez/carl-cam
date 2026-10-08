"""Benchmark an approved ONNX/CUDA region adapter on generated pixels only.

This fails closed while the approval list is empty. It never approves a runtime,
opens a camera, compares against a held SDK, or claims real-person quality.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from carl_cam.gpu_regions import GpuRegionEngine
from carl_cam.records import FramePacket
from carl_cam.vision_privacy import VisionPrivacyError


def summary(values):
    # Keep the segmentation experiment independent of Qt/rendering tooling.
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument('--experiment', action='store_true',
                        help='Test only the pinned experimental inventory; never promote it')
    parser.add_argument('--disable-tf32', action='store_true',
                        help='Measure full FP32 math rather than the CUDA TF32 default')
    arguments = parser.parse_args()
    try:
        engine = GpuRegionEngine(arguments.candidate, experiment=arguments.experiment)
        if arguments.disable_tf32:
            engine.session.set_providers([('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0})])
            engine.session.disable_fallback()
            if engine.session.get_provider_options().get('CUDAExecutionProvider', {}).get('use_tf32') != '0':
                engine.close()
                raise ValueError('Requested full FP32 precision was not honored')
    except VisionPrivacyError as error:
        print(f"Required GPU benchmark held: {error}")
        return 2
    root = Path(__file__).resolve().parents[1]
    output = root / "build" / ("gpu-region-benchmark-" + uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
              "physical_camera_used": False, "virtual_output_used": False,
              "generated_pixels_only": True, "quality_evaluated": False,
              "native_executable_benchmarked": False, "live_fps_gate_passed": False,
              'experimental': arguments.experiment, 'runtime_promoted': False,
              'candidate_evidence_sha256': engine.candidate.evidence_sha256,
              'tf32_disabled': arguments.disable_tf32,
              'provider_options': engine.session.get_provider_options(),
              "fixtures": {}, "completed": False, "sources": {}}
    for relative in ("src/carl_cam/gpu_regions.py", "src/carl_cam/gpu_dependencies.py",
                     "src/carl_cam/vision.py", "src/carl_cam/camera.py",
                     "tools/requirements-gpu-experiment.txt", "tools/benchmark_gpu_regions.py"):
        report["sources"][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    rng = np.random.default_rng(17)
    images = {"black": np.zeros((360, 640, 3), np.uint8),
              "white": np.full((360, 640, 3), 255, np.uint8),
              "noise": rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)}
    identifier = 0
    try:
        for name, rgb in images.items():
            rows = {key: [] for key in ("total", "preprocess", "inference", "postprocess")}
            for index in range(-12, 181):
                packet = FramePacket(identifier, monotonic_ns(), rgb)
                result = engine.process(packet)
                identifier += 1
                if (result.frame_id != packet.frame_id
                        or result.capture_time_ns != packet.capture_time_ns
                        or result.confidence_masks.shape != (360, 640, 6)
                        or not np.isfinite(result.confidence_masks).all()):
                    raise ValueError("GPU region result lost its source-frame contract")
                if index >= 0:
                    rows["total"].append(result.inference_ms)
                    for stage, value in engine.last_stage_ms.items():
                        rows[stage].append(value)
            report["fixtures"][name] = {stage: summary(values) for stage, values in rows.items()}
            print(f"Completed generated GPU region fixture: {name}", flush=True)
        report["completed"] = True
        return 0
    finally:
        engine.close()
        (output / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"GPU timing evidence: {output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
