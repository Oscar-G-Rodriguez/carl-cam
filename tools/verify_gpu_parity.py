"""Compare CPU/GPU confidence masks on identical generated inputs, never a camera.

Each backend runs in its own interpreter to preserve its package/ABI boundary.
All retained arrays are numerical masks of generated pixels, not webcam media.
The result cannot establish real-person segmentation quality or live FPS.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def fixtures():
    y, x = np.indices((360, 640))
    gradient = np.stack((x * 255 // 639, y * 255 // 359, (x + y) * 255 // 998), axis=-1)
    checker = np.repeat((((x // 32 + y // 32) % 2) * 255)[..., None], 3, axis=-1)
    return {
        'black': np.zeros((360, 640, 3), np.uint8),
        'white': np.full((360, 640, 3), 255, np.uint8),
        'gradient': gradient.astype(np.uint8),
        'checker': checker.astype(np.uint8),
        'noise': np.random.default_rng(17).integers(0, 256, (360, 640, 3), dtype=np.uint8),
    }


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare_masks(cpu, gpu, criteria):
    """Reject incompatible arrays before applying generated-pixel tolerances."""
    for masks in (cpu, gpu):
        if (not isinstance(masks, np.ndarray) or masks.shape != (360, 640, 6)
                or masks.dtype != np.float32 or not np.isfinite(masks).all()
                or masks.min() < 0 or masks.max() > 1
                or not np.allclose(masks.sum(axis=-1), 1., rtol=0., atol=2e-5)):
            raise ValueError('Incompatible confidence-map evidence')
    delta = np.abs(cpu - gpu)
    metrics = {'max_absolute_error': float(delta.max()),
               'p99_absolute_error': float(np.percentile(delta, 99)),
               'mean_absolute_error': float(delta.mean()),
               'category_disagreement_fraction': float(np.mean(
                   cpu.argmax(axis=-1) != gpu.argmax(axis=-1)))}
    metrics['passed'] = all(metrics[key] <= limit for key, limit in criteria.items())
    return metrics


def cuda_profile_counts(events):
    """Require actual CUDA operations with no CPU-provider operation events."""
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError('Incompatible operator-profile evidence')
    counts = {}
    for event in events:
        arguments = event.get('args', {})
        if not isinstance(arguments, dict):
            raise TypeError('Incompatible operator-profile arguments')
        provider = arguments.get('provider')
        if event.get('cat') == 'Node' and provider:
            if not isinstance(provider, str):
                raise ValueError('Incompatible operator-profile provider')
            counts[provider] = counts.get(provider, 0) + 1
    if not counts.get('CUDAExecutionProvider') or any(
            provider != 'CUDAExecutionProvider' for provider in counts):
        raise ValueError('Operator profile did not prove exclusive CUDA execution')
    return counts


def worker(backend, output, candidate, *, disable_tf32=False, profile=False):
    from carl_cam.records import FramePacket
    if backend == 'cpu':
        from carl_cam.vision import RegionEngine
        engine = RegionEngine()
    else:
        from carl_cam.gpu_regions import GpuRegionEngine
        engine = GpuRegionEngine(candidate, experiment=True)
        if disable_tf32:
            # Public ORT API recreates the session with its existing options.
            # Keep the pinned model/DLL loader and CPU-fallback policy intact.
            engine.session.set_providers([('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0})])
            engine.session.disable_fallback()
            if engine.session.get_provider_options().get('CUDAExecutionProvider', {}).get('use_tf32') != '0':
                engine.close()
                raise ValueError('Requested full FP32 precision was not honored')
        if profile:
            import onnxruntime as ort
            options = engine.session.get_session_options()
            options.enable_profiling = True
            options.profile_file_prefix = str(output / 'gpu-operators')
            provider_options = engine.session.get_provider_options()['CUDAExecutionProvider']
            engine.session = ort.InferenceSession(str(engine.candidate.model), sess_options=options,
                                                 providers=[('CUDAExecutionProvider', provider_options)])
            engine.session.disable_fallback()
    masks, inputs = {}, {}
    try:
        for identifier, (name, rgb) in enumerate(fixtures().items()):
            result = engine.process(FramePacket(identifier, monotonic_ns(), rgb))
            if (result.frame_id != identifier or result.confidence_masks.shape != (360, 640, 6)
                    or result.confidence_masks.dtype != np.float32
                    or not np.isfinite(result.confidence_masks).all()):
                raise ValueError('Backend lost the numerical/source contract')
            masks[name] = result.confidence_masks
            inputs[name] = hashlib.sha256(rgb.tobytes()).hexdigest()
        np.savez_compressed(output / (backend + '-masks.npz'), **masks)
        report = {'backend': backend, 'inputs': inputs, 'camera_used': False,
                  'interpreter': str(Path(sys.executable).absolute())}
        if backend == 'gpu':
            report['providers'] = engine.session.get_providers()
            report['provider_options'] = engine.session.get_provider_options()
            report['candidate_evidence_sha256'] = engine.candidate.evidence_sha256
            if profile:
                profile_path = Path(engine.session.end_profiling())
                if (not profile_path.is_relative_to(output) or not profile_path.is_file()
                        or profile_path.stat().st_size > 16 * 1024 * 1024):
                    raise ValueError('Unexpected operator-profile output')
                events = json.loads(profile_path.read_text())
                counts = cuda_profile_counts(events)
                report['profile_sha256'] = digest(profile_path)
                report['profile_kernel_counts'] = counts
        (output / (backend + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--worker', choices=('cpu', 'gpu'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--disable-tf32', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.worker:
        output = args.output.absolute()
        if (not output.is_relative_to(ROOT / 'build') or not output.is_dir()
                or any(p.is_symlink() or p.is_junction() for p in (output, *output.parents))
                or any((output / (args.worker + suffix)).exists() for suffix in ('.json', '-masks.npz'))):
            raise ValueError('Worker needs new artifacts in the private build directory')
        worker(args.worker, output, args.candidate.absolute(),
               disable_tf32=args.disable_tf32, profile=args.profile)
        return 0
    output = ROOT / 'build' / ('gpu-parity-' + uuid4().hex)
    output.mkdir(exist_ok=False)
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'generated_pixels_only': True, 'physical_camera_used': False,
              'quality_evaluated': False, 'live_fps_gate_passed': False,
              'privacy_verified': False, 'runtime_promoted': False,
              'tf32_disabled': args.disable_tf32, 'criteria_scope': 'developer generated-pixel experiment',
              'operator_profiling_requested': args.profile,
              'completed': False, 'numerical_parity_verified': False,
              'criteria': {'max_absolute_error': 0.02, 'p99_absolute_error': 0.001,
                           'category_disagreement_fraction': 0.01},
              'source_sha256': digest(Path(__file__)), 'fixtures': {}}
    try:
        for backend, interpreter in (
                ('cpu', ROOT / '.venv/Scripts/python.exe'), ('gpu', Path(sys.executable).absolute())):
            command = [str(interpreter), str(Path(__file__).absolute()),
                       '--candidate', str(args.candidate.absolute()), '--worker', backend,
                       '--output', str(output)]
            if args.disable_tf32:
                command.append('--disable-tf32')
            if args.profile:
                command.append('--profile')
            process = subprocess.run(command, cwd=ROOT, check=False, timeout=180, capture_output=True)
            (output / (backend + '-stdout.log')).write_bytes(process.stdout)
            (output / (backend + '-stderr.log')).write_bytes(process.stderr)
            if process.returncode != 0:
                raise RuntimeError(f'{backend} worker failed with code {process.returncode}; logs retained')
        cpu_record = json.loads((output / 'cpu.json').read_text())
        gpu_record = json.loads((output / 'gpu.json').read_text())
        if cpu_record['inputs'] != gpu_record['inputs']:
            raise ValueError('Backends processed different generated pixels')
        if 'CUDAExecutionProvider' not in gpu_record['providers']:
            raise ValueError('GPU backend did not retain its CUDA provider')
        if args.disable_tf32 and gpu_record['provider_options']['CUDAExecutionProvider'].get('use_tf32') != '0':
            raise ValueError('GPU record did not retain full FP32 precision')
        if args.profile:
            counts = gpu_record.get('profile_kernel_counts', {})
            if set(counts) != {'CUDAExecutionProvider'} or counts['CUDAExecutionProvider'] <= 0:
                raise ValueError('GPU record did not retain exclusive CUDA execution')
        passed = True
        with np.load(output / 'cpu-masks.npz', allow_pickle=False) as cpu, \
                np.load(output / 'gpu-masks.npz', allow_pickle=False) as gpu:
            if set(cpu.files) != set(fixtures()) or set(gpu.files) != set(cpu.files):
                raise ValueError('Incomplete paired mask inventory')
            for name in cpu.files:
                metrics = compare_masks(cpu[name], gpu[name], report['criteria'])
                passed = passed and metrics['passed']
                report['fixtures'][name] = metrics
        report.update(completed=True, numerical_parity_verified=passed,
                      cpu_masks_sha256=digest(output / 'cpu-masks.npz'),
                      gpu_masks_sha256=digest(output / 'gpu-masks.npz'),
                      cpu_report_sha256=digest(output / 'cpu.json'),
                      gpu_report_sha256=digest(output / 'gpu.json'))
        return 0 if passed else 2
    finally:
        (output / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Paired numerical evidence: ' + str(output), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
