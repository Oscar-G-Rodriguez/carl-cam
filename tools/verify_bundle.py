"""Camera-free native bundle checks and an exact generated file manifest."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from carl_cam.gpu_bundle import HELPER_DIRECTORY, verified_helper
from carl_cam.vision_privacy import (
    APPROVED_VISION_RUNTIMES,
    RUNTIME_POLICY,
    SUPPORTED_VISION_RUNTIMES,
)

NATIVE_CHECKS = frozenset({
    'Qt_widgets_and_bundled_font', 'original_graphics', 'bundled_vision_models',
    'PyTorch_checksum_load_and_inference', 'CPU_neural_training_and_logistic_comparison',
    'camera_off_by_default_and_finite_shutdown',
    'optional_paired_canned_model',
    'CUDA_training_runtime_and_CPU_default',
})
REQUIRED_FILES = (
    'mediapipe/tasks/c/libmediapipe.dll', 'carl_cam/assets/models/manifest.json',
    'carl_cam/assets/graphics/manifest.json', 'carl_cam/assets/fonts/Arimo.ttf',
    'carl_cam/assets/fonts/OFL.txt', 'carl_cam/assets/models/Apache-2.0.txt',
    'help/Quick Start.html', 'licenses/inventory.json',
    'msvcp140.dll', 'vcomp140.dll', 'msvcp140_atomic_wait.dll', 'vcruntime140_threads.dll',
)
TORCH_HEADER_DIRECTORY = 'torch/include/torch/csrc/api/include/torch/data/datasets'
TORCH_DATA_HEADERS = frozenset({'base.h', 'chunk.h', 'map.h', 'mnist.h', 'shared.h',
                                'stateful.h', 'tensor.h'})
ORT_DATASET_MODULE = 'gpu-helper/onnxruntime/datasets/__init__.py'
ORT_DATASET_MODULE_SHA256 = '0ea45da4c7d1b437e1564090bb995399f490f861b87444c73567680787c0ee55'
GPU_NATIVE_CHECK = 'installed_GPU_helper_and_prepared_frame_pairing'
GPU_TRAINING_CHECK = 'CUDA_neural_training_and_CPU_model_reload'


def _published_torch_header(relative, checksum):
    """Dataset API headers are vendor source, not collected personal samples."""
    if relative.parent.as_posix() != TORCH_HEADER_DIRECTORY or relative.name not in TORCH_DATA_HEADERS:
        return False
    original = Path(importlib.metadata.distribution('torch').locate_file(relative.as_posix()))
    if not original.is_file() or original.is_symlink():
        return False
    with original.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == checksum


def _normal_tree(path: Path) -> bool:
    # Reject a linked parent too, before resolve() hides that indirection.
    return all(not item.is_symlink() for item in (path, *path.parents))


def verify(bundle: Path, report_path: Path, *, selftest: bool = True, require_gpu_training: bool = False) -> dict:
    if require_gpu_training and not selftest:
        raise ValueError('Actual GPU training requires native execution')
    if not _normal_tree(bundle) or not _normal_tree(report_path):
        raise ValueError('Bundle and evidence paths must not use symbolic links')
    bundle = bundle.resolve()
    report_path = report_path.resolve()
    if report_path.is_relative_to(bundle):
        raise ValueError('Keep verification evidence outside the installation bundle')
    if report_path.exists() or report_path.with_name('bundled-selftest.json').exists():
        raise ValueError('Use a new report directory so old evidence cannot be reused')
    exe = bundle / 'Carl-Cam.exe'
    if not exe.is_file() or exe.is_symlink():
        raise ValueError('The standalone Carl-Cam.exe is missing')
    missing = [name for name in REQUIRED_FILES if not (bundle / name).is_file()]
    if missing:
        raise ValueError('Bundle is missing required offline files: ' + ', '.join(missing))
    files = []
    gpu_present = (bundle / HELPER_DIRECTORY).exists()
    if gpu_present:
        verified_helper(bundle / HELPER_DIRECTORY)
    for path in sorted(bundle.rglob('*')):
        if path.is_symlink() or not path.resolve().is_relative_to(bundle):
            raise ValueError('Bundle cannot contain links outside its installation directory')
        if path.is_file():
            relative_path = path.relative_to(bundle)
            relative = relative_path.as_posix()
            vendor_header = (relative_path.parent.as_posix() == TORCH_HEADER_DIRECTORY
                             and relative_path.name in TORCH_DATA_HEADERS)
            vendor_ort = relative == ORT_DATASET_MODULE
            if (any(part in ('.venv', 'test-recordings') for part in relative_path.parts)
                    or 'datasets' in relative_path.parts and not (vendor_header or vendor_ort)):
                raise ValueError('Personal recordings or development environment entered the bundle')
            with path.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            if vendor_header and not _published_torch_header(relative_path, checksum):
                raise ValueError('Vendor dataset header does not match the locked PyTorch wheel')
            if vendor_ort and checksum != ORT_DATASET_MODULE_SHA256:
                raise ValueError('Vendor ORT module does not match the locked GPU wheel')
            files.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': checksum})
    # Inspect the complete tree before launching anything from it. Otherwise a
    # linked native dependency could be loaded before the link is discovered.
    vision_hash = next(entry['sha256'] for entry in files
                       if entry['path'] == 'mediapipe/tasks/c/libmediapipe.dll')
    privacy_verified = vision_hash in APPROVED_VISION_RUNTIMES
    result = None
    if selftest:
        if vision_hash not in SUPPORTED_VISION_RUNTIMES | APPROVED_VISION_RUNTIMES:
            raise ValueError('Vision runtime integrity hold: exact supported SDK bytes are required')
        test_path = report_path.with_name('bundled-selftest.json')
        test_path.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.run([str(exe), '--self-test', '--self-test-report', str(test_path)],
                                 cwd=bundle, check=False, timeout=120)
        if process.returncode != 0 or not test_path.is_file():
            raise ValueError('The bundled executable failed its offline native self-test')
        result = json.loads(test_path.read_text(encoding='utf-8'))
        if (result.get('passed') is not True or result.get('camera_opened') is not False
                or result.get('virtual_output_started') is not False
                or result.get('schema_version') != 1
                or not isinstance(result.get('checks'), dict)
                or any(result['checks'].get(name) is not True for name in NATIVE_CHECKS)
                or gpu_present and result['checks'].get(GPU_NATIVE_CHECK) is not True
                or require_gpu_training and result['checks'].get(GPU_TRAINING_CHECK) is not True):
            raise ValueError('Bundle self-test has invalid or failing evidence')
    report = {'schema_version': 1, 'selftest_passed': result is not None,
              'vision_runtime_privacy_verified': privacy_verified,
              'vision_runtime_policy': RUNTIME_POLICY,
              'telemetry_free_required': False,
              'native_checks': sorted(NATIVE_CHECKS | ({GPU_NATIVE_CHECK} if gpu_present else set())
                                      | ({GPU_TRAINING_CHECK} if require_gpu_training else set())) if result else [],
              'gpu_training_required': require_gpu_training,
              'gpu_helper_present': gpu_present,
              'bundle_bytes': sum(entry['bytes'] for entry in files), 'files': files,
              'human_camera_OBS_clean_machine_tests': 'not performed',
              'public_release_approved': False}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path('dist/Carl-Cam.dist'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--require-gpu-training', action='store_true',
                        help='Require actual CUDA fitting and portable CPU model reload on this test machine')
    args = parser.parse_args()
    try:
        result = verify(args.bundle, args.report, require_gpu_training=args.require_gpu_training)
        print(f"Verified standalone native loading; {result['bundle_bytes']} installed file bytes.")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(1, str(error) + '\n')
