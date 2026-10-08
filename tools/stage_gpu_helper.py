"""Stage exact original GPU wheel payloads beside the compiled private helper.

The compiled metadata resolver inventories these unchanged files before loading
CUDA. This command does not launch the helper, load GPU libraries or grant public
redistribution approval. Conflicting files are preserved and stop staging.
"""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from carl_cam import gpu_dependencies, gpu_regions
from carl_cam.gpu_process import DEVELOPMENT_CANDIDATE

BUNDLE = ROOT / 'deployment/gpu-helper/Carl-Cam-GPU.dist'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_exact(source, target):
    for path in (source, target):
        if any(part.is_symlink() or part.is_junction() for part in (path, *path.parents)):
            raise ValueError('GPU staging paths must not use links')
    if not source.is_file():
        raise ValueError('GPU staging source is missing')
    expected = digest(source)
    if target.exists():
        if not target.is_file() or digest(target) != expected:
            raise ValueError('Existing GPU bundle file differs: ' + target.name)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if digest(target) != expected:
        raise ValueError('GPU payload copy failed verification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-record', type=Path, required=True)
    args = parser.parse_args()
    record_path = args.build_record.absolute()
    if (not record_path.is_relative_to(ROOT / 'build') or record_path.name != 'inputs.json'
            or not record_path.parent.name.startswith('gpu-native-input-')):
        raise ValueError('Select the exact GPU compilation receipt')
    build = json.loads(record_path.read_text(encoding='utf-8'))
    executable = BUNDLE / 'Carl-Cam-GPU.exe'
    if (build.get('compiler_returncode') != 0
            or build.get('source_matches_initial_inventory') is not True
            or not executable.is_file()
            or digest(executable) != build.get('executable_sha256')):
        raise ValueError('Actual GPU compilation has not passed')
    # Inspection is static: imports no ORT or NVIDIA DLLs.
    identity = gpu_regions.runtime_inventory()
    candidate = ROOT / 'build' / DEVELOPMENT_CANDIDATE
    parsed = gpu_regions.inspect_candidate(candidate)
    if (parsed.evidence_sha256, identity) not in gpu_regions.EXPERIMENTAL_GPU_CONFIGURATIONS:
        raise ValueError('Source GPU runtime differs from the experimental pinned inventory')
    nvidia = gpu_dependencies.inspect_nvidia_dependencies()
    for item in nvidia.files:
        copy_exact(item.path, BUNDLE / item.relative)
    distribution = importlib.metadata.distribution('onnxruntime-gpu')
    for relative in distribution.files or ():
        name = PurePosixPath(str(relative))
        if name.is_absolute() or '..' in name.parts or '\\' in str(relative):
            # Original wheel console launchers are not part of this helper.
            continue
        metadata = name.parts[0] == 'onnxruntime_gpu-1.30.0.dist-info'
        payload = (name.parts[0] == 'onnxruntime' and
                   (name.suffix.lower() in ('.py', '.dll', '.pyd')
                    or name.name in ('LICENSE', 'Privacy.md', 'ThirdPartyNotices.txt')))
        if metadata or payload:
            copy_exact(Path(distribution.locate_file(relative)), BUNDLE / str(relative))
    for name in ('evidence.json', 'selfie_multiclass_fp32.onnx'):
        copy_exact(candidate / name, BUNDLE / 'candidate' / name)
    # Resolve the staged wheel metadata exactly as the compiled helper will.
    original_file = gpu_dependencies.__file__
    try:
        gpu_dependencies.__compiled__ = True
        gpu_dependencies.__file__ = str(BUNDLE / 'carl_cam/gpu_dependencies.py')
        staged_identity = gpu_regions.runtime_inventory()
    finally:
        gpu_dependencies.__file__ = original_file
        del gpu_dependencies.__compiled__
    if staged_identity != identity:
        raise ValueError('Staged GPU inventory differs from the original wheels')
    items = []
    for path in sorted(BUNDLE.rglob('*')):
        if any(part.is_symlink() or part.is_junction() for part in (path, *path.parents)):
            raise ValueError('GPU bundle contains a linked entry')
        if path.is_file() and path.name != 'gpu-helper-staging.json':
            items.append({'path': path.relative_to(BUNDLE).as_posix(),
                          'bytes': path.stat().st_size, 'sha256': digest(path)})
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'build_record': str(record_path.relative_to(ROOT)),
              'build_record_sha256': digest(record_path),
              'executable_sha256': digest(executable), 'runtime_inventory': identity,
              'candidate_evidence_sha256': parsed.evidence_sha256,
              'files': items, 'total_bytes': sum(item['bytes'] for item in items),
              'runtime_payload_verified': True, 'native_loading_verified': False,
              'public_redistribution_approved': False}
    (BUNDLE / 'gpu-helper-staging.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'staged_files': len(items), 'total_bytes': report['total_bytes'],
                      'runtime_inventory': identity, 'native_loading_verified': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
