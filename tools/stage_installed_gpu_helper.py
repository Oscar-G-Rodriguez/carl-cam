"""Copy the native-tested GPU helper into the main private app bundle.

Uses the tested closed manifest; conflicts and links stop rather than overwrite.
No helper execution, camera, Windows policy change or public approval occurs.
"""

import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from carl_cam.gpu_bundle import HELPER_DIRECTORY, STAGING_SHA256, verified_helper

NATIVE_EVIDENCE = 'gpu-process-check-e5699b3750e64b19996c8377e5c8c3d6/evidence.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    source = ROOT / 'deployment/gpu-helper/Carl-Cam-GPU.dist'
    target = ROOT / 'deployment/Carl-Cam.dist' / HELPER_DIRECTORY
    executable, _ = verified_helper(source)
    evidence = json.loads((ROOT / 'build' / NATIVE_EVIDENCE).read_text(encoding='utf-8'))
    expected = {'black', 'white', 'gradient', 'checker', 'noise'}
    if (evidence.get('passed') is not True or evidence.get('native_helper_used') is not True
            or evidence.get('native_helper_sha256') != digest(executable)
            or evidence.get('staging_record_sha256') != STAGING_SHA256
            or evidence.get('child_exited') is not True
            or evidence.get('shared_buffer_closed') is not True
            or set(evidence.get('fixtures', {})) != expected
            or not all(item.get('passed') is True for item in evidence['fixtures'].values())):
        raise ValueError('Native GPU helper comparison is not complete')
    if not (target.parent / 'Carl-Cam.exe').is_file():
        raise ValueError('Compile the main application before GPU staging')
    for path in (target, *target.parents):
        if path.is_symlink() or path.is_junction():
            raise ValueError('Installed GPU destination must not use links')
    for original in source.rglob('*'):
        if not original.is_file():
            continue
        destination = target / original.relative_to(source)
        if any(path.is_symlink() or path.is_junction() for path in (destination, *destination.parents)):
            raise ValueError('Installed GPU file must not use links')
        if destination.exists():
            if not destination.is_file() or digest(destination) != digest(original):
                raise ValueError('Existing installed GPU file differs')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
    verified_helper(target)
    output = ROOT / 'build' / ('installed-gpu-staging-' + uuid4().hex)
    output.mkdir(exist_ok=False)
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'native_evidence': NATIVE_EVIDENCE, 'native_evidence_sha256': digest(ROOT / 'build' / NATIVE_EVIDENCE),
              'staging_record_sha256': STAGING_SHA256, 'closed_payload_verified': True,
              'main_application_gpu_executed': False, 'installed_application_verified': False,
              'public_release_approved': False}
    (output / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Installed GPU payload evidence: ' + str(output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
