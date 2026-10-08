"""Identify the exact same-machine GPU helper that passed native comparisons.

The parent does not search PATH, load CUDA, or accept a user-selected executable.
Only a fixed sibling directory with the pinned closed inventory can be launched.
These pins establish tested bytes, not public redistribution or camera quality.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

STAGING_SHA256 = 'fc576b38018b5450f53163b54812b0bc3bab57fc0fac31710602f961da65dfd9'
EXECUTABLE_SHA256 = '8037e14f2d77fafba9b81391ca6296cdb515fceed9088a2b48249c631c6c8cda'
RUNTIME_INVENTORY = 'a838900a566bba0124e38c086a63aa59e3a2a292e0b67d642ab6e5f8c3507d02'
CANDIDATE_SHA256 = 'ab07f9774b2d1e64106886be7f3aed1ebfb9d4440a29d4d67f1897ef62e126f8'
HELPER_DIRECTORY = 'gpu-helper'


def _ordinary(path: Path) -> bool:
    return not any(item.is_symlink() or item.is_junction() for item in (path, *path.parents))


def _digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_helper(bundle: Path) -> tuple[Path, Path]:
    """Check all helper files before process creation; never repair a mismatch."""
    bundle = bundle.absolute()
    manifest_path = bundle / 'gpu-helper-staging.json'
    if (not _ordinary(bundle) or not bundle.is_dir() or not _ordinary(manifest_path)
            or not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= 1024**2
            or _digest(manifest_path) != STAGING_SHA256):
        raise ValueError('GPU helper inventory is unavailable or changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1
            or manifest.get('runtime_payload_verified') is not True
            or manifest.get('executable_sha256') != EXECUTABLE_SHA256
            or manifest.get('runtime_inventory') != RUNTIME_INVENTORY
            or manifest.get('candidate_evidence_sha256') != CANDIDATE_SHA256
            or not isinstance(manifest.get('files'), list)
            or not 1 <= len(manifest['files']) <= 4096):
        raise ValueError('GPU helper inventory is unsupported')
    recorded = set()
    total = 0
    for entry in manifest['files']:
        relative = entry.get('path')
        size = entry.get('bytes')
        checksum = entry.get('sha256')
        if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
                or any(part in ('', '.', '..') for part in relative.split('/'))
                or PurePosixPath(relative).is_absolute()
                or relative == manifest_path.name or relative.casefold() in recorded
                or type(size) is not int or not 0 <= size <= 2 * 1024**3
                or not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum)):
            raise ValueError('GPU helper file record is invalid')
        path = bundle / relative
        if (not _ordinary(path) or not path.is_file() or path.stat().st_size != size
                or _digest(path) != checksum):
            raise ValueError('GPU helper payload is missing or changed')
        recorded.add(relative.casefold())
        total += size
    actual = set()
    for path in bundle.rglob('*'):
        if not _ordinary(path):
            raise ValueError('GPU helper contains a linked entry')
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(bundle).as_posix().casefold())
    if (actual != recorded or type(manifest.get('total_bytes')) is not int
            or total != manifest['total_bytes']):
        raise ValueError('GPU helper contains unexpected or missing files')
    executable = bundle / 'Carl-Cam-GPU.exe'
    candidate = bundle / 'candidate'
    if (not executable.is_file() or _digest(executable) != EXECUTABLE_SHA256
            or not (candidate / 'evidence.json').is_file()
            or _digest(candidate / 'evidence.json') != CANDIDATE_SHA256
            or not (candidate / 'selfie_multiclass_fp32.onnx').is_file()):
        raise ValueError('GPU helper entry point or model is unsupported')
    return executable, candidate
