"""Closed installed-helper inventory checks, without native code execution."""

import hashlib
import json

import pytest

from carl_cam import gpu_bundle, gpu_process


def fixture(tmp_path, monkeypatch):
    bundle = tmp_path / 'gpu-helper'
    contents = {'Carl-Cam-GPU.exe': b'not an executable',
                'candidate/evidence.json': b'{}',
                'candidate/selfie_multiclass_fp32.onnx': b'not a model',
                'runtime.dll': b'not a library'}
    entries = []
    for name, data in contents.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    monkeypatch.setattr(gpu_bundle, 'EXECUTABLE_SHA256', entries[0]['sha256'])
    monkeypatch.setattr(gpu_bundle, 'CANDIDATE_SHA256', entries[1]['sha256'])
    manifest = {'schema_version': 1, 'runtime_payload_verified': True,
                'runtime_inventory': gpu_bundle.RUNTIME_INVENTORY,
                'executable_sha256': gpu_bundle.EXECUTABLE_SHA256,
                'candidate_evidence_sha256': gpu_bundle.CANDIDATE_SHA256,
                'files': entries, 'total_bytes': sum(entry['bytes'] for entry in entries)}
    publish(bundle, manifest, monkeypatch)
    return bundle, manifest


def publish(bundle, manifest, monkeypatch):
    payload = json.dumps(manifest).encode()
    (bundle / 'gpu-helper-staging.json').write_bytes(payload)
    monkeypatch.setattr(gpu_bundle, 'STAGING_SHA256', hashlib.sha256(payload).hexdigest())


def test_complete_inventory_accepts_only_fixed_entry_point(tmp_path, monkeypatch):
    bundle, _ = fixture(tmp_path, monkeypatch)
    assert gpu_bundle.verified_helper(bundle) == (bundle / 'Carl-Cam-GPU.exe', bundle / 'candidate')


@pytest.mark.parametrize('change', ['missing', 'changed', 'extra', 'manifest'])
def test_changed_closed_inventory_cannot_launch(tmp_path, monkeypatch, change):
    bundle, _ = fixture(tmp_path, monkeypatch)
    if change == 'missing':
        (bundle / 'runtime.dll').unlink()
    elif change == 'extra':
        (bundle / 'unexpected.dll').write_bytes(b'changed')
    else:
        (bundle / ('gpu-helper-staging.json' if change == 'manifest' else 'runtime.dll')).write_bytes(b'changed')
    with pytest.raises(ValueError):
        gpu_bundle.verified_helper(bundle)


@pytest.mark.parametrize('path', ['../elsewhere', '/absolute', 'C:/outside', 'runtime.dll:stream',
                                 'candidate\\escape', './runtime.dll', 'candidate//evidence.json'])
def test_manifest_paths_are_strict_relative_files(tmp_path, monkeypatch, path):
    bundle, manifest = fixture(tmp_path, monkeypatch)
    manifest['files'][0]['path'] = path
    publish(bundle, manifest, monkeypatch)
    with pytest.raises(ValueError):
        gpu_bundle.verified_helper(bundle)


@pytest.mark.parametrize('field,value', [('schema_version', True), ('runtime_payload_verified', 1),
                                       ('total_bytes', True), ('runtime_inventory', 'changed')])
def test_manifest_types_and_runtime_identity_are_required(tmp_path, monkeypatch, field, value):
    bundle, manifest = fixture(tmp_path, monkeypatch)
    manifest[field] = value
    publish(bundle, manifest, monkeypatch)
    with pytest.raises(ValueError):
        gpu_bundle.verified_helper(bundle)


def test_duplicate_case_insensitive_record_is_rejected(tmp_path, monkeypatch):
    bundle, manifest = fixture(tmp_path, monkeypatch)
    manifest['files'].append({**manifest['files'][0], 'path': 'carl-cam-gpu.EXE'})
    publish(bundle, manifest, monkeypatch)
    with pytest.raises(ValueError):
        gpu_bundle.verified_helper(bundle)


def test_frozen_launcher_uses_fixed_sibling_not_developer_python(tmp_path, monkeypatch):
    bundle, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(gpu_process, '__compiled__', True, raising=False)
    monkeypatch.setattr(gpu_process, '__file__', str(tmp_path / 'carl_cam/gpu_process.py'))
    monkeypatch.setattr(gpu_process, 'development_launch', lambda: pytest.fail('Frozen app searched developer Python'))
    launch = gpu_process.runtime_launch()
    assert launch.command == (str(bundle / 'Carl-Cam-GPU.exe'),)
    assert launch.candidate == bundle / 'candidate'
    (bundle / 'runtime.dll').write_bytes(b'changed')
    assert gpu_process.runtime_launch() is None


def test_source_launcher_retains_separate_development_environment(monkeypatch):
    monkeypatch.delattr(gpu_process, '__compiled__', raising=False)
    sentinel = object()
    monkeypatch.setattr(gpu_process, 'development_launch', lambda: sentinel)
    assert gpu_process.runtime_launch() is sentinel
