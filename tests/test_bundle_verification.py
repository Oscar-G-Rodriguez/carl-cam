"""Packaging evidence must come from all actual native checks, not a summary flag."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    'carlcam_verify_bundle', Path(__file__).resolve().parents[1] / 'tools/verify_bundle.py')
verify_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_tool)


def _fixture(tmp_path):
    bundle = tmp_path / 'bundle'
    for name in ('Carl-Cam.exe', *verify_tool.REQUIRED_FILES):
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'fixture only; never an executable')
    return bundle, tmp_path / 'evidence/manifest.json'


def _runner(monkeypatch, *, checks=None):
    # A fixture hash is test-only authorization; no installed DLL is approved.
    monkeypatch.setattr(verify_tool, 'APPROVED_VISION_RUNTIMES', frozenset({
        hashlib.sha256(b'fixture only; never an executable').hexdigest()}))
    def fake_run(command, **kwargs):
        assert kwargs['cwd'].name == 'bundle' and kwargs['timeout'] == 120
        report = {'schema_version': 1, 'passed': True, 'camera_opened': False,
                  'virtual_output_started': False,
                  'checks': checks if checks is not None else dict.fromkeys(verify_tool.NATIVE_CHECKS, True)}
        Path(command[-1]).write_text(json.dumps(report), encoding='utf-8')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(verify_tool.subprocess, 'run', fake_run)


def test_bundle_check_requires_all_native_checks_and_retains_file_hashes(tmp_path, monkeypatch):
    bundle, report = _fixture(tmp_path)
    _runner(monkeypatch)
    result = verify_tool.verify(bundle, report)
    assert result['selftest_passed'] is True
    assert result['vision_runtime_privacy_verified'] is True
    assert len(result['native_checks']) == 8
    assert len(result['files']) == 1 + len(verify_tool.REQUIRED_FILES)
    assert all(len(item['sha256']) == 64 for item in result['files'])
    assert result['public_release_approved'] is False
    with pytest.raises(ValueError, match='new report'):
        verify_tool.verify(bundle, report)


@pytest.mark.parametrize('gpu_training', [True, False, 1, None])
def test_gpu_training_gate_requires_actual_boolean_native_result(tmp_path, monkeypatch, gpu_training):
    bundle, report = _fixture(tmp_path)
    checks = dict.fromkeys(verify_tool.NATIVE_CHECKS, True)
    if gpu_training is not None:
        checks[verify_tool.GPU_TRAINING_CHECK] = gpu_training
    _runner(monkeypatch, checks=checks)
    if gpu_training is True:
        result = verify_tool.verify(bundle, report, require_gpu_training=True)
        assert result['gpu_training_required'] is True
        assert verify_tool.GPU_TRAINING_CHECK in result['native_checks']
    else:
        with pytest.raises(ValueError, match='invalid or failing'):
            verify_tool.verify(bundle, report, require_gpu_training=True)


def test_gpu_training_cannot_be_verified_without_native_execution(tmp_path):
    bundle, report = _fixture(tmp_path)
    with pytest.raises(ValueError, match='native execution'):
        verify_tool.verify(bundle, report, selftest=False, require_gpu_training=True)


def test_unknown_vision_runtime_is_rejected_before_native_launch(tmp_path, monkeypatch):
    bundle, report = _fixture(tmp_path)
    def unexpected_run(*args, **kwargs):
        pytest.fail('Privacy hold must prevent native inference before launch')
    monkeypatch.setattr(verify_tool.subprocess, 'run', unexpected_run)
    with pytest.raises(ValueError, match='integrity hold'):
        verify_tool.verify(bundle, report)
    assert not report.exists()
    assert not report.with_name('bundled-selftest.json').exists()


def test_supported_metrics_bundle_keeps_privacy_unverified(tmp_path, monkeypatch):
    bundle, report = _fixture(tmp_path)
    _runner(monkeypatch)
    checksum = hashlib.sha256(b'fixture only; never an executable').hexdigest()
    monkeypatch.setattr(verify_tool, 'APPROVED_VISION_RUNTIMES', frozenset())
    monkeypatch.setattr(verify_tool, 'SUPPORTED_VISION_RUNTIMES', frozenset({checksum}))
    result = verify_tool.verify(bundle, report)
    assert result['selftest_passed'] is True
    assert result['vision_runtime_privacy_verified'] is False
    assert result['telemetry_free_required'] is False
    assert result['vision_runtime_policy'] == verify_tool.RUNTIME_POLICY


def test_inventory_without_inference_cannot_certify_privacy(tmp_path):
    bundle, report = _fixture(tmp_path)
    result = verify_tool.verify(bundle, report, selftest=False)
    assert result['vision_runtime_privacy_verified'] is False
    assert result['selftest_passed'] is False


@pytest.mark.parametrize('checks', [{}, {'bundled_vision_models': True},
    dict.fromkeys(verify_tool.NATIVE_CHECKS, 1)])
def test_summary_pass_cannot_hide_missing_or_invalid_native_checks(tmp_path, monkeypatch, checks):
    bundle, report = _fixture(tmp_path)
    _runner(monkeypatch, checks=checks)
    with pytest.raises(ValueError, match='invalid or failing'):
        verify_tool.verify(bundle, report)
    assert not report.exists()


def test_packaging_rejects_personal_data_and_in_bundle_evidence(tmp_path, monkeypatch):
    bundle, report = _fixture(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        verify_tool.verify(bundle, bundle / 'manifest.json')
    def unexpected_run(*args, **kwargs):
        pytest.fail('An invalid bundle must be rejected before running an executable')
    monkeypatch.setattr(verify_tool.subprocess, 'run', unexpected_run)
    personal = bundle / 'datasets/samples.json'
    personal.parent.mkdir()
    personal.touch()
    with pytest.raises(ValueError, match='Personal recordings'):
        verify_tool.verify(bundle, report)
    assert not report.exists()


def test_offline_help_and_licenses_are_required_before_native_launch(tmp_path, monkeypatch):
    bundle, report = _fixture(tmp_path)
    (bundle / 'help/Quick Start.html').unlink()
    def unexpected_run(*args, **kwargs):
        pytest.fail('A bundle with missing offline instructions must not be launched')
    monkeypatch.setattr(verify_tool.subprocess, 'run', unexpected_run)
    with pytest.raises(ValueError, match='required offline files'):
        verify_tool.verify(bundle, report)


def test_only_exact_published_torch_dataset_headers_are_allowed(tmp_path, monkeypatch):
    bundle, report = _fixture(tmp_path)
    original = tmp_path / 'vendor/base.h'
    original.parent.mkdir()
    original.write_bytes(b'published header fixture; no personal samples')
    header = bundle / verify_tool.TORCH_HEADER_DIRECTORY / 'base.h'
    header.parent.mkdir(parents=True)
    header.write_bytes(original.read_bytes())
    monkeypatch.setattr(verify_tool.importlib.metadata, 'distribution',
        lambda name: SimpleNamespace(locate_file=lambda relative: original))
    _runner(monkeypatch)
    assert verify_tool.verify(bundle, report)['selftest_passed']
    header.write_bytes(b'changed or private content')
    with pytest.raises(ValueError, match='does not match'):
        verify_tool.verify(bundle, tmp_path / 'fresh/manifest.json')
    header.unlink()
    (header.parent / 'private-samples.json').write_bytes(b'not a published header')
    with pytest.raises(ValueError, match='Personal recordings'):
        verify_tool.verify(bundle, tmp_path / 'another/manifest.json')
