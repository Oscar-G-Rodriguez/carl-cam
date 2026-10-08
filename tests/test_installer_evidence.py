"""The private installer must pin all actual bundle checks and unchanged bytes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
SPEC = importlib.util.spec_from_file_location('carlcam_installed_check', TOOLS / 'check_installed_app.py')
installer_tool = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(TOOLS))
try:
    SPEC.loader.exec_module(installer_tool)
finally:
    sys.path.pop(0)


def fixture(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    executable = bundle / 'Carl-Cam.exe'
    executable.write_bytes(b'fixture; not executable')
    runtime = bundle / 'mediapipe/tasks/c/libmediapipe.dll'
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b'fixture; not an SDK')
    monkeypatch.setattr(installer_tool, 'APPROVED_VISION_RUNTIMES',
                        frozenset({installer_tool.digest(runtime)}))
    report = tmp_path / 'manifest.json'
    report.write_text(json.dumps({'selftest_passed': True, 'native_checks': sorted(installer_tool.NATIVE_CHECKS),
        'vision_runtime_privacy_verified': True,
        'vision_runtime_policy': installer_tool.RUNTIME_POLICY, 'telemetry_free_required': False,
        'files': [{'path': path.relative_to(bundle).as_posix(), 'bytes': path.stat().st_size,
                   'sha256': installer_tool.digest(path)} for path in (executable, runtime)]}), encoding='utf-8')
    native = report.with_name('bundled-selftest.json')
    native.write_text(json.dumps({'passed': True, 'camera_opened': False, 'virtual_output_started': False,
        'checks': dict.fromkeys(installer_tool.NATIVE_CHECKS, True)}), encoding='utf-8')
    return bundle, report, native


def test_installer_accepts_complete_exact_evidence(tmp_path, monkeypatch):
    bundle, report, _ = fixture(tmp_path, monkeypatch)
    assert installer_tool.validate_gate(bundle, report)['selftest_passed'] is True


@pytest.mark.parametrize('gpu_training', [True, False, 1, None])
def test_installed_gpu_training_requires_actual_boolean_result(tmp_path, monkeypatch, gpu_training):
    bundle, report, native = fixture(tmp_path, monkeypatch)
    data = json.loads(report.read_text())
    data['gpu_training_required'] = True
    data['native_checks'].append(installer_tool.GPU_TRAINING_CHECK)
    report.write_text(json.dumps(data))
    data = json.loads(native.read_text())
    if gpu_training is not None:
        data['checks'][installer_tool.GPU_TRAINING_CHECK] = gpu_training
    native.write_text(json.dumps(data))
    if gpu_training is True:
        assert installer_tool.validate_gate(bundle, report)['gpu_training_required'] is True
    else:
        with pytest.raises(ValueError, match='Complete native'):
            installer_tool.validate_gate(bundle, report)


@pytest.mark.parametrize('gpu_check', [True, False, 1, None])
def test_gpu_installation_requires_an_actual_gpu_result(tmp_path, monkeypatch, gpu_check):
    bundle, report, native = fixture(tmp_path, monkeypatch)
    (bundle / installer_tool.HELPER_DIRECTORY).mkdir()
    monkeypatch.setattr(installer_tool, 'verified_helper', lambda path: None)
    data = json.loads(report.read_text())
    data['native_checks'].append(installer_tool.GPU_NATIVE_CHECK)
    report.write_text(json.dumps(data))
    data = json.loads(native.read_text())
    if gpu_check is not None:
        data['checks'][installer_tool.GPU_NATIVE_CHECK] = gpu_check
    native.write_text(json.dumps(data))
    if gpu_check is True:
        assert installer_tool.validate_gate(bundle, report)['selftest_passed'] is True
    else:
        with pytest.raises(ValueError, match='Complete native'):
            installer_tool.validate_gate(bundle, report)


@pytest.mark.parametrize('mutation', ['change', 'add', 'delete', 'duplicate'])
def test_installer_refuses_changed_bundle_or_duplicate_inventory(tmp_path, monkeypatch, mutation):
    bundle, report, _ = fixture(tmp_path, monkeypatch)
    if mutation == 'change':
        (bundle / 'Carl-Cam.exe').write_bytes(b'changed')
    elif mutation == 'add':
        (bundle / 'unverified.dll').touch()
    elif mutation == 'delete':
        (bundle / 'Carl-Cam.exe').unlink()
    else:
        data = json.loads(report.read_text())
        data['files'] *= 2
        report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='changed'):
        installer_tool.validate_gate(bundle, report)


@pytest.mark.parametrize('mutation', ['missing_check', 'integer_check', 'camera', 'output', 'summary'])
def test_installer_rejects_incomplete_or_unsafe_native_evidence(tmp_path, monkeypatch, mutation):
    bundle, report, native = fixture(tmp_path, monkeypatch)
    data = json.loads(native.read_text())
    if mutation == 'missing_check':
        data['checks'].pop(next(iter(data['checks'])))
    elif mutation == 'integer_check':
        data['checks'][next(iter(data['checks']))] = 1
    elif mutation == 'camera':
        data['camera_opened'] = True
    elif mutation == 'output':
        data['virtual_output_started'] = True
    else:
        data['passed'] = False
    native.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Complete native'):
        installer_tool.validate_gate(bundle, report)


@pytest.mark.parametrize('mutation', ['missing_policy', 'wrong_policy', 'integer_flag', 'unapproved_hash', 'missing_runtime'])
def test_installer_rejects_legacy_or_unreviewed_privacy_evidence(tmp_path, monkeypatch, mutation):
    bundle, report, _ = fixture(tmp_path, monkeypatch)
    data = json.loads(report.read_text())
    if mutation == 'missing_policy':
        data.pop('vision_runtime_policy')
    elif mutation == 'wrong_policy':
        data['vision_runtime_policy'] = 'old_no_telemetry_requirement'
    elif mutation == 'integer_flag':
        data['telemetry_free_required'] = 0
    elif mutation == 'unapproved_hash':
        monkeypatch.setattr(installer_tool, 'APPROVED_VISION_RUNTIMES', frozenset())
    else:
        data['files'] = [item for item in data['files'] if item['path'] == 'Carl-Cam.exe']
        (bundle / 'mediapipe/tasks/c/libmediapipe.dll').unlink()
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='policy|supported vision runtime'):
        installer_tool.validate_gate(bundle, report)


def test_installer_accepts_disclosed_metrics_without_claiming_privacy(tmp_path, monkeypatch):
    bundle, report, _ = fixture(tmp_path, monkeypatch)
    data = json.loads(report.read_text())
    data['vision_runtime_privacy_verified'] = False
    checksum = installer_tool.digest(bundle / 'mediapipe/tasks/c/libmediapipe.dll')
    monkeypatch.setattr(installer_tool, 'APPROVED_VISION_RUNTIMES', frozenset())
    monkeypatch.setattr(installer_tool, 'SUPPORTED_VISION_RUNTIMES', frozenset({checksum}))
    report.write_text(json.dumps(data))
    assert installer_tool.validate_gate(bundle, report)['vision_runtime_privacy_verified'] is False


def test_policy_block_has_a_safe_code_without_exception_text_or_paths():
    error = OSError('sensitive message and local path must not enter evidence')
    error.winerror = 4551
    assert installer_tool.failure_evidence(error) == {
        'error_type': 'OSError', 'windows_error_code': 4551, 'application_control_blocked': True}
    assert installer_tool.failure_evidence(ValueError('sensitive detail')) == {'error_type': 'ValueError'}
