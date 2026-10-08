"""Isolated install, native check and uninstall of a separately identified build.

Never install/uninstall the normal Carl-Cam registration during this check.
The program, compiler output and controlled data sentinel live in a fresh
repository-local directory. This is not a clean-machine or physical-camera test.
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from verify_bundle import (
    APPROVED_VISION_RUNTIMES,
    GPU_NATIVE_CHECK,
    GPU_TRAINING_CHECK,
    HELPER_DIRECTORY,
    NATIVE_CHECKS,
    RUNTIME_POLICY,
    SUPPORTED_VISION_RUNTIMES,
    _normal_tree,
    verified_helper,
)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unlinked(path):
    return _normal_tree(path) and all(not item.is_junction() for item in (path, *path.parents))


def validate_gate(bundle, report_path):
    """Pin this installation check to the exact previously tested bundle."""
    if not unlinked(bundle) or not unlinked(report_path) or not unlinked(report_path.with_name('bundled-selftest.json')):
        raise ValueError('Installer evidence must not use symbolic links')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    native = json.loads(report_path.with_name('bundled-selftest.json').read_text(encoding='utf-8'))
    gpu_present = (bundle / HELPER_DIRECTORY).exists()
    required = NATIVE_CHECKS | ({GPU_NATIVE_CHECK} if gpu_present else set())
    if report.get('gpu_training_required') is True:
        required |= {GPU_TRAINING_CHECK}
    elif report.get('gpu_training_required', False) is not False:
        raise ValueError('Invalid GPU training requirement')
    if gpu_present:
        verified_helper(bundle / HELPER_DIRECTORY)
    if (report.get('selftest_passed') is not True or set(report.get('native_checks', [])) != required
            or native.get('passed') is not True or native.get('camera_opened') is not False
            or native.get('virtual_output_started') is not False
            or any(native.get('checks', {}).get(name) is not True for name in required)):
        raise ValueError('Complete native bundle evidence is required before installation')
    entries = report.get('files', [])
    names = {item['path'] for item in entries}
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob('*') if path.is_file()}
    if len(names) != len(entries) or names != actual:
        raise ValueError('Bundle contents changed since native verification')
    runtime = [item for item in entries if item.get('path') == 'mediapipe/tasks/c/libmediapipe.dll']
    if (len(runtime) != 1
            or runtime[0].get('sha256') not in SUPPORTED_VISION_RUNTIMES | APPROVED_VISION_RUNTIMES):
        raise ValueError('Exact supported vision runtime evidence is required before installation')
    if (report.get('vision_runtime_policy') != RUNTIME_POLICY
            or report.get('telemetry_free_required') is not False):
        raise ValueError('Current disclosed-metrics policy evidence is required before installation')
    for item in entries:
        path = bundle / item['path']
        if (not unlinked(path) or not path.resolve().is_relative_to(bundle.resolve())
                or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']):
            raise ValueError('Bundle bytes changed since native verification')
    return report


def run_logged(command, *, cwd, log_path, timeout, env=None):
    with log_path.open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def failure_evidence(error):
    """Keep actionable policy codes without exception messages or local paths."""
    result = {'error_type': type(error).__name__}
    if getattr(error, 'winerror', None) == 4551:
        result.update(windows_error_code=4551, application_control_blocked=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-report', type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = root / 'deployment/Carl-Cam.dist'
    bundle_report = validate_gate(bundle, arguments.bundle_report)
    required_checks = set(bundle_report['native_checks'])
    identity = str(uuid4()).upper()
    evidence = root / 'build' / ('installed-check-' + identity.lower())
    if not unlinked(evidence) or evidence.exists():
        raise ValueError('Install check needs a fresh unlinked directory')
    evidence.mkdir(parents=True, exist_ok=False)
    program, setup = evidence / 'program', evidence / 'setup'
    group = 'Carl-Cam private check ' + identity
    report = {'schema_version': 1, 'started_utc': datetime.now(UTC).isoformat(),
              'private_app_id': identity, 'program_path': str(program),
              'bundle_manifest_sha256': digest(arguments.bundle_report),
              'camera_opened': False, 'virtual_output_started': False,
              'vision_runtime_privacy_verified': bundle_report.get('vision_runtime_privacy_verified') is True,
              'vision_runtime_policy': RUNTIME_POLICY,
              'telemetry_free_required': False,
              'gpu_helper_present': (bundle / HELPER_DIRECTORY).exists(),
              'gpu_training_required': bundle_report.get('gpu_training_required') is True,
              'clean_machine_verified': False, 'public_release_approved': False,
              'checks': {}, 'cleanup_completed': False, 'passed': False}
    result_path = evidence / 'result.json'
    def save():
        result_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    print(f'Isolated installer evidence: {evidence}', flush=True)
    try:
        import winreg
        key = f'Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{{{identity}}}_is1'
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
                raise ValueError('Private installer identity unexpectedly already exists')
        except FileNotFoundError:
            pass
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders') as folders:
            programs_path = Path(winreg.QueryValueEx(folders, 'Programs')[0])
        shortcuts = programs_path / group
        if not programs_path.is_absolute() or not unlinked(shortcuts) or shortcuts.exists():
            raise ValueError('Private shortcut group needs a fresh unlinked directory')
        compiler = root / 'build/tools/inno/ISCC.exe'
        run_logged([str(compiler), '--quiet', f'--define=BundleDir={bundle}',
                    '--define=CarlCamAppId={{' + identity + '}',
                    f'--define=CarlCamInstallDir={program}', f'--define=CarlCamGroup={group}',
                    f'--output-dir={setup}', '--output-filename=Carl-Cam-isolated-check',
                    str(root / 'packaging/Carl-Cam.iss')], cwd=root,
                   log_path=evidence / 'compiler.log', timeout=1800)
        private_installer = setup / 'Carl-Cam-isolated-check.exe'
        report['private_installer_sha256'] = digest(private_installer)
        report['checks']['private_identity_installer_compiled'] = True
        save()
        print('Private installer compiled; installing only into its isolated directory.', flush=True)
        sentinel = evidence / 'controlled-userdata/CarlCam/preserve.txt'
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b'Carl-Cam controlled uninstall preservation fixture')
        sentinel_digest = digest(sentinel)
        run_logged([str(private_installer), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                    '/SP-', f'/DIR={program}', f'/GROUP={group}', '/TASKS=',
                    f'/LOG={evidence / "install.log"}'], cwd=evidence,
                   log_path=evidence / 'install-console.log', timeout=240)
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as registration:
            location = winreg.QueryValueEx(registration, 'InstallLocation')[0]
        if Path(location).resolve() != program.resolve():
            raise ValueError('Private registration did not resolve to the isolated program directory')
        installed = program / 'Carl-Cam.exe'
        for item in bundle_report['files']:
            path = program / item['path']
            if (not unlinked(path) or not path.resolve().is_relative_to(program.resolve())
                    or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']):
                raise ValueError('Installed files differ from the verified bundle')
        if not all((shortcuts / name).is_file() for name in ('Carl-Cam.lnk', 'Carl-Cam Quick Start.lnk')):
            raise ValueError('Private Start Menu shortcuts are missing')
        report['checks']['isolated_installation'] = True
        save()
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        installed_report = evidence / 'installed-selftest.json'
        run_logged([str(installed), '--self-test', '--self-test-report', str(installed_report)],
                   cwd=program, log_path=evidence / 'installed-console.log', timeout=120, env=environment)
        native = json.loads(installed_report.read_text(encoding='utf-8'))
        if (native.get('passed') is not True or native.get('camera_opened') is not False
                or native.get('virtual_output_started') is not False
                or any(native.get('checks', {}).get(name) is not True for name in required_checks)):
            raise ValueError('The installed executable failed its native checks')
        report['checks']['installed_native_checks'] = True
        save()
        uninstaller = program / 'unins000.exe'
        if (not unlinked(uninstaller) or not program.resolve().is_relative_to(evidence.resolve())
                or program.parent.resolve() != evidence.resolve() or not uninstaller.is_file()):
            raise ValueError('Uninstall target is not this exact isolated installation')
        print('Installed native checks passed; uninstalling only the private registration.', flush=True)
        run_logged([str(uninstaller), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                    f'/LOG={evidence / "uninstall.log"}'], cwd=evidence,
                   log_path=evidence / 'uninstall-console.log', timeout=240)
        if installed.exists() or digest(sentinel) != sentinel_digest:
            raise ValueError('Uninstall did not remove its program or preserve the controlled data sentinel')
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
                raise ValueError('Private uninstall registration was left behind')
        except FileNotFoundError:
            pass
        if shortcuts.exists():
            raise ValueError('Private Start Menu shortcut group was left behind')
        report['checks']['private_program_removed_controlled_data_preserved'] = True
        report['checks']['private_registration_and_shortcuts_removed'] = True
        report['cleanup_completed'] = True
        report['passed'] = True
        print('Isolated install, all required installed model checks and controlled-data preservation passed.', flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report.update(failure_evidence(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(UTC).isoformat()
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
