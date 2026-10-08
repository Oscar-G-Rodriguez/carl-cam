import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location('carlcam_bundle_check', ROOT / 'tools/verify_bundle.py')
CHECK = module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_bundle_cannot_skip_missing_native_assets(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'Carl-Cam.exe').write_bytes(b'non-executable test fixture')
    with pytest.raises(ValueError, match='missing required offline'):
        CHECK.verify(bundle, tmp_path / 'report.json', selftest=False)


def make_bundle(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    for name in ('Carl-Cam.exe', *CHECK.REQUIRED_FILES):
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'non-executable test fixture')
    return bundle


def test_manifest_does_not_claim_skipped_selftest_passed(tmp_path):
    bundle = make_bundle(tmp_path)
    report = CHECK.verify(bundle, tmp_path / 'report.json', selftest=False)
    assert report['selftest_passed'] is False
    assert report['public_release_approved'] is False
    assert report['human_camera_OBS_clean_machine_tests'] == 'not performed'
    assert len(report['files']) == 1 + len(CHECK.REQUIRED_FILES)
    assert all(len(entry['sha256']) == 64 for entry in report['files'])


def test_bundle_rejects_numerical_recordings(tmp_path):
    bundle = make_bundle(tmp_path)
    folder = bundle / 'datasets'
    folder.mkdir()
    (folder / 'private.json').write_text('{}')
    with pytest.raises(ValueError, match='Personal recordings'):
        CHECK.verify(bundle, tmp_path / 'report.json', selftest=False)


def test_bundle_check_wont_reuse_old_native_result(tmp_path):
    bundle = make_bundle(tmp_path)
    (tmp_path / 'bundled-selftest.json').write_text(json.dumps({'passed': True}))
    with pytest.raises(ValueError, match='old evidence'):
        CHECK.verify(bundle, tmp_path / 'report.json')


def test_installer_defaults_are_per_user_and_preserve_data():
    script = (ROOT / 'packaging/Carl-Cam.iss').read_text()
    assert 'PrivilegesRequired=lowest' in script
    assert 'Flags: nowait postinstall skipifsilent unchecked' in script
    assert '#define CarlCamInstallDir "{localappdata}\\Programs\\Carl-Cam"' in script
    assert 'DefaultDirName={#CarlCamInstallDir}' in script
    assert '#define CarlCamGroup "Carl-Cam"' in script
    assert '\n[UninstallDelete]' not in script
    assert 'Source: "{#BundleDir}\\*"' in script
    assert 'CloseApplications=no' in script
    assert '#define CarlCamAppId "{{C10BE185-6F59-4D19-9B14-483B2391E3A4}"' in script
    assert 'AppId={#CarlCamAppId}' in script


def test_dynamic_scipy_numpy_compatibility_modules_are_explicitly_packaged():
    import configparser

    config = configparser.ConfigParser()
    config.read(ROOT / 'pysidedeploy.spec')
    assert '--include-package=scipy._external.array_api_compat.numpy' in config['nuitka']['extra_args'].split()
