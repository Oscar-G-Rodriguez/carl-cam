import builtins
import hashlib
from time import sleep

import pytest

from carl_cam import vision_privacy
from carl_cam.channels import LatestValue
from carl_cam.comparison_vision import ComparisonEngine
from carl_cam.vision import HandHeadEngine, RegionEngine, VisionWorker


def test_compiled_runtime_resolves_only_its_bundled_dll(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_privacy, '__compiled__', object(), raising=False)
    monkeypatch.setattr(vision_privacy, '__file__', str(tmp_path / 'carl_cam/vision_privacy.py'))
    monkeypatch.setattr(vision_privacy.importlib.metadata, 'distribution',
                        lambda _: pytest.fail('Compiled runtime must not use synthesized metadata paths'))
    assert vision_privacy.runtime_library_path() == tmp_path / 'mediapipe/tasks/c/libmediapipe.dll'


@pytest.fixture(autouse=True)
def no_installed_runtime_in_policy_fixtures(monkeypatch):
    # These tests exercise refusal with fixtures, never the real installed SDK.
    monkeypatch.setattr(vision_privacy, 'SUPPORTED_VISION_RUNTIMES', frozenset())


def test_supported_metrics_runtime_does_not_fabricate_privacy(tmp_path, monkeypatch):
    library = tmp_path / 'libmediapipe.dll'
    library.write_bytes(b'exact supported runtime fixture; never executable')
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    monkeypatch.setattr(vision_privacy, 'SUPPORTED_VISION_RUNTIMES', frozenset({digest}))
    assert digest not in vision_privacy.APPROVED_VISION_RUNTIMES
    assert vision_privacy.require_verified_vision_runtime(library) == digest
    library.write_bytes(b'changed fixture')
    with pytest.raises(vision_privacy.VisionPrivacyError):
        vision_privacy.require_verified_vision_runtime(library)


def test_unknown_library_is_rejected_even_without_telemetry_strings(tmp_path):
    library = tmp_path / 'libmediapipe.dll'
    library.write_bytes(b'unreviewed native library fixture')
    with pytest.raises(vision_privacy.VisionPrivacyError, match='not verified'):
        vision_privacy.require_verified_vision_runtime(library)


def test_only_exact_reviewed_bytes_are_accepted(tmp_path, monkeypatch):
    library = tmp_path / 'libmediapipe.dll'
    library.write_bytes(b'reviewed fixture only; not an SDK')
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    monkeypatch.setattr(vision_privacy, 'APPROVED_VISION_RUNTIMES', frozenset({digest}))
    assert vision_privacy.require_verified_vision_runtime(library) == digest
    library.write_bytes(b'different bytes; same filename')
    with pytest.raises(vision_privacy.VisionPrivacyError):
        vision_privacy.require_verified_vision_runtime(library)


@pytest.mark.parametrize('kind', ['missing', 'empty', 'directory'])
def test_missing_or_invalid_runtime_fails_closed(tmp_path, kind):
    library = tmp_path / 'libmediapipe.dll'
    if kind == 'empty':
        library.touch()
    elif kind == 'directory':
        library.mkdir()
    with pytest.raises(vision_privacy.VisionPrivacyError):
        vision_privacy.require_verified_vision_runtime(library)


@pytest.mark.parametrize('engine', [HandHeadEngine, RegionEngine, ComparisonEngine])
def test_all_engine_paths_check_privacy_before_importing_sdk(monkeypatch, engine):
    original = builtins.__import__
    def no_sdk_import(name, *args, **kwargs):
        assert not name.startswith('mediapipe'), 'An unverified SDK must not be imported'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_sdk_import)
    with pytest.raises(vision_privacy.VisionPrivacyError):
        engine(None) if engine is ComparisonEngine else engine()


def test_worker_reports_privacy_hold_and_stops_without_an_sdk():
    worker = VisionWorker(LatestValue())
    worker.start()
    try:
        for _ in range(100):
            if (worker.status.read() or '').startswith('Vision unavailable'):
                break
            sleep(.005)
        assert worker.status.read() == 'Vision unavailable: ' + vision_privacy.PRIVACY_HOLD
        assert worker.results.read() is None and worker.completed.read() is None
    finally:
        assert worker.stop()
