"""Numerical live-test observer is testable without opening a physical camera."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    'carlcam_live_probe', Path(__file__).resolve().parents[1] / 'tools/check_live_pipeline.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_live_probe_deduplicates_and_retains_only_measurements():
    samples = PROBE.TimingSamples()
    rendered = SimpleNamespace(frame_id=4, render_ms=10., capture_time_ns=1_000_000,
                               render_time_ns=21_000_000, preview_rgb='private_pixels')
    hands = SimpleNamespace(frame_id=4, inference_ms=15., landmarks='private_landmarks')
    for _ in range(3):
        samples.consume(rendered, hands, None)
    samples.consume(SimpleNamespace(frame_id=-1), None, None)
    report = samples.evidence(2.)
    assert report['observed_distinct_render_fps'] == .5
    assert report['stage_ms']['capture_to_render']['median_ms'] == 20.
    assert report['stage_ms']['hands']['samples'] == 1
    assert report['stage_ms']['regions']['samples'] == 0
    assert 'private' not in str(vars(samples))


def test_live_probe_counts_stale_completion_without_claiming_preview_time():
    samples = PROBE.TimingSamples()
    samples.consume(SimpleNamespace(frame_id=1, render_ms=10., capture_time_ns=0,
                                     render_time_ns=251_000_000), None, None)
    assert samples.evidence(1.)['stale_render_completions'] == 1


def test_live_probe_requires_consent_before_calling_any_camera_code(monkeypatch):
    monkeypatch.setattr('sys.argv', ['check_live_pipeline.py'])
    monkeypatch.setattr(PROBE, 'run', lambda _: pytest.fail('Camera check called without opt-in'))
    with pytest.raises(SystemExit) as stopped:
        PROBE.main()
    assert stopped.value.code == 2
