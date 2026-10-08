"""Installed GPU check cleanup and result gates using inert engine fixtures."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from carl_cam import gpu_process, selftest, vision
from carl_cam.dsp import signal_stages
from carl_cam.themes import regions_from_masks


def engines(monkeypatch, fault=None):
    closed = []

    def masks():
        values = np.zeros((360, 640, 6), np.float32)
        values[:, :, 0] = 1
        values.flags.writeable = False
        return values

    class Cpu:
        def process(self, packet):
            return SimpleNamespace(confidence_masks=masks())

        def close(self):
            closed.append('cpu')

    class Gpu:
        frames = object()
        process_handle = SimpleNamespace(poll=lambda: 0)

        def __init__(self, launch):
            assert launch == 'fixture'

        def initialize(self, cancel):
            if fault == 'initialization':
                raise ValueError('fixture initialization failure')

        def process(self, packet):
            values = masks()
            working = cv2.resize(vision.mirrored_canvas(packet), (640, 360), interpolation=cv2.INTER_AREA)
            stages = signal_stages(working)
            for field in ('luminance', 'denoised', 'edge_magnitude'):
                getattr(stages, field).flags.writeable = False
            if fault == 'writable':
                stages.luminance.flags.writeable = True
            return SimpleNamespace(frame_id=packet.frame_id + (fault == 'identity'),
                                   capture_time_ns=packet.capture_time_ns, confidence_masks=values,
                                   prepared_signals=stages, prepared_regions=regions_from_masks(values, (360, 640)))

        def close(self):
            self.frames = None
            closed.append('gpu')

    monkeypatch.setattr(gpu_process, 'runtime_launch', lambda: 'fixture')
    monkeypatch.setattr(gpu_process, 'GpuProcessRegionEngine', Gpu)
    monkeypatch.setattr(vision, 'RegionEngine', Cpu)
    return closed


def test_installed_gpu_check_requires_all_fixtures_and_finite_cleanup(monkeypatch):
    closed = engines(monkeypatch)
    result = selftest.check_installed_gpu()
    assert set(result['fixtures']) == {'black', 'white', 'noise'}
    assert result['child_exited'] is result['shared_buffer_closed'] is True
    assert result['physical_camera_used'] is result['live_fps_gate_passed'] is False
    assert closed == ['cpu', 'gpu']


@pytest.mark.parametrize('fault', ['identity', 'writable', 'initialization'])
def test_installed_gpu_failure_always_closes_owned_engines(monkeypatch, fault):
    closed = engines(monkeypatch, fault)
    with pytest.raises((AssertionError, ValueError)):
        selftest.check_installed_gpu()
    assert closed == (['gpu'] if fault == 'initialization' else ['cpu', 'gpu'])


def test_missing_installed_launcher_is_not_a_gpu_pass(monkeypatch):
    monkeypatch.setattr(gpu_process, 'runtime_launch', lambda: None)
    with pytest.raises(ValueError, match='unavailable'):
        selftest.check_installed_gpu()
