"""Same-frame preparation is immutable and preserves the original filter math."""

from types import SimpleNamespace

import numpy as np
import pytest

from carl_cam import render
from carl_cam.dsp import SignalStages, signal_stages
from carl_cam.gpu_ipc import (
    INPUT_SHAPE,
    OUTPUT_SHAPE,
    PROTOCOL_VERSION,
    REGIONS_SHAPE,
    SIGNALS_SHAPE,
)
from carl_cam.gpu_process import GpuProcessError, GpuProcessRegionEngine
from carl_cam.records import FramePacket
from carl_cam.render import SceneState, compose_palette
from carl_cam.themes import regions_from_masks


def client_fixture(monkeypatch):
    engine = GpuProcessRegionEngine(None)
    engine.frames = SimpleNamespace(input=np.zeros(INPUT_SHAPE, np.uint8),
                                   output=np.full(OUTPUT_SHAPE, 1 / 6, np.float32),
                                   prepared_regions=np.zeros(REGIONS_SHAPE, np.float32),
                                   prepared_signals=np.full(SIGNALS_SHAPE, .5, np.float32))
    response = {'schema_version': PROTOCOL_VERSION, 'kind': 'result', 'frame_id': 9,
                'capture_time_ns': 100, 'inference_ms': 19., 'regions_reliable': False}
    monkeypatch.setattr(engine, '_send', lambda _: None)
    monkeypatch.setattr(engine, '_reply', lambda _: response)
    return engine, FramePacket(9, 100, np.zeros((360, 640, 3), np.uint8)), response


def test_client_copies_all_prepared_planes_before_shared_memory_can_change(monkeypatch):
    engine, packet, _ = client_fixture(monkeypatch)
    result = engine.process(packet)
    assert result.frame_id == 9 and result.capture_time_ns == 100
    assert not result.prepared_regions.reliable
    for name in ('person', 'hair', 'skin', 'clothes'):
        plane = getattr(result.prepared_regions, name)
        assert not plane.flags.writeable and plane.flags.c_contiguous
        assert not np.shares_memory(plane, engine.frames.prepared_regions)
    for name in ('luminance', 'denoised', 'edge_magnitude'):
        plane = getattr(result.prepared_signals, name)
        assert not plane.flags.writeable and plane.flags.c_contiguous
        assert not np.shares_memory(plane, engine.frames.prepared_signals)
    engine.frames.prepared_signals[:] = .9
    np.testing.assert_array_equal(result.prepared_signals.luminance, .5)


@pytest.mark.parametrize('field,value', [('prepared_regions', np.nan), ('prepared_regions', -.01),
                                      ('prepared_signals', np.inf), ('prepared_signals', 1.01),
                                      ('prepared_signals', -.01), ('prepared_regions', .2)])
def test_invalid_prepared_buffers_are_rejected_for_cpu_fallback(monkeypatch, field, value):
    engine, packet, _ = client_fixture(monkeypatch)
    getattr(engine.frames, field)[0, 0, 0] = value
    with pytest.raises(GpuProcessError, match='invalid_preparation'):
        engine.process(packet)


@pytest.mark.parametrize('flag', [1, 'false', None])
def test_prepared_reliability_is_an_exact_boolean(monkeypatch, flag):
    engine, packet, response = client_fixture(monkeypatch)
    response['regions_reliable'] = flag
    with pytest.raises(GpuProcessError, match='mismatch'):
        engine.process(packet)


@pytest.mark.parametrize('theme', ['red', 'yellow', 'blue'])
def test_cached_signal_and_region_arrays_preserve_all_filter_pixels(theme, monkeypatch):
    image = np.random.default_rng(73).integers(0, 256, (720, 1280, 3), np.uint8)
    masks = np.zeros(OUTPUT_SHAPE, np.float32)
    masks[..., 0] = 1
    masks[90:250, 220:420, 0] = 0
    masks[90:130, 220:420, 1] = 1
    masks[130:180, 220:420, 3] = 1
    masks[180:250, 220:420, 4] = 1
    state = SceneState(theme, ((300, 150), (980, 150), (980, 570), (300, 570)))
    original, expected_signals = compose_palette(image, state, masks=masks, seconds=1.25)
    prepared = regions_from_masks(masks, (360, 640))
    for plane in (expected_signals.luminance, expected_signals.denoised, expected_signals.edge_magnitude):
        plane.flags.writeable = False
    monkeypatch.setattr(render, 'signal_stages', lambda _: pytest.fail('Prepared signals must not be recomputed'))
    actual, returned_signals = compose_palette(image, state, regions=prepared,
                                               stages=expected_signals, seconds=1.25)
    np.testing.assert_array_equal(actual, original)
    assert returned_signals is expected_signals


def test_wrong_sized_prepared_stages_cannot_enter_the_renderer():
    image = np.zeros((720, 1280, 3), np.uint8)
    malformed = SignalStages(*(np.zeros((20, 20), np.float32) for _ in range(3)))
    with pytest.raises(ValueError, match='processing canvas'):
        compose_palette(image, SceneState(), stages=malformed)


def test_prepared_signal_planes_match_direct_canonical_processing():
    image = np.zeros((360, 640, 3), np.uint8)
    stages = signal_stages(image)
    assert all(plane.shape == (360, 640) and plane.dtype == np.float32
               for plane in (stages.luminance, stages.denoised, stages.edge_magnitude))
