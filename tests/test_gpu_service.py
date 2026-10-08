"""GPU service protocol fixtures, independent of actual NVIDIA execution."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from carl_cam import gpu_service
from carl_cam.dsp import signal_stages
from carl_cam.gpu_ipc import (
    INPUT_SHAPE,
    OUTPUT_SHAPE,
    PROTOCOL_VERSION,
    REGIONS_SHAPE,
    SIGNALS_SHAPE,
    decode_message,
    encode_message,
)
from carl_cam.themes import regions_from_masks
from carl_cam.vision import mirrored_canvas


def fixture_service(monkeypatch, *, precision='0'):
    frames = SimpleNamespace(
        input=np.random.default_rng(81).integers(0, 256, INPUT_SHAPE, np.uint8),
        output=np.full(OUTPUT_SHAPE, np.nan, np.float32),
        prepared_regions=np.full(REGIONS_SHAPE, np.nan, np.float32),
        prepared_signals=np.full(SIGNALS_SHAPE, np.nan, np.float32), closed=False,
    )
    engine = SimpleNamespace(
        session=SimpleNamespace(get_provider_options=lambda: {
            'CUDAExecutionProvider': {'use_tf32': precision}}),
        packets=[], closed=False,
    )
    masks = np.zeros(OUTPUT_SHAPE, np.float32)
    masks[..., 0] = 1
    masks[100:260, 180:450, 0] = 0
    masks[100:150, 180:450, 1] = 1
    masks[150:200, 180:450, 3] = 1
    masks[200:260, 180:450, 4] = 1

    def process(packet):
        engine.packets.append(packet)
        return SimpleNamespace(confidence_masks=masks, inference_ms=17.0)

    engine.process = process
    engine.close = lambda: setattr(engine, 'closed', True)
    frames.close = lambda: setattr(frames, 'closed', True)
    monkeypatch.setattr(gpu_service, 'SharedFrames', lambda _: frames)
    monkeypatch.setattr(gpu_service, 'GpuRegionEngine', lambda *args, **kwargs: engine)
    return frames, engine, masks


def messages(*items):
    return BytesIO(b''.join(encode_message({'schema_version': PROTOCOL_VERSION, **item})
                            for item in items))


def replies(stream):
    return [decode_message(line + b'\n') for line in stream.getvalue().splitlines()]


def test_service_prepares_all_arrays_from_exact_same_frame_and_closes(monkeypatch):
    frames, engine, masks = fixture_service(monkeypatch)
    incoming = messages({'kind': 'frame', 'frame_id': 41, 'capture_time_ns': 1234},
                        {'kind': 'stop'})
    outgoing = BytesIO()
    assert gpu_service.serve('fixture', Path('candidate'), incoming, outgoing) == 0
    assert engine.closed and frames.closed and len(engine.packets) == 4
    packet = engine.packets[-1]
    assert packet.frame_id == 41 and packet.capture_time_ns == 1234
    assert not packet.rgb.flags.writeable
    assert not np.shares_memory(packet.rgb, frames.input)
    expected_signals = signal_stages(cv2.resize(mirrored_canvas(packet), (640, 360),
                                               interpolation=cv2.INTER_AREA))
    expected_regions = regions_from_masks(masks, (360, 640))
    np.testing.assert_array_equal(frames.output, masks)
    for index, name in enumerate(('person', 'hair', 'skin', 'clothes')):
        np.testing.assert_array_equal(frames.prepared_regions[index], getattr(expected_regions, name))
    for index, name in enumerate(('luminance', 'denoised', 'edge_magnitude')):
        np.testing.assert_array_equal(frames.prepared_signals[index], getattr(expected_signals, name))
    assert replies(outgoing) == [
        {'schema_version': PROTOCOL_VERSION, 'kind': 'ready', 'backend': 'cuda_fp32'},
        {'schema_version': PROTOCOL_VERSION, 'kind': 'result', 'frame_id': 41,
         'capture_time_ns': 1234, 'inference_ms': 17.0,
         'regions_reliable': expected_regions.reliable},
        {'schema_version': PROTOCOL_VERSION, 'kind': 'stopped'},
    ]


@pytest.mark.parametrize('precision', ['1', None])
def test_service_rejects_precision_changes_before_ready(monkeypatch, precision):
    frames, engine, _ = fixture_service(monkeypatch, precision=precision)
    outgoing = BytesIO()
    assert gpu_service.serve('fixture', Path('candidate'), BytesIO(), outgoing) == 2
    assert engine.closed and frames.closed and not engine.packets
    assert np.isnan(frames.output).all()
    assert replies(outgoing) == [{'schema_version': PROTOCOL_VERSION, 'kind': 'error',
                                 'code': 'gpu_service_unavailable'}]


def test_service_rejects_repeated_frame_before_second_inference(monkeypatch):
    frames, engine, _ = fixture_service(monkeypatch)
    frame = {'kind': 'frame', 'frame_id': 41, 'capture_time_ns': 1234}
    outgoing = BytesIO()
    assert gpu_service.serve('fixture', Path('candidate'), messages(frame, frame), outgoing) == 2
    assert engine.closed and frames.closed and len(engine.packets) == 4
    assert [reply['kind'] for reply in replies(outgoing)] == ['ready', 'result', 'error']
