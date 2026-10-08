"""Shared-buffer and fallback fixtures; native GPU execution is separate."""

from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from carl_cam.gpu_ipc import (
    BUFFER_BYTES,
    INPUT_SHAPE,
    OUTPUT_SHAPE,
    PROTOCOL_VERSION,
    REGIONS_SHAPE,
    SIGNALS_SHAPE,
    SharedFrames,
    decode_message,
    encode_message,
    validate_frame_request,
)
from carl_cam.gpu_process import AdaptiveRegionEngine, GpuProcessError, GpuProcessRegionEngine
from carl_cam.records import FramePacket


def request():
    return {'schema_version': PROTOCOL_VERSION, 'kind': 'frame', 'frame_id': 9, 'capture_time_ns': 100}


def test_fixed_protocol_roundtrip_preserves_exact_frame_identity():
    decoded = decode_message(encode_message(request()))
    assert validate_frame_request(decoded) == (9, 100)


@pytest.mark.parametrize('payload', [b'', b'{}', b'[]\n', b'{}\n',
                                    b'{"schema_version":true}\n',
                                    b'{"schema_version":1}\n', b'x' * 1025 + b'\n'])
def test_invalid_or_unbounded_control_messages_are_rejected(payload):
    with pytest.raises(ValueError):
        decode_message(payload)


@pytest.mark.parametrize('key,value', [('frame_id', True), ('frame_id', -1),
                                     ('frame_id', 2**63), ('frame_id', 9.0),
                                     ('capture_time_ns', False), ('capture_time_ns', 0),
                                     ('capture_time_ns', 2**63), ('kind', 'evaluate'),
                                     ('unexpected', 'pixel-data')])
def test_frame_metadata_cannot_change_type_range_or_command(key, value):
    changed = request()
    changed[key] = value
    with pytest.raises(ValueError):
        validate_frame_request(changed)


def test_shared_buffer_has_exact_rgb_and_six_float_planes_and_single_owner():
    name = 'CarlCamGpu_' + uuid4().hex
    owner = SharedFrames(name, create=True)
    peer = None
    try:
        peer = SharedFrames(name)
        assert owner.memory.size == BUFFER_BYTES
        assert owner.input.shape == INPUT_SHAPE and owner.input.dtype == np.uint8
        assert peer.output.shape == OUTPUT_SHAPE and peer.output.dtype == np.float32
        assert peer.prepared_regions.shape == REGIONS_SHAPE
        assert peer.prepared_signals.shape == SIGNALS_SHAPE
        owner.input[2, 3] = [17, 31, 43]
        np.testing.assert_array_equal(peer.input[2, 3], [17, 31, 43])
        peer.output[4, 5] = [.1, .1, .1, .1, .1, .5]
        np.testing.assert_allclose(owner.output[4, 5], [.1, .1, .1, .1, .1, .5])
        peer.close()
        peer.close()
        reopened = SharedMemory(name=name)
        reopened.close()
    finally:
        if peer is not None:
            peer.close()
        owner.close()
        owner.close()
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


@pytest.mark.parametrize('name', ['', '../arbitrary', 'Global\\OtherApp',
                                  'CarlCamGpu_' + 'z' * 32, 'CarlCamGpu_' + '0' * 33])
def test_unrecognized_shared_memory_names_cannot_be_opened(name):
    with pytest.raises(ValueError):
        SharedFrames(name)


def test_wrong_sized_peer_is_rejected_without_unlinking_its_owner():
    name = 'CarlCamGpu_' + uuid4().hex
    owner = SharedMemory(name=name, create=True, size=12)
    try:
        with pytest.raises(ValueError, match='size'):
            SharedFrames(name)
        reopened = SharedMemory(name=name)
        reopened.close()
    finally:
        owner.close()
        owner.unlink()


class FakeCpu:
    def __init__(self):
        self.closed = False

    def process(self, packet):
        return ('cpu', packet)

    def close(self):
        self.closed = True


class FakeGpu(FakeCpu):
    backend_status = 'GPU — full FP32 (experimental)'

    def __init__(self, launch):
        super().__init__()

    def initialize(self, cancel):
        self.cancel = cancel

    def process(self, packet):
        return ('gpu', packet)


def test_cpu_preference_never_probes_or_launches_gpu():
    def refuse_launch():
        pytest.fail('CPU mode must not launch GPU')
    engine = AdaptiveRegionEngine('cpu', launcher=refuse_launch, cpu_factory=FakeCpu)
    engine.initialize(Event())
    assert engine.process('frame') == ('cpu', 'frame') and engine.backend_status == 'CPU'
    cpu = engine.engine
    engine.close()
    engine.close()
    assert cpu.closed


def test_missing_gpu_component_visibly_falls_back_to_cpu():
    engine = AdaptiveRegionEngine('gpu', launcher=lambda: None, cpu_factory=FakeCpu)
    engine.initialize(Event())
    assert engine.backend_status == 'CPU — GPU unavailable'
    assert engine.process('frame') == ('cpu', 'frame')
    engine.close()


def test_gpu_init_failure_closes_gpu_and_uses_cpu():
    gpu = FakeGpu(None)
    def fail(cancel):
        raise GpuProcessError('fixture_failure')
    gpu.initialize = fail
    engine = AdaptiveRegionEngine('gpu', launcher=lambda: object(),
                                  gpu_factory=lambda _: gpu, cpu_factory=FakeCpu)
    engine.initialize(Event())
    assert gpu.closed and engine.backend_status.startswith('CPU')
    assert engine.process('frame') == ('cpu', 'frame')
    engine.close()


def test_gpu_frame_failure_closes_child_and_reprocesses_same_frame_on_cpu():
    gpu = FakeGpu(None)
    engine = AdaptiveRegionEngine('gpu', launcher=lambda: object(),
                                  gpu_factory=lambda _: gpu, cpu_factory=FakeCpu)
    engine.initialize(Event())
    assert engine.process('first') == ('gpu', 'first')
    def fail(packet):
        raise GpuProcessError('fixture_failure')
    gpu.process = fail
    frame = object()
    assert engine.process(frame) == ('cpu', frame)
    assert gpu.closed and engine.backend_status.startswith('CPU — GPU stopped')
    engine.close()


def test_cancelled_initialization_does_not_start_cpu_fallback():
    token = Event()
    token.set()
    engine = AdaptiveRegionEngine('gpu', launcher=lambda: object(),
                                  cpu_factory=lambda: pytest.fail('Cancellation must not load CPU'))
    with pytest.raises(GpuProcessError, match='cancelled'):
        engine.initialize(token)
    engine.close()


@pytest.mark.parametrize('field,value', [('frame_id', 2), ('frame_id', True),
                                      ('capture_time_ns', 101), ('inference_ms', float('nan')),
                                      ('inference_ms', -1), ('inference_ms', True),
                                      ('kind', 'ready'), ('unexpected', 'field')])
def test_gpu_client_rejects_mismatched_and_invalid_responses(field, value, monkeypatch):
    engine = GpuProcessRegionEngine(None)
    engine.frames = SimpleNamespace(input=np.zeros(INPUT_SHAPE, np.uint8))
    response = {'schema_version': PROTOCOL_VERSION, 'kind': 'result', 'frame_id': 9,
                'capture_time_ns': 100, 'inference_ms': 17.0, 'regions_reliable': False}
    response[field] = value
    monkeypatch.setattr(engine, '_send', lambda _: None)
    monkeypatch.setattr(engine, '_reply', lambda _: response)
    with pytest.raises(GpuProcessError, match='mismatch'):
        engine.process(FramePacket(9, 100, np.zeros((360, 640, 3), np.uint8)))


def test_verified_development_launcher_is_disabled_in_compiled_app(monkeypatch):
    from carl_cam import gpu_process
    monkeypatch.setattr(gpu_process, '__compiled__', object(), raising=False)
    monkeypatch.setattr(gpu_process, '__file__', str(Path('untrusted.py')))
    assert gpu_process.development_launch() is None
