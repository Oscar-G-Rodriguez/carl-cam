"""GPU adapter fixtures: no native inference, model evaluation or camera."""

import builtins
import hashlib
import json
import os
import sys
from time import monotonic_ns
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from carl_cam import gpu_regions
from carl_cam.records import FramePacket
from carl_cam.vision_privacy import VisionPrivacyError


@pytest.fixture
def candidate(tmp_path):
    model = tmp_path / "selfie_multiclass_fp32.onnx"
    model.write_bytes(b"not a model; private validation fixture")
    report = {
        "schema_version": 1, "source_model_sha256": gpu_regions.SOURCE_MODEL_SHA256,
        "conversion_verified": True, "source_model_unchanged": True,
        "source_contract": {"normalization": {"mean": [127.5], "std": [127.5]},
                            "activation": "SOFTMAX", "activation_explicit": True,
                            "associated_files": {"labels.txt":
                                "background\nhair\nbody-skin\nface-skin\nclothes\nothers\n"}},
        "onnx_contract": {"inputs": [{"name": "input_29", "shape": [1, 256, 256, 3],
                                      "dtype": "FLOAT32"}],
                          "outputs": [{"name": "Identity", "shape": [1, 256, 256, 6],
                                       "dtype": "FLOAT32"}]},
        "onnx_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "onnx_bytes": model.stat().st_size,
    }
    (tmp_path / "evidence.json").write_text(json.dumps(report))
    return tmp_path, report


def test_candidate_provenance_validation_does_not_approve_execution(candidate):
    directory, _ = candidate
    parsed = gpu_regions.inspect_candidate(directory)
    assert parsed.input_name == "input_29" and parsed.output_name == "Identity"
    assert parsed.model.parent == directory
    assert not gpu_regions.APPROVED_GPU_CONFIGURATIONS
    with pytest.raises(VisionPrivacyError):
        gpu_regions.require_verified_gpu_candidate(directory)


def test_empty_python_package_markers_are_hashed_but_empty_binaries_refused(tmp_path):
    marker = tmp_path / '__init__.py'
    marker.touch()
    assert gpu_regions._plain_file(marker, 32, allow_empty_python=True) == b''
    with pytest.raises(ValueError):
        gpu_regions._plain_file(marker, 32)
    binary = tmp_path / 'empty.dll'
    binary.touch()
    with pytest.raises(ValueError):
        gpu_regions._plain_file(binary, 32, allow_empty_python=True)


def test_experiment_requires_exact_inventory_and_never_promotes(candidate, monkeypatch):
    directory, _ = candidate
    parsed = gpu_regions.inspect_candidate(directory)
    monkeypatch.setattr(gpu_regions, 'runtime_inventory', lambda: 'pinned-fixture-inventory')
    monkeypatch.setattr(gpu_regions, 'EXPERIMENTAL_GPU_CONFIGURATIONS', frozenset({
        (parsed.evidence_sha256, 'pinned-fixture-inventory')}))
    assert gpu_regions.require_verified_gpu_candidate(directory, experiment=True) == parsed
    assert not gpu_regions.APPROVED_GPU_CONFIGURATIONS
    with pytest.raises(VisionPrivacyError):
        gpu_regions.require_verified_gpu_candidate(directory)
    monkeypatch.setattr(gpu_regions, 'runtime_inventory', lambda: 'changed-inventory')
    with pytest.raises(VisionPrivacyError):
        gpu_regions.require_verified_gpu_candidate(directory, experiment=True)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("source_model_sha256", "wrong"),
    ("conversion_verified", False), ("conversion_verified", "true"),
    ("source_model_unchanged", False), ("onnx_sha256", "wrong"),
    ("onnx_bytes", 0), ("source_contract", None), ("onnx_contract", []),
])
def test_candidate_rejects_altered_evidence(candidate, field, value):
    directory, report = candidate
    report[field] = value
    (directory / "evidence.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        gpu_regions.inspect_candidate(directory)


@pytest.mark.parametrize("field,value", [
    ("normalization", {"mean": [0.], "std": [255.]}),
    ("activation", "NONE"), ("activation_explicit", False),
    ("associated_files", None), ("associated_files", {"labels.txt": "hair\nbackground\n"}),
])
def test_processing_contract_cannot_silently_change(candidate, field, value):
    directory, report = candidate
    report["source_contract"][field] = value
    (directory / "evidence.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        gpu_regions.inspect_candidate(directory)


@pytest.mark.parametrize("field,value", [("shape", [1, 256, 256, 2]),
                                         ("dtype", "FLOAT16"), ("name", "")])
def test_output_shape_precision_and_name_are_checked(candidate, field, value):
    directory, report = candidate
    report["onnx_contract"]["outputs"][0][field] = value
    (directory / "evidence.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        gpu_regions.inspect_candidate(directory)


def test_changed_model_bytes_are_rejected(candidate):
    directory, _ = candidate
    (directory / "selfie_multiclass_fp32.onnx").write_bytes(b"different")
    with pytest.raises(ValueError, match="checksum"):
        gpu_regions.inspect_candidate(directory)


def test_privacy_hold_precedes_runtime_import_and_file_lookup(monkeypatch, tmp_path):
    original = builtins.__import__

    def no_runtime(name, *arguments, **keywords):
        assert not name.startswith(("onnxruntime", "mediapipe", "tensorflow"))
        return original(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", no_runtime)
    with pytest.raises(VisionPrivacyError, match="GPU segmentation"):
        gpu_regions.GpuRegionEngine(tmp_path / "absent")


@pytest.mark.parametrize("value,expected", [(0, -1.), (255, 1.), (128, 128 / 127.5 - 1)])
def test_input_normalization_black_white_midpoint(value, expected):
    rgb = np.full((360, 640, 3), value, dtype=np.uint8)
    rgb.flags.writeable = False
    tensor = gpu_regions.prepare_tensor(rgb)
    assert tensor.shape == (1, 256, 256, 3) and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous and not np.shares_memory(tensor, rgb)
    # OpenCV's float32 multiply-add differs from a float64 oracle by up to
    # two float32 ULPs. This is a recipe fixture, not native-model parity.
    np.testing.assert_allclose(tensor, expected, rtol=0., atol=2e-7)
    assert np.all(rgb == value)


def test_projection_preserves_channels_and_uses_full_roi_not_resize_half_pixels():
    rgb = np.zeros((360, 640, 3), np.uint8)
    rgb[:, :, 0] = np.arange(640, dtype=np.uint16)[None] % 256
    rgb[:, :, 1] = 73
    rgb[:, :, 2] = np.arange(360, dtype=np.uint16)[:, None] % 256
    before = rgb.copy()
    actual = gpu_regions.prepare_tensor(rgb)[0]
    transform = np.diag([256 / 640, 256 / 360, 1.])
    oracle = cv2.warpPerspective(rgb, transform, (256, 256), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    expected = oracle.astype(np.float64) * (2 / 255) - 1
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-7)
    np.testing.assert_array_equal(rgb, before)
    resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    assert not np.array_equal(oracle, resized)


@pytest.mark.parametrize("rgb", [None, np.zeros((360, 640), np.uint8),
                                  np.zeros((360, 640, 4), np.uint8),
                                  np.zeros((360, 640, 3), np.float32)])
def test_invalid_input_is_rejected(rgb):
    with pytest.raises(ValueError):
        gpu_regions.prepare_tensor(rgb)


def test_softmax_is_stable_normalized_and_keeps_six_class_order():
    logits = np.empty((1, 256, 256, 6), np.float32)
    logits[:] = [1000, 1001, 1002, 1003, 1004, 1005]
    before = logits.copy()
    logits.flags.writeable = False
    masks = gpu_regions.confidence_masks(logits)
    weights = np.exp(np.arange(6, dtype=np.float64) - 5)
    expected = weights / weights.sum()
    np.testing.assert_allclose(masks[180, 320], expected, atol=1e-7)
    np.testing.assert_allclose(masks.sum(axis=-1), 1., atol=2e-7)
    assert masks.shape == (360, 640, 6) and masks.dtype == np.float32
    assert not masks.flags.writeable and not np.shares_memory(masks, logits)
    np.testing.assert_array_equal(logits, before)


def test_activation_precedes_mask_resize():
    logits = np.zeros((1, 256, 256, 6), np.float32)
    logits[0, :, :128, 1] = 8
    logits[0, :, 128:, 2] = 8
    probabilities = np.exp(logits[0].astype(np.float64))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.stack([cv2.resize(probabilities[..., i].astype(np.float32), (640, 360),
                                    interpolation=cv2.INTER_LINEAR) for i in range(6)], axis=-1)
    np.testing.assert_allclose(gpu_regions.confidence_masks(logits), expected, atol=2e-7)


@pytest.mark.parametrize("logits", [None, np.zeros((1, 256, 256, 2), np.float32),
                                     np.zeros((1, 256, 256, 6), np.float16),
                                     np.full((1, 256, 256, 6), np.nan, np.float32),
                                     np.full((1, 256, 256, 6), np.inf, np.float32)])
def test_incompatible_logits_are_rejected(logits):
    with pytest.raises(ValueError):
        gpu_regions.confidence_masks(logits)


def fake_runtime(monkeypatch, candidate, *, available=True, initialized=True):
    directory, _ = candidate
    parsed = gpu_regions.inspect_candidate(directory)
    monkeypatch.setattr(gpu_regions, "require_verified_gpu_candidate", lambda _, **kwargs: parsed)
    state = {"telemetry_disabled": False, "fallback_disabled": False, "options": {},
             "resources_closed": False}

    def load_dependencies(selected, **kwargs):
        assert selected is parsed
        state["dependencies_loaded"] = True
        return (SimpleNamespace(close=lambda: state.update(resources_closed=True)),)

    monkeypatch.setattr(gpu_regions, "_load_cuda_dependencies", load_dependencies)

    class Options:
        def add_session_config_entry(self, name, value):
            state["options"][name] = value

    class Session:
        def __init__(self, model, *, sess_options, providers):
            assert state["telemetry_disabled"] and state["dependencies_loaded"]
            state["providers"] = providers

        def disable_fallback(self):
            state["fallback_disabled"] = True

        def get_providers(self):
            return ["CUDAExecutionProvider"] if initialized else ["CPUExecutionProvider"]

        def run(self, names, feeds):
            assert names == ["Identity"] and list(feeds) == ["input_29"]
            assert feeds["input_29"].shape == (1, 256, 256, 3)
            return [np.zeros((1, 256, 256, 6), np.float32)]

    runtime = SimpleNamespace(SessionOptions=Options, InferenceSession=Session,
                              get_available_providers=lambda: ["CUDAExecutionProvider"]
                                  if available else ["CPUExecutionProvider"],
                              disable_telemetry_events=lambda: state.update(telemetry_disabled=True))
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    return directory, state


def test_fixture_engine_keeps_frame_pair_and_explicit_cuda_configuration(monkeypatch, candidate):
    directory, state = fake_runtime(monkeypatch, candidate)
    engine = gpu_regions.GpuRegionEngine(directory)
    packet = FramePacket(19, monotonic_ns(), np.zeros((360, 640, 3), np.uint8))
    result = engine.process(packet)
    assert result.frame_id == packet.frame_id and result.capture_time_ns == packet.capture_time_ns
    assert result.confidence_masks.shape == (360, 640, 6)
    assert not result.confidence_masks.flags.writeable
    assert state["options"]["session.disable_cpu_ep_fallback"] == "1"
    assert state["fallback_disabled"]
    assert state["providers"] == [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})]
    assert set(engine.last_stage_ms) == {"preprocess", "inference", "postprocess"}
    engine.close()
    engine.close()
    assert engine.session is None
    assert engine._cuda_libraries == ()
    assert state["resources_closed"]


@pytest.mark.parametrize("available,initialized", [(False, True), (True, False)])
def test_missing_gpu_never_claims_a_cpu_session_is_gpu(monkeypatch, candidate, available, initialized):
    directory, state = fake_runtime(monkeypatch, candidate,
                                    available=available, initialized=initialized)
    with pytest.raises(RuntimeError):
        gpu_regions.GpuRegionEngine(directory)
    assert state["resources_closed"]


def test_actual_gpu_model_loads_and_processes_without_a_camera():
    if not gpu_regions.APPROVED_GPU_CONFIGURATIONS:
        pytest.skip("Production GPU packaging/integration is unverified; explicit developer GPU evidence is separate")
    selected = os.environ.get("CARLCAM_GPU_CANDIDATE")
    assert selected, "Select the exact reviewed GPU candidate for the required integration check"
    from pathlib import Path
    engine = gpu_regions.GpuRegionEngine(Path(selected))
    try:
        packet = FramePacket(7, monotonic_ns(), np.zeros((360, 640, 3), np.uint8))
        result = engine.process(packet)
        assert result.frame_id == packet.frame_id
        assert result.capture_time_ns == packet.capture_time_ns
        assert result.confidence_masks.shape == (360, 640, 6)
        assert np.isfinite(result.confidence_masks).all()
        assert not result.confidence_masks.flags.writeable
    finally:
        engine.close()
