import copy
from threading import Event

import numpy as np
import pytest
import torch

from carl_cam.storage import LABELS
from carl_cam.training import (
    ActionResult,
    GestureNetwork,
    Standardizer,
    _partitions,
    fit_candidates,
    gpu_training_status,
    select_action_winner,
)


def rows():
    examples = []
    rng = np.random.default_rng(17)
    for label_index, label in enumerate(LABELS):
        for round_index in range(5):
            for _ in range(3):
                vector = rng.normal(0, 0.03, 180).astype(np.float32)
                vector[label_index] += 2
                examples.append({
                    "label": label, "round_index": round_index,
                    "session_id": f"{label}-{round_index}",
                    "features": {str(width): vector[:width].tolist() for width in (112, 138, 180)},
                })
    return examples


def test_sealed_measurements_never_enter_fitting_or_scaling():
    original = rows()
    changed = copy.deepcopy(original)
    for row in changed:
        if row["round_index"] == 4:
            row["features"] = {"poison": "must not be read"}
    first, first_sessions = _partitions(original, 180, True)
    second, second_sessions = _partitions(changed, 180, True)
    assert first_sessions == second_sessions
    for split in first:
        np.testing.assert_array_equal(first[split][0], second[split][0])
    scaler = Standardizer.fit(first["train"][0])
    np.testing.assert_array_equal(scaler.mean, first["train"][0].mean(axis=0))


def test_one_recording_cannot_cross_partitions():
    examples = rows()
    examples[-1]["session_id"] = examples[0]["session_id"]
    with pytest.raises(ValueError, match="cannot span"):
        _partitions(examples, 112, False)


def test_network_contract_and_rejects_nonfinite_measurements():
    assert GestureNetwork(180, (128, 64))(torch.zeros(2, 180)).shape == (2, 6)
    with pytest.raises(ValueError):
        GestureNetwork(181, (128, 64))
    with pytest.raises(ValueError):
        Standardizer.fit(np.full((2, 112), np.nan))


@pytest.fixture(scope="module")
def fitted():
    return fit_candidates(rows(), include_depth=True, max_epochs=2)


def test_fits_six_neural_candidates_and_three_comparison_baselines(fitted):
    assert len(fitted.candidates) == 6 and len(fitted.baselines) == 3
    assert {candidate.width for candidate in fitted.candidates} == {112, 138, 180}
    candidate = fitted.candidates[-1]
    probabilities = candidate.scores(np.zeros(180, dtype=np.float32))
    assert probabilities.shape == (1, 6)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert fitted.requested_device == "cpu" and not fitted.cpu_fallback
    assert all(candidate.training_device == "cpu" for candidate in fitted.candidates)


def test_four_candidates_when_depth_is_unavailable():
    result = fit_candidates(rows(), include_depth=False, max_epochs=1)
    assert len(result.candidates) == 4 and not result.depth_used


def test_candidate_selection_requires_every_pose_and_no_accidental_command(fitted):
    smaller, larger = fitted.candidates[:2]
    assert select_action_winner([ActionResult(smaller, (5, 5, 5, 5, 3), 0, 300)]) is None
    assert select_action_winner([ActionResult(smaller, (5, 5, 5, 5, 5), 1, 300)]) is None
    assert select_action_winner([ActionResult(smaller, (5, 5, 5, 5, 5), 0, float("nan"))]) is None
    assert select_action_winner([
        ActionResult(smaller, (4, 4, 4, 4, 4), 0, 340),
        ActionResult(larger, (4, 4, 4, 4, 4), 0, 300),
    ]) is smaller
    assert select_action_winner([
        ActionResult(smaller, (4, 4, 4, 4, 4), 0, 351),
        ActionResult(larger, (4, 4, 4, 4, 4), 0, 300),
    ]) is larger


def test_cancel_keeps_training_from_producing_a_candidate():
    cancel = Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        fit_candidates(rows(), include_depth=False, cancel=cancel, max_epochs=1)


@pytest.mark.parametrize("compiled, available, expected", [(None, True, False), ("13.0", False, False), ("13.0", True, True)])
def test_gpu_availability_reports_runtime_and_driver_separately(monkeypatch, compiled, available, expected):
    monkeypatch.setattr(torch.version, "cuda", compiled)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    supported, message = gpu_training_status()
    assert supported is expected
    assert message and ("CPU" in message)


@pytest.mark.parametrize("error", [RuntimeError, OSError])
def test_failed_gpu_probe_does_not_break_cpu_training(monkeypatch, error):
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    def unavailable():
        raise error("fixture GPU initialization failure")
    monkeypatch.setattr(torch.cuda, "is_available", unavailable)
    assert gpu_training_status()[0] is False


def test_explicit_gpu_request_reports_real_cpu_fallback(monkeypatch):
    from carl_cam import training
    monkeypatch.setattr(training, "gpu_training_status", lambda: (False, "No compatible GPU"))
    messages = []
    result = fit_candidates(rows(), include_depth=False, use_cuda=True, progress=messages.append, max_epochs=1)
    assert result.requested_device == "cuda" and result.cpu_fallback
    assert all(candidate.training_device == "cpu" for candidate in result.candidates)
    assert "CPU fallback" in result.backend_summary
    assert any("unavailable" in message for message in messages)


def test_gpu_failure_retries_on_cpu_once_then_keeps_using_cpu(monkeypatch):
    from carl_cam import training
    original = training._fit_one
    devices = []
    def fail_gpu(*args, device, **kwargs):
        devices.append(device)
        if device == "cuda":
            raise RuntimeError("fixture out of memory")
        return original(*args, device=device, **kwargs)
    monkeypatch.setattr(training, "gpu_training_status", lambda: (True, "Available"))
    monkeypatch.setattr(training, "_fit_one", fail_gpu)
    result = fit_candidates(rows(), include_depth=False, use_cuda=True, max_epochs=1)
    assert devices == ["cuda", "cpu", "cpu", "cpu", "cpu"]
    assert result.cpu_fallback and all(candidate.training_device == "cpu" for candidate in result.candidates)


def test_canceled_gpu_fit_never_probes_or_allocates_gpu(monkeypatch):
    from carl_cam import training
    def unexpected():
        pytest.fail("Canceled fitting must not probe GPU availability")
    monkeypatch.setattr(training, "gpu_training_status", unexpected)
    cancel = Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        fit_candidates(rows(), include_depth=True, use_cuda=True, cancel=cancel)
def test_depth_coverage_ignores_sealed_values_and_tolerates_isolated_missing_rows():
    from carl_cam.storage import LABELS
    from carl_cam.training import depth_coverage

    class ForbiddenFeatures:
        def get(self, *args):
            raise AssertionError("Sealed features were read before winner selection")
    rows = []
    for label in LABELS:
        for index in (0, 3):
            rows.extend({"label": label, "round_index": index, "features": {"180": [0.] * 180}} for _ in range(10))
            rows.append({"label": label, "round_index": index, "features": {}})
        rows.append({"label": label, "round_index": 4, "features": ForbiddenFeatures()})
    assert all(counts == (10, 10) for counts in depth_coverage(rows).values())
