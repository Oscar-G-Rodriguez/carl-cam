import json
from dataclasses import replace

import numpy as np
import pytest

from carl_cam.model_store import ModelStore, PersonalGate, fingerprint_dataset
from carl_cam.storage import Settings, Store
from carl_cam.training import GestureNetwork, NeuralCandidate, Standardizer


def candidate():
    return NeuralCandidate(112, (64, 32), GestureNetwork(112, (64, 32)).eval(),
                           Standardizer(np.zeros(112, dtype=np.float32), np.ones(112, dtype=np.float32)),
                           0.5, (0.9,) * 6, 2, {"train": ("one",), "validation": ("two",), "sealed": ("three",)})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_device_provenance_preserves_cpu_model_loading(tmp_path, device):
    models = ModelStore(Store(tmp_path / "CarlCam"))
    version = models.stage(replace(candidate(), training_device=device), dataset_fingerprint=fingerprint_dataset([]), report={})
    restored = models.load(version)
    assert restored.training_device == device
    assert all(parameter.device.type == "cpu" for parameter in restored.network.parameters())
    manifest = json.loads((models.store.models / version / "manifest.json").read_text())
    assert manifest["training"]["device"] == device


def test_legacy_training_device_is_unknown_not_fabricated(tmp_path):
    models = ModelStore(Store(tmp_path / "CarlCam"))
    version = models.stage(candidate(), dataset_fingerprint=fingerprint_dataset([]), report={})
    path = models.store.models / version / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training"].pop("device")
    path.write_text(json.dumps(manifest))
    assert models.load(version).training_device == "unknown"


def test_invalid_training_device_is_rejected(tmp_path):
    models = ModelStore(Store(tmp_path / "CarlCam"))
    with pytest.raises(ValueError, match="training device"):
        models.stage(replace(candidate(), training_device="fake_gpu"), dataset_fingerprint=fingerprint_dataset([]), report={})


def passed_gate():
    return PersonalGate((4,) * 5, 0, 120, (0.8,) * 5, 0, (8,) * 5, 0, 120)


def test_model_round_trip_preserves_probabilities_and_manifest(tmp_path):
    models = ModelStore(Store(tmp_path / "CarlCam"))
    trained = candidate()
    version = models.stage(trained, dataset_fingerprint=fingerprint_dataset([]), report={"status": "staged"})
    loaded = models.load(version)
    np.testing.assert_array_equal(trained.scores(np.zeros(112)), loaded.scores(np.zeros(112)))
    manifest = json.loads((models.store.models / version / "manifest.json").read_text())
    assert manifest["training"]["seed"] == 17
    assert manifest["labels"][-1] == "other"
    assert models.store.load_settings()[0].active_model_kind == "none"


def test_failed_gate_or_missing_confirmation_cannot_replace_working_model(tmp_path):
    store = Store(tmp_path / "CarlCam")
    models = ModelStore(store)
    settings = Settings(active_model_kind="personal", active_model_version="previous")
    store.save_settings(settings)
    version = models.stage(candidate(), dataset_fingerprint=fingerprint_dataset([]), report={})
    for gate, confirmed in ((passed_gate(), False), (replace(passed_gate(), rehearsal_false_commands=1), True)):
        with pytest.raises(ValueError, match="explicit confirmation"):
            models.activate(version, settings, gate, user_confirmed=confirmed)
        assert store.load_settings()[0].active_model_version == "previous"
    updated = models.activate(version, settings, passed_gate(), user_confirmed=True)
    assert updated.active_model_version == version
    assert settings.active_model_version == "previous"  # Caller snapshot is not mutated.


def test_checksum_and_schema_failures_preserve_active_pointer(tmp_path):
    store = Store(tmp_path / "CarlCam")
    models = ModelStore(store)
    settings = Settings(active_model_version="previous")
    store.save_settings(settings)
    version = models.stage(candidate(), dataset_fingerprint=fingerprint_dataset([]), report={})
    scaler = store.models / version / "scaler.json"
    scaler.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        models.activate(version, settings, passed_gate(), user_confirmed=True)
    assert store.load_settings()[0].active_model_version == "previous"
    manifest_path = store.models / version / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_schema"] = "future-unknown"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema"):
        models.load(version)


def test_versions_cannot_escape_owned_storage(tmp_path):
    models = ModelStore(Store(tmp_path / "CarlCam"))
    with pytest.raises(ValueError):
        models.load("../not-ours")
