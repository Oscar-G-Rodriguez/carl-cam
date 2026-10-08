import json

import pytest
from test_features import hand

from carl_cam.collection import CollectionRound, RoundPrompt, collection_plan
from carl_cam.features import FeatureExtractor
from carl_cam.storage import Settings, Store


def test_settings_recover_invalid_fields_without_erasing_personal_data(tmp_path) -> None:
    store = Store(tmp_path / "CarlCam")
    store.save_settings(Settings(selected_theme="blue", setup_complete=True))
    source = json.loads(store.settings_path.read_text())
    source["hold_ms"] = 1
    store.settings_path.write_text(json.dumps(source))
    settings, warnings = store.load_settings()
    assert settings.selected_theme == "blue" and settings.setup_complete
    assert settings.hold_ms == 300 and warnings


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_backend_setting_roundtrip(backend, tmp_path):
    store = Store(tmp_path / 'CarlCam')
    store.save_settings(Settings(region_backend=backend))
    loaded, warnings = store.load_settings()
    assert loaded.region_backend == backend and not warnings


@pytest.mark.parametrize('backend', ['unreviewed', None, {}, True])
def test_invalid_backend_falls_back_without_erasing_settings(backend, tmp_path):
    store = Store(tmp_path / 'CarlCam')
    store.save_settings(Settings(selected_theme='blue'))
    source = json.loads(store.settings_path.read_text())
    source['region_backend'] = backend
    store.settings_path.write_text(json.dumps(source))
    loaded, warnings = store.load_settings()
    assert loaded.region_backend == 'cpu' and loaded.selected_theme == 'blue' and warnings


def test_collection_has_thirty_whole_sessions_and_rate_limit() -> None:
    plan = collection_plan()
    assert len(plan) == 30 and {prompt.round_index for prompt in plan} == set(range(5))
    round_ = CollectionRound(RoundPrompt("peace", 0, ("center", "shift", "angle"), 12), dataset_id="test", start_seconds=0)
    result = FeatureExtractor().extract((hand(),))
    assert round_.accept(result, 0, "center")
    assert not round_.accept(result, 0.1, "center")
    assert not round_.accept(result, 0.2, "center")
    assert "nearly identical" in round_.last_hint
    shifted = FeatureExtractor().extract((hand(offset=(304, 350)),))
    assert round_.accept(shifted, 0.2, "center")
    assert len(round_.rows) == 2 and not round_.complete


def test_deleting_samples_keeps_models_and_unrelated_files(tmp_path) -> None:
    store = Store(tmp_path / "CarlCam")
    round_ = CollectionRound(collection_plan()[5], dataset_id="test", start_seconds=0)
    round_.accept(FeatureExtractor().extract((hand(),)), 0, round_.prompt.variations[0])
    store.save_round("test", round_.session_id, round_.rows)
    store.models.mkdir()
    model = store.models / "weights.pt"
    model.write_bytes(b"test")
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")
    store.delete_samples()
    assert not store.datasets.exists() and model.exists() and unrelated.exists()
    with pytest.raises(ValueError):
        store.save_round("../outside", "session", [{"label": "peace"}])


def test_dataset_round_trip_rejects_extra_payload_and_nonfinite_data(tmp_path):
    store = Store(tmp_path / "CarlCam")
    round_ = CollectionRound(collection_plan()[5], dataset_id="test", start_seconds=0)
    round_.accept(FeatureExtractor().extract((hand(),)), 0, round_.prompt.variations[0])
    store.save_round("test", round_.session_id, round_.rows)
    assert len(store.load_dataset("test")) == 1
    row = round_.rows[0]
    row["pixels"] = "unexpected payload"
    with pytest.raises(ValueError, match="anonymous numerical"):
        store.save_round("test", round_.session_id, round_.rows)
    del row["pixels"]
    row["features"]["112"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        store.save_round("test", round_.session_id, round_.rows)
