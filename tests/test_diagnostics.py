import json

import pytest

from carl_cam.diagnostics import LocalDiagnostics
from carl_cam.storage import Settings, Store


def test_local_diagnostics_rotate_with_bounded_allowlisted_content(tmp_path):
    store = Store(tmp_path / "CarlCam")
    log = LocalDiagnostics(store, maximum_bytes=256)
    for _ in range(30):
        assert log.record("render_slow")
    files = list((store.root / "logs").glob("*"))
    assert len(files) == 3 and all(path.stat().st_size <= 256 for path in files)
    for path in files:
        for line in path.read_text().splitlines():
            assert set(json.loads(line)) == {"schema_version", "time_utc", "code"}
    with pytest.raises(ValueError, match="error codes"):
        log.record("camera C:/private/path; image or landmarks here")


def test_reset_all_removes_only_named_owned_data_and_preserves_other_files(tmp_path):
    store = Store(tmp_path / "CarlCam")
    store.save_settings(Settings())
    store.save_settings(Settings())
    for name in ("datasets", "models", "logs"):
        folder = store.root / name
        folder.mkdir()
        (folder / "owned-example").touch()
    unrelated = store.root / "keep-user-file.txt"
    unrelated.touch()
    store.reset_all()
    assert list(store.root.iterdir()) == [unrelated]
    assert store.load_settings()[0].active_model_kind == "none"


def test_invalid_diagnostic_target_cannot_block_app_or_overwrite_unrelated_directory(tmp_path):
    store = Store(tmp_path / "CarlCam")
    folder = store.root / "logs" / "events.jsonl"
    folder.mkdir(parents=True)
    assert not LocalDiagnostics(store).record("model_invalid")
    assert folder.is_dir()
