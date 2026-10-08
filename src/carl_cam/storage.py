"""Atomic, versioned local settings and numerical training data.

Only known Carl-Cam directories are deletion targets. No image array is accepted
by the sample API, and model export contains artifacts rather than examples.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

LABELS = ("frame", "peace", "thumbs_up", "open_palm", "ily", "other")
THEMES = ("red", "yellow", "blue")


def valid_camera_key(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(slots=True)
class Settings:
    schema_version: int = 1
    camera_device: int = 0
    camera_key: str | None = None
    region_backend: str = 'cpu'
    selected_theme: str = "red"
    assignments: dict[str, str] = field(default_factory=dict)
    hold_ms: int = 300
    reduced_motion: bool = False
    high_contrast: bool = False
    active_model_kind: str = "none"
    active_model_version: str | None = None
    privacy_accepted: bool = False
    setup_complete: bool = False
    window_width: int = 1280
    window_height: int = 800


def atomic_json(path: Path, value: object, *, keep_backup: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A short unique component avoids exceeding Windows path limits merely
    # because an already-valid destination has a long descriptive filename.
    temporary = path.with_name(f".{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        if keep_backup and path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "CarlCam"
        self.datasets = self.root / "datasets"
        self.models = self.root / "models"
        self.settings_path = self.root / "settings.json"
        self._settings_read_only = False

    def load_settings(self) -> tuple[Settings, list[str]]:
        settings = Settings()
        from carl_cam.accessibility import os_reduced_motion
        settings.reduced_motion = os_reduced_motion()
        warnings = []
        if not self.settings_path.exists():
            return settings, warnings
        try:
            with self.settings_path.open(encoding="utf-8") as stream:
                source = json.load(stream)
        except (OSError, ValueError):
            return settings, ["Settings could not be read. Your saved gesture data was kept."]
        if not isinstance(source, dict) or source.get("schema_version") != 1:
            self._settings_read_only = True
            return settings, ["This settings version is unsupported. Saved data has not been changed."]
        validators = {
            "camera_device": lambda value: type(value) is int and 0 <= value <= 99,
            "camera_key": lambda value: value is None or valid_camera_key(value),
            "region_backend": lambda value: isinstance(value, str) and value in ('cpu', 'gpu'),
            "selected_theme": lambda value: value in THEMES,
            "assignments": valid_assignments,
            "hold_ms": lambda value: type(value) is int and value in (300, 400, 600),
            "reduced_motion": lambda value: type(value) is bool,
            "high_contrast": lambda value: type(value) is bool,
            "active_model_kind": lambda value: value in ("none", "demo", "personal"),
            "active_model_version": lambda value: value is None or safe_identifier(value),
            "privacy_accepted": lambda value: type(value) is bool,
            "setup_complete": lambda value: type(value) is bool,
            "window_width": lambda value: type(value) is int and 1024 <= value <= 7680,
            "window_height": lambda value: type(value) is int and 640 <= value <= 4320,
        }
        for key, validate in validators.items():
            if key in source:
                if validate(source[key]):
                    setattr(settings, key, source[key])
                else:
                    warnings.append(f"The saved {key.replace('_', ' ')} setting was reset.")
        return settings, warnings

    def save_settings(self, settings: Settings) -> None:
        if self._settings_read_only:
            raise ValueError("Unsupported settings version must remain read-only")
        atomic_json(self.settings_path, asdict(settings), keep_backup=True)

    def save_round(self, dataset_id: str, session_id: str, rows: list[dict]) -> Path:
        if not safe_identifier(dataset_id) or not safe_identifier(session_id):
            raise ValueError("Invalid dataset or session identifier")
        if not rows:
            raise ValueError("A round must contain numerical observations")
        for row in rows:
            validate_sample(row, dataset_id, session_id)
        path = self.datasets / dataset_id / f"{session_id}.json"
        atomic_json(path, {"schema_version": 1, "dataset_id": dataset_id, "session_id": session_id, "rows": rows})
        return path

    def load_dataset(self, dataset_id: str) -> list[dict]:
        if not safe_identifier(dataset_id):
            raise ValueError("Invalid dataset identifier")
        directory = self.datasets / dataset_id
        if directory.is_symlink() or not directory.resolve().is_relative_to(self.datasets.resolve()):
            raise ValueError("Dataset leaves Carl-Cam storage")
        rows = []
        for path in sorted(directory.glob("*.json")):
            if path.name == "dataset-manifest.json":
                continue
            if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Invalid or oversized recording round")
            with path.open(encoding="utf-8") as stream:
                content = json.load(stream)
            if not isinstance(content, dict) or content.get("schema_version") != 1 or content.get("dataset_id") != dataset_id or not safe_identifier(content.get("session_id")):
                raise ValueError("Recording round has an unsupported identity or schema")
            session = content["session_id"]
            if path.stem != session or not isinstance(content.get("rows"), list) or len(content["rows"]) > 10000:
                raise ValueError("Recording round has invalid observations")
            for row in content["rows"]:
                validate_sample(row, dataset_id, session)
                rows.append(row)
        return rows

    def delete_samples(self) -> None:
        self._remove_owned_directory(self.datasets)

    def delete_personal_models(self) -> None:
        self._remove_owned_directory(self.models)

    def reset_all(self) -> None:
        """Remove only the named app-owned state, never the storage root itself."""
        for directory in (self.datasets, self.models, self.root / "logs"):
            self._remove_owned_directory(directory)
        for name in ("settings.json", "settings.json.bak"):
            path = self.root / name
            if path.is_symlink() or path.resolve().parent != self.root.resolve():
                raise ValueError("Settings point outside Carl-Cam storage")
            path.unlink(missing_ok=True)
        self._settings_read_only = False

    def occupied_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file() and not path.is_symlink())

    def _remove_owned_directory(self, path: Path) -> None:
        if path.name not in ("datasets", "models", "logs") or path.parent != self.root:
            raise ValueError("Not a Carl-Cam data directory")
        if not path.exists():
            return
        if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Data directory points outside Carl-Cam storage")
        shutil.rmtree(path)


def safe_identifier(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 80 and all(character.isalnum() or character in "-_" for character in value)


def valid_assignments(value: object) -> bool:
    return isinstance(value, dict) and (not value or (
        set(value) == {"thumbs_up", "open_palm", "ily"}
        and all(isinstance(theme, str) for theme in value.values())
        and set(value.values()) == set(THEMES)
    ))


def finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_sample(row: dict, dataset_id: str, session_id: str) -> None:
    """An allowlist prevents accidental image, video, or identity persistence."""
    fields = {"schema_version", "dataset_id", "session_id", "label", "round_index", "time_seconds",
              "variation", "features", "landmarks", "handedness", "quality"}
    if (not isinstance(row, dict) or set(row) != fields or row.get("schema_version") != 1
            or row.get("dataset_id") != dataset_id or row.get("session_id") != session_id
            or row.get("label") not in LABELS or type(row.get("round_index")) is not int
            or not 0 <= row["round_index"] < 5
            or not finite_number(row.get("time_seconds")) or row["time_seconds"] < 0
            or row.get("variation") not in {"center", "shift", "angle", "distance", "lighting",
                                            "rest_typing", "touch_face", "ordinary_movement", "point_pinch", "entry_exit"}):
        raise ValueError("Only supported anonymous numerical observations may be stored")
    features = row["features"]
    if not isinstance(features, dict) or not {"112", "138"} <= features.keys() or not features.keys() <= {"112", "138", "180"}:
        raise ValueError("Unsupported feature schema")
    for key, values in features.items():
        if not isinstance(values, (list, tuple)) or len(values) != int(key) or not all(finite_number(value) for value in values):
            raise ValueError("Measurements must be finite and match their feature schema")
    hands, identities, quality = row["landmarks"], row["handedness"], row["quality"]
    if (not isinstance(hands, (list, tuple)) or not 1 <= len(hands) <= 2
            or not isinstance(identities, (list, tuple)) or len(identities) != len(hands)
            or any(identity not in ("Left", "Right") for identity in identities)
            or not isinstance(quality, dict) or set(quality) != {"two_hand_stable", "valid_frame"}
            or not all(type(value) is bool for value in quality.values())):
        raise ValueError("Recording has invalid hand or quality metadata")
    for hand in hands:
        if not isinstance(hand, (list, tuple)) or len(hand) != 21:
            raise ValueError("Recording has incomplete landmarks")
        for point in hand:
            if (not isinstance(point, (list, tuple)) or len(point) != 3
                    or not all(finite_number(value) for value in point[:2])
                    or point[2] is not None and not finite_number(point[2])):
                raise ValueError("Recording has invalid landmarks")
