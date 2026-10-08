"""App-owned model versions, staged verification, and explicit atomic activation.

No checkpoint import is offered. Only bounded, checksummed tensors written into
Carl-Cam's own model directory are loaded, using PyTorch's weights-only loader.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from carl_cam import __version__
from carl_cam.features import FEATURE_SCHEMAS
from carl_cam.storage import LABELS, Settings, Store, atomic_json, safe_identifier, valid_camera_key
from carl_cam.training import GestureNetwork, NeuralCandidate, Standardizer

MODEL_FILES = ("weights.pt", "scaler.json")
MAX_FILE_BYTES = 4 * 1024 * 1024


def fingerprint_dataset(rows: list[dict]) -> str:
    encoded = json.dumps(rows, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PersonalGate:
    action_successes: tuple[int, int, int, int, int]
    action_false_commands: int
    action_negative_seconds: float
    sealed_pose_recalls: tuple[float, float, float, float, float]
    sealed_false_commands: int
    rehearsal_successes: tuple[int, int, int, int, int]
    rehearsal_false_commands: int
    rehearsal_negative_seconds: float

    @property
    def passed(self) -> bool:
        return (
            len(self.action_successes) == len(self.rehearsal_successes) == len(self.sealed_pose_recalls) == 5
            and all(type(value) is int and 4 <= value <= 5 for value in self.action_successes)
            and all(type(value) is int and 8 <= value <= 10 for value in self.rehearsal_successes)
            and all(math.isfinite(value) and 0.80 <= value <= 1 for value in self.sealed_pose_recalls)
            and self.action_false_commands == self.sealed_false_commands == self.rehearsal_false_commands == 0
            and math.isfinite(self.action_negative_seconds) and self.action_negative_seconds >= 120
            and math.isfinite(self.rehearsal_negative_seconds) and self.rehearsal_negative_seconds >= 120
        )


class ModelStore:
    def __init__(self, store: Store) -> None:
        self.store = store

    def _directory(self, version: str) -> Path:
        if not safe_identifier(version):
            raise ValueError("Invalid model version")
        path = self.store.models / version
        if path.is_symlink() or not path.resolve().is_relative_to(self.store.models.resolve()):
            raise ValueError("Model path leaves Carl-Cam storage")
        return path

    def stage(self, candidate: NeuralCandidate, *, dataset_fingerprint: str, report: dict) -> str:
        if candidate.training_device not in ("cpu", "cuda", "unknown"):
            raise ValueError("Unsupported training device")
        if len(dataset_fingerprint) != 64 or any(character not in "0123456789abcdef" for character in dataset_fingerprint):
            raise ValueError("A verified dataset fingerprint is required")
        version = uuid4().hex
        directory = self._directory(version)
        directory.mkdir(parents=True, exist_ok=False)
        temporary = directory / ".weights.tmp"
        try:
            torch.save(candidate.network.cpu().state_dict(), temporary)
            os.replace(temporary, directory / "weights.pt")
        finally:
            temporary.unlink(missing_ok=True)
        atomic_json(directory / "scaler.json", {
            "mean": candidate.scaler.mean.tolist(), "scale": candidate.scaler.scale.tolist(),
        })
        manifest = {
            "schema_version": 1, "model_version": version, "app_version": __version__,
            "created_utc": datetime.now(UTC).isoformat(),
            "feature_schema": FEATURE_SCHEMAS[candidate.width], "width": candidate.width,
            "hidden": list(candidate.hidden), "labels": list(LABELS),
            "threshold": candidate.threshold, "margin": candidate.margin,
            "validation_loss": candidate.validation_loss,
            "validation_recall": list(candidate.validation_recall), "epochs": candidate.epochs,
            "session_ids": candidate.session_ids, "dataset_fingerprint": dataset_fingerprint,
            "training": {"seed": 17, "optimizer": "AdamW", "learning_rate": 0.001,
                         "weight_decay": 0.01, "dropout": 0.10, "batch_size": 64,
                         "max_epochs": 100, "patience": 10, "minimum_improvement": 0.001,
                         "device": candidate.training_device},
            "libraries": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scikit-learn")},
            "checksums": {name: sha256_file(directory / name) for name in MODEL_FILES},
            "report": report,
        }
        atomic_json(directory / "manifest.json", manifest)
        self.load(version)  # Schema, checksum, tensor shape, and finite smoke test.
        return version

    def load(self, version: str) -> NeuralCandidate:
        directory = self._directory(version)
        for name in (*MODEL_FILES, "manifest.json"):
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Model is missing, oversized, or points outside its version")
        with (directory / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        width, hidden = manifest.get("width"), manifest.get("hidden")
        if (manifest.get("schema_version") != 1 or manifest.get("model_version") != version
                or type(width) is not int or width not in FEATURE_SCHEMAS
                or manifest.get("feature_schema") != FEATURE_SCHEMAS[width]
                or manifest.get("labels") != list(LABELS)
                or hidden not in ([64, 32], [128, 64])):
            raise ValueError("Unsupported model or feature schema; please retrain")
        threshold, margin = manifest.get("threshold"), manifest.get("margin")
        if (not isinstance(threshold, (int, float)) or not 0.8 <= threshold <= 1
                or not isinstance(margin, (int, float)) or not 0.15 <= margin <= 1):
            raise ValueError("Model uses an unsupported command gate")
        checksums = manifest.get("checksums", {})
        if any(checksums.get(name) != sha256_file(directory / name) for name in MODEL_FILES):
            raise ValueError("Model checksum failed; your previous version has been kept")
        with (directory / "scaler.json").open(encoding="utf-8") as stream:
            scaler_values = json.load(stream)
        mean = np.asarray(scaler_values.get("mean"), dtype=np.float32)
        scale = np.asarray(scaler_values.get("scale"), dtype=np.float32)
        if mean.shape != (width,) or scale.shape != (width,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale < 1e-6):
            raise ValueError("Model scaler is invalid")
        network = GestureNetwork(width, tuple(hidden))
        weights = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError("Model weights are invalid")
        network.load_state_dict(weights, strict=True)
        device = manifest.get("training", {}).get("device", "unknown")
        if device not in ("cpu", "cuda", "unknown"):
            raise ValueError("Unsupported training device")
        candidate = NeuralCandidate(width, tuple(hidden), network.eval(), Standardizer(mean, scale),
                                    manifest["validation_loss"], tuple(manifest["validation_recall"]),
                                    manifest["epochs"], manifest["session_ids"], threshold, margin,
                                    device)
        if not np.isfinite(candidate.scores(mean)).all():
            raise ValueError("Model smoke test failed")
        return candidate

    def activate(self, version: str, settings: Settings, gate: PersonalGate, *, user_confirmed: bool) -> Settings:
        if not user_confirmed or not gate.passed:
            raise ValueError("Activation needs passed tests and your explicit confirmation")
        self.load(version)
        directory = self._directory(version)
        atomic_json(directory / "activation.json", {"schema_version": 1, "model_version": version, "gate": asdict(gate), "explicit_confirmation": True,
                                                    "camera_key": settings.camera_key})
        updated = replace(settings, active_model_kind="personal", active_model_version=version)
        self.store.save_settings(updated)  # Atomic pointer switch only after verification.
        return updated

    def load_tested(self, version: str) -> NeuralCandidate:
        path = self._directory(version) / "activation.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("This personal model has not passed activation checks")
        with path.open(encoding="utf-8") as stream:
            source = json.load(stream)
        try:
            gate = PersonalGate(**source["gate"])
            if source.get("schema_version") != 1 or source.get("model_version") != version or source.get("explicit_confirmation") is not True or not gate.passed:
                raise ValueError("This personal model has not passed activation checks")
        except (TypeError, KeyError) as error:
            raise ValueError("Invalid personal-model activation record") from error
        return self.load(version)

    def camera_verified(self, version: str, camera_key: str) -> bool:
        if not valid_camera_key(camera_key):
            return False
        directory = self._directory(version)
        activation_path = directory / "activation.json"
        if activation_path.is_symlink() or not activation_path.is_file() or activation_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Invalid personal-model activation record")
        activation = json.loads(activation_path.read_text(encoding="utf-8"))
        try:
            gate = PersonalGate(**activation["gate"])
            if (activation.get("schema_version") != 1 or activation.get("model_version") != version
                    or activation.get("explicit_confirmation") is not True or not gate.passed):
                raise ValueError("Invalid personal-model activation record")
        except (TypeError, KeyError, AttributeError) as error:
            raise ValueError("Invalid personal-model activation record") from error
        if activation.get("camera_key") == camera_key:
            return True  # Full personal activation happened on this camera.
        path = directory / f"camera-{camera_key}.json"
        if not path.is_file():
            return False
        if path.is_symlink() or path.stat().st_size > 128 * 1024:
            raise ValueError("Invalid camera recheck receipt")
        source = json.loads(path.read_text(encoding="utf-8"))
        from carl_cam.evaluation import ExerciseReport
        try:
            report = ExerciseReport(**source["report"])
            return (source.get("schema_version") == 1 and source.get("model_version") == version
                    and source.get("camera_key") == camera_key and source.get("explicit_confirmation") is True
                    and report.attempts_per_pose == 5 and report.passed)
        except (TypeError, KeyError) as error:
            raise ValueError("Invalid camera recheck receipt") from error

    def confirm_camera(self, version: str, camera_key: str, report, *, user_confirmed: bool):
        if not user_confirmed or not valid_camera_key(camera_key) or report.attempts_per_pose != 5 or not report.passed:
            raise ValueError("A camera recheck needs all five poses, a clean ordinary-movement check and confirmation")
        self.load_tested(version)
        atomic_json(self._directory(version) / f"camera-{camera_key}.json", {
            "schema_version": 1, "model_version": version, "camera_key": camera_key,
            "explicit_confirmation": True, "report": asdict(report),
        })

    def tested_versions(self) -> tuple[tuple[str, str], ...]:
        """List activation-receipted versions; selecting one still verifies weights."""
        entries = []
        for directory in sorted(self.store.models.glob("*")):
            try:
                directory = self._directory(directory.name)
                path = directory / "activation.json"
                manifest_path = directory / "manifest.json"
                if (not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES
                        or not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.stat().st_size > MAX_FILE_BYTES):
                    continue
                source = json.loads(path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                gate = PersonalGate(**source["gate"])
                if (source.get("schema_version") != 1 or source.get("model_version") != directory.name
                        or source.get("explicit_confirmation") is not True or not gate.passed
                        or manifest.get("model_version") != directory.name):
                    continue
                entries.append((directory.name, f"Personal {directory.name[:8]} · {manifest.get('created_utc', 'date unavailable')}"))
            except (ValueError, OSError, KeyError, TypeError):
                continue
        return tuple(entries)
