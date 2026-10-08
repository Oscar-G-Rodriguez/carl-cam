"""Prepare app-owned model artifacts for later creator-demo release review.

An export is a candidate, never a bundled/publicly approved demo. Numerical
recordings and personal activation receipts are deliberately excluded.
"""

import json
import os
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from carl_cam.model_store import MODEL_FILES, ModelStore, sha256_file
from carl_cam.storage import Store


def export_candidate(store: Store, version: str, destination: Path, *, user_confirmed: bool) -> Path:
    if not user_confirmed:
        raise ValueError("Confirm creator-candidate export first")
    models = ModelStore(store)
    models.load_tested(version)
    directory = models._directory(version)
    if destination.suffix.casefold() != ".zip" or destination.exists() or destination.is_symlink():
        raise ValueError("Choose a new ZIP filename; existing files are not overwritten")
    candidate = {
        "schema_version": 1, "model_version": version, "candidate_only": True,
        "public_release_approved": False, "unseen_person_validation_complete": False,
        "contains_training_recordings": False,
        "artifacts": {name: sha256_file(directory / name) for name in (*MODEL_FILES, "manifest.json")},
        "next_gate": "Owner selection, unchanged-candidate unseen-person evaluation, clean-machine and licensing checks",
    }
    # Build the entire archive beside its destination, then publish a complete
    # file with an exclusive hard link. Failure never leaves a partial chosen
    # archive or overwrites a file another process created during the export.
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
                for name in (*MODEL_FILES, "manifest.json"):
                    archive.write(directory / name, name)
                archive.writestr("candidate.json", json.dumps(candidate, indent=2, allow_nan=False))
                archive.writestr("README.txt", "Carl-Cam creator-demo candidate\n\nThis is not an approved public demo. It contains model weights, a normalizer, schema and integrity metadata, without training recordings, landmarks, photos or video.\n\nKeep this candidate unchanged during unseen-person tests. The owner must select it and complete the release checks before developers bundle it. Carl-Cam does not import arbitrary model archives.\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
