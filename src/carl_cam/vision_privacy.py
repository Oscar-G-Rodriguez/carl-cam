"""Pin vision runtime integrity separately from telemetry-free verification.

On October 3, 2026 the owner removed the no-telemetry requirement. The exact
locked MediaPipe 0.10.35 Windows DLL may run with its metrics disclosed. Unknown
binaries still fail closed; permission to run never establishes privacy.
"""

import hashlib
import importlib.metadata
from pathlib import Path

APPROVED_VISION_RUNTIMES: frozenset[str] = frozenset()
# Matches the locked Windows wheel's RECORD entry, 28,736,000 bytes.
SUPPORTED_VISION_RUNTIMES: frozenset[str] = frozenset({
    'aa8e6c1b618c30cd3a6ad584dee1b2f2c99c3f3025d683bada36e1566d9092b7',
})
RUNTIME_POLICY = 'standard_mediapipe_metrics_disclosed'
PRIVACY_HOLD = "Vision runtime is unavailable or not verified. Mouse and keyboard still work."


class VisionPrivacyError(ValueError):
    """The installed SDK is missing or outside the supported runtime inventory."""


def runtime_library_path() -> Path:
    """Resolve the actual SDK layout in source and Nuitka directory builds.

Nuitka synthesizes distribution metadata rather than a normal site-packages
tree. Its ctypes DLL is explicitly shipped at the application-root path.
Never substitute a nearby source file for a source installation's SDK.
"""
    if '__compiled__' in globals():
        return Path(__file__).absolute().parents[1] / 'mediapipe/tasks/c/libmediapipe.dll'
    return Path(importlib.metadata.distribution('mediapipe').locate_file(
        'mediapipe/tasks/c/libmediapipe.dll'))


def require_verified_vision_runtime(library: Path | None = None) -> str:
    """Check exact supported bytes without importing or loading the library.

The hash identifies an allowed runtime, not a privacy or functional-test pass.
"""
    try:
        path = library if library is not None else runtime_library_path()
        if (any(item.is_symlink() or item.is_junction() for item in (path, *path.parents))
                or not path.is_file() or not 0 < path.stat().st_size <= 512 * 1024 * 1024):
            raise VisionPrivacyError(PRIVACY_HOLD)
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    except (OSError, importlib.metadata.PackageNotFoundError) as error:
        raise VisionPrivacyError(PRIVACY_HOLD) from error
    if checksum not in SUPPORTED_VISION_RUNTIMES | APPROVED_VISION_RUNTIMES:
        raise VisionPrivacyError(PRIVACY_HOLD)
    return checksum
