"""Bounded local error codes only: never images, landmarks or error messages.

Three 128 KiB files at most. Retention is deliberately small and local; callers
may continue safely if diagnostics cannot be written. No upload/export API.
"""

import json
from datetime import UTC, datetime
from threading import Lock

EVENTS = frozenset({"camera_unavailable", "vision_unavailable", "output_unavailable",
                    "model_invalid", "render_slow", "shutdown_delayed"})
NAMES = ("events.jsonl", "events.1.jsonl", "events.2.jsonl")


class LocalDiagnostics:
    def __init__(self, store, *, maximum_bytes=128 * 1024):
        if type(maximum_bytes) is not int or not 256 <= maximum_bytes <= 128 * 1024:
            raise ValueError("Invalid diagnostic retention bound")
        self.root, self.maximum_bytes = store.root / "logs", maximum_bytes
        self._lock = Lock()

    def record(self, code):
        if code not in EVENTS:
            raise ValueError("Only supported error codes may enter local diagnostics")
        row = (json.dumps({"schema_version": 1, "time_utc": datetime.now(UTC).isoformat(),
                           "code": code}, separators=(",", ":")) + "\n").encode()
        try:
            with self._lock:
                if self.root.is_symlink():
                    return False
                self.root.mkdir(parents=True, exist_ok=True)
                paths = [self.root / name for name in NAMES]
                if any(path.is_symlink() or path.exists() and not path.is_file() for path in paths):
                    return False
                if paths[0].exists() and paths[0].stat().st_size + len(row) > self.maximum_bytes:
                    paths[2].unlink(missing_ok=True)
                    if paths[1].exists():
                        paths[1].replace(paths[2])
                    paths[0].replace(paths[1])
                with paths[0].open("ab") as stream:
                    stream.write(row)
            return True
        except OSError:
            return False  # A diagnostic failure must never interrupt live safety.
