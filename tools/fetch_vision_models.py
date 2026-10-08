"""Developer-only acquisition; the installed app never downloads models."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / "src/carl_cam/assets/models"
ASSETS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "selfie_multiclass_256x256.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, url in ASSETS.items():
        path = ROOT / name
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                data = response.read(32 * 1024 * 1024 + 1)
            if not 1000 <= len(data) <= 32 * 1024 * 1024:
                raise ValueError(f"Unexpected model size: {name}")
            path.write_bytes(data)
        records[name] = {"origin": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "bytes": path.stat().st_size, "license": "Apache-2.0"}
        print(f"Verified {name}: {path.stat().st_size:,} bytes")
    license_path = ROOT / "Apache-2.0.txt"
    if not license_path.exists():
        with urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt", timeout=30) as response:
            license_path.write_bytes(response.read(65536))
    (ROOT / "manifest.json").write_text(json.dumps({"schema_version": 1, "models": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
