"""Developer-only acquisition of the unchanged OFL font used by the app/Figma."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / "src/carl_cam/assets/fonts"
ORIGIN = "https://raw.githubusercontent.com/google/fonts/main/ofl/arimo/"
FILES = {"Arimo.ttf": "Arimo%5Bwght%5D.ttf", "OFL.txt": "OFL.txt"}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name, remote_name in FILES.items():
        path = ROOT / name
        if not path.exists():
            with urlopen(ORIGIN + remote_name, timeout=30) as response:
                data = response.read(2 * 1024 * 1024 + 1)
            if not 1000 <= len(data) <= 2 * 1024 * 1024:
                raise ValueError(f"Unexpected font asset size: {name}")
            path.write_bytes(data)
        assets[name] = {
            "origin": ORIGIN + remote_name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "license": "OFL-1.1",
        }
        print(f"Verified {name}: {path.stat().st_size:,} bytes")
    (ROOT / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "assets": assets}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
