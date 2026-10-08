"""Mechanical provenance/integrity manifest for genuine retained Figma exports."""

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/carl_cam/assets/graphics"


def main():
    ledger = json.loads((ROOT / "design/exported-nodes.json").read_text(encoding="utf-8"))
    assets = {}
    for entry in ledger["exported"]:
        name = entry["name"] + ".svg"
        data = (ASSETS / name).read_bytes()
        tree = ElementTree.fromstring(data)
        if any(node.tag.rsplit("}", 1)[-1] in ("image", "script", "foreignObject") for node in tree.iter()):
            raise ValueError("Only original vector paths and shapes belong in these exports")
        if b"https://" in data or b"http://" in data.replace(b"http://www.w3.org/2000/svg", b"").replace(b"http://www.w3.org/1999/xlink", b""):
            raise ValueError("Graphics may not depend on a remote resource")
        assets[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                        "figma_node": entry["nodeId"], "width": entry["width"], "height": entry["height"]}
    manifest = {"schema_version": 1, "figma_file": ledger["fileKey"],
                "origin": "Original Carl-Cam editable Figma artwork, retained SVG exports",
                "assets": assets, "contains_live_filter_pixels": False}
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(assets)} original offline SVG exports")


if __name__ == "__main__":
    main()
