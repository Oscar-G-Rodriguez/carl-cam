"""Verify transferred bytes using only Python's standard library.

This does not import the SDK, load the candidate, start an observer or pass any
runtime test. The manifest SHA-256 must be carried separately from the bundle.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def file_hash(path):
    if (any(p.is_symlink() or p.is_junction() for p in (path, *path.parents))
            or not path.is_file()):
        raise ValueError("Expected an ordinary transferred file")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, manifest_hash):
    root = Path(root).absolute()
    manifest = root / "environment.json"
    if (len(manifest_hash) != 64 or any(c not in "0123456789abcdef" for c in manifest_hash)
            or file_hash(manifest) != manifest_hash):
        raise ValueError("Environment manifest does not match its independent hash")
    record: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    if (type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("preparation_verified") is not True
            or any(record.get(k) is not False for k in
                   ("runtime_loaded", "abi_verified", "privacy_verified", "numerical_verified",
                    "promoted", "gpu_benchmarked", "guest_environment_executed",
                    "network_enabled", "physical_camera_used", "host_runtime_modified"))):
        raise ValueError("Expected a static, unapproved environment manifest")
    names = set()
    for entry in record["files"]:
        name = entry["relative"]
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or any(p in ("", ".", "..") or p.endswith((".", " ")) for p in name.split("/"))
                or name.casefold() in names or name.casefold() == "environment.json"):
            raise ValueError("Unsafe or duplicate environment path")
        names.add(name.casefold())
        path = root / name
        if (type(entry.get("size")) is not int or path.stat().st_size != entry["size"]
                or file_hash(path) != entry["sha256"]):
            raise ValueError("Transferred bytes differ from manifest")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() or path.is_junction():
            raise ValueError("Transferred directory contains a link")
        if path.is_file():
            name = path.relative_to(root).as_posix().casefold()
            if name in actual:
                raise ValueError("Case-colliding environment files")
            actual.add(name)
    if actual != names | {"environment.json"}:
        raise ValueError("Unexpected or missing transferred file")
    if file_hash(manifest) != manifest_hash:
        raise ValueError("Manifest changed during verification")
    return {"files_verified": len(names), "bytes_verified": sum(e["size"] for e in record["files"]),
            "runtime_loaded": False, "privacy_verified": False, "abi_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.environment, args.manifest_sha256), indent=2), flush=True)


if __name__ == "__main__":
    main()
