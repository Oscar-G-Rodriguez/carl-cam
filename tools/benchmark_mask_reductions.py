"""Camera-free experiment; compare six-label reductions without loading vision.

This measures only confidence summation and category winner selection. It is
not full rendering, segmentation inference, live FPS or permission to promote
the held SDK. No application algorithm is changed by running this tool.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np

from carl_cam.themes import _category_winner, _foreground_channels


def reference(masks):
    return masks[..., 1:].sum(axis=-1), np.argmax(masks, axis=-1)


def candidate(masks):
    confidence, channels = _foreground_channels(masks)
    return confidence, _category_winner(masks, channels)


def fixtures():
    rng = np.random.default_rng(20261002)
    for dtype in (np.float32, np.float64):
        for shape in ((1, 1, 6), (9, 23, 6), (360, 640, 6)):
            values = rng.random(shape).astype(dtype)
            yield f"noise-{dtype.__name__}-{shape}", values
        tied = np.zeros((40, 80, 6), dtype)
        tied[::2] = .55
        tied[1::2, ::2, (1, 3, 5)] = .55
        yield f"ties-{dtype.__name__}", tied
        yield f"noncontiguous-{dtype.__name__}", values[::2, ::-2]
        confidence_boundary = np.full((1, 9, 6), .11, dtype)
        confidence_boundary[..., 0] = 0
        confidence_boundary[0, 0, 1] = np.nextafter(dtype(.11), dtype(0))
        confidence_boundary[0, 1, 1] = np.nextafter(dtype(.11), dtype(1))
        yield f"threshold-{dtype.__name__}", confidence_boundary


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "build" / ("mask-reduction-experiment-" + uuid4().hex)
    output.mkdir()
    source = Path(__file__)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report = {
        "created_utc": datetime.now(UTC).isoformat(), "source_sha256": source_hash,
        "theme_source_sha256": hashlib.sha256((root / "src/carl_cam/themes.py").read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(), "physical_camera_used": False,
        "vision_runtime_loaded": False, "application_mutated_by_tool": False,
        "live_performance_verified": False, "fixture_results": [],
        "timing_scope": ["six_category_winner", "foreground_confidence_sum"],
        "warmup_iterations": 12, "iterations_per_variant": 181,
    }
    for name, masks in fixtures():
        masks.flags.writeable = False
        before = masks.copy()
        expected, actual = reference(masks), candidate(masks)
        report["fixture_results"].append({
            "name": name, "confidence_exact": bool(np.array_equal(expected[0], actual[0])),
            "winner_exact": bool(np.array_equal(expected[1], actual[1])),
            "input_unchanged": bool(np.array_equal(masks, before)),
        })
    # Time finite clipped model-like masks separately from fixture assertions.
    masks = np.random.default_rng(19).random((360, 640, 6), dtype=np.float32)
    masks.flags.writeable = False
    timings = {"reference": [], "candidate": []}
    functions = {"reference": reference, "candidate": candidate}
    for iteration in range(-12, 181):
        order = ("candidate", "reference") if iteration % 2 else ("reference", "candidate")
        for variant in order:
            started = perf_counter()
            functions[variant](masks)
            elapsed = (perf_counter() - started) * 1000
            if iteration >= 0:
                timings[variant].append(elapsed)
    report["raw_ms"] = timings
    report["timings"] = {variant: {
        "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
    } for variant, values in timings.items()}
    report["all_exact"] = all(all(result[key] for key in
        ("confidence_exact", "winner_exact", "input_unchanged"))
        for result in report["fixture_results"])
    report["source_unchanged"] = (
        hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        and hashlib.sha256((root / "src/carl_cam/themes.py").read_bytes()).hexdigest()
        == report["theme_source_sha256"])
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (output / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"evidence": str(output / "evidence.json"), "all_exact": report["all_exact"],
                      "timings": report["timings"]}))
    return 0 if report["all_exact"] and report["source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
