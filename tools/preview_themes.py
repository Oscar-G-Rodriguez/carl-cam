"""Developer-only synthetic compositor review; never records a camera image.

The schematic seated figure and exact hand-written category masks exercise
layering. This is not evidence of real-person segmentation quality.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np

from carl_cam.dsp import AdaptiveRange
from carl_cam.graphics import asset_pixels, draw_border
from carl_cam.render import SceneState, compose_palette


def synthetic_scene():
    """Exact procedural fixture shared by review and timing tools; no camera."""
    image = np.zeros((720, 1280, 3), np.uint8)
    image[:] = (140, 151, 163)
    masks = np.zeros((360, 640, 6), np.float32)
    masks[..., 0] = 1
    # An intentionally simple seated person, drawn procedurally for inspection.
    category = np.zeros((720, 1280), np.uint8)
    cv2.ellipse(category, (640, 315), (103, 136), 0, 0, 360, 3, -1)
    cv2.ellipse(category, (640, 223), (116, 62), 0, 0, 360, 1, -1)
    cv2.fillPoly(category, [np.array(((580, 430), (700, 430), (852, 676), (424, 676)))], 4)
    image[category == 3] = (202, 178, 158)
    image[category == 1] = (40, 37, 47)
    image[category == 4] = (85, 91, 104)
    cv2.ellipse(image, (580, 300), (19, 8), 0, 0, 360, (33, 32, 37), -1)
    cv2.ellipse(image, (698, 300), (19, 8), 0, 0, 360, (33, 32, 37), -1)
    cv2.line(image, (644, 319), (659, 353), (117, 92, 85), 4)
    cv2.ellipse(image, (640, 368), (33, 12), 0, 0, 180, (40, 31, 42), 4)
    cv2.ellipse(image, (610, 216), (31, 10), -25, 0, 360, (165, 160, 168), -1)
    small = cv2.resize(category, (640, 360), interpolation=cv2.INTER_NEAREST)
    for index in range(6):
        masks[..., index] = small == index
    quad = ((265, 111), (1010, 149), (981, 619), (278, 589))
    return image, masks, quad


def main():
    output = Path(__file__).resolve().parents[1] / "build"
    output.mkdir(exist_ok=True)
    review = output / ('synthetic-render-' + uuid4().hex)
    review.mkdir()
    image, masks, quad = synthetic_scene()
    timings = {}
    for theme in ("red", "yellow", "blue"):
        asset_pixels("border-" + theme)
        elapsed = []
        adaptive = AdaptiveRange()
        for frame in range(12):
            started = perf_counter()
            result, _ = compose_palette(image, SceneState(theme, quad), masks=masks,
                                        adaptive=adaptive, seconds=frame / 30)
            result = draw_border(result, theme, quad, seconds=frame / 30)
            elapsed.append((perf_counter() - started) * 1000)
        timings[theme] = {"median_ms": float(np.median(elapsed[2:])), "max_ms": max(elapsed),
                          "steady_samples_ms": elapsed[2:]}
        cv2.imwrite(str(output / f"theme-{theme}.png"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(review / f"theme-{theme}.png"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    (review / 'evidence.json').write_text(json.dumps({'created_utc': datetime.now(UTC).isoformat(),
        'schematic_subject_exact_masks': True, 'physical_camera_used': False,
        'capture_and_inference_included': False, 'animated_border': True,
        'timings': timings}, indent=2), encoding='utf-8')
    (output / "synthetic-render-timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
