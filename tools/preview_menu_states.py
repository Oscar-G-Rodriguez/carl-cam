"""Render genuine offline menu exports for camera-free visual comparison."""

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from carl_cam.graphics import asset_pixels, menu_card, place_asset
from carl_cam.menu import perimeter
from carl_cam.render import draw_menu


def main():
    app = QApplication.instance() or QApplication([])
    root = Path(__file__).resolve().parents[1]
    evidence = root / "build" / ("menu-state-review-" + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    contact = np.full((530, 1370, 3), (16, 18, 27), np.uint8)
    for row, theme in enumerate(("red", "yellow", "blue")):
        for col, state in enumerate(("default", "hover", "focused", "selected")):
            contact = place_asset(contact, asset_pixels(f"menu-{theme}-{state}"),
                                  30 + 335 * col, 42 + 157 * row)
    path = evidence / "all-menu-states.png"
    cv2.imwrite(str(path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
    # The flat field is deliberately not a person's image. It verifies the
    # actual menu renderer and both view orientations without opening a camera.
    field = np.full((720, 1280, 3), (70, 75, 85), np.uint8)
    for name, layout in (("preview", perimeter()), ("output", perimeter().output())):
        rendered = draw_menu(field, layout, selected="yellow", hover="yellow", focus="blue")
        cv2.imwrite(str(evidence / f"{name}-menu.png"), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    stacked = menu_card("yellow", True, interaction="hover")
    cv2.imwrite(str(evidence / "selected-and-hovered.png"),
                cv2.cvtColor(stacked, cv2.COLOR_RGBA2BGRA))
    record = {"physical_camera_used": False, "vision_sdk_loaded": False,
              "assets_manifest": json.loads((root / "src/carl_cam/assets/graphics/manifest.json").read_text()),
              "contact_sheet": str(path), "menu_orientations_rendered": ["preview", "output"],
              "real_head_tracking_or_native_window_verified": False}
    (evidence / "evidence.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"review": str(evidence), "contact_sheet": str(path)}))
    return 0 if app is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
