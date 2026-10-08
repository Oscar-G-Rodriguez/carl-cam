"""Rasterize the original Figma pointer glyph into Windows icon sizes."""

from pathlib import Path

from PIL import Image

from carl_cam.graphics import ROOT, svg_pixels


def main():
    pixels = svg_pixels((ROOT / "pointer.svg").read_bytes(), 256, 256)
    image = Image.fromarray(pixels)
    target = Path(__file__).resolve().parents[1] / "src/carl_cam/assets/graphics/carl-cam.ico"
    image.save(target, format="ICO", sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)])
    print("Created Windows icon from original Figma vector")


if __name__ == "__main__":
    main()
