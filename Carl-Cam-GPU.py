"""Private inference-helper entry point, launched only by the desktop app."""

from carl_cam.gpu_service import main

if __name__ == '__main__':
    raise SystemExit(main())
