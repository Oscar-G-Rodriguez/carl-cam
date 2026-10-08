"""Developer-only OpenCV thread-count comparison for the reviewed GPU service."""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from carl_cam.gpu_service import main

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--opencv-threads', type=int, choices=(1, 2, 4, 8), required=True)
arguments, remaining = parser.parse_known_args()
cv2.setNumThreads(arguments.opencv_threads)
sys.argv = [sys.argv[0], *remaining]
raise SystemExit(main())
