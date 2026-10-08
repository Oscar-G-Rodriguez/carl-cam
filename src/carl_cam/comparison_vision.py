"""Same-frame, alternating-order perception for the private paired diagnostic.

No pixels/landmarks leave this engine's transient input processing. The canned
recognizer is created only when the user opens the optional exercise.
"""

from dataclasses import replace
from time import monotonic_ns, perf_counter

import numpy as np

from carl_cam.comparison import ClassifierResult, PairedObservation, canned_result
from carl_cam.features import FeatureExtractor, observed_quad
from carl_cam.storage import LABELS
from carl_cam.vision import convert_hands, mirrored_canvas, model_path
from carl_cam.vision_privacy import require_verified_vision_runtime


class ComparisonEngine:
    def __init__(self, candidate):
        require_verified_vision_runtime()
        import mediapipe as mp
        self._mp = mp
        self.candidate, self.extractor = candidate, FeatureExtractor()
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(model_path('hand_landmarker.task'))),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
            min_tracking_confidence=.5))
        try:
            self.canned = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
                base_options=base(model_asset_path=str(model_path('gesture_recognizer.task'))),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
                min_tracking_confidence=.5))
        except Exception:
            self.hand.close()
            raise
        self.timestamp_ms = -1

    def _neural(self, image, timestamp):
        start = perf_counter()
        detected = self.hand.detect_for_video(image, timestamp)
        hands = convert_hands(detected)
        features = self.extractor.extract(hands)
        vector = features.vectors.get(self.candidate.width)
        if vector is None:
            scores, outcome = None, 'unusable' if detected.hand_landmarks else 'no_hand'
        else:
            scores = tuple(float(value) for value in self.candidate.scores(vector)[0])
            outcome = LABELS[int(np.argmax(scores))]
        result = ClassifierResult(scores, outcome, bool(detected.hand_landmarks),
                                  (perf_counter() - start) * 1000, monotonic_ns())
        return result, observed_quad(features) is not None

    def _canned(self, image, timestamp):
        start = perf_counter()
        detected = self.canned.recognize_for_video(image, timestamp)
        return replace(canned_result(detected, (perf_counter() - start) * 1000),
                       completed_time_ns=monotonic_ns())

    def process(self, packet):
        mp = self._mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mirrored_canvas(packet))
        timestamp = max(self.timestamp_ms + 1, packet.capture_time_ns // 1_000_000)
        self.timestamp_ms = timestamp
        if packet.frame_id % 2:
            canned = self._canned(image, timestamp)
            neural, valid_frame = self._neural(image, timestamp)
        else:
            neural, valid_frame = self._neural(image, timestamp)
            canned = self._canned(image, timestamp)
        return PairedObservation(packet.frame_id, packet.capture_time_ns, monotonic_ns(),
                                 neural, canned, valid_frame)

    def close(self):
        self.hand.close()
        self.canned.close()
