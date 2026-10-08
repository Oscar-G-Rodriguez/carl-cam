from types import SimpleNamespace

import numpy as np
import pytest

from carl_cam.evaluation import DwellMonitor, Exercise, ExerciseReport, evaluate_sealed
from carl_cam.features import FeatureResult
from carl_cam.storage import LABELS


class Candidate:
    width = 112
    threshold = 0.8
    margin = 0.15

    def __init__(self):
        self.session_ids = {"sealed": tuple(f"{label}-4" for label in LABELS)}

    def scores(self, values):
        scores = np.full((1, 6), 0.01)
        scores[0, int(values[0])] = 0.95
        return scores


def feature(label):
    vector = np.zeros(112, dtype=np.float32)
    vector[0] = label
    return FeatureResult({112: vector}, (SimpleNamespace(),), True)


def scores(label):
    values = [0.01] * 6
    values[label] = 0.95
    return values


def test_dwell_counts_one_command_until_real_release_and_rejects_tracking_gaps():
    monitor = DwellMonitor()
    fired = [monitor.update(scores(2), i * 0.05, has_hand=True, valid_frame=True) for i in range(21)]
    assert fired.count("thumbs_up") == 1
    for time in (1.1, 1.15, 1.21):
        monitor.update(scores(5), time, has_hand=True, valid_frame=True)
    assert monitor.update(scores(2), 1.25, has_hand=True, valid_frame=True) is None
    assert monitor.update(scores(2), 2, has_hand=True, valid_frame=True) is None
    for time in (2.05, 2.1, 2.15, 2.2, 2.25):
        assert monitor.update(scores(2), time, has_hand=True, valid_frame=True) is None
    assert monitor.update(scores(2), 2.3, has_hand=True, valid_frame=True) == "thumbs_up"


def test_frame_score_without_valid_geometry_is_not_a_success():
    monitor = DwellMonitor()
    assert all(monitor.update(scores(0), i * 0.05, has_hand=True, valid_frame=False) is None for i in range(20))


def test_complete_commands_disabled_exercise_and_negative_camera_gaps():
    exercise = Exercise((Candidate(),), attempts_per_pose=5)
    time = 1.0
    for label, pose in enumerate(LABELS[:5]):
        for _ in range(5):
            exercise.start_trial(pose, time)
            for offset in range(10):
                now = time + offset * 0.05
                exercise.observe(feature(label), now, now, valid_frame=True)
            exercise.finish_trial()
            time += 1
    exercise.start_negative()
    exercise.observe(feature(5), time, time, valid_frame=False)
    exercise.observe(feature(5), time + 5, time + 5, valid_frame=False, camera_live=False)
    assert exercise.negative_seconds == 0
    time += 6
    for offset in range(2403):
        now = time + offset * 0.05
        exercise.observe(feature(5), now, now, valid_frame=False)
    report, = exercise.reports()
    assert report.passed and report.pose_successes == (5,) * 5
    assert 250 <= report.p95_delay_ms <= 351
    assert exercise.action_results()[0].unintended_commands == 0


def test_trial_marker_rejects_earlier_measurements_and_wrong_first_action():
    exercise = Exercise((Candidate(),), attempts_per_pose=5)
    exercise.start_trial("peace", 2)
    exercise.observe(feature(1), 1.99, 2, valid_frame=False)
    for index in range(10):
        now = 2 + index * 0.05
        exercise.observe(feature(2), now, now, valid_frame=False)
    exercise.finish_trial()
    assert exercise.reports()[0].pose_successes[1] == 0
    assert exercise.reports()[0].wrong_actions == 1
    with pytest.raises(ValueError, match="incomplete"):
        exercise.action_results()


def sealed_rows():
    return [{"label": label, "round_index": 4, "session_id": f"{label}-4", "time_seconds": index * 0.2,
             "features": {"112": [label_index] + [0] * 111},
             "quality": {"valid_frame": True, "two_hand_stable": True}}
            for label_index, label in enumerate(LABELS) for index in range(5)]


def test_sealed_report_requires_each_pose_and_detects_negative_dwell():
    rows = sealed_rows()
    report = evaluate_sealed(Candidate(), rows)
    assert report.passed and report.recalls == (1,) * 6
    assert len(report.confusion[0]) == 7
    for row in rows:
        if row["label"] == "other":
            row["features"]["112"][0] = 1
    report = evaluate_sealed(Candidate(), rows)
    assert not report.passed and report.false_commands == 1
    with pytest.raises(ValueError, match="recordings"):
        evaluate_sealed(Candidate(), rows[:-5])


def test_activation_report_cannot_pass_with_nonfinite_or_impossible_receipt_values():
    from dataclasses import replace
    report = ExerciseReport((5,) * 5, 5, 0, 120, (300,), 0, 0)
    assert report.passed
    for change in ({"pose_successes": (999,) * 5}, {"negative_seconds": float("inf")},
                   {"pose_successes": (True,) * 5}, {"delays_ms": (float("nan"),)},
                   {"false_commands": False}, {"wrong_actions": -1}):
        assert not replace(report, **change).passed
