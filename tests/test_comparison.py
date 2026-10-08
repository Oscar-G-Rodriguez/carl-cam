from dataclasses import replace
from types import SimpleNamespace

import pytest

from carl_cam.comparison import (
    PAIRED_POSES,
    ClassifierResult,
    ComparisonExercise,
    PairedObservation,
    canned_result,
    summarize,
)
from carl_cam.storage import LABELS, Store


def result(pose, *, confidence=.91):
    if pose == 'no_hand':
        return ClassifierResult(None, pose, False, 1.)
    scores = [0.] * 6
    scores[LABELS.index(pose)] = confidence
    return ClassifierResult(tuple(scores), pose, True, 1.)


class Source:
    def __init__(self, exercise):
        self.exercise, self.id, self.time = exercise, 0, 1_000_000_000

    def send(self, pose, *, step=50_000_000, confidence=.91):
        self.id += 1
        self.time += step
        pair = PairedObservation(self.id, self.time, self.time + 20_000_000,
                                 result(pose, confidence=confidence), result(pose, confidence=confidence))
        self.exercise.observe(pair, now_ns=pair.completed_time_ns)
        return pair

    def release(self):
        for _ in range(3):
            self.send('no_hand')
        assert self.exercise.ready

    def portion(self):
        for pose in PAIRED_POSES:
            for _ in range(5):
                self.release()
                self.exercise.start_trial(pose, self.time + 1)
                for _ in range(9):
                    self.send(pose)
                self.exercise.finish_trial()
        self.exercise.start_negative()
        while self.exercise.negative_seconds < 60:
            self.send('other', confidence=.99)
        self.exercise.finish_portion()


def test_thresholds_freeze_before_untouched_portion_and_no_model_is_mutated(tmp_path):
    exercise = ComparisonExercise()
    source = Source(exercise)
    source.portion()
    assert exercise.phase == 'test'
    assert exercise.thresholds == {'neural': (.9, .15), 'canned': (.9, 0.)}
    assert not exercise.rows['test'] and exercise.reports is None
    frozen = dict(exercise.thresholds)
    with pytest.raises(ValueError, match='Complete'):
        exercise.save(Store(tmp_path / 'CarlCam'), user_consented=True)
    source.portion()
    assert exercise.phase == 'complete' and exercise.thresholds == frozen
    for report in exercise.reports.values():
        assert set(report['successes'].values()) == {5}
        assert report['paired_accidental_commands'] == report['wrong_actions'] == 0
        assert report['p95_delay_ms'] >= 250
    store = Store(tmp_path / 'CarlCam')
    with pytest.raises(ValueError, match='consent'):
        exercise.save(store, user_consented=False)
    assert not store.root.exists()
    path = exercise.save(store, user_consented=True)
    assert path.is_file() and not store.models.exists() and not store.settings_path.exists()
    text = path.read_text()
    assert 'landmarks' not in text and 'rgb' not in text and 'features' not in text
    with pytest.raises(ValueError, match='already'):
        exercise.save(store, user_consented=True)


@pytest.mark.parametrize('category,outcome', [('None', 'none'), ('Unknown', 'unknown'),
    ('Closed_Fist', 'unsupported'), ('Victory', 'peace'), ('ILoveYou', 'ily')])
def test_finite_canned_outcomes_are_not_mislabeled_as_correct_other(category, outcome):
    detected = SimpleNamespace(hand_landmarks=[object()],
        gestures=[[SimpleNamespace(category_name=category, score=.9)]])
    parsed = canned_result(detected, 2.)
    assert parsed.outcome == outcome and parsed.has_hand
    no_hand = canned_result(SimpleNamespace(hand_landmarks=[]), 2.)
    assert no_hand.outcome == 'no_hand' and no_hand.scores is None


def test_repeated_stale_future_or_invalid_frames_do_not_advance_time():
    exercise = ComparisonExercise()
    source = Source(exercise)
    source.release()
    exercise.start_trial('peace', source.time + 1)
    pair = source.send('peace')
    assert len(exercise.rows['validation']) == 1
    assert not exercise.observe(pair, now_ns=pair.completed_time_ns)
    for invalid in (replace(pair, frame_id=10, completed_time_ns=pair.capture_time_ns + 81_000_000),
                    replace(pair, frame_id=10, neural=replace(pair.neural, scores=(float('nan'),) * 6)),
                    replace(pair, frame_id=10, capture_time_ns=True)):
        assert not exercise.observe(invalid, now_ns=pair.completed_time_ns)
    assert not exercise.observe(replace(pair, frame_id=10), now_ns=pair.capture_time_ns - 1)
    assert len(exercise.rows['validation']) == 1
    exercise.finish_trial()
    with pytest.raises(ValueError, match='Release'):
        exercise.start_trial('peace', source.time + 1)


def test_frame_counts_are_separate_from_four_pose_negative_metric():
    pair = PairedObservation(1, 1_000_000_000, 1_010_000_000, result('frame'), result('other'), True)
    from dataclasses import asdict
    rows = []
    for index in range(9):
        row = asdict(replace(pair, frame_id=index, capture_time_ns=1_000_000_000 + index * 50_000_000,
                            completed_time_ns=1_010_000_000 + index * 50_000_000))
        row.update(truth='other', trial='other-0', formed_time_ns=0)
        rows.append(row)
    neural = summarize(rows, 'neural', .8, .15, 300)
    canned = summarize(rows, 'canned', .5, 0., 300)
    assert neural['frame_commands_separate'] == 1
    assert neural['paired_accidental_commands'] == 0
    assert canned['frame_commands_separate'] == 0


def test_stale_camera_cannot_reuse_an_old_hands_released_marker():
    exercise = ComparisonExercise()
    source = Source(exercise)
    source.release()
    exercise.pause()
    assert not exercise.ready
    with pytest.raises(ValueError, match='Release'):
        exercise.start_trial('peace', source.time + 1)
    source.release()
    pair = source.send('no_hand')
    assert not exercise.observe(replace(pair, frame_id=pair.frame_id + 1),
                                now_ns=pair.capture_time_ns + 90_000_000)
    assert not exercise.ready
