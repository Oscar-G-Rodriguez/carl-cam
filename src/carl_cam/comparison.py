"""Opt-in paired numerical diagnostic; never dispatches or trains a model.

Both classifiers see one immutable camera packet. Validation chooses independent
score thresholds; the untouched portion uses those frozen values. Carl-Cam's
two-hand frame is counted separately because the canned vocabulary has no peer.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from uuid import uuid4

import numpy as np

from carl_cam.evaluation import DwellMonitor
from carl_cam.interaction import winning_pose
from carl_cam.storage import LABELS, atomic_json

PAIRED_POSES = ('peace', 'thumbs_up', 'open_palm', 'ily')
CANNED_MAP = {'Victory': 'peace', 'Thumb_Up': 'thumbs_up',
              'Open_Palm': 'open_palm', 'ILoveYou': 'ily'}
OUTCOMES = (*LABELS, 'none', 'unknown', 'unsupported', 'no_hand', 'unusable', 'abstain')


@dataclass(frozen=True, slots=True)
class ClassifierResult:
    scores: tuple[float, ...] | None
    outcome: str
    has_hand: bool
    inference_ms: float
    completed_time_ns: int = 0


@dataclass(frozen=True, slots=True)
class PairedObservation:
    frame_id: int
    capture_time_ns: int
    completed_time_ns: int
    neural: ClassifierResult
    canned: ClassifierResult
    valid_frame: bool = False

    @property
    def fresh(self):
        return (type(self.capture_time_ns) is int and type(self.completed_time_ns) is int
                and self.capture_time_ns >= 0
                and 0 <= self.completed_time_ns - self.capture_time_ns <= 80_000_000)


def canned_result(result, inference_ms):
    """Keep None, Unknown and no-hand distinct; do not call them correct other."""
    if not result.hand_landmarks:
        return ClassifierResult(None, 'no_hand', False, inference_ms)
    if len(result.hand_landmarks) != 1:
        return ClassifierResult(None, 'unusable', True, inference_ms)
    categories = result.gestures[0] if result.gestures else ()
    if not categories:
        return ClassifierResult(None, 'unknown', True, inference_ms)
    category = max(categories, key=lambda value: value.score)
    score = float(category.score)
    if not isfinite(score) or not 0 <= score <= 1:
        return ClassifierResult(None, 'unusable', True, inference_ms)
    name = category.category_name
    outcome = CANNED_MAP.get(name, 'none' if name == 'None' else 'unknown' if name == 'Unknown' else 'unsupported')
    scores = [0.] * 6
    # Nonpaired categories may establish release when confidently observed,
    # but remain named outcomes in confusion/abstention counts.
    scores[LABELS.index(outcome) if outcome in PAIRED_POSES else 5] = score
    return ClassifierResult(tuple(scores), outcome, True, inference_ms)


def _valid_result(result):
    return (isinstance(result, ClassifierResult) and result.outcome in OUTCOMES
            and type(result.has_hand) is bool and type(result.inference_ms) in (float, int)
            and isfinite(result.inference_ms) and result.inference_ms >= 0
            and type(result.completed_time_ns) is int and result.completed_time_ns >= 0
            and (result.scores is None or isinstance(result.scores, tuple)
                 and len(result.scores) == 6 and all(type(value) in (float, int)
                    and isfinite(value) and 0 <= value <= 1 for value in result.scores)))


def summarize(rows, system, threshold, margin, hold_ms):
    """Replay one portion without inspecting any other portion or live model."""
    monitors = {}
    successes = dict.fromkeys(PAIRED_POSES, 0)
    wrong = false = frame_commands = 0
    delays = []
    confusion = {pose: dict.fromkeys(OUTCOMES, 0) for pose in (*PAIRED_POSES, 'other')}
    missing = dict.fromkeys(('no_hand', 'unusable', 'unknown', 'none'), 0)
    resolved = set()
    for row in rows:
        result = row[system]
        truth = row['truth']
        outcome = result['outcome']
        scores = result['scores']
        predicted = winning_pose(scores, threshold=threshold, margin=margin) if result['has_hand'] else None
        named = predicted if predicted else outcome if outcome not in PAIRED_POSES else 'abstain'
        confusion[truth][named] += 1
        if outcome in missing:
            missing[outcome] += 1
        trial = row['trial']
        monitor = monitors.setdefault(trial, DwellMonitor(hold_ms=hold_ms, threshold=threshold, margin=margin))
        command = monitor.update(scores, row['capture_time_ns'] / 1e9,
                                 has_hand=result['has_hand'], valid_frame=row['valid_frame'])
        if command == 'frame':
            frame_commands += 1
        if command is None:
            continue
        if truth == 'other':
            if command in PAIRED_POSES:
                false += 1
        elif trial not in resolved:
            resolved.add(trial)
            if command == truth:
                successes[truth] += 1
                completed = result['completed_time_ns'] or row['completed_time_ns']
                delays.append(max(0., (completed - row['formed_time_ns']) / 1e6))
            else:
                wrong += 1
    return {'successes': successes, 'attempts_per_pose': 5,
            'misses': {pose: 5 - count for pose, count in successes.items()},
            'wrong_actions': wrong, 'paired_accidental_commands': false,
            'frame_commands_separate': frame_commands, 'delays_ms': delays,
            'p95_delay_ms': float(np.percentile(delays, 95)) if delays else None,
            'confusion': confusion, 'missing_outcomes': missing, 'score_threshold': threshold,
            'margin': margin, 'mean_inference_ms': float(np.mean([row[system]['inference_ms'] for row in rows])) if rows else None}


class ComparisonExercise:
    """Twenty marked attempts and sixty usable negative seconds per portion."""
    def __init__(self, *, threshold=.8, margin=.15, hold_ms=300):
        if not .8 <= threshold <= 1 or not 0 <= margin <= 1 or hold_ms not in (300, 400, 600):
            raise ValueError('Unsupported comparison policy')
        self.session_id = uuid4().hex
        self.phase = 'validation'
        self.hold_ms = hold_ms
        self.neural_floor, self.neural_margin = threshold, margin
        self.thresholds = None
        self.rows = {'validation': [], 'test': []}
        self.attempts = dict.fromkeys(PAIRED_POSES, 0)
        self.negative_seconds = 0.
        self.trial = None
        self.negative = False
        self._last_frame = -1
        self._last_negative = None
        self._released_since = None
        self.ready = False
        self.stale_frames = 0
        self.reports = None

    def start_trial(self, pose, formed_time_ns):
        if (self.phase not in self.rows or self.trial or self.negative or not self.ready
                or pose not in PAIRED_POSES or self.attempts[pose] >= 5
                or type(formed_time_ns) is not int or formed_time_ns < 0):
            raise ValueError('Release both hands from view before the next marked attempt')
        self.trial = (pose, self.attempts[pose], formed_time_ns)
        self.ready = False
        self._released_since = None

    def finish_trial(self):
        if not self.trial:
            raise ValueError('No active comparison attempt')
        self.attempts[self.trial[0]] += 1
        self.trial = None
        self.ready = False
        self._released_since = None

    def start_negative(self):
        if self.trial or self.phase not in self.rows or any(value != 5 for value in self.attempts.values()):
            raise ValueError('Complete five marked attempts for every paired pose first')
        self.negative = True
        self._last_negative = None

    def observe(self, pair, *, now_ns, camera_live=True):
        if self.phase not in self.rows:
            return False
        if (not isinstance(pair, PairedObservation) or camera_live is not True or not pair.fresh
                or type(now_ns) is not int or now_ns < pair.completed_time_ns
                or not 0 <= now_ns - pair.capture_time_ns <= 80_000_000):
            self.stale_frames += 1
            self.pause()
            return False
        if (type(pair.frame_id) is not int or pair.frame_id <= self._last_frame
                or type(pair.valid_frame) is not bool
                or not _valid_result(pair.neural) or not _valid_result(pair.canned)
                or any(value.completed_time_ns and not pair.capture_time_ns <= value.completed_time_ns <= pair.completed_time_ns
                       for value in (pair.neural, pair.canned))):
            self.pause()
            return False
        self._last_frame = pair.frame_id
        source = pair.capture_time_ns / 1e9
        if self.trial is None and not self.negative:
            if not pair.neural.has_hand and not pair.canned.has_hand:
                if self._released_since is None or source - self._released_since > .2:
                    self._released_since = source
                if source - self._released_since >= .1 - 1e-9:
                    self.ready = True
            else:
                self._released_since = None
            return False
        if self.trial and pair.capture_time_ns < self.trial[2]:
            return False
        rows = self.rows[self.phase]
        if len(rows) >= 20000:
            raise ValueError('This diagnostic reached its numerical-record limit; start a new exercise')
        if self.negative:
            if self._last_negative is not None and 0 < source - self._last_negative <= .1:
                self.negative_seconds += source - self._last_negative
            self._last_negative = source
        pose, index, formed = self.trial or ('other', 0, pair.capture_time_ns)
        row = asdict(pair)
        row.update(truth=pose, trial=f'{pose}-{index}', formed_time_ns=formed)
        rows.append(row)
        return True

    def pause(self):
        self._last_negative = self._released_since = None
        self.ready = False

    def finish_portion(self):
        if self.trial or not self.negative or self.negative_seconds < 60:
            raise ValueError('Complete all attempts and sixty usable ordinary-movement seconds')
        if self.phase == 'validation':
            chosen = {}
            for system, thresholds, margin in (
                ('neural', sorted({self.neural_floor, *[value for value in (.8, .85, .9, .95) if value >= self.neural_floor]}), self.neural_margin),
                ('canned', (.5, .6, .7, .8, .9, .95), 0.)):
                candidates = [summarize(self.rows['validation'], system, value, margin, self.hold_ms) for value in thresholds]
                best = min(candidates, key=lambda report: (
                    report['paired_accidental_commands'], -min(report['successes'].values()),
                    -sum(report['successes'].values()),
                    report['p95_delay_ms'] if report['p95_delay_ms'] is not None else float('inf'),
                    -report['score_threshold']))
                chosen[system] = (best['score_threshold'], margin)
            self.thresholds = chosen
            self.phase = 'test'
            self.attempts = dict.fromkeys(PAIRED_POSES, 0)
            self.negative_seconds = 0.
            self.negative = self.ready = False
            self._last_negative = self._released_since = None
        elif self.phase == 'test':
            self.reports = {system: summarize(self.rows['test'], system, *policy, self.hold_ms)
                            for system, policy in self.thresholds.items()}
            self.phase = 'complete'
            self.negative = False
        else:
            raise ValueError('No active comparison portion')

    def save(self, store, *, user_consented):
        if user_consented is not True or self.phase != 'complete':
            raise ValueError('Complete the diagnostic and explicitly consent before saving')
        root = store.datasets / 'comparison-results'
        if any(path.is_symlink() for path in (store.root, store.datasets, root)):
            raise ValueError('Comparison storage points outside Carl-Cam')
        path = root / f'{self.session_id}.json'
        if path.exists():
            raise ValueError('This diagnostic has already been saved')
        atomic_json(path, {'schema_version': 1, 'kind': 'preliminary-mediapipe-comparison',
                          'session_id': self.session_id, 'numerical_data_consent': True,
                          'thresholds': self.thresholds, 'hold_ms': self.hold_ms,
                          'validation_and_test': self.rows, 'reports': self.reports,
                          'stale_frames': self.stale_frames, 'public_release_approved': False})
        return path
