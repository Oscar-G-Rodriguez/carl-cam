"""Commands-disabled exercises and the single sealed recording check.

Live exercises observe the same measurements for every candidate. Timing starts
at an explicit formation marker, never at a classifier's first confident guess.
No filter/menu dispatcher is called by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from carl_cam.features import FeatureResult
from carl_cam.interaction import winning_pose
from carl_cam.storage import LABELS
from carl_cam.training import ActionResult, NeuralCandidate


class DwellMonitor:
    def __init__(self, *, hold_ms: int = 300, threshold: float = 0.80, margin: float = 0.15, maximum_gap: float = 0.100) -> None:
        self.hold_ms, self.threshold, self.margin, self.maximum_gap = hold_ms, threshold, margin, maximum_gap
        self.reset()

    def reset(self) -> None:
        self._pending = None
        self._since = 0.0
        self._last_time = None
        self._needs_release = False
        self._release_since = None

    def update(self, scores, time: float, *, has_hand: bool, valid_frame: bool) -> str | None:
        if not isfinite(time) or self._last_time is not None and time <= self._last_time:
            return None
        if self._last_time is not None and time - self._last_time > self.maximum_gap:
            self._pending = None
            self._release_since = None
        self._last_time = time
        pose = winning_pose(scores, threshold=self.threshold, margin=self.margin) if has_hand else None
        if pose == "frame" and not valid_frame:
            pose = None
        released = not has_hand or (scores is not None and len(scores) == 6 and np.isfinite(scores).all() and scores[5] >= self.threshold)
        if released:
            if self._release_since is None:
                self._release_since = time
            elif time - self._release_since >= 0.100:
                self._needs_release = False
        else:
            self._release_since = None
        if pose is None or self._needs_release:
            self._pending = None
            return None
        if self._pending != pose:
            self._pending, self._since = pose, time
            return None
        dwell = (0.250 if pose == "peace" else 0.300) if self.hold_ms == 300 else self.hold_ms / 1000
        if time - self._since + 1e-9 < dwell:
            return None
        self._pending = None
        self._needs_release = True
        return pose


@dataclass(frozen=True, slots=True)
class ExerciseReport:
    pose_successes: tuple[int, ...]
    attempts_per_pose: int
    false_commands: int
    negative_seconds: float
    delays_ms: tuple[float, ...]
    wrong_actions: int
    no_usable_hand_frames: int

    @property
    def p95_delay_ms(self) -> float:
        return float(np.percentile(self.delays_ms, 95)) if self.delays_ms else float("inf")

    @property
    def median_delay_ms(self) -> float:
        return float(np.median(self.delays_ms)) if self.delays_ms else float("inf")

    @property
    def passed(self) -> bool:
        minimum = 4 if self.attempts_per_pose == 5 else 8 if self.attempts_per_pose == 10 else None
        return (type(self.attempts_per_pose) is int and minimum is not None
                and isinstance(self.pose_successes, (tuple, list)) and len(self.pose_successes) == 5
                and all(type(value) is int and minimum <= value <= self.attempts_per_pose for value in self.pose_successes)
                and type(self.false_commands) is int and self.false_commands == 0
                and type(self.negative_seconds) in (int, float) and isfinite(self.negative_seconds) and self.negative_seconds >= 120
                and isinstance(self.delays_ms, (tuple, list))
                and all(type(value) in (int, float) and isfinite(value) and value >= 0 for value in self.delays_ms)
                and type(self.wrong_actions) is int and self.wrong_actions >= 0
                and type(self.no_usable_hand_frames) is int and self.no_usable_hand_frames >= 0)


class Exercise:
    """A formation-marked trial and continuous two-minute negative exercise."""
    def __init__(self, candidates: tuple[NeuralCandidate, ...], *, attempts_per_pose: int, hold_ms: int = 300) -> None:
        if attempts_per_pose not in (5, 10) or not candidates:
            raise ValueError("Unsupported exercise")
        self.candidates = candidates
        self.attempts_per_pose = attempts_per_pose
        self.monitors = [DwellMonitor(hold_ms=hold_ms, threshold=c.threshold, margin=c.margin) for c in candidates]
        self._successes = [[0] * 5 for _ in candidates]
        self._attempts = [0] * 5
        self._delays = [[] for _ in candidates]
        self._wrong = [0] * len(candidates)
        self._false = [0] * len(candidates)
        self._resolved = [False] * len(candidates)
        self._trial: tuple[str, float] | None = None
        self._negative = False
        self.negative_seconds = 0.0
        self._last_negative_time = None
        self.no_usable_hand_frames = 0

    def start_trial(self, pose: str, formed_at: float) -> None:
        if pose not in LABELS[:5] or not isfinite(formed_at) or self._trial is not None or self._negative:
            raise ValueError("Finish the current attempt before starting another")
        if self._attempts[LABELS.index(pose)] >= self.attempts_per_pose:
            raise ValueError("All attempts for this pose are already recorded")
        self._trial = pose, formed_at
        self._resolved = [False] * len(self.candidates)
        for monitor in self.monitors:
            monitor.reset()

    def finish_trial(self) -> None:
        if self._trial is None:
            raise ValueError("No active attempt")
        self._attempts[LABELS.index(self._trial[0])] += 1
        self._trial = None

    def start_negative(self) -> None:
        if self._trial is not None or any(count != self.attempts_per_pose for count in self._attempts):
            raise ValueError("Complete every deliberate attempt first")
        self._negative = True
        self._last_negative_time = None
        for monitor in self.monitors:
            monitor.reset()

    def observe(self, result: FeatureResult, source_time: float, now: float, *, valid_frame: bool, camera_live: bool = True) -> None:
        if not camera_live or not 0 <= now - source_time <= 0.080:
            self._last_negative_time = None
            return
        if self._trial is not None and source_time < self._trial[1]:
            return
        if self._negative:
            if self._last_negative_time is not None and 0 < source_time - self._last_negative_time <= 0.100:
                self.negative_seconds += source_time - self._last_negative_time
            self._last_negative_time = source_time
        if not result.vectors:
            self.no_usable_hand_frames += 1
        for index, (candidate, monitor) in enumerate(zip(self.candidates, self.monitors, strict=True)):
            vector = result.vectors.get(candidate.width)
            scores = tuple(candidate.scores(vector)[0]) if vector is not None else None
            command = monitor.update(scores, source_time, has_hand=bool(result.hands),
                                     valid_frame=valid_frame and result.two_hand_stable)
            if command is None:
                continue
            if self._negative:
                self._false[index] += 1
            elif self._trial is not None and not self._resolved[index]:
                self._resolved[index] = True
                pose, formed_at = self._trial
                if command == pose:
                    self._successes[index][LABELS.index(pose)] += 1
                    self._delays[index].append(max(0, (source_time - formed_at) * 1000))
                else:
                    self._wrong[index] += 1

    def reports(self) -> tuple[ExerciseReport, ...]:
        return tuple(ExerciseReport(tuple(successes), self.attempts_per_pose, false,
                                    self.negative_seconds, tuple(delays), wrong, self.no_usable_hand_frames)
                     for successes, false, delays, wrong in zip(self._successes, self._false, self._delays, self._wrong, strict=True))

    def action_results(self) -> tuple[ActionResult, ...]:
        if self.attempts_per_pose != 5 or self.negative_seconds < 120 or self._trial is not None:
            raise ValueError("The comparison exercise is incomplete")
        return tuple(ActionResult(candidate, report.pose_successes, report.false_commands, report.p95_delay_ms)
                     for candidate, report in zip(self.candidates, self.reports(), strict=True))


@dataclass(frozen=True, slots=True)
class SealedReport:
    confusion: tuple[tuple[int, ...], ...]  # Six true labels × six predictions plus abstain.
    recalls: tuple[float, ...]
    precision: tuple[float, ...]
    false_commands: int
    session_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return len(self.recalls) == 6 and min(self.recalls[:5]) >= 0.80 and self.false_commands == 0


def evaluate_sealed(candidate: NeuralCandidate, rows: list[dict], *, hold_ms: int = 300) -> SealedReport:
    sealed = [row for row in rows if row.get("round_index") == 4]
    sessions = set(candidate.session_ids["sealed"])
    if {row.get("session_id") for row in sealed} != sessions:
        raise ValueError("The final recordings do not match the frozen model")
    confusion = np.zeros((6, 7), dtype=np.int64)
    false_commands = 0
    for session in sorted(sessions):
        monitor = DwellMonitor(hold_ms=hold_ms, threshold=candidate.threshold, margin=candidate.margin, maximum_gap=0.350)
        for row in sorted((row for row in sealed if row["session_id"] == session), key=lambda row: row["time_seconds"]):
            label = LABELS.index(row["label"])
            values = row["features"].get(str(candidate.width))
            if values is None:
                continue  # Unusable depth is not invented; missing class rows fail below.
            vector = np.asarray(values, dtype=np.float32)
            scores = tuple(candidate.scores(vector)[0])
            order = np.argsort(scores)
            predicted = int(order[-1]) if scores[order[-1]] >= candidate.threshold and scores[order[-1]] - scores[order[-2]] >= candidate.margin else 6
            quality = row["quality"]
            valid_frame = quality["valid_frame"] and quality["two_hand_stable"]
            if predicted == 0 and not valid_frame:
                predicted = 6
            confusion[label, predicted] += 1
            command = monitor.update(scores, row["time_seconds"], has_hand=True, valid_frame=valid_frame)
            if label == 5 and command is not None:
                false_commands += 1
    if np.any(confusion.sum(axis=1) == 0):
        raise ValueError("Every label needs usable final-check recordings")
    recalls = tuple(float(confusion[label, label] / confusion[label].sum()) for label in range(6))
    precision = tuple(float(confusion[label, label] / max(1, confusion[:, label].sum())) for label in range(6))
    return SealedReport(tuple(tuple(int(value) for value in row) for row in confusion), recalls, precision, false_commands, tuple(sorted(sessions)))
