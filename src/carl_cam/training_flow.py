"""The in-app collect → compare → sealed check → rehearse → activate workflow.

Completed and interrupted rounds remain numerical local data. A revealed final
check is permanently marked before evaluation, so a retry cannot tune against
it. Targeted retries keep earlier training examples and replace revealed tests.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from threading import Event, Thread
from uuid import uuid4

from carl_cam import __version__
from carl_cam.channels import LatestValue
from carl_cam.collection import (
    NEGATIVE_VARIATIONS,
    VARIATIONS,
    CollectionRound,
    RoundPrompt,
    collection_plan,
)
from carl_cam.evaluation import Exercise, evaluate_sealed
from carl_cam.features import FeatureResult
from carl_cam.model_store import ModelStore, PersonalGate, fingerprint_dataset
from carl_cam.storage import (
    LABELS,
    Settings,
    Store,
    atomic_json,
    safe_identifier,
    valid_assignments,
)
from carl_cam.training import FitResult, depth_coverage, fit_candidates, select_action_winner


def read_resume_manifest(store: Store, dataset_id: str) -> dict:
    """Read only a bounded owned workflow manifest, never a foreign model."""
    if not safe_identifier(dataset_id):
        raise ValueError("Invalid recording identifier")
    directory = store.datasets / dataset_id
    path = directory / "dataset-manifest.json"
    if directory.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(store.datasets.resolve()):
        raise ValueError("Recording leaves Carl-Cam storage")
    if not path.is_file() or path.stat().st_size > 128 * 1024:
        raise ValueError("This recording cannot be resumed")
    with path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    if (not isinstance(source, dict) or source.get("schema_version") != 2
            or source.get("dataset_id") != dataset_id or source.get("numerical_data_consent") is not True
            or not source.get("assignments") or not valid_assignments(source["assignments"])
            or source.get("phase") not in ("collection", "review", "fitting", "comparison", "sealed", "rehearsal", "ready", "needs_examples")):
        raise ValueError("This recording has an unsupported or completed workflow. Its data was kept.")
    plan = source.get("plan")
    if not isinstance(plan, list) or not 1 <= len(plan) <= 60:
        raise ValueError("Invalid recording plan")
    for prompt in plan:
        if (not isinstance(prompt, dict) or prompt.get("label") not in LABELS
                or type(prompt.get("round_index")) is not int or not 0 <= prompt["round_index"] <= 4
                or prompt.get("target_seconds") != (15 if prompt["label"] == "other" else 12)
                or not isinstance(prompt.get("variations"), list) or len(prompt["variations"]) != 3
                or any(not isinstance(value, str) for value in prompt["variations"])
                or len(set(prompt["variations"])) != 3
                or any(value not in (*VARIATIONS, *NEGATIVE_VARIATIONS) for value in prompt["variations"])):
            raise ValueError("Invalid recording prompt")
    index = source.get("index")
    sessions = source.get("selected_sessions")
    if (type(index) is not int or not 0 <= index <= len(plan)
            or not isinstance(sessions, list) or len(sessions) > 120
            or any(not safe_identifier(value) for value in sessions)
            or len(set(sessions)) != len(sessions)
            or type(source.get("sealed_revealed")) is not bool
            or type(source.get("lighting_skipped")) is not bool
            or not isinstance(source.get("revealed_checks"), list)):
        raise ValueError("Invalid recording progress")
    for check in source["revealed_checks"]:
        if (not isinstance(check, dict) or not safe_identifier(check.get("model_version"))
                or not isinstance(check.get("session_ids"), list)
                or any(not safe_identifier(value) for value in check["session_ids"])):
            raise ValueError("Invalid final-check history")
    return source


def resumable_recordings(store: Store) -> tuple[tuple[str, str], ...]:
    entries = []
    for directory in sorted(store.datasets.glob("*")):
        try:
            source = read_resume_manifest(store, directory.name)
            entries.append((directory.name, f"Recording {directory.name[:8]} — {source['index']}/{len(source['plan'])} rounds in this collection"))
        except (OSError, ValueError, TypeError, KeyError):
            continue  # Unknown or damaged files remain untouched.
    return tuple(entries)


class FitWorker:
    def __init__(self, rows: list[dict], *, include_depth: bool, use_cuda: bool) -> None:
        self.rows, self.include_depth, self.use_cuda = rows, include_depth, use_cuda
        self.progress = LatestValue[str]()
        self.result = LatestValue[FitResult]()
        self.error = LatestValue[str]()
        self.cancel = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        self._thread = Thread(target=self._run, name="CarlCamTraining", daemon=True)
        self._thread.start()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 3) -> bool:
        self.cancel.set()
        if self._thread:
            self._thread.join(timeout)
        return not self.running

    def _run(self) -> None:
        try:
            result = fit_candidates(self.rows, include_depth=self.include_depth, use_cuda=self.use_cuda,
                                    cancel=self.cancel, progress=self.progress.publish)
            if not self.cancel.is_set():
                self.result.publish(result)
        except (ValueError, RuntimeError, OSError, InterruptedError) as error:
            self.error.publish(str(error))


class TrainingFlow:
    def __init__(self, store: Store, settings: Settings, *, assignments: dict[str, str], resume_id: str | None = None) -> None:
        saved = read_resume_manifest(store, resume_id) if resume_id is not None else None
        if saved:
            assignments = saved["assignments"]
        if not assignments or not valid_assignments(assignments):
            raise ValueError("Assign the three direct poses to distinct filters first")
        self.store, self.settings = store, replace(settings, assignments=dict(assignments))
        self.dataset_id = resume_id or uuid4().hex
        self.phase = "collection"
        self.plan = list(collection_plan())
        self.index = 0
        self.current: CollectionRound | None = None
        self.selected_sessions: list[str] = []
        self.lighting_skipped = False
        self.worker: FitWorker | None = None
        self.fitted: FitResult | None = None
        self.exercise: Exercise | None = None
        self.action_reports = None
        self.sealed_report = None
        self.rehearsal_report = None
        self.winner = None
        self.version: str | None = None
        self.sealed_revealed = False
        self.revealed_checks: list[dict] = []
        self._manifest_path = store.datasets / self.dataset_id / "dataset-manifest.json"
        if saved:
            self.plan = [RoundPrompt(prompt["label"], prompt["round_index"], tuple(prompt["variations"]), prompt["target_seconds"]) for prompt in saved["plan"]]
            self.index = saved["index"]
            self.selected_sessions = list(saved["selected_sessions"])
            self.lighting_skipped = saved["lighting_skipped"]
            self.sealed_revealed = saved["sealed_revealed"]
            self.revealed_checks = saved["revealed_checks"]
            available = {row["session_id"] for row in store.load_dataset(self.dataset_id)}
            if not set(self.selected_sessions).issubset(available):
                raise ValueError("A completed recording is missing or damaged. Data was kept; start a new collection.")
            self.phase = "collection" if self.index < len(self.plan) else "review"
            if self.sealed_revealed:
                # Never re-fit against a revealed final check after reopening.
                # Restart the checks with new held-out sessions and fresh extra
                # training examples; retain the old evidence for comparison.
                self.phase = "needs_examples"
                self.retry_weak_cases()
        self._save_manifest()

    def _save_manifest(self) -> None:
        atomic_json(self._manifest_path, {
            "schema_version": 2, "dataset_id": self.dataset_id, "app_version": __version__,
            "numerical_data_consent": True, "updated_utc": datetime.now(UTC).isoformat(),
            "selected_sessions": self.selected_sessions, "lighting_skipped": self.lighting_skipped,
            "sealed_revealed": self.sealed_revealed, "sealed_model_version": self.version if self.sealed_revealed else None,
            "revealed_checks": self.revealed_checks,
            "phase": self.phase, "index": self.index, "plan": [asdict(prompt) for prompt in self.plan],
            "assignments": self.settings.assignments,
        })

    @property
    def prompt(self) -> RoundPrompt | None:
        return self.plan[self.index] if self.index < len(self.plan) else None

    def skip_lighting(self) -> None:
        if self.phase != "collection" or self.current is not None:
            raise ValueError("Choose lighting coverage between rounds")
        if self.prompt and "lighting" in self.prompt.variations:
            other = next(name for name in ("center", "shift", "angle", "distance") if name not in self.prompt.variations)
            variations = tuple(other if name == "lighting" else name for name in self.prompt.variations)
            self.plan[self.index] = replace(self.prompt, variations=variations)
            self.lighting_skipped = True
            self._save_manifest()

    def start_round(self, now: float) -> None:
        if self.phase != "collection" or self.prompt is None or self.current is not None:
            raise ValueError("No recording round is ready")
        self.current = CollectionRound(self.prompt, dataset_id=self.dataset_id, start_seconds=now)

    @property
    def variation(self) -> str | None:
        if self.current is None:
            return None
        target = self.current.prompt.target_seconds / 3
        return next((name for name in self.current.prompt.variations if self.current.variation_seconds(name) < target - 1e-6), self.current.prompt.variations[-1])

    def observe(self, result: FeatureResult, source_time: float, now: float, *, valid_frame: bool) -> bool:
        if self.current is None or source_time < self.current.start_seconds or not 0 <= now - source_time <= 0.100:
            return False
        return self.current.accept(result, source_time, self.variation, valid_frame=valid_frame)

    def finish_round(self) -> None:
        if self.current is None or not self.current.complete:
            raise ValueError("This round needs more usable, varied examples")
        self.store.save_round(self.dataset_id, self.current.session_id, self.current.rows)
        self.selected_sessions.append(self.current.session_id)
        self.current = None
        self.index += 1
        if self.index == len(self.plan):
            self.phase = "review"
        self._save_manifest()

    def interrupt_round(self) -> None:
        if self.current and self.current.rows:
            self.store.save_round(self.dataset_id, self.current.session_id, self.current.rows)
        self.current = None  # Interrupted data is retained, but not silently fitted.
        self._save_manifest()

    def rows(self) -> list[dict]:
        selected = set(self.selected_sessions)
        return [row for row in self.store.load_dataset(self.dataset_id) if row["session_id"] in selected]

    def begin_fit(self, *, include_depth: bool, use_cuda: bool) -> None:
        if self.phase != "review":
            raise ValueError("Review completed recording rounds first")
        rows = self.rows()
        if not rows:
            raise ValueError("There are no completed rounds")
        if include_depth and any(min(counts) < 10 for counts in depth_coverage(rows).values()):
            raise ValueError("Depth needs another collection attempt. Use Retry depth rounds, or explicitly uncheck depth to compare 112/138 measurements.")
        self.worker = FitWorker(rows, include_depth=include_depth, use_cuda=use_cuda)
        self.phase = "fitting"
        self._save_manifest()
        self.worker.start()

    def retry_depth_rounds(self) -> None:
        if self.phase != "review":
            raise ValueError("Review recordings before retrying depth")
        prompts = []
        for label, (training_count, validation_count) in depth_coverage(self.rows()).items():
            for index, count in ((0, training_count), (3, validation_count)):
                if count < 10:
                    variations = NEGATIVE_VARIATIONS[:3] if label == "other" else ("shift", "angle", "distance")
                    prompts.append(RoundPrompt(label, index, variations, 15 if label == "other" else 12))
        if not prompts:
            raise ValueError("Enough fit data has usable depth; no depth retry is needed")
        # Retain old numerical rounds and all untouched final sessions. New
        # sessions supplement only the deficient training/validation portions.
        self.plan, self.index, self.phase = prompts, 0, "collection"
        self._save_manifest()

    def poll_fit(self) -> str | None:
        if self.phase != "fitting" or self.worker is None:
            return None
        result, error = self.worker.result.read(), self.worker.error.read()
        if result:
            self.fitted = result
            self.exercise = Exercise(result.candidates, attempts_per_pose=5, hold_ms=self.settings.hold_ms)
            self.phase = "comparison"
            self._save_manifest()
            return result.backend_summary + " Candidates ready. Complete the commands-disabled comparison."
        if error:
            self.phase = "review"
            self._save_manifest()
            return error
        return self.worker.progress.read()

    def finish_comparison(self) -> bool:
        if self.phase != "comparison" or self.exercise is None:
            raise ValueError("No comparison exercise is active")
        results = self.exercise.action_results()
        self.action_reports = self.exercise.reports()
        self.winner = select_action_winner(results)
        if self.winner is None:
            self.phase = "needs_examples"
            self._save_manifest()
            return False
        rows = self.rows()
        report = {"action_comparison": [asdict(result) for result in self.action_reports],
                  "baselines": [{"width": baseline.width, "validation_recall": baseline.validation_recall} for baseline in self.fitted.baselines],
                  "lighting_skipped": self.lighting_skipped}
        self.version = ModelStore(self.store).stage(self.winner, dataset_fingerprint=fingerprint_dataset(rows), report=report)
        self.phase = "sealed"
        self._save_manifest()
        return True

    def check_sealed(self) -> bool:
        if self.phase != "sealed" or self.sealed_revealed or self.winner is None or self.version is None:
            raise ValueError("The final check can be revealed only once for the fixed winner")
        self.sealed_revealed = True
        self.revealed_checks.append({"model_version": self.version, "session_ids": self.winner.session_ids["sealed"]})
        self._save_manifest()  # Mark before reading any final feature values.
        try:
            selected = ModelStore(self.store).load(self.version)
            self.sealed_report = evaluate_sealed(selected, self.rows(), hold_ms=self.settings.hold_ms)
            self._save_report()
        except (ValueError, OSError, RuntimeError, KeyError, TypeError):
            # A read/evaluation failure still consumes this fixed final check.
            # Persist the recovery route before returning the error to the UI;
            # another click cannot reveal or tune against these sessions again.
            self.phase = "needs_examples"
            self._save_manifest()
            raise
        if not self.sealed_report.passed:
            self.phase = "needs_examples"
            self._save_manifest()
            return False
        self.exercise = Exercise((selected,), attempts_per_pose=10, hold_ms=self.settings.hold_ms)
        self.phase = "rehearsal"
        self._save_manifest()
        return True

    def finish_rehearsal(self) -> bool:
        if self.phase != "rehearsal" or self.exercise is None:
            raise ValueError("No activation rehearsal is active")
        self.rehearsal_report, = self.exercise.reports()
        if self.exercise.negative_seconds < 120:
            raise ValueError("Complete the two-minute negative rehearsal")
        self.phase = "ready" if self.rehearsal_report.passed else "needs_examples"
        self._save_manifest()
        self._save_report()
        return self.rehearsal_report.passed

    def _save_report(self) -> None:
        if self.version:
            atomic_json(self.store.models / self.version / "evaluation.json", {
                "schema_version": 1, "model_version": self.version,
                "sealed": asdict(self.sealed_report) if self.sealed_report else None,
                "rehearsal": asdict(self.rehearsal_report) if self.rehearsal_report else None,
            })

    def activate(self, *, user_confirmed: bool) -> Settings:
        if self.phase != "ready":
            raise ValueError("Finish all checks before activating")
        winner_index = next(index for index, candidate in enumerate(self.fitted.candidates) if candidate is self.winner)
        action = self.action_reports[winner_index]
        rehearsal = self.rehearsal_report
        gate = PersonalGate(action.pose_successes, action.false_commands, action.negative_seconds,
                            self.sealed_report.recalls[:5], self.sealed_report.false_commands,
                            rehearsal.pose_successes, rehearsal.false_commands, rehearsal.negative_seconds)
        self.settings = ModelStore(self.store).activate(self.version, self.settings, gate, user_confirmed=user_confirmed)
        self.phase = "activated"
        self._save_manifest()
        return self.settings

    def retry_weak_cases(self) -> None:
        if self.phase != "needs_examples":
            raise ValueError("A retry needs failed checks")
        weak = set()
        if self.rehearsal_report:
            weak.update(LABELS[index] for index, count in enumerate(self.rehearsal_report.pose_successes) if count < 8)
            if self.rehearsal_report.false_commands:
                weak.add("other")
        elif self.sealed_report:
            weak.update(LABELS[index] for index, recall in enumerate(self.sealed_report.recalls[:5]) if recall < 0.8)
            if self.sealed_report.false_commands:
                weak.add("other")
        elif self.action_reports:
            best = max(self.action_reports, key=lambda report: (min(report.pose_successes), -report.false_commands))
            weak.update(LABELS[index] for index, count in enumerate(best.pose_successes) if count < 4)
            if best.false_commands:
                weak.add("other")
        weak = weak or set(LABELS)
        self.plan = [RoundPrompt(label, 0, ("rest_typing", "touch_face", "ordinary_movement") if label == "other" else ("shift", "angle", "distance"), 15 if label == "other" else 12) for label in LABELS if label in weak]
        if self.sealed_revealed:
            old_rows = self.rows()
            old_sealed = {row["session_id"] for row in old_rows if row["round_index"] == 4}
            self.selected_sessions = [session for session in self.selected_sessions if session not in old_sealed]
            self.plan.extend(prompt for prompt in collection_plan() if prompt.round_index == 4)
        self.index = 0
        self.phase = "collection"
        self.winner = self.version = self.sealed_report = self.rehearsal_report = None
        self.sealed_revealed = False
        self._save_manifest()

    def stop(self, *, timeout: float = 3) -> bool:
        if self.worker and not self.worker.stop(timeout=timeout):
            return False
        self.interrupt_round()
        return True
