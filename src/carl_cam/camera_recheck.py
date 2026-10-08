"""Recheck an unchanged tested personal model after changing physical camera.

Uses the Training screen's commands-disabled exercise, without collecting
training rows or changing weights, thresholds or previously recorded tests.
"""

from dataclasses import replace

from carl_cam.evaluation import Exercise
from carl_cam.model_store import ModelStore
from carl_cam.storage import valid_camera_key


class CameraRecheckFlow:
    def __init__(self, store, settings, version):
        if not valid_camera_key(settings.camera_key):
            raise ValueError("Choose and start a physical camera before rechecking")
        self.store, self.settings, self.version = store, settings, version
        self.winner = ModelStore(store).load_tested(version)
        self.phase = "rehearsal"
        self.exercise = Exercise((self.winner,), attempts_per_pose=5, hold_ms=settings.hold_ms)
        self.action_reports = self.sealed_report = self.rehearsal_report = None

    def finish_rehearsal(self):
        self.rehearsal_report, = self.exercise.reports()
        if self.exercise.negative_seconds < 120:
            raise ValueError("Complete the two-minute ordinary-movement recheck")
        self.phase = "ready" if self.rehearsal_report.passed else "needs_examples"
        return self.rehearsal_report.passed

    def retry_weak_cases(self):
        self.exercise = Exercise((self.winner,), attempts_per_pose=5, hold_ms=self.settings.hold_ms)
        self.rehearsal_report = None
        self.phase = "rehearsal"

    def activate(self, *, user_confirmed):
        if self.phase != "ready":
            raise ValueError("Finish the camera recheck before enabling commands")
        ModelStore(self.store).confirm_camera(self.version, self.settings.camera_key, self.rehearsal_report, user_confirmed=user_confirmed)
        updated = replace(self.settings, active_model_kind="personal", active_model_version=self.version)
        self.store.save_settings(updated)
        self.phase = "activated"
        return updated

    def stop(self, *, timeout=3):
        return True  # No capture/training worker or sample persistence is owned here.
