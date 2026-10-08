from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from test_features import hand

from carl_cam.collection import RoundPrompt
from carl_cam.evaluation import ExerciseReport
from carl_cam.features import FeatureExtractor
from carl_cam.storage import LABELS, Settings, Store
from carl_cam.training_flow import TrainingFlow, read_resume_manifest, resumable_recordings

ASSIGNMENTS = {"thumbs_up": "red", "open_palm": "yellow", "ily": "blue"}


def test_assignment_is_required_before_a_dataset_is_created(tmp_path):
    store = Store(tmp_path / "CarlCam")
    with pytest.raises(ValueError):
        TrainingFlow(store, Settings(), assignments={})
    assert not store.datasets.exists()


def test_completed_rounds_are_selected_but_interrupted_examples_are_only_retained(tmp_path):
    store = Store(tmp_path / "CarlCam")
    flow = TrainingFlow(store, Settings(), assignments=ASSIGNMENTS)
    flow.plan = [RoundPrompt("peace", 0, ("center", "shift", "angle"), 12)]
    flow.start_round(1)
    result = FeatureExtractor().extract((hand(),))
    assert not flow.observe(result, 0.99, 1, valid_frame=False)
    assert flow.observe(result, 1, 1, valid_frame=False)
    flow.interrupt_round()
    assert len(store.load_dataset(flow.dataset_id)) == 1 and flow.rows() == []
    flow.start_round(2)
    now = 2
    for sample in range(70):
        varied = FeatureExtractor().extract((hand(offset=(300 + (sample % 2) * 4, 350)),))
        flow.observe(varied, now, now, valid_frame=False)
        now += 0.2
        if flow.current.complete:
            break
    assert flow.current.complete
    flow.finish_round()
    assert flow.phase == "review" and len(flow.rows()) >= 60
    assert flow.rows()[0]["session_id"] != store.load_dataset(flow.dataset_id)[0]["session_id"] or len(store.load_dataset(flow.dataset_id)) > len(flow.rows())


def test_lighting_skip_is_explicit_and_recorded(tmp_path):
    flow = TrainingFlow(Store(tmp_path / "CarlCam"), Settings(), assignments=ASSIGNMENTS)
    flow.plan[0] = replace(flow.prompt, variations=("angle", "distance", "lighting"))
    flow.skip_lighting()
    assert flow.lighting_skipped and "lighting" not in flow.prompt.variations


def test_depth_retry_supplements_fit_partitions_and_never_reveals_sealed(tmp_path):
    flow = TrainingFlow(Store(tmp_path / "CarlCam"), Settings(), assignments=ASSIGNMENTS)
    flow.phase = "review"
    flow.selected_sessions = ["old-training", "untouched-sealed"]
    flow.retry_depth_rounds()
    assert flow.phase == "collection" and len(flow.plan) == 12
    assert {prompt.round_index for prompt in flow.plan} == {0, 3}
    assert flow.selected_sessions == ["old-training", "untouched-sealed"]
    assert not flow.sealed_revealed


def test_failure_collects_only_weak_cases_and_keeps_previous_settings(tmp_path):
    flow = TrainingFlow(Store(tmp_path / "CarlCam"), Settings(active_model_version="previous"), assignments=ASSIGNMENTS)
    flow.phase = "needs_examples"
    flow.action_reports = (ExerciseReport((5, 5, 3, 5, 5), 5, 0, 120, (300,), 0, 0),)
    flow.retry_weak_cases()
    assert [prompt.label for prompt in flow.plan] == ["thumbs_up"]
    assert flow.phase == "collection" and flow.settings.active_model_version == "previous"
    with pytest.raises(ValueError, match="checks"):
        flow.activate(user_confirmed=True)


def test_revealed_check_requires_fresh_final_sessions_on_retry(tmp_path):
    flow = TrainingFlow(Store(tmp_path / "CarlCam"), Settings(), assignments=ASSIGNMENTS)
    flow.phase = "needs_examples"
    flow.sealed_revealed = True
    flow.revealed_checks = [{"model_version": "old", "session_ids": ["revealed"]}]
    flow.rehearsal_report = ExerciseReport((10, 10, 7, 10, 10), 10, 0, 120, (300,), 0, 0)
    flow.retry_weak_cases()
    assert {prompt.label for prompt in flow.plan if prompt.round_index == 4} == set(LABELS)
    assert flow.revealed_checks[0]["session_ids"] == ["revealed"]
    assert not flow.sealed_revealed


def test_resume_keeps_completed_round_and_restarts_only_interrupted_round(tmp_path):
    store = Store(tmp_path / "CarlCam")
    flow = TrainingFlow(store, Settings(), assignments=ASSIGNMENTS)
    flow.plan = [RoundPrompt("peace", 0, ("center", "shift", "angle"), 12),
                 RoundPrompt("peace", 1, ("shift", "angle", "distance"), 12)]
    result = FeatureExtractor().extract((hand(),))
    flow.start_round(1)
    now = 1.
    sample = 0
    while not flow.current.complete:
        varied = FeatureExtractor().extract((hand(offset=(300 + (sample % 2) * 4, 350)),))
        flow.observe(varied, now, now, valid_frame=False)
        now += .2
        sample += 1
    first_session = flow.current.session_id
    flow.finish_round()
    flow.start_round(now)
    flow.observe(result, now, now, valid_frame=False)
    interrupted = flow.current.session_id
    assert flow.stop()
    assert resumable_recordings(store)[0][0] == flow.dataset_id
    resumed = TrainingFlow(store, Settings(), assignments=ASSIGNMENTS, resume_id=flow.dataset_id)
    assert resumed.index == 1 and resumed.prompt.round_index == 1 and resumed.current is None
    assert resumed.selected_sessions == [first_session]
    assert interrupted not in {row["session_id"] for row in resumed.rows()}
    assert interrupted in {row["session_id"] for row in store.load_dataset(flow.dataset_id)}


def test_resume_after_revealed_test_demands_fresh_sealed_sessions(tmp_path):
    store = Store(tmp_path / "CarlCam")
    flow = TrainingFlow(store, Settings(), assignments=ASSIGNMENTS)
    flow.index = len(flow.plan)
    flow.phase = "rehearsal"
    flow.sealed_revealed = True
    flow.revealed_checks = [{"model_version": "fixedwinner", "session_ids": ["oldsealed"]}]
    flow._save_manifest()
    resumed = TrainingFlow(store, Settings(), assignments=ASSIGNMENTS, resume_id=flow.dataset_id)
    assert resumed.phase == "collection" and resumed.index == 0
    assert sum(prompt.round_index == 4 for prompt in resumed.plan) == 6
    assert resumed.revealed_checks == flow.revealed_checks
    assert not resumed.sealed_revealed and resumed.settings.active_model_kind == "none"


def test_resume_rejects_damaged_future_manifest_without_overwriting(tmp_path):
    store = Store(tmp_path / "CarlCam")
    flow = TrainingFlow(store, Settings(), assignments=ASSIGNMENTS)
    path = store.datasets / flow.dataset_id / "dataset-manifest.json"
    original = path.read_text()
    path.write_text(original.replace('"schema_version":2', '"schema_version":999'))
    changed = path.read_bytes()
    assert resumable_recordings(store) == ()
    with pytest.raises(ValueError, match="unsupported"):
        TrainingFlow(store, Settings(), assignments=ASSIGNMENTS, resume_id=flow.dataset_id)
    assert path.read_bytes() == changed


def test_unhashable_manifest_values_are_a_recoverable_validation_error(tmp_path):
    import json

    store = Store(tmp_path / "CarlCam")
    flow = TrainingFlow(store, Settings(), assignments=ASSIGNMENTS)
    path = store.datasets / flow.dataset_id / "dataset-manifest.json"
    content = json.loads(path.read_text())
    content["plan"][0]["variations"][0] = []
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="prompt"):
        TrainingFlow(store, Settings(), assignments=ASSIGNMENTS, resume_id=flow.dataset_id)
    assert resumable_recordings(store) == ()


def test_resume_preserves_lighting_skip_and_original_pose_mapping(tmp_path):
    store = Store(tmp_path / "CarlCam")
    flow = TrainingFlow(store, Settings(), assignments=ASSIGNMENTS)
    flow.plan[0] = replace(flow.prompt, variations=("angle", "distance", "lighting"))
    flow.skip_lighting()
    different = {"thumbs_up": "yellow", "open_palm": "red", "ily": "blue"}
    resumed = TrainingFlow(store, Settings(), assignments=different, resume_id=flow.dataset_id)
    assert resumed.lighting_skipped and "lighting" not in resumed.prompt.variations
    assert resumed.settings.assignments == ASSIGNMENTS


@pytest.mark.parametrize('failure_stage', ['load', 'evaluation', 'report'])
def test_failed_final_check_is_consumed_and_persisted_as_recoverable(
    tmp_path, monkeypatch, failure_stage,
):
    from carl_cam.evaluation import SealedReport
    from carl_cam.model_store import ModelStore

    flow = TrainingFlow(Store(tmp_path / 'CarlCam'), Settings(active_model_version='previous'),
                        assignments=ASSIGNMENTS)
    flow.phase = 'sealed'
    flow.version = 'fixedcandidate'
    flow.winner = SimpleNamespace(session_ids={'sealed': ('untouchedsession',)})
    calls = []

    def fail():
        raise OSError('Fixture final-check read failure')

    def load(*args):
        calls.append('load')
        return fail() if failure_stage == 'load' else flow.winner

    def evaluate(*args, **kwargs):
        calls.append('evaluation')
        return (fail() if failure_stage == 'evaluation'
                else SealedReport((), (1.0,) * 6, (1.0,) * 6, 0, ('untouchedsession',)))

    def save_report():
        calls.append('report')
        if failure_stage == 'report':
            fail()

    monkeypatch.setattr(ModelStore, 'load', load)
    monkeypatch.setattr('carl_cam.training_flow.evaluate_sealed', evaluate)
    monkeypatch.setattr(flow, '_save_report', save_report)
    with pytest.raises(OSError, match='Fixture final-check'):
        flow.check_sealed()
    assert flow.phase == 'needs_examples' and flow.sealed_revealed
    saved = read_resume_manifest(flow.store, flow.dataset_id)
    assert saved['phase'] == 'needs_examples' and saved['sealed_revealed']
    assert saved['revealed_checks'] == [
        {'model_version': 'fixedcandidate', 'session_ids': ['untouchedsession']},
    ]
    assert flow.settings.active_model_version == 'previous' and not flow.store.models.exists()
    with pytest.raises(ValueError, match='only once'):
        flow.check_sealed()
    assert calls.count('load') == 1


def test_zero_timeout_stop_does_not_close_over_a_running_worker(tmp_path):
    from carl_cam.training_flow import FitWorker

    flow = TrainingFlow(Store(tmp_path / 'CarlCam'), Settings(), assignments=ASSIGNMENTS)
    worker = FitWorker([], include_depth=False, use_cuda=False)
    entered, release = Event(), Event()

    def wait_without_training():
        entered.set()
        release.wait(5)

    worker._thread = Thread(target=wait_without_training)
    flow.worker = worker
    before = flow._manifest_path.read_bytes()
    worker._thread.start()
    try:
        assert entered.wait(1) and worker.running
        assert not flow.stop(timeout=0)
        assert worker.cancel.is_set() and worker.running
        assert flow._manifest_path.read_bytes() == before
        release.set()
        worker._thread.join(1)
        assert not worker.running and flow.stop(timeout=0)
        assert flow.current is None and flow.rows() == []
        assert not flow.store.models.exists()
    finally:
        release.set()
        worker.stop(timeout=1)
