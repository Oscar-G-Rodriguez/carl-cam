"""Exercise the real asynchronous training worker using generated measurements.

No camera, user's recordings, active model, settings or OBS device is opened.
This is a developer functional check, not a gesture-quality or speed benchmark.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import uuid4

import numpy as np

from carl_cam.storage import LABELS
from carl_cam.training import gpu_training_status
from carl_cam.training_flow import FitWorker


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'build' / ('gpu-training-worker-' + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'started_utc': datetime.now(UTC).isoformat(),
              'camera_opened': False, 'personal_recordings_used': False,
              'active_model_changed': False, 'virtual_output_started': False,
              'recognition_quality_verified': False, 'passed': False}
    worker = None
    try:
        if not gpu_training_status()[0]:
            raise RuntimeError('An actual compatible NVIDIA device is required for this check')
        rng = np.random.default_rng(17)
        rows = []
        for label_index, label in enumerate(LABELS):
            for round_index in range(5):
                for _ in range(12):
                    values = rng.normal(0, 0.03, 180).astype(np.float32)
                    values[label_index] += 2
                    rows.append({'label': label, 'round_index': round_index,
                                 'session_id': f'generated-{label}-{round_index}',
                                 'features': {str(width): values[:width].tolist()
                                              for width in (112, 138, 180)}})
        worker = FitWorker(rows, include_depth=True, use_cuda=True)
        started = monotonic()
        worker.start()
        # Finite joins wait for the actual worker; they do not drive a GUI or
        # substitute a fitted result. The ordinary worker uses its 100-pass cap.
        while worker.running and monotonic() - started < 120:
            worker._thread.join(0.25)
        if worker.running:
            raise RuntimeError('Training worker did not finish within the functional-check budget')
        result = worker.result.read()
        if worker.error.read() is not None or result is None:
            raise RuntimeError('Training worker did not return a successful fit')
        assert result.requested_device == 'cuda' and not result.cpu_fallback
        assert len(result.candidates) == 6 and len(result.baselines) == 3
        assert all(candidate.training_device == 'cuda' for candidate in result.candidates)
        assert all(all(parameter.device.type == 'cpu' for parameter in candidate.network.parameters())
                   for candidate in result.candidates)
        assert all(np.isfinite(candidate.scores(np.zeros(candidate.width, np.float32))).all()
                   for candidate in result.candidates)
        report.update(passed=True, actual_gpu_worker_executed=True,
                      cpu_fallback_used=False, all_models_returned_to_cpu=True,
                      worker_finished=True, result_channel_received=True,
                      default_epoch_cap=100,
                      candidates=[{'width': candidate.width, 'hidden': list(candidate.hidden),
                                   'epochs': candidate.epochs, 'fit_device': candidate.training_device}
                                  for candidate in result.candidates],
                      final_progress=worker.progress.read(),
                      elapsed_seconds=monotonic() - started)
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        report['error_type'] = type(error).__name__
    finally:
        report['worker_stopped'] = worker.stop(timeout=3) if worker is not None else True
        report['passed'] = report['passed'] and report['worker_stopped']
        report['finished_utc'] = datetime.now(UTC).isoformat()
        (evidence / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'evidence': str(evidence), 'passed': report['passed']}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
