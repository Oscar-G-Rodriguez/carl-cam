"""Exercise the application's GPU region worker using generated pixels only.

This uses the same latest-value worker and atomic source/result pairs as the
desktop renderer. It cannot establish camera, rendering, gesture or OBS speed.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from time import monotonic, monotonic_ns, perf_counter, perf_counter_ns, sleep
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from carl_cam.channels import LatestValue
from carl_cam.gpu_process import AdaptiveRegionEngine, development_launch
from carl_cam.gpu_regions import GpuRegionEngine
from carl_cam.records import FramePacket
from carl_cam.vision import VisionWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--isolated-process', action='store_true',
                        help='Use the app GPU process and CPU fallback, not an in-process test')
    args = parser.parse_args()
    output = ROOT / 'build' / ('gpu-worker-' + uuid4().hex)
    output.mkdir(exist_ok=False)
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'physical_camera_used': False, 'virtual_output_used': False,
              'generated_pixels_only': True, 'native_executable_benchmarked': False,
              'live_fps_gate_passed': False, 'runtime_promoted': False,
              'completed': False, 'passed': False, 'checks': {}, 'sources': {}}
    report['isolated_process'] = args.isolated_process
    for name in ('tools/check_gpu_worker.py', 'src/carl_cam/gpu_regions.py',
                 'src/carl_cam/vision.py', 'src/carl_cam/channels.py'):
        report['sources'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    pause = Event()
    frames = LatestValue()
    engine = None
    worker = None
    producer = None
    producer_stop = Event()
    producer_finished = Event()

    def factory():
        nonlocal engine
        if args.isolated_process:
            launch = development_launch()
            if launch is None or launch.candidate != args.candidate.absolute():
                raise ValueError('Select the exact managed GPU candidate')
            engine = AdaptiveRegionEngine('gpu')
            return engine
        engine = GpuRegionEngine(args.candidate.absolute(), experiment=True)
        engine.session.set_providers([('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0})])
        engine.session.disable_fallback()
        if engine.session.get_provider_options()['CUDAExecutionProvider'].get('use_tf32') != '0':
            engine.close()
            raise ValueError('Full FP32 configuration was not retained')
        report['candidate_evidence_sha256'] = engine.candidate.evidence_sha256
        # Prime native convolution algorithms before measuring worker cadence.
        blank = np.zeros((360, 640, 3), np.uint8)
        for index in range(6):
            engine.process(FramePacket(index, monotonic_ns(), blank))
        report['generated_warmup_frames'] = 6
        return engine

    try:
        worker = VisionWorker(frames, regions=True, factory=factory)
        worker.start()
        deadline = monotonic() + 45
        while worker.status.read() != 'Local vision ready' and monotonic() < deadline:
            status = worker.status.read() or ''
            if status.startswith('Vision unavailable:'):
                raise RuntimeError('GPU worker initialization failed')
            pause.wait(.01)
        if worker.status.read() != 'Local vision ready':
            raise TimeoutError('GPU worker initialization timed out')
        if args.isolated_process and not engine.backend_status.startswith('GPU'):
            raise RuntimeError('The experiment fell back to CPU instead of initializing GPU')
        report['backend_status'] = worker.backend_status.read()
        report['checks']['initialized_full_fp32'] = True
        images = [np.zeros((360, 640, 3), np.uint8),
                  np.full((360, 640, 3), 255, np.uint8),
                  np.random.default_rng(17).integers(0, 256, (360, 640, 3), dtype=np.uint8)]
        for image in images:
            image.flags.writeable = False
        delays, completions, observed = [], [], set()
        submitted = {}

        def produce():
            start = perf_counter()
            try:
                for identifier in range(60):
                    remaining = start + identifier / 30 - perf_counter()
                    if remaining > 0:
                        sleep(remaining)
                    if producer_stop.is_set():
                        break
                    packet = FramePacket(identifier, monotonic_ns(), images[identifier % 3])
                    submitted[identifier] = (packet, perf_counter_ns())
                    frames.publish(packet)
            finally:
                producer_finished.set()

        producer = Thread(target=produce, name='GeneratedInput', daemon=True)
        producer.start()
        deadline, drained_at = perf_counter() + 4, None
        while perf_counter() < deadline:
            pair = worker.completed.read()
            if pair is not None and pair[0].frame_id not in observed:
                source, result = pair
                identifier = source.frame_id
                packet, submitted_at = submitted[identifier]
                if (source is not packet or result.frame_id != identifier
                        or result.capture_time_ns != packet.capture_time_ns
                        or result.confidence_masks.shape != (360, 640, 6)
                        or result.confidence_masks.dtype != np.float32
                        or result.confidence_masks.flags.writeable
                        or not np.isfinite(result.confidence_masks).all()):
                    raise ValueError('GPU worker lost its atomic source/result contract')
                finished = perf_counter_ns()
                delays.append((finished - submitted_at) / 1e6)
                completions.append(finished)
                observed.add(identifier)
            if producer_finished.is_set():
                drained_at = drained_at or perf_counter()
                if perf_counter() - drained_at > .25 or len(observed) == 60:
                    break
            sleep(.001)
        producer.join(timeout=1)
        report['checks']['input_producer_stopped'] = not producer.is_alive()
        report['checks']['input_producer_completed'] = len(submitted) == 60 and producer_finished.is_set()
        report.update(measured_frames=len(delays), submitted_frames=len(submitted),
                      unobserved_frame_ids=sorted(set(submitted) - observed),
                      requested_generated_input_hz=30)
        timestamps = [timestamp for _, timestamp in submitted.values()]
        if len(timestamps) > 1:
            report['generated_submission_hz'] = (len(timestamps) - 1) * 1e9 / (
                timestamps[-1] - timestamps[0])
        if delays:
            report['completion_latency_ms'] = {
                'median': float(np.median(delays)), 'p95': float(np.percentile(delays, 95)),
                'max': float(max(delays))}
        if len(completions) > 1:
            report['generated_completion_hz'] = (len(completions) - 1) * 1e9 / (
                completions[-1] - completions[0])
        # The app intentionally replaces old inputs. Report unobserved frames
        # separately; transport correctness must not invent a live-FPS pass.
        report['checks']['atomic_readonly_frame_pairs'] = bool(observed)
        frames.publish(FramePacket(67, monotonic_ns() - 300_000_000, images[0]))
        deadline = monotonic() + .5
        while worker.completed.read() is not None and monotonic() < deadline:
            pause.wait(.005)
        report['checks']['stale_output_cleared'] = all(channel.read() is None for channel in (
            worker.results, worker.source, worker.completed))
        report['completed'] = True
    finally:
        producer_stop.set()
        if producer is not None:
            producer.join(timeout=1)
        report['checks']['worker_stopped'] = worker is None or worker.stop(timeout=3)
        report['checks']['gpu_resources_closed'] = engine is None or (
            engine.engine is None if args.isolated_process else
            engine.session is None and not engine._cuda_libraries)
        report['passed'] = report['completed'] and all(report['checks'].values())
        (output / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Generated worker evidence: ' + str(output), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
