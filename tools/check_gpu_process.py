"""Verify the isolated GPU process on paired generated CPU/GPU frames only."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import monotonic_ns, perf_counter
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from verify_gpu_parity import compare_masks, fixtures

from carl_cam.dsp import signal_stages
from carl_cam.gpu_process import GpuProcessRegionEngine, LaunchSpec, development_launch
from carl_cam.records import FramePacket
from carl_cam.themes import regions_from_masks
from carl_cam.vision import RegionEngine, mirrored_canvas


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-helper', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'build' / ('gpu-process-check-' + uuid4().hex)
    output.mkdir(exist_ok=False)
    report = {'schema_version': 1, 'created_utc': datetime.now(UTC).isoformat(),
              'physical_camera_used': False, 'virtual_output_used': False,
              'generated_pixels_only': True, 'live_fps_gate_passed': False,
              'native_executable_benchmarked': False, 'runtime_promoted': False,
              'completed': False, 'passed': False, 'fixtures': {}, 'sources': {}}
    for name in ('tools/check_gpu_process.py', 'tools/verify_gpu_parity.py',
                 'src/carl_cam/gpu_ipc.py', 'src/carl_cam/gpu_service.py',
                 'src/carl_cam/gpu_process.py', 'src/carl_cam/gpu_regions.py',
                 'src/carl_cam/vision.py', 'src/carl_cam/camera.py',
                 'src/carl_cam/dsp.py', 'src/carl_cam/themes.py'):
        report['sources'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    gpu = cpu = None
    process_handle = None
    try:
        launch = development_launch()
        if args.native_helper:
            bundle = ROOT / 'deployment/gpu-helper/Carl-Cam-GPU.dist'
            stage_path = bundle / 'gpu-helper-staging.json'
            stage = json.loads(stage_path.read_text(encoding='utf-8'))
            if stage.get('runtime_payload_verified') is not True:
                raise ValueError('GPU helper staging has not passed')
            recorded = set()
            for item in stage['files']:
                path = bundle / item['path']
                if (not path.absolute().is_relative_to(bundle)
                        or '..' in Path(item['path']).parts
                        or any(part.is_symlink() or part.is_junction() for part in (path, *path.parents))
                        or not path.is_file() or path.stat().st_size != item['bytes']
                        or file_digest(path) != item['sha256']):
                    raise ValueError('GPU helper staged file differs')
                recorded.add(path)
            actual = {path for path in bundle.rglob('*') if path.is_file() and path != stage_path}
            if actual != recorded:
                raise ValueError('GPU helper has unrecorded or missing files')
            executable = bundle / 'Carl-Cam-GPU.exe'
            if file_digest(executable) != stage['executable_sha256']:
                raise ValueError('GPU helper executable identity differs')
            launch = LaunchSpec((str(executable),), bundle / 'candidate', bundle)
            report['native_helper_used'] = True
            report['native_helper_sha256'] = stage['executable_sha256']
            report['staging_record_sha256'] = hashlib.sha256(stage_path.read_bytes()).hexdigest()
        if launch is None:
            raise ValueError('Verified development GPU environment is unavailable')
        gpu = GpuProcessRegionEngine(launch)
        started = perf_counter()
        gpu.initialize(Event())
        report['initialization_ms'] = (perf_counter() - started) * 1000
        process_handle = gpu.process_handle
        report['backend'] = gpu.backend_status
        cpu = RegionEngine()
        criteria = {'max_absolute_error': .02, 'p99_absolute_error': .001,
                    'category_disagreement_fraction': .01}
        report['criteria'] = criteria
        for identifier, (fixture_name, rgb) in enumerate(fixtures().items()):
            rgb.flags.writeable = False
            packet = FramePacket(identifier, monotonic_ns(), rgb)
            accelerated = gpu.process(packet)
            reference = cpu.process(packet)
            if (accelerated.frame_id != packet.frame_id
                    or accelerated.capture_time_ns != packet.capture_time_ns
                    or accelerated.confidence_masks.flags.writeable):
                raise ValueError('GPU process lost the source frame contract')
            metrics = compare_masks(reference.confidence_masks, accelerated.confidence_masks, criteria)
            working = cv2.resize(mirrored_canvas(packet), (640, 360), interpolation=cv2.INTER_AREA)
            expected_signals = signal_stages(working)
            expected_regions = regions_from_masks(accelerated.confidence_masks, (360, 640))
            signal_comparison = {}
            for name in ('luminance', 'denoised', 'edge_magnitude'):
                actual = getattr(accelerated.prepared_signals, name)
                expected = getattr(expected_signals, name)
                signal_comparison[name] = {
                    'differing_values': int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
                    'maximum_absolute_error': float(np.max(np.abs(actual - expected))),
                    'readonly': not actual.flags.writeable}
            region_match = (accelerated.prepared_regions.reliable == expected_regions.reliable and
                            all(np.array_equal(getattr(accelerated.prepared_regions, name),
                                               getattr(expected_regions, name))
                                and not getattr(accelerated.prepared_regions, name).flags.writeable
                                for name in ('person', 'hair', 'skin', 'clothes')))
            metrics['prepared_signals'] = signal_comparison
            metrics['prepared_regions_match'] = region_match
            metrics['passed'] = metrics['passed'] and region_match and all(
                item['differing_values'] == 0 and item['readonly']
                for item in signal_comparison.values())
            metrics['gpu_transport_total_ms'] = accelerated.inference_ms
            metrics['cpu_total_ms'] = reference.inference_ms
            report['fixtures'][fixture_name] = metrics
        report['completed'] = set(report['fixtures']) == set(fixtures())
    finally:
        if cpu is not None:
            cpu.close()
        if gpu is not None:
            gpu.close()
        report['child_exited'] = process_handle is not None and process_handle.poll() is not None
        report['shared_buffer_closed'] = gpu is not None and gpu.frames is None
        report['passed'] = (report['completed'] and report['child_exited']
                            and report['shared_buffer_closed']
                            and all(result['passed'] for result in report['fixtures'].values()))
        (output / 'evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Paired GPU-process evidence: ' + str(output), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
