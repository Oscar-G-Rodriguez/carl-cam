"""Developer-only installed-bundle checks; no camera or virtual sender opens."""

import importlib.metadata
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic_ns, perf_counter

import numpy as np

from carl_cam import __version__
from carl_cam.camera import CameraStatus
from carl_cam.channels import LatestValue
from carl_cam.records import FramePacket
from carl_cam.storage import Store, atomic_json


def failure_codes(error):
    """Developer diagnostics contain identifiers, never paths, locals or text."""
    identifier = re.compile(r'[A-Za-z_][A-Za-z0-9_.]{0,159}')
    codes = {'error_type': type(error).__name__}
    missing = getattr(error, 'name', None) if isinstance(error, ModuleNotFoundError) else None
    if isinstance(missing, str) and identifier.fullmatch(missing):
        codes['missing_module'] = missing
    frames = []
    trace = error.__traceback__
    while trace:
        module = trace.tb_frame.f_globals.get('__name__')
        function = trace.tb_frame.f_code.co_name
        if isinstance(module, str) and identifier.fullmatch(module):
            frames.append({'module': module, 'function': function if identifier.fullmatch(function) else 'module'})
        trace = trace.tb_next
    codes['failure_frames'] = frames[-8:]
    return codes


class SyntheticCamera:
    def __init__(self):
        self.frames, self.status = LatestValue(), LatestValue()
        self.running = False
        self.quality_serial = 0
        self.frame_id = 0

    def start(self):
        self.running = True
        pixels = np.full((360, 640, 3), (120, 160, 180), np.uint8)
        pixels.flags.writeable = False
        self.frames.publish(FramePacket(0, monotonic_ns(), pixels))
        self.status.publish(CameraStatus("live", "Synthetic self-test", (640, 360), 30))

    def stop(self, timeout=3):
        self.running = False
        self.frames.clear()
        return True

    def request_quality_level(self, level):
        self.quality_serial += 1
        self.frame_id += 1
        pixels = np.full((360, 640, 3), (120, 160, 180), np.uint8)
        pixels.flags.writeable = False
        self.frames.publish(FramePacket(self.frame_id, monotonic_ns(), pixels))
        self.status.publish(CameraStatus('live', 'Synthetic self-test', (640, 360), 30,
                                         level, self.quality_serial))
        return self.quality_serial


def check_filter_rendering():
    """Exercise shipped compositors with generated pixels and exact labels.

    This is a frozen-dependency/geometry check, not camera appearance evidence.
    No image or landmark record leaves this function. Borders intentionally
    straddle the frame edge, so exterior preservation is checked before them.
    """
    from carl_cam.geometry import polygon_mask
    from carl_cam.graphics import asset_pixels, draw_border
    from carl_cam.render import SceneState, compose_palette

    width, height = 1280, 720
    y, x = np.indices((height, width))
    pixels = np.stack(((x + y) % 256, (x // 3 + 71) % 256,
                       (y // 2 + 37) % 256), axis=-1).astype(np.uint8)
    original = pixels.copy()
    pixels.flags.writeable = False
    masks = np.zeros((height // 2, width // 2, 6), np.float32)
    masks[..., 0] = 1.
    for label, bounds in ((1, (slice(65, 115), slice(245, 390))),
                          (3, (slice(115, 210), slice(265, 375))),
                          (4, (slice(210, 320), slice(210, 420)))):
        masks[bounds] = 0.
        masks[bounds + (label,)] = 1.
    masks.flags.writeable = False
    quad = ((265, 111), (1010, 149), (981, 619), (278, 589))
    alpha = polygon_mask(quad, (width, height))
    outside, interior = alpha == 0, alpha == 1
    assert outside.any() and interior.any()
    occlusion = np.ones((height, width), np.float32)
    summaries = {}
    for theme in ("red", "yellow", "blue"):
        assert asset_pixels("border-" + theme).shape == (360, 640, 4)
        for seconds, reduced_motion, labels in ((0., False, masks),
                                               (.37, False, masks),
                                               (1., True, masks),
                                               (0., False, None)):
            styled, stages = compose_palette(
                pixels, SceneState(theme, quad, reduced_motion=reduced_motion),
                masks=labels, seconds=seconds)
            assert styled.shape == pixels.shape and styled.dtype == np.uint8
            assert np.isfinite(stages.edge_magnitude).all()
            np.testing.assert_array_equal(styled[outside], pixels[outside])
            assert np.any(styled[interior] != pixels[interior])
            bordered = draw_border(styled, theme, quad, seconds=seconds,
                                   reduced_motion=reduced_motion)
            assert bordered.shape == pixels.shape and bordered.dtype == np.uint8
            assert np.any(bordered != styled)
            covered = draw_border(styled, theme, quad, occlusion,
                                  seconds=seconds, reduced_motion=reduced_motion)
            np.testing.assert_array_equal(covered, styled)
        summaries[theme] = {"cases": 4, "bounded_stylization": True,
                            "border_rendered": True, "hand_occlusion": True}
    np.testing.assert_array_equal(pixels, original)
    return summaries


def check_installed_gpu():
    """Exercise the frozen sibling helper with inert pixels and exact frame pairing."""
    from threading import Event

    import cv2

    from carl_cam.dsp import signal_stages
    from carl_cam.gpu_process import GpuProcessRegionEngine, runtime_launch
    from carl_cam.themes import regions_from_masks
    from carl_cam.vision import RegionEngine, mirrored_canvas

    launch = runtime_launch()
    if launch is None:
        raise ValueError('The installed GPU helper is unavailable')
    gpu, cpu = GpuProcessRegionEngine(launch), None
    handle = None
    comparisons = {}
    try:
        gpu.initialize(Event())
        handle = gpu.process_handle
        cpu = RegionEngine()
        rng = np.random.default_rng(19)
        for identifier, (name, pixels) in enumerate((
                ('black', np.zeros((360, 640, 3), np.uint8)),
                ('white', np.full((360, 640, 3), 255, np.uint8)),
                ('noise', rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)))):
            pixels.flags.writeable = False
            packet = FramePacket(identifier, monotonic_ns(), pixels)
            accelerated, reference = gpu.process(packet), cpu.process(packet)
            assert (accelerated.frame_id, accelerated.capture_time_ns) == (packet.frame_id, packet.capture_time_ns)
            error = np.abs(accelerated.confidence_masks - reference.confidence_masks)
            assert np.max(error) <= .02 and np.percentile(error, 99) <= .001
            assert np.mean(np.argmax(accelerated.confidence_masks, axis=2)
                           != np.argmax(reference.confidence_masks, axis=2)) <= .01
            expected = signal_stages(cv2.resize(mirrored_canvas(packet), (640, 360), interpolation=cv2.INTER_AREA))
            for field in ('luminance', 'denoised', 'edge_magnitude'):
                actual = getattr(accelerated.prepared_signals, field)
                assert not actual.flags.writeable
                np.testing.assert_array_equal(actual, getattr(expected, field))
            expected_regions = regions_from_masks(accelerated.confidence_masks, (360, 640))
            assert accelerated.prepared_regions.reliable == expected_regions.reliable
            for field in ('person', 'hair', 'skin', 'clothes'):
                np.testing.assert_array_equal(getattr(accelerated.prepared_regions, field),
                                              getattr(expected_regions, field))
            comparisons[name] = {'max_absolute_error': float(np.max(error)), 'prepared_values_exact': True}
    finally:
        if cpu is not None:
            cpu.close()
        gpu.close()
    assert handle is not None and handle.poll() is not None and gpu.frames is None
    return {'fixtures': comparisons, 'child_exited': True, 'shared_buffer_closed': True,
            'physical_camera_used': False, 'live_fps_gate_passed': False}


def run_selftest(report_path: Path) -> int:
    started = perf_counter()
    report = {"schema_version": 1, "app_version": __version__, "camera_opened": False,
              "virtual_output_started": False, "human_validation": "not performed", "checks": {}}
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import torch
        from PySide6.QtGui import QFont, QRawFont
        from PySide6.QtWidgets import QApplication

        from carl_cam.comparison_vision import ComparisonEngine
        from carl_cam.model_store import ModelStore, fingerprint_dataset
        from carl_cam.performance import PerformanceFallback
        from carl_cam.storage import LABELS
        from carl_cam.training import (
            GestureNetwork,
            NeuralCandidate,
            Standardizer,
            fit_candidates,
            gpu_training_status,
        )
        from carl_cam.typography import FONT_FAMILY, configure_typography
        from carl_cam.ui import CameraWindow
        from carl_cam.vision import HandHeadEngine, RegionEngine

        app = QApplication.instance() or QApplication([])
        configure_typography(strict=True)
        with TemporaryDirectory(prefix="carlcam-selftest-") as directory:
            store = Store(Path(directory) / "CarlCam")
            camera = SyntheticCamera()
            window = CameraWindow(test_camera=camera, store=store)
            assert not camera.running
            window.show()
            app.processEvents()
            font = QRawFont.fromFont(QFont("Arimo", 11))
            assert font.isValid() and font.familyName() == FONT_FAMILY
            assert all(font.glyphIndexesForString("Carl-Cam setup 0123456789"))
            report["checks"]["Qt_widgets_and_bundled_font"] = True
            report["filter_render_checks"] = check_filter_rendering()
            report["checks"]["original_graphics"] = True
            packet = FramePacket(0, monotonic_ns(), np.zeros((360, 640, 3), np.uint8))
            hands, regions = None, None
            try:
                hands, regions = HandHeadEngine(), RegionEngine()
                assert hands.process(packet).frame_id == 0
                masks = regions.process(packet)
                assert masks.confidence_masks.shape == (360, 640, 6)
                report["checks"]["bundled_vision_models"] = True
            finally:
                if hands:
                    hands.close()
                if regions:
                    regions.close()
            network = GestureNetwork(112, (64, 32)).eval()
            candidate = NeuralCandidate(112, (64, 32), network,
                                        Standardizer(np.zeros(112, np.float32), np.ones(112, np.float32)),
                                        .5, (.9,) * 6, 1, {"train": (), "validation": (), "sealed": ()})
            models = ModelStore(store)
            version = models.stage(candidate, dataset_fingerprint=fingerprint_dataset([]), report={"synthetic_selftest": True})
            assert np.isfinite(models.load(version).scores(np.zeros(112))).all()
            report["checks"]["PyTorch_checksum_load_and_inference"] = True
            paired = ComparisonEngine(candidate)
            try:
                comparison = paired.process(FramePacket(1, monotonic_ns(), packet.rgb))
                assert comparison.frame_id == 1
                assert comparison.neural.outcome == comparison.canned.outcome == "no_hand"
                report["checks"]["optional_paired_canned_model"] = True
            finally:
                paired.close()
            rng = np.random.default_rng(17)
            synthetic_rows = []
            for label_index, label in enumerate(LABELS):
                for round_index in range(5):
                    for _ in range(8):
                        values = rng.normal(0, .01, 180).astype(np.float32)
                        values[label_index] = 1.
                        synthetic_rows.append({"label": label, "round_index": round_index,
                                               "session_id": f"synthetic-{label}-{round_index}",
                                               "features": {str(width): values[:width].tolist() for width in (112, 138, 180)}})
            fitted = fit_candidates(synthetic_rows, include_depth=True, use_cuda=False, max_epochs=2)
            assert len(fitted.candidates) == 6 and len(fitted.baselines) == 3
            assert all(np.isfinite(candidate.scores(np.zeros(candidate.width))).all() for candidate in fitted.candidates)
            report["checks"]["CPU_neural_training_and_logistic_comparison"] = True
            assert fitted.requested_device == "cpu" and not fitted.cpu_fallback
            assert all(candidate.training_device == "cpu" for candidate in fitted.candidates)
            assert torch.version.cuda == "13.0"
            report["checks"]["CUDA_training_runtime_and_CPU_default"] = True
            available, _ = gpu_training_status()
            accelerated = fit_candidates(synthetic_rows, include_depth=True, use_cuda=True, max_epochs=2)
            assert len(accelerated.candidates) == 6 and len(accelerated.baselines) == 3
            assert accelerated.requested_device == "cuda"
            assert accelerated.cpu_fallback is not available
            assert all(candidate.training_device == ("cuda" if available else "cpu")
                       for candidate in accelerated.candidates)
            for trained in accelerated.candidates:
                assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32
                           for parameter in trained.network.parameters())
                before_reload = trained.scores(np.zeros(trained.width, np.float32))
                saved = models.stage(trained, dataset_fingerprint=fingerprint_dataset(synthetic_rows),
                                     report={"synthetic_training_selftest": True})
                restored = models.load(saved)
                assert restored.training_device == trained.training_device
                np.testing.assert_array_equal(restored.scores(np.zeros(trained.width, np.float32)), before_reload)
            report['gpu_training_checks'] = {
                'compiled_cuda': torch.version.cuda, 'gpu_available': available,
                'actual_gpu_training_executed': available,
                'cpu_fallback_used': accelerated.cpu_fallback,
                'candidate_count': 6, 'all_models_returned_to_cpu': True,
                'cpu_save_reload_predictions_exact': True,
                'physical_camera_used': False, 'personal_recordings_used': False}
            if available:
                report['checks']['CUDA_neural_training_and_CPU_model_reload'] = True
            window.start_camera()
            assert camera.running
            fallback = PerformanceFallback()
            for step in range(10):
                assert fallback.observe(step / 2, step * 10, 100., 0) is None
            assert fallback.observe(5., 100, 100., 0) == 1
            for step in range(11, 30):
                assert fallback.observe(step / 2, step * 10, 100., 1) is None
            assert fallback.observe(15., 300, 100., 1) == 2
            window._request_input_quality(1)
            assert window._quality_pending_request == 1
            window._observe_performance(window.renderer.results.read(), camera.status.read())
            assert window._quality_pending_request is None
            window.restore_input_quality()
            assert window._quality_pending_request == 2
            window._observe_performance(window.renderer.results.read(), camera.status.read())
            assert window._quality_pending_request is None
            report['performance_fallback_checks'] = {
                'five_second_window': True, 'ten_second_cooldown': True,
                'explicit_restore': True, 'capture_request_acknowledgement': True,
                'physical_format_changes_evaluated': False}
            window.close()
            assert not camera.running and window.renderer is None
            report["checks"]["camera_off_by_default_and_finite_shutdown"] = True
            from carl_cam.gpu_bundle import HELPER_DIRECTORY
            if (Path(__file__).absolute().parents[1] / HELPER_DIRECTORY).exists():
                report['gpu_helper_checks'] = check_installed_gpu()
                report['checks']['installed_GPU_helper_and_prepared_frame_pairing'] = True
            report["libraries"] = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scikit-learn", "PySide6", "mediapipe", "pyvirtualcam")}
        report["passed"] = True
    except Exception as error:  # noqa: BLE001 - top-level developer diagnostic boundary
        # A developer report records a type/code, not exception contents that
        # may contain paths. It does not change any real user's model/settings.
        report["passed"] = False
        report.update(failure_codes(error))
    report["elapsed_seconds"] = perf_counter() - started
    atomic_json(report_path, report)
    print(json.dumps(report))
    return 0 if report["passed"] else 1
