from dataclasses import replace
from time import monotonic_ns, sleep
from types import SimpleNamespace

import numpy as np
import pytest
from test_features import hand

from carl_cam.channels import LatestValue
from carl_cam.dsp import AdaptiveRange, SignalStages
from carl_cam.records import FramePacket, PerceptionPacket
from carl_cam.render import (
    RenderWorker,
    SceneState,
    compose_palette,
    hand_occlusion,
    matched_region,
)
from carl_cam.themes import PALETTES, motif_overlay, regions_from_masks, subject_tones

QUAD = ((300, 150), (980, 150), (980, 570), (300, 570))


def masks():
    result = np.zeros((360, 640, 6), np.float32)
    result[..., 0] = 1
    result[95:255, 235:405, 0] = 0
    result[95:125, 235:405, 1] = 1  # Hair
    result[125:190, 235:405, 3] = 1  # Face skin
    result[190:255, 235:405, 4] = 1  # Clothes
    return result


def test_regions_reject_ambiguous_nonfinite_and_wrong_shapes():
    ambiguous = np.full((360, 640, 6), .1, np.float32)
    ambiguous[..., 0] = .5
    for bad in (None, np.full((360, 640, 6), np.nan), ambiguous, np.zeros((20, 20, 6))):
        assert not regions_from_masks(bad, (360, 640)).reliable
    regions = regions_from_masks(masks(), (360, 640))
    assert regions.reliable and regions.hair[110, 300] == 1
    assert regions.skin[150, 300] == regions.clothes[220, 300] == 1
    assert 0 < regions.person[94, 300] < 1


@pytest.mark.parametrize("theme", ("red", "yellow", "blue"))
def test_motifs_stay_behind_person_and_inside_quad(theme, monkeypatch):
    image = np.full((720, 1280, 3), 160, np.uint8)
    state = SceneState(theme, QUAD, reduced_motion=True)
    plain, _ = compose_palette(image, state)
    decorated, _ = compose_palette(image, state, masks=masks())
    # Compare with the same region tones but no motifs, not a label-free palette.
    monkeypatch.setattr("carl_cam.render.motif_overlay", lambda *args: np.zeros((360, 640, 4), np.float32))
    subject_only, _ = compose_palette(image, state, masks=masks())
    np.testing.assert_array_equal(decorated[270:360, 500:770], subject_only[270:360, 500:770])
    np.testing.assert_array_equal(decorated[410:480, 500:770], subject_only[410:480, 500:770])
    np.testing.assert_array_equal(decorated[:, :290], image[:, :290])
    assert not np.array_equal(decorated[160:500, 310:450], plain[160:500, 310:450])
    unreliable, _ = compose_palette(image, state, masks=np.full((360, 640, 6), 1 / 6))
    np.testing.assert_array_equal(unreliable, plain)


@pytest.mark.parametrize("theme", ("red", "yellow", "blue"))
def test_fine_regions_use_flat_approved_tones_and_abstain_independently(theme):
    regions = regions_from_masks(masks(), (360, 640))
    lightness = np.full((360, 640), .66, np.float32)
    edges = np.zeros_like(lightness)
    stages = SignalStages(lightness, lightness, edges)
    tones = subject_tones(stages, lightness, theme, regions)
    # Clothing's highlight threshold is higher; skin remains a cream highlight.
    np.testing.assert_array_equal(tones[150, 300], PALETTES[theme][2])
    np.testing.assert_array_equal(tones[220, 300], PALETTES[theme][1])
    if theme in ("red", "yellow"):
        base = np.full_like(lightness, .45)
        np.testing.assert_array_equal(subject_tones(stages, base, theme, regions)[110, 300], PALETTES[theme][0])
    if theme == "blue":
        base = np.full_like(lightness, .45)
        np.testing.assert_array_equal(subject_tones(stages, base, theme, regions)[150, 300], (66, 216, 238))
    # Broad foreground can be reliable while fine labels have no confident winner.
    uncertain = masks()
    uncertain[125:190, 235:405, 1:] = .2
    uncertain_regions = regions_from_masks(uncertain, (360, 640))
    assert uncertain_regions.reliable and uncertain_regions.skin[150, 300] == 0
    fallback = subject_tones(stages, lightness, theme, uncertain_regions)
    label_free = subject_tones(stages, lightness, theme, regions_from_masks(None, (360, 640)))
    np.testing.assert_array_equal(fallback[150, 300], label_free[150, 300])


def test_red_hair_stays_black_cream_and_blue_panels_are_translucent():
    image = np.full((720, 1280, 3), 160, np.uint8)
    result, _ = compose_palette(image, SceneState("red", QUAD), masks=masks())
    pixel = result[220, 600]
    assert abs(int(pixel[0]) - int(pixel[1])) < 10
    overlay = motif_overlay("blue", QUAD, (1280, 720), 0, True)
    assert np.any((overlay[..., 3] > .4) & (overlay[..., 3] < .95))


@pytest.mark.parametrize("theme", ("red", "yellow", "blue"))
def test_animation_is_elapsed_time_deterministic_and_reduced_motion_is_static(theme):
    first = motif_overlay(theme, QUAD, (1280, 720), 0)
    np.testing.assert_array_equal(first, motif_overlay(theme, QUAD, (1280, 720), 0))
    assert not np.array_equal(first, motif_overlay(theme, QUAD, (1280, 720), 1.25))
    np.testing.assert_array_equal(motif_overlay(theme, QUAD, (1280, 720), 0, True),
                                  motif_overlay(theme, QUAD, (1280, 720), 99, True))


def test_blue_panels_have_distinct_bounded_speeds_and_both_drive_the_render(monkeypatch):
    from carl_cam import themes

    motion = themes.BLUE_PANEL_MOTION
    assert all(0 < distance <= 5 and period == 6 for distance, period, _ in motion)
    speeds = [distance / period for distance, period, _ in motion]
    assert speeds[1] / speeds[0] == pytest.approx(.9)
    moving = motif_overlay('blue', QUAD, (1280, 720), 1.25)
    static = motif_overlay('blue', QUAD, (1280, 720), 1.25, True)
    for index in (0, 1):
        changed = list(motion)
        changed[index] = (0., 6., motion[index][2])
        monkeypatch.setattr(themes, 'BLUE_PANEL_MOTION', tuple(changed))
        assert not np.array_equal(moving, motif_overlay('blue', QUAD, (1280, 720), 1.25))
        np.testing.assert_array_equal(static, motif_overlay('blue', QUAD, (1280, 720), 1.25, True))


def test_lighting_range_ema_limits_bright_object_change_and_resets():
    adaptive = AdaptiveRange()
    gradient = np.tile(np.linspace(.1, .7, 640, dtype=np.float32), (360, 1))
    adaptive.normalize(gradient, 10.)
    old = adaptive.bounds
    adaptive.normalize(np.clip(gradient + .3, 0, 1), 10.033)
    assert 0 < adaptive.bounds[0] - old[0] < .03
    adaptive.reset()
    assert adaptive.bounds is None and adaptive.last_time is None


def test_hand_hulls_have_finite_grace_and_conservative_expansion():
    packet = FramePacket(1, 2_000_000_000, np.zeros((360, 640, 3), np.uint8))
    observation = PerceptionPacket(1, packet.capture_time_ns, (hand(),))
    mask = hand_occlusion(observation, packet)
    assert mask.shape == (720, 1280) and mask[350, 300] > .9
    assert mask[:, 600:].max() == 0
    assert hand_occlusion(replace(observation, capture_time_ns=1_899_999_999), packet) is None
    np.testing.assert_array_equal(hand_occlusion(observation, packet, np.zeros((360, 640))), np.zeros((720, 1280)))


def test_exact_region_pair_never_joins_mismatched_ids_or_timestamps():
    packet = FramePacket(5, 1_000_000_000, np.zeros((360, 640, 3), np.uint8))
    region = SimpleNamespace(frame_id=5, capture_time_ns=packet.capture_time_ns, confidence_masks=masks())
    assert matched_region((packet, region), 1_100_000_000)[0] is packet
    assert matched_region((replace(packet, frame_id=6), region), 1_100_000_000) is None
    assert matched_region((replace(packet, capture_time_ns=packet.capture_time_ns + 1), region), 1_100_000_000) is None
    assert matched_region((packet, region), 1_201_000_000) is None


def test_render_uses_the_region_source_pixels_not_the_latest_unmatched_image(monkeypatch):
    frames, pairs = LatestValue(), LatestValue()
    now = monotonic_ns()
    # This fixture isolates pairing, not CPU throughput. Real-clock ageing is
    # covered separately; a loaded test runner must not invalidate its pixels.
    monkeypatch.setattr("carl_cam.render.monotonic_ns", lambda: now)
    source = FramePacket(5, now, np.full((360, 640, 3), (20, 60, 100), np.uint8))
    frames.publish(FramePacket(6, now + 1, np.full((360, 640, 3), 240, np.uint8)))
    pairs.publish((source, SimpleNamespace(frame_id=5, capture_time_ns=now, confidence_masks=masks())))
    worker = RenderWorker(frames, region_pairs=pairs)
    worker.scene.publish(SceneState("red", QUAD, reduced_motion=True))
    worker.start()
    try:
        for _ in range(100):
            result = worker.results.read()
            if result is not None:
                break
            sleep(.002)
        assert result.frame_id == 5 and result.capture_time_ns == now
        np.testing.assert_array_equal(result.preview_rgb[0, 0], (20, 60, 100))
        assert not result.preview_rgb.flags.writeable and not result.output_rgb.flags.writeable
    finally:
        assert worker.stop()
