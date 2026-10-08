"""Worker-owned signal processing and inside-quadrilateral palette composition."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Event, Thread
from time import monotonic_ns, perf_counter

import cv2
import numpy as np

from carl_cam.camera import CANVAS_SIZE, canvas_rgb
from carl_cam.channels import LatestValue
from carl_cam.dsp import AdaptiveRange, SignalStages, premultiplied_rgb_over, signal_stages
from carl_cam.features import palm_ruler
from carl_cam.geometry import Quad, polygon_mask, validate_quad
from carl_cam.graphics import asset_pixels, draw_border, menu_card, place_asset
from carl_cam.menu import MenuLayout
from carl_cam.records import FramePacket, PerceptionPacket, RenderPacket, ThemeName
from carl_cam.themes import Regions, feather, motif_overlay, regions_from_masks, subject_tones


@dataclass(frozen=True, slots=True)
class SceneState:
    theme: ThemeName = "red"
    quad: Quad | None = None
    mode: str = "camera"
    menu: MenuLayout | None = None
    pointer: tuple[float, float] | None = None
    hover: str | None = None
    focus: str | None = None
    reduced_motion: bool = False


@dataclass(frozen=True, slots=True)
class InspectionSnapshot:
    frame_id: int
    capture_time_ns: int
    original_rgb: np.ndarray
    stages: SignalStages
    person_mask: np.ndarray
    overlay_rgb: np.ndarray
    final_rgb: np.ndarray
    segmentation_reliable: bool


class MenuTransition:
    """Opacity-only 180/120 ms open/close; logical hit areas never move."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.menu = None
        self.visible = False
        self.opacity = 0.
        self.started = 0.
        self.close_from = 0.

    def update(self, state: SceneState, now: float):
        opening = state.mode == "menu" and state.menu is not None
        if state.mode not in ("camera", "framed", "menu"):
            self.reset()  # Never retain a performance overlay in private modes.
            return None, 0.
        if opening:
            if not self.visible:
                self.started = now - self.opacity * .180
            self.visible, self.menu = True, state
            self.opacity = 1. if state.reduced_motion else min(1., max(0., (now - self.started) / .180))
        else:
            if self.visible:
                self.started, self.close_from = now, self.opacity
            self.visible = False
            self.opacity = 0. if state.reduced_motion else max(0., self.close_from * (1 - (now - self.started) / .120))
        return (replace(self.menu, theme=state.theme), self.opacity) if self.menu and self.opacity > 0 else (None, 0.)


def diagnostic_overlay(rgb, state, observation, packet):
    result = rgb.copy()
    if state.quad:
        cv2.polylines(result, [np.asarray(state.quad).round().astype(np.int32)], True, (245, 201, 21), 2, cv2.LINE_AA)
    if observation and abs(observation.capture_time_ns - packet.capture_time_ns) <= 100_000_000:
        for hand in observation.hands:
            points = np.asarray(hand.landmarks)[:, :2]
            if not np.isfinite(points).all():
                continue
            for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 10, 11, 12), (9, 13, 14, 15, 16), (13, 17, 18, 19, 20), (0, 17)):
                cv2.polylines(result, [points[list(chain)].round().astype(np.int32)], False, (74, 201, 232), 2, cv2.LINE_AA)
    return result


def compose_palette(rgb: np.ndarray, state: SceneState, *, masks=None,
                    adaptive: AdaptiveRange | None = None, seconds: float = 0.,
                    hand_mask: np.ndarray | None = None,
                    regions: Regions | None = None,
                    stages: SignalStages | None = None) -> tuple[np.ndarray, SignalStages]:
    """Tone-map the unwarped live image while retaining original exterior pixels."""
    width, height = rgb.shape[1], rgb.shape[0]
    shape = (height // 2, width // 2)
    if stages is None:
        small = cv2.resize(rgb, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        stages = signal_stages(small)
    elif (not isinstance(stages, SignalStages)
          or any(not isinstance(plane, np.ndarray) or plane.shape != shape or plane.dtype != np.float32
                 for plane in (stages.luminance, stages.denoised, stages.edge_magnitude))):
        raise ValueError('Prepared signal stages must match the current processing canvas')
    if state.mode not in ("camera", "framed") or state.quad is None or not validate_quad(state.quad, (width, height)):
        return rgb.copy(), stages
    regions = regions if regions is not None else regions_from_masks(masks, shape)
    if regions.person.shape != shape:
        raise ValueError('Prepared regions must match the current processing canvas')
    lightness = adaptive.normalize(stages.denoised, seconds, regions.person if regions.reliable else None) if adaptive else stages.denoised
    tones = subject_tones(stages, lightness, state.theme, regions)
    if regions.reliable:
        person = regions.person
        if hand_mask is not None:
            person = np.maximum(person, cv2.resize(hand_mask, (width // 2, height // 2), interpolation=cv2.INTER_AREA))
        local_quad = tuple((x / 2, y / 2) for x, y in state.quad)
        decoration = motif_overlay(state.theme, local_quad, (width // 2, height // 2), seconds, state.reduced_motion)
        decoration *= (1 - person)[..., None]
        tones = premultiplied_rgb_over(tones, decoration)
    styled = cv2.resize(tones, (width, height), interpolation=cv2.INTER_LINEAR)
    # Blend only the quad's padded bounds. No full-canvas float RGB temporary
    # is needed, and every exterior pixel is copied bit-for-bit.
    points = np.asarray(state.quad)
    left, top = np.maximum(0, np.floor(points.min(axis=0) - 8)).astype(int)
    right, bottom = np.minimum((width, height), np.ceil(points.max(axis=0) + 9)).astype(int)
    translated = tuple((x - left, y - top) for x, y in state.quad)
    alpha = polygon_mask(translated, (right - left, bottom - top))
    final = rgb.copy()
    area = np.s_[top:bottom, left:right]
    final[area] = cv2.blendLinear(styled[area], rgb[area], alpha, 1 - alpha)
    return final, stages


def hand_occlusion(observation: PerceptionPacket | None, packet: FramePacket,
                   person: np.ndarray | None = None) -> np.ndarray | None:
    """Never extrapolate a stale hand hull; preserve current styled fingers."""
    if observation is None or abs(packet.capture_time_ns - observation.capture_time_ns) > 100_000_000:
        return None
    mask = np.zeros((360, 640), np.float32)
    for hand in observation.hands:
        points = np.asarray(hand.landmarks, np.float32)
        if points.shape != (21, 3) or not np.isfinite(points[:, :2]).all() or hand.score < .5:
            continue
        ruler = palm_ruler(points)
        if ruler < 36:
            continue
        hull = cv2.convexHull((points[:, :2] / 2).round().astype(np.int32))
        layer = np.zeros_like(mask)
        cv2.fillConvexPoly(layer, hull, 1)
        palm_width = np.linalg.norm(points[5, :2] - points[17, :2])
        radius = max(1, round(.12 * palm_width / 2))
        layer = cv2.dilate(layer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2))
        mask = np.maximum(mask, layer)
    if person is not None:
        mask *= person
    return cv2.resize(feather(mask), CANVAS_SIZE, interpolation=cv2.INTER_LINEAR)


def matched_region(pair, now: int):
    """Validate the immutable segmentation/source pair before pixel composition."""
    if pair is None:
        return None
    packet, region = pair
    if packet.frame_id != region.frame_id or packet.capture_time_ns != region.capture_time_ns:
        return None
    if not 0 <= now - packet.capture_time_ns <= 200_000_000:
        return None
    return packet, region.confidence_masks


def draw_menu(rgb: np.ndarray, layout: MenuLayout, *, selected: str, hover: str | None = None,
              focus: str | None = None, pointer=None) -> np.ndarray:
    """Place genuine Figma state exports after each view's coordinate conversion."""
    result = rgb.copy()
    for name, rect in layout.cards:
        start = (round(rect.x), round(rect.y))
        end = (round(rect.x + rect.width), round(rect.y + rect.height))
        interaction = "focused" if name == focus else "hover" if name == hover else "default"
        result = place_asset(result, menu_card(name, name == selected, interaction=interaction),
                             rect.x, rect.y, rect.width, rect.height)
        # Back has one original export, so its interactive chrome stays live.
        if name == "back" and name == hover:
            cv2.rectangle(result, (start[0] + 8, start[1] + 8), (end[0] - 8, end[1] - 8), (74, 201, 232), 3)
        if name == "back" and name == focus:
            cv2.rectangle(result, (start[0] + 14, start[1] + 14), (end[0] - 14, end[1] - 14), (245, 201, 21), 2)
    if pointer is not None:
        result = place_asset(result, asset_pixels("pointer"), pointer[0] - 16, pointer[1] - 16)
    return result


def recovery_menu(rgb: np.ndarray, layout: MenuLayout) -> np.ndarray:
    """Readable temporary controls when a vector fails; never final artwork."""
    result = rgb.copy()
    for name, rect in layout.cards:
        x, y, width, height = round(rect.x), round(rect.y), round(rect.width), round(rect.height)
        cv2.rectangle(result, (x, y), (x + width, y + height), (17, 16, 20), -1)
        cv2.rectangle(result, (x, y), (x + width, y + height), (74, 201, 232), 2)
        label = "BACK TO CAMERA" if name == "back" else name.upper()
        cv2.putText(result, label, (x + 16, y + 58), cv2.FONT_HERSHEY_SIMPLEX, .65, (241, 236, 226), 2, cv2.LINE_AA)
    return result


class RenderWorker:
    """Process new frame IDs on a background thread, with finite stale-image grace."""

    def __init__(self, frames: LatestValue[FramePacket], *, region_pairs=None,
                 region_status=None, perceptions=None) -> None:
        self.frames = frames
        self.region_pairs = region_pairs
        self.region_status = region_status
        self.perceptions = perceptions
        self.scene: LatestValue[SceneState] = LatestValue()
        self.scene.publish(SceneState())
        self.results: LatestValue[RenderPacket] = LatestValue()
        self.signals: LatestValue[SignalStages] = LatestValue()
        self.status = LatestValue[str]()
        self.inspection = LatestValue[InspectionSnapshot]()
        self.adaptive = AdaptiveRange()
        self.menu_transition = MenuTransition()
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="CarlCamRender", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()

    def _run(self) -> None:
        last_id, black = -1, False
        render_sequence = 0
        last_publish = 0
        while not self._stop.is_set():
            packet = self.frames.read()
            now = monotonic_ns()
            state = self.scene.read() or SceneState()
            if packet is None or now - packet.capture_time_ns > 200_000_000:
                if not black:
                    blank = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
                    self.results.publish(RenderPacket(-1, blank, blank, now, 0))
                    self.adaptive.reset()
                    self.menu_transition.reset()
                    black = True
                self._stop.wait(0.01)
                continue
            masks = None
            prepared_regions = prepared_signals = None
            decorated_mode = state.mode in ("camera", "framed", "inspect") and state.quad is not None
            region_failed = self.region_status and (self.region_status.read() or "").startswith("Vision unavailable")
            if decorated_mode and self.region_pairs is not None and not region_failed:
                region_pair = self.region_pairs.read()
                compatible = matched_region(region_pair, now)
                if compatible and compatible[0].frame_id > last_id:
                    packet, masks = compatible
                    prepared_regions = getattr(region_pair[1], 'prepared_regions', None)
                    prepared_signals = getattr(region_pair[1], 'prepared_signals', None)
                elif now - last_publish < 100_000_000:
                    # At most 100 ms waiting for a same-frame region result;
                    # never queue frames or borrow an old mask for new pixels.
                    self._stop.wait(.003)
                    continue
            if packet.frame_id <= last_id:
                self._stop.wait(0.003)
                continue
            started = perf_counter()
            unmirrored = canvas_rgb(packet.rgb)
            preview = np.ascontiguousarray(unmirrored[:, ::-1])
            regions = (prepared_regions if prepared_regions is not None
                       else regions_from_masks(masks, (360, 640)))
            observation = self.perceptions.read() if self.perceptions else None
            hands = hand_occlusion(observation, packet,
                                   regions.person if regions.reliable else None)
            filter_state = replace(state, mode="framed") if state.mode == "inspect" else state
            final, stages = compose_palette(preview, filter_state, masks=masks, adaptive=self.adaptive,
                                            seconds=packet.capture_time_ns / 1e9, hand_mask=hands,
                                            regions=regions, stages=prepared_signals)
            if decorated_mode:
                try:
                    final = draw_border(final, state.theme, state.quad, hands,
                                        seconds=packet.capture_time_ns / 1e9,
                                        reduced_motion=state.reduced_motion)
                    self.status.publish("Filter ready" if regions.reliable else "Palette only — background masking unavailable")
                except (OSError, ValueError, RuntimeError, cv2.error):
                    self.status.publish("Border unavailable. Palette remains usable; repair the installation.")
            output = np.ascontiguousarray(final[:, ::-1])
            if state.mode == "inspect":
                # Diagnostics and even the inspect-only filter comparison never
                # enter output; the recipient gets current ordinary camera RGB.
                output = unmirrored
                overlay = diagnostic_overlay(preview, state, observation, packet)
                for array in (preview, overlay, regions.person, stages.luminance, stages.denoised, stages.edge_magnitude):
                    array.flags.writeable = False
                final.flags.writeable = False
                self.inspection.publish(InspectionSnapshot(packet.frame_id, packet.capture_time_ns,
                                                           preview, stages, regions.person, overlay,
                                                           final, regions.reliable))
            menu_state, opacity = self.menu_transition.update(state, packet.capture_time_ns / 1e9)
            if menu_state:
                output_pointer = (1279 - menu_state.pointer[0], menu_state.pointer[1]) if menu_state.pointer else None
                try:
                    decorated = draw_menu(final, menu_state.menu, selected=state.theme, hover=menu_state.hover, focus=menu_state.focus, pointer=menu_state.pointer)
                    outgoing = draw_menu(output, menu_state.menu.output(), selected=state.theme, hover=menu_state.hover, focus=menu_state.focus, pointer=output_pointer)
                except (OSError, ValueError, RuntimeError, cv2.error):
                    decorated = recovery_menu(final, menu_state.menu)
                    outgoing = recovery_menu(output, menu_state.menu.output())
                    self.status.publish("Menu graphics unavailable. Temporary controls are usable; repair the installation.")
                final = cv2.addWeighted(final, 1 - opacity, decorated, opacity, 0)
                output = cv2.addWeighted(output, 1 - opacity, outgoing, opacity, 0)
            final.flags.writeable = False
            output.flags.writeable = False
            self.signals.publish(stages)
            completed = monotonic_ns()
            if completed - packet.capture_time_ns > 250_000_000:
                final = np.zeros_like(final)
                output = np.zeros_like(output)
                final.flags.writeable = output.flags.writeable = False
                self.status.publish("Processing is too slow — showing a black safety frame")
            render_sequence += 1
            self.results.publish(RenderPacket(
                packet.frame_id, final, output, completed, (perf_counter() - started) * 1000,
                packet.capture_time_ns,
                render_sequence,
            ))
            last_id, black = packet.frame_id, False
            last_publish = monotonic_ns()
