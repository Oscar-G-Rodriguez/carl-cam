"""Private GPU inference process: numerical shared buffers, no camera or files.

The service is launched by Carl-Cam, not by a user-facing terminal workflow.
The candidate and runtime inventories remain exactly checksum-gated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The isolated interpreter intentionally ignores external PYTHONPATH (-I).
if '__compiled__' not in globals():
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

import cv2
import numpy as np

from carl_cam.dsp import signal_stages
from carl_cam.gpu_ipc import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    SharedFrames,
    decode_message,
    encode_message,
    validate_frame_request,
)
from carl_cam.gpu_regions import GpuRegionEngine
from carl_cam.records import FramePacket
from carl_cam.themes import regions_from_masks
from carl_cam.vision import mirrored_canvas


def serve(buffer_name: str, candidate: Path, incoming, outgoing) -> int:
    frames = engine = None

    def send(kind, **fields):
        outgoing.write(encode_message({'schema_version': PROTOCOL_VERSION, 'kind': kind, **fields}))
        outgoing.flush()

    try:
        frames = SharedFrames(buffer_name)
        engine = GpuRegionEngine(candidate, experiment=True)
        precision = engine.session.get_provider_options()['CUDAExecutionProvider'].get('use_tf32')
        if precision != '0':
            raise ValueError('GPU service did not retain full FP32 precision')
        # Native algorithm selection must finish before the ready handshake.
        blank = np.zeros((720, 1280, 3), np.uint8)
        for identifier in range(3):
            engine.process(FramePacket(identifier, 1, blank))
        send('ready', backend='cuda_fp32')
        last_id = -1
        while True:
            payload = incoming.readline(MAX_MESSAGE_BYTES + 1)
            if not payload:
                return 0
            message = decode_message(payload)
            if message == {'schema_version': PROTOCOL_VERSION, 'kind': 'stop'}:
                send('stopped')
                return 0
            identifier, timestamp = validate_frame_request(message)
            if identifier <= last_id:
                raise ValueError('GPU service received a repeated or reordered frame')
            rgb = frames.input.copy()
            rgb.flags.writeable = False
            packet = FramePacket(identifier, timestamp, rgb)
            result = engine.process(packet)
            regions = regions_from_masks(result.confidence_masks, (360, 640))
            working = cv2.resize(mirrored_canvas(packet), (640, 360), interpolation=cv2.INTER_AREA)
            signals = signal_stages(working)
            np.copyto(frames.output, result.confidence_masks)
            for index, plane in enumerate((regions.person, regions.hair, regions.skin, regions.clothes)):
                np.copyto(frames.prepared_regions[index], plane)
            for index, plane in enumerate((signals.luminance, signals.denoised, signals.edge_magnitude)):
                np.copyto(frames.prepared_signals[index], plane)
            send('result', frame_id=identifier, capture_time_ns=timestamp,
                 inference_ms=result.inference_ms, regions_reliable=regions.reliable)
            last_id = identifier
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, ImportError):
        # No pixels, measurements, paths or exception text in diagnostics.
        send('error', code='gpu_service_unavailable')
        return 2
    finally:
        if engine is not None:
            engine.close()
        if frames is not None:
            frames.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--buffer', required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    arguments = parser.parse_args()
    return serve(arguments.buffer, arguments.candidate, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == '__main__':
    raise SystemExit(main())
