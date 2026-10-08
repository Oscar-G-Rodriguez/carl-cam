"""A fixed-size, live-only frame/mask buffer and bounded JSON control protocol.

The desktop and GPU process never exchange Python objects or pickle payloads.
Only the process that creates a shared buffer may unlink it.
"""

from __future__ import annotations

import json
import re
from multiprocessing.shared_memory import SharedMemory

import numpy as np

INPUT_SHAPE = (720, 1280, 3)
OUTPUT_SHAPE = (360, 640, 6)
INPUT_BYTES = 720 * 1280 * 3
OUTPUT_BYTES = 360 * 640 * 6 * 4
REGIONS_SHAPE = (4, 360, 640)
SIGNALS_SHAPE = (3, 360, 640)
REGIONS_BYTES = 4 * 360 * 640 * 4
SIGNALS_BYTES = 3 * 360 * 640 * 4
BUFFER_BYTES = INPUT_BYTES + OUTPUT_BYTES + REGIONS_BYTES + SIGNALS_BYTES
PROTOCOL_VERSION = 2
MAX_MESSAGE_BYTES = 1024
NAME_PATTERN = re.compile(r'CarlCamGpu_[0-9a-f]{32}')


def encode_message(message: dict) -> bytes:
    payload = json.dumps(message, separators=(',', ':'), allow_nan=False).encode('ascii') + b'\n'
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError('GPU control message is too large')
    return payload


def decode_message(payload: bytes) -> dict:
    if (not isinstance(payload, bytes) or not payload.endswith(b'\n')
            or not 0 < len(payload) <= MAX_MESSAGE_BYTES):
        raise ValueError('Invalid GPU control message')
    value = json.loads(payload, parse_constant=lambda _: None)
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
            or value['schema_version'] != PROTOCOL_VERSION):
        raise ValueError('Unsupported GPU control protocol')
    return value


def validate_frame_request(message: dict) -> tuple[int, int]:
    if (set(message) != {'schema_version', 'kind', 'frame_id', 'capture_time_ns'}
            or message['kind'] != 'frame' or type(message['schema_version']) is not int
            or message['schema_version'] != PROTOCOL_VERSION):
        raise ValueError('Unexpected GPU frame request')
    identifier, timestamp = message['frame_id'], message['capture_time_ns']
    if (type(identifier) is not int or not 0 <= identifier < 2**63
            or type(timestamp) is not int or not 0 < timestamp < 2**63):
        raise ValueError('Invalid GPU frame identity')
    return identifier, timestamp


class SharedFrames:
    def __init__(self, name: str, *, create: bool = False):
        if not isinstance(name, str) or NAME_PATTERN.fullmatch(name) is None:
            raise ValueError('Invalid Carl-Cam GPU buffer name')
        self.memory = SharedMemory(name=name, create=create, size=BUFFER_BYTES if create else 0)
        self.owner = create
        self.input = None
        self.output = None
        self.prepared_regions = None
        self.prepared_signals = None
        try:
            if self.memory.size != BUFFER_BYTES:
                raise ValueError('GPU buffer has an incompatible size')
            self.input = np.ndarray(INPUT_SHAPE, np.uint8, buffer=self.memory.buf)
            self.output = np.ndarray(OUTPUT_SHAPE, np.float32, buffer=self.memory.buf, offset=INPUT_BYTES)
            self.prepared_regions = np.ndarray(REGIONS_SHAPE, np.float32, buffer=self.memory.buf,
                                               offset=INPUT_BYTES + OUTPUT_BYTES)
            self.prepared_signals = np.ndarray(SIGNALS_SHAPE, np.float32, buffer=self.memory.buf,
                                               offset=INPUT_BYTES + OUTPUT_BYTES + REGIONS_BYTES)
        except BaseException:
            self.close()
            raise

    def close(self):
        memory, self.memory = self.memory, None
        self.input = self.output = None
        self.prepared_regions = self.prepared_signals = None
        if memory is not None:
            memory.close()
            if self.owner:
                memory.unlink()
