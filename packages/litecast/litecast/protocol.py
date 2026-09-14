"""Pure-Python wire framing for the Litecast data plane.

Transport code is intentionally kept out of this module.  A UCXX endpoint can
receive :data:`HEADER_SIZE` bytes, validate the header with ``decode_header``,
and then receive exactly the declared payload size.
"""

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Mapping, Tuple


MAGIC = b"SCST"
PROTOCOL_VERSION = 1
MAX_PAYLOAD_SIZE = 64 * 1024 * 1024
MAX_TEXT_SIZE = 64 * 1024
MAX_VERSION_NAME_SIZE = 1024
MAX_BATCH_SHARDS = 64

# magic, protocol version, message type, flags, payload length
HEADER = struct.Struct("!4sBBHQ")
HEADER_SIZE = HEADER.size
CHUNK_PREFIX = struct.Struct("!I")


class ProtocolError(ValueError):
    """Raised when a wire frame is malformed or violates protocol limits."""


class MessageType(IntEnum):
    GET_VERSION = 1
    VERSION_METADATA = 2
    METADATA = 2
    CHUNK_DATA = 3
    ERROR = 4
    SHUTDOWN = 5
    CLEAN_SHUTDOWN = 5
    GET_SHARD = 6
    SHARD_METADATA = 7
    WAIT_VERSION = 8
    VERSION_AVAILABLE = 9
    GET_SHARDS = 10
    SHARDS_METADATA = 11


@dataclass(frozen=True)
class Frame:
    """A decoded protocol frame."""

    message_type: MessageType
    payload: bytes = b""
    flags: int = 0
    protocol_version: int = PROTOCOL_VERSION


def _coerce_payload(payload: Any) -> bytes:
    try:
        view = memoryview(payload)
    except TypeError:
        raise TypeError("payload must support the buffer protocol")
    if not view.contiguous:
        return view.tobytes()
    return bytes(view)


def encode_frame(
    message_type: MessageType,
    payload: Any = b"",
    flags: int = 0,
    protocol_version: int = PROTOCOL_VERSION,
    max_payload_size: int = MAX_PAYLOAD_SIZE,
) -> bytes:
    """Encode one complete length-prefixed frame."""
    body = _coerce_payload(payload)
    _validate_limit(max_payload_size)
    if len(body) > max_payload_size:
        raise ProtocolError("payload exceeds maximum size")
    try:
        kind = MessageType(message_type)
    except (TypeError, ValueError):
        raise ProtocolError("unknown message type")
    if not 0 <= flags <= 0xFFFF:
        raise ProtocolError("flags must fit in uint16")
    if not 0 <= protocol_version <= 0xFF:
        raise ProtocolError("protocol version must fit in uint8")
    return HEADER.pack(MAGIC, protocol_version, int(kind), flags, len(body)) + body


def decode_header(
    header: Any, max_payload_size: int = MAX_PAYLOAD_SIZE
) -> Tuple[MessageType, int, int]:
    """Validate a fixed-size header and return type, flags, and payload size."""
    raw = _coerce_payload(header)
    if len(raw) != HEADER_SIZE:
        raise ProtocolError("invalid header size")
    _validate_limit(max_payload_size)
    magic, version, raw_type, flags, payload_size = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ProtocolError("invalid protocol magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version: {}".format(version))
    try:
        message_type = MessageType(raw_type)
    except ValueError:
        raise ProtocolError("unknown message type: {}".format(raw_type))
    if payload_size > max_payload_size:
        raise ProtocolError("declared payload exceeds maximum size")
    return message_type, flags, payload_size


def decode_frame(data: Any, max_payload_size: int = MAX_PAYLOAD_SIZE) -> Frame:
    """Decode exactly one frame, rejecting truncation and trailing bytes."""
    raw = _coerce_payload(data)
    if len(raw) < HEADER_SIZE:
        raise ProtocolError("truncated frame header")
    message_type, flags, payload_size = decode_header(
        raw[:HEADER_SIZE], max_payload_size=max_payload_size
    )
    expected = HEADER_SIZE + payload_size
    if len(raw) < expected:
        raise ProtocolError("truncated frame payload")
    if len(raw) > expected:
        raise ProtocolError("trailing bytes after frame")
    frame = Frame(message_type, raw[HEADER_SIZE:], flags)
    _validate_message(frame)
    return frame


class FrameDecoder:
    """Incrementally decode frames from an arbitrary byte stream."""

    def __init__(self, max_payload_size: int = MAX_PAYLOAD_SIZE) -> None:
        _validate_limit(max_payload_size)
        self._maximum = max_payload_size
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, data: Any) -> List[Frame]:
        incoming = _coerce_payload(data)
        self._buffer.extend(incoming)
        frames = []
        while len(self._buffer) >= HEADER_SIZE:
            kind, flags, payload_size = decode_header(
                self._buffer[:HEADER_SIZE], self._maximum
            )
            frame_size = HEADER_SIZE + payload_size
            if len(self._buffer) < frame_size:
                break
            body = bytes(self._buffer[HEADER_SIZE:frame_size])
            del self._buffer[:frame_size]
            frame = Frame(kind, body, flags)
            _validate_message(frame)
            frames.append(frame)
        return frames


def encode_get_version(version: str) -> bytes:
    return encode_frame(MessageType.GET_VERSION, _encode_text(version, "version", MAX_VERSION_NAME_SIZE))


def decode_get_version(frame: Frame) -> str:
    _require_type(frame, MessageType.GET_VERSION)
    return _decode_text(frame.payload, "version", MAX_VERSION_NAME_SIZE)


def encode_get_shard(version: str, index: int) -> bytes:
    if (
        not isinstance(version, str)
        or not version
        or len(version.encode("utf-8")) > MAX_VERSION_NAME_SIZE
    ):
        raise ProtocolError("invalid version")
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index <= 0xFFFFFFFF
    ):
        raise ProtocolError("shard index must fit in uint32")
    payload = json.dumps(
        {"version": version, "index": index},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return encode_frame(MessageType.GET_SHARD, payload)


def decode_get_shard(frame: Frame) -> Tuple[str, int]:
    _require_type(frame, MessageType.GET_SHARD)
    try:
        value = json.loads(frame.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid shard request") from exc
    if not isinstance(value, dict) or set(value) != {"version", "index"}:
        raise ProtocolError("invalid shard request")
    version = value["version"]
    index = value["index"]
    if (
        not isinstance(version, str)
        or not version
        or len(version.encode("utf-8")) > MAX_VERSION_NAME_SIZE
        or not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index <= 0xFFFFFFFF
    ):
        raise ProtocolError("invalid shard request")
    return version, index


def encode_wait_version(version: str) -> bytes:
    return encode_frame(
        MessageType.WAIT_VERSION,
        _encode_text(version, "version", MAX_VERSION_NAME_SIZE),
    )


def decode_wait_version(frame: Frame) -> str:
    _require_type(frame, MessageType.WAIT_VERSION)
    return _decode_text(frame.payload, "version", MAX_VERSION_NAME_SIZE)


def encode_version_available(version: str) -> bytes:
    return encode_frame(
        MessageType.VERSION_AVAILABLE,
        _encode_text(version, "version", MAX_VERSION_NAME_SIZE),
    )


def decode_version_available(frame: Frame) -> str:
    _require_type(frame, MessageType.VERSION_AVAILABLE)
    return _decode_text(frame.payload, "version", MAX_VERSION_NAME_SIZE)


def encode_get_shards(version: str, indices: Iterable[int]) -> bytes:
    values = list(indices)
    if (
        not isinstance(version, str)
        or not version
        or len(version.encode("utf-8")) > MAX_VERSION_NAME_SIZE
        or not values
        or len(values) > MAX_BATCH_SHARDS
        or len(set(values)) != len(values)
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index <= 0xFFFFFFFF
            for index in values
        )
    ):
        raise ProtocolError("invalid batched shard request")
    payload = json.dumps(
        {"version": version, "indices": values},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return encode_frame(MessageType.GET_SHARDS, payload)


def decode_get_shards(frame: Frame) -> Tuple[str, Tuple[int, ...]]:
    _require_type(frame, MessageType.GET_SHARDS)
    try:
        value = json.loads(frame.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid batched shard request") from exc
    if not isinstance(value, dict) or set(value) != {"version", "indices"}:
        raise ProtocolError("invalid batched shard request")
    version = value["version"]
    indices = value["indices"]
    if (
        not isinstance(version, str)
        or not version
        or len(version.encode("utf-8")) > MAX_VERSION_NAME_SIZE
        or not isinstance(indices, list)
        or not indices
        or len(indices) > MAX_BATCH_SHARDS
        or len(set(indices)) != len(indices)
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index <= 0xFFFFFFFF
            for index in indices
        )
    ):
        raise ProtocolError("invalid batched shard request")
    return version, tuple(indices)


def encode_metadata(metadata: Mapping[str, Any]) -> bytes:
    """Encode version metadata as bounded, deterministic UTF-8 JSON."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    try:
        payload = json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("metadata is not JSON serializable") from exc
    if len(payload) > MAX_TEXT_SIZE:
        raise ProtocolError("metadata exceeds maximum size")
    return encode_frame(MessageType.METADATA, payload)


def decode_metadata(frame: Frame) -> Dict[str, Any]:
    _require_type(frame, MessageType.METADATA)
    if len(frame.payload) > MAX_TEXT_SIZE:
        raise ProtocolError("metadata exceeds maximum size")
    try:
        value = json.loads(frame.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid metadata JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("metadata must decode to an object")
    return value


def encode_shard_metadata(metadata: Mapping[str, Any]) -> bytes:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    payload = json.dumps(
        dict(metadata), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(payload) > MAX_TEXT_SIZE:
        raise ProtocolError("metadata exceeds maximum size")
    return encode_frame(MessageType.SHARD_METADATA, payload)


def decode_shard_metadata(frame: Frame) -> Dict[str, Any]:
    _require_type(frame, MessageType.SHARD_METADATA)
    try:
        value = json.loads(frame.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid shard metadata") from exc
    if not isinstance(value, dict):
        raise ProtocolError("shard metadata must decode to an object")
    return value


def encode_shards_metadata(metadata: Mapping[str, Any]) -> bytes:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    payload = json.dumps(
        dict(metadata), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(payload) > MAX_TEXT_SIZE:
        raise ProtocolError("metadata exceeds maximum size")
    return encode_frame(MessageType.SHARDS_METADATA, payload)


def decode_shards_metadata(frame: Frame) -> Dict[str, Any]:
    _require_type(frame, MessageType.SHARDS_METADATA)
    try:
        value = json.loads(frame.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid batched shard metadata") from exc
    if not isinstance(value, dict):
        raise ProtocolError("batched shard metadata must decode to an object")
    return value


def encode_chunk_data(index: int, data: Any) -> bytes:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= 0xFFFFFFFF:
        raise ProtocolError("chunk index must fit in uint32")
    body = _coerce_payload(data)
    if len(body) > MAX_PAYLOAD_SIZE - CHUNK_PREFIX.size:
        raise ProtocolError("chunk exceeds maximum size")
    return encode_frame(MessageType.CHUNK_DATA, CHUNK_PREFIX.pack(index) + body)


def decode_chunk_data(frame: Frame) -> Tuple[int, memoryview]:
    _require_type(frame, MessageType.CHUNK_DATA)
    if len(frame.payload) < CHUNK_PREFIX.size:
        raise ProtocolError("chunk frame is missing its index")
    index = CHUNK_PREFIX.unpack(frame.payload[: CHUNK_PREFIX.size])[0]
    return index, memoryview(frame.payload)[CHUNK_PREFIX.size:]


def encode_error(message: str) -> bytes:
    return encode_frame(MessageType.ERROR, _encode_text(message, "error", MAX_TEXT_SIZE))


def decode_error(frame: Frame) -> str:
    _require_type(frame, MessageType.ERROR)
    return _decode_text(frame.payload, "error", MAX_TEXT_SIZE)


def encode_shutdown() -> bytes:
    return encode_frame(MessageType.SHUTDOWN)


def decode_shutdown(frame: Frame) -> None:
    _require_type(frame, MessageType.SHUTDOWN)
    if frame.payload:
        raise ProtocolError("shutdown frame must have an empty payload")


def decode_frames(
    data: Any, max_payload_size: int = MAX_PAYLOAD_SIZE
) -> Iterable[Frame]:
    """Decode a complete sequence of frames; reject an incomplete final frame."""
    decoder = FrameDecoder(max_payload_size)
    frames = decoder.feed(data)
    if decoder.buffered_bytes:
        raise ProtocolError("truncated final frame")
    return frames


def _validate_message(frame: Frame) -> None:
    if frame.message_type == MessageType.SHUTDOWN and frame.payload:
        raise ProtocolError("shutdown frame must have an empty payload")
    if frame.message_type == MessageType.GET_VERSION:
        _decode_text(frame.payload, "version", MAX_VERSION_NAME_SIZE)
    elif frame.message_type == MessageType.GET_SHARD:
        decode_get_shard(frame)
    elif frame.message_type == MessageType.GET_SHARDS:
        decode_get_shards(frame)
    elif frame.message_type == MessageType.WAIT_VERSION:
        decode_wait_version(frame)
    elif frame.message_type == MessageType.VERSION_AVAILABLE:
        decode_version_available(frame)
    elif frame.message_type == MessageType.CHUNK_DATA and len(frame.payload) < CHUNK_PREFIX.size:
        raise ProtocolError("chunk frame is missing its index")
    elif frame.message_type == MessageType.METADATA:
        if len(frame.payload) > MAX_TEXT_SIZE:
            raise ProtocolError("text payload exceeds maximum size")
        decode_metadata(frame)
    elif frame.message_type == MessageType.SHARD_METADATA:
        decode_shard_metadata(frame)
    elif frame.message_type == MessageType.SHARDS_METADATA:
        decode_shards_metadata(frame)
    elif frame.message_type == MessageType.ERROR:
        _decode_text(frame.payload, "error", MAX_TEXT_SIZE)


def _encode_text(value: str, label: str, maximum: int) -> bytes:
    if not isinstance(value, str):
        raise TypeError("{} must be str".format(label))
    payload = value.encode("utf-8")
    if not payload:
        raise ProtocolError("{} must not be empty".format(label))
    if len(payload) > maximum:
        raise ProtocolError("{} exceeds maximum size".format(label))
    return payload


def _decode_text(payload: bytes, label: str, maximum: int) -> str:
    if not payload:
        raise ProtocolError("{} must not be empty".format(label))
    if len(payload) > maximum:
        raise ProtocolError("{} exceeds maximum size".format(label))
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("{} is not valid UTF-8".format(label)) from exc


def _require_type(frame: Frame, expected: MessageType) -> None:
    if not isinstance(frame, Frame) or frame.message_type != expected:
        raise ProtocolError("expected {} frame".format(expected.name))


def _validate_limit(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("max_payload_size must be a non-negative integer")
