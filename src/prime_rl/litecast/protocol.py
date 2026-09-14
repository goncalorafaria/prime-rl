"""Immutable adapter bundles and version identities shared by publishers and workers."""

import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

MAGIC = b"SC-LORA1"
MAX_CONFIG_BYTES = 1024 * 1024


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("run_id must contain 1-80 letters, digits, underscores or hyphens")
    return run_id


def base_alias(run_id: str) -> str:
    return f"sc-{validate_run_id(run_id)}-base"


def desired_key(run_id: str) -> str:
    return f"prime_rl:litecast:{validate_run_id(run_id)}:desired"


def peer_service(run_id: str, digest: str) -> str:
    return f"litecast:{validate_run_id(run_id)}:{digest}"


def pack_adapter(directory: Path, max_bytes: int) -> bytes:
    config = directory / "adapter_config.json"
    weights = directory / "adapter_model.safetensors"
    if config.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("adapter configuration is too large")
    if len(MAGIC) + 8 + config.stat().st_size + weights.stat().st_size > max_bytes:
        raise ValueError("adapter bundle exceeds max_adapter_bytes")
    payload = MAGIC + struct.pack("!Q", config.stat().st_size) + config.read_bytes() + weights.read_bytes()
    split_adapter(payload)
    return payload


def split_adapter(payload: bytes) -> tuple[bytes, bytes]:
    if len(payload) < 16 or payload[:8] != MAGIC:
        raise ValueError("invalid adapter bundle")
    length = struct.unpack("!Q", payload[8:16])[0]
    if not 0 < length <= MAX_CONFIG_BYTES or 16 + length >= len(payload):
        raise ValueError("invalid adapter configuration length")
    config, weights = payload[16 : 16 + length], payload[16 + length :]
    value = json.loads(config)
    if not isinstance(value, dict) or value.get("peft_type") != "LORA":
        raise ValueError("only PEFT LoRA adapters are supported")
    return config, weights


@dataclass(frozen=True)
class Publication:
    run_id: str
    base_model: str
    step: int
    digest: str
    size: int

    def __post_init__(self):
        validate_run_id(self.run_id)
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if type(self.size) is not int or self.size <= 16:
            raise ValueError("invalid bundle size")
        if not re.fullmatch(r"[0-9a-f]{64}", self.digest):
            raise ValueError("invalid bundle digest")
        if not self.base_model:
            raise ValueError("base_model is required")

    @property
    def model_name(self) -> str:
        return f"sc-{self.run_id}-s{self.step}-{self.digest}"

    def to_dict(self) -> dict:
        return asdict(self)

    def verify(self, payload: bytes) -> None:
        if len(payload) != self.size or hashlib.sha256(payload).hexdigest() != self.digest:
            raise ValueError("adapter bundle integrity failure")
        split_adapter(payload)
