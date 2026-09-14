import hashlib
import json
import struct

import pytest

from prime_rl.shardcast.protocol import Publication, base_alias, pack_adapter, split_adapter


def test_bundle_round_trip_and_corruption(tmp_path):
    (tmp_path / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}')
    (tmp_path / "adapter_model.safetensors").write_bytes(b"tensor-payload")
    payload = pack_adapter(tmp_path, 1024)
    publication = Publication("run1", "Qwen/Qwen3.5-2B", 3, hashlib.sha256(payload).hexdigest(), len(payload))
    publication.verify(payload)
    assert split_adapter(payload)[1] == b"tensor-payload"
    assert publication.model_name != base_alias("run1")
    with pytest.raises(ValueError, match="integrity"):
        publication.verify(payload[:-1] + b"x")
    with pytest.raises(ValueError, match="max_adapter_bytes"):
        pack_adapter(tmp_path, 20)


@pytest.mark.parametrize(
    "payload", [b"bad", b"SC-LORA1" + struct.pack("!Q", 9999999) + b"x", b"SC-LORA1" + struct.pack("!Q", 2) + b"[]x"]
)
def test_reject_malformed_bundle(payload):
    with pytest.raises(ValueError):
        split_adapter(payload)


def test_names_isolate_runs_and_versions():
    a = Publication("run1", "model", 1, "a" * 64, 100)
    b = Publication("run2", "model", 1, "a" * 64, 100)
    c = Publication("run1", "model", 2, "b" * 64, 100)
    assert len({a.model_name, b.model_name, c.model_name}) == 3
    assert Publication(**json.loads(json.dumps(a.to_dict()))) == a
    with pytest.raises(ValueError):
        base_alias("../escape")
