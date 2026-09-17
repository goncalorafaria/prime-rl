import hashlib
import json
import struct

import pytest

from prime_rl.litecast.protocol import Publication, base_alias, pack_adapter, split_adapter


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


def test_hybrid_cache_shares_only_matching_histories():
    from prime_rl.inference.vllm.hybrid_lora import cache_namespace

    v7 = [{"version": "v7", "computed_tokens": 0, "output_tokens": 0}]
    hybrid = v7 + [{"version": "v8", "computed_tokens": 1024, "output_tokens": 32}]
    assert cache_namespace("tenant", "A", v7) == cache_namespace("tenant", "A", list(v7))
    assert cache_namespace("tenant", "A", v7) != cache_namespace("tenant", "B", v7)
    assert cache_namespace("tenant", "A", v7) != cache_namespace("other", "A", v7)
    assert cache_namespace("tenant", "A", hybrid) != cache_namespace("tenant", "A", v7)
    assert cache_namespace("tenant", "A", hybrid) != cache_namespace("tenant", "A", [{"version": "v8", "computed_tokens": 0, "output_tokens": 0}])
    other_boundary = v7 + [{"version": "v8", "computed_tokens": 1040, "output_tokens": 48}]
    assert cache_namespace("tenant", "A", hybrid) != cache_namespace("tenant", "A", other_boundary)


def test_live_lineage_rejects_missing_or_invalid_token_history():
    from prime_rl.litecast.lineage import validate_lineage
    record = {'semantics':'retained_state','output_tokens':10,
              'segments':[{'output_start':0,'version':'v7'},{'output_start':4,'version':'v8'}]}
    validate_lineage(record,10)
    with pytest.raises(ValueError):
        validate_lineage(None,10)
    with pytest.raises(ValueError):
        validate_lineage(record,9)
    record['segments'][1]['output_start']=11
    with pytest.raises(ValueError):
        validate_lineage(record,10)
