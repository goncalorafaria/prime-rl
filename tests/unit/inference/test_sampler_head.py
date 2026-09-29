"""Packed score-centering sampler heads (``sampler_head.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pybase64

from prime_rl.inference.vllm.sampler_head import pack_sampler_head, wants_packed_sampler_head


def _flat(positions):
    """A FlatLogprobs-shaped container from per-position [(token_id, logprob), ...] lists."""
    starts, ends, token_ids, logprobs = [], [], [], []
    for entries in positions:
        starts.append(len(token_ids))
        for token_id, logprob in entries:
            token_ids.append(token_id)
            logprobs.append(logprob)
        ends.append(len(token_ids))
    return SimpleNamespace(start_indices=starts, end_indices=ends, token_ids=token_ids, logprobs=logprobs)


def _decode(head):
    counts = np.frombuffer(pybase64.b64decode(head["counts"]), dtype=np.int32)
    ids = np.frombuffer(pybase64.b64decode(head["ids"]), dtype=np.int32)
    logprobs = np.frombuffer(pybase64.b64decode(head["logprobs"]), dtype=np.float32)
    bounds = np.cumsum(counts)[:-1]
    return [list(zip(i.tolist(), lp.tolist())) for i, lp in zip(np.split(ids, bounds), np.split(logprobs, bounds))]


def test_pack_keeps_sampled_first_and_drops_masked_and_duplicates():
    inf = float("-inf")
    positions = [
        # sampled 7 (also rank 1), one real alternative, masked tail (-inf and the -9999 clamp)
        [(7, -0.25), (7, -0.25), (9, -1.5), (0, inf), (1, -9999.0)],
        # sampled token outside the finite top-k entries, all alternatives masked
        [(8, -0.0), (3, inf), (4, inf)],
    ]
    head, starts = pack_sampler_head(_flat(positions))

    assert head["num_positions"] == 2
    assert starts.tolist() == [0, 5]
    assert _decode(head) == [[(7, -0.25), (9, -1.5)], [(8, -0.0)]]


def test_pack_accepts_dict_logprobs():
    logprob = lambda value: SimpleNamespace(logprob=value)
    positions = [{5: logprob(-0.5), 6: logprob(-1.0), 2: logprob(float("-inf"))}]
    head, starts = pack_sampler_head(positions)
    assert starts.tolist() == [0]
    assert _decode(head) == [[(5, -0.5), (6, -1.0)]]


def test_wants_packed_sampler_head_requires_opt_in_and_logprobs():
    assert wants_packed_sampler_head(SimpleNamespace(extra_args={"pack_sampler_head": True}, logprobs=128))
    assert not wants_packed_sampler_head(SimpleNamespace(extra_args={"pack_sampler_head": True}, logprobs=None))
    assert not wants_packed_sampler_head(SimpleNamespace(extra_args=None, logprobs=128))
