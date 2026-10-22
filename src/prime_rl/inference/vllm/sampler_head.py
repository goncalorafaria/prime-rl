"""Compact export of score-centering sampler heads on ``/inference/v1/generate``.

Score centering asks vLLM for the top-k processed logprobs of every generated token.
Upstream turns each of the k alternatives into a ``Logprob`` object and a JSON entry,
and under top-p most of them are masked (``-inf``, serialized as the ``-9999`` clamp),
so decode throughput collapses on the per-alternative Python work. Requests that opt
in via ``sampling_params.extra_args["pack_sampler_head"]`` instead get the head as
base64 arrays — ``counts`` (entries per position), ``ids`` and ``logprobs`` — holding
the sampled token first and then the remaining finite alternatives, and the per-token
logprobs the upstream handler serializes are trimmed to the sampled token.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pybase64

PACK_SAMPLER_HEAD = "pack_sampler_head"
# vLLM clamps serialized logprobs to this value; entries at or below it carry no mass.
LOGPROB_FLOOR = -9999.0


def wants_packed_sampler_head(sampling_params: Any) -> bool:
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return bool(extra_args.get(PACK_SAMPLER_HEAD)) and bool(getattr(sampling_params, "logprobs", None))


def _flat_arrays(logprobs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(starts, ends, token_ids, logprobs)`` from ``FlatLogprobs`` or ``list[dict[int, Logprob]]``."""
    if hasattr(logprobs, "start_indices"):
        return (
            np.asarray(logprobs.start_indices, dtype=np.int64),
            np.asarray(logprobs.end_indices, dtype=np.int64),
            np.asarray(logprobs.token_ids, dtype=np.int64),
            np.asarray(logprobs.logprobs, dtype=np.float32),
        )
    starts, ends, token_ids, values = [], [], [], []
    for position in logprobs:
        starts.append(len(token_ids))
        for token_id, logprob in (position or {}).items():
            token_ids.append(token_id)
            values.append(logprob.logprob)
        ends.append(len(token_ids))
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(token_ids, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
    )


def pack_sampler_head(logprobs: Any) -> tuple[dict[str, Any], np.ndarray]:
    """Encode a request's sample logprobs; also return each position's first entry index.

    vLLM stores the sampled token first at every position, followed by the top-k
    (which may repeat the sampled token). The packed head keeps the sampled entry, then
    the other top-k entries with finite mass, in order.
    """
    starts, ends, token_ids, values = _flat_arrays(logprobs)
    sizes = ends - starts
    if (sizes < 1).any():
        raise ValueError("Every generated token needs its sampled logprob to pack a sampler head")
    position = np.repeat(np.arange(len(starts)), sizes)
    first = np.zeros(len(token_ids), dtype=bool)
    first[starts] = True
    sampled = token_ids[starts]
    keep = first | ((values > LOGPROB_FLOOR) & (token_ids != sampled[position]))
    counts = np.bincount(position[keep], minlength=len(starts)).astype(np.int32)
    head = {
        "counts": pybase64.b64encode(memoryview(np.ascontiguousarray(counts))).decode("ascii"),
        "ids": pybase64.b64encode(memoryview(np.ascontiguousarray(token_ids[keep].astype(np.int32)))).decode("ascii"),
        "logprobs": pybase64.b64encode(memoryview(np.ascontiguousarray(values[keep]))).decode("ascii"),
        "num_positions": int(len(starts)),
    }
    return head, starts
