"""Sanity tests for the prime-RL ``ServingTokens`` subclass.

The full happy-path is owned upstream by vLLM's
``vllm/entrypoints/serve/disagg`` test suite. We only cover the prime-RL
deltas here:
    * ``serialize_routed_experts`` round-trips a compact raw-byte payload.
    * The subclass overrides ``serve_tokens_full_generator`` without
      monkey-patching the parent.
    * ``post_process`` swaps in the compact routed_experts while preserving
      the rest of the upstream response (``usage`` included).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pybase64
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateResponse, GenerateResponseChoice
from vllm.entrypoints.serve.engine.protocol import UsageInfo

from prime_rl.inference.vllm.routed_experts import serialize_routed_experts
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlServingTokens,
    _GenerateRoutedExpertsCapture,
    _SamplerHeadCapture,
)


def _decode_routed_experts(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=np.uint8,
    ).reshape(encoded["shape"])


async def _empty_request_outputs():
    if False:
        yield


def test_subclass_overrides_serve_tokens_full_generator():
    upstream = PrimeRlServingTokens.__mro__[1]
    assert PrimeRlServingTokens.serve_tokens_full_generator is not upstream.serve_tokens_full_generator


def test_serialize_routed_experts_uses_compact_raw_payload():
    routed_experts = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int64,
    )

    encoded = serialize_routed_experts(routed_experts)
    assert encoded is not None

    decoded = _decode_routed_experts(encoded)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, routed_experts)


def test_generate_response_post_process_replaces_upstream_routed_experts():
    compact_routed_experts = {"data": "AQID", "shape": [1, 1, 3], "start": 0}
    capture = _GenerateRoutedExpertsCapture(_empty_request_outputs())
    capture.routed_experts[0] = compact_routed_experts
    usage = UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7)
    response = GenerateResponse(
        request_id="request-id",
        model="test-model",
        choices=[
            GenerateResponseChoice(
                index=0,
                token_ids=[1, 2, 3],
                routed_experts="upstream-npy-payload",
            )
        ],
        usage=usage,
    )

    processed = capture.post_process(response)

    assert processed.choices[0].routed_experts == compact_routed_experts
    assert processed.model == "test-model"
    assert processed.usage == usage
    # The compact object form must survive JSON serialization (the parent
    # declares ``routed_experts`` as a base64 string).
    payload = processed.model_dump(mode="json")
    assert payload["choices"][0]["routed_experts"] == compact_routed_experts
    assert payload["usage"]["total_tokens"] == 7


def test_sampler_head_capture_packs_head_and_trims_logprobs():
    from vllm.logprobs import FlatLogprobs

    flat = FlatLogprobs()
    flat.append_fast([7, 7, 9, 0], [-0.25, -0.25, -1.5, float("-inf")], iter([2, 1, 2, 3]), [None] * 4)
    flat.append_fast([8, 3, 4], [-0.0, float("-inf"), float("-inf")], iter([1, 1, 2]), [None] * 3)
    output = SimpleNamespace(index=0, logprobs=flat)

    async def outputs():
        yield SimpleNamespace(outputs=[output])

    capture = _SamplerHeadCapture(outputs())

    async def drain():
        return [item async for item in capture]

    asyncio.run(drain())

    head = capture.sampler_heads[0]
    counts = np.frombuffer(pybase64.b64decode(head["counts"]), dtype=np.int32)
    ids = np.frombuffer(pybase64.b64decode(head["ids"]), dtype=np.int32)
    assert counts.tolist() == [2, 1]
    assert ids.tolist() == [7, 9, 8]
    # Upstream now serializes only the sampled token at each position.
    assert len(output.logprobs) == 2
    assert [list(output.logprobs[i].keys()) for i in range(2)] == [[7], [8]]
    assert output.logprobs[0][7].logprob == -0.25

    response = GenerateResponse(
        request_id="request-id",
        model="test-model",
        choices=[GenerateResponseChoice(index=0, token_ids=[7, 8])],
        usage=UsageInfo(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )
    payload = capture.post_process(response).model_dump(mode="json")
    assert payload["choices"][0]["sampler_head"] == head
    assert payload["choices"][0]["token_ids"] == [7, 8]
