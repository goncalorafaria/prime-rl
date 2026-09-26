"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.scale_out.token_in_token_out.serving.ServingTokens`` that covers
prefix-cache salting, lora dispatch, multimodal features, prompt logprobs,
priority, ``data_parallel_rank`` header routing, server-side ``max_tokens``
defaulting and ``usage`` reporting. We subclass it for the one bit still
missing from the upstream handler: compact ``routed_experts`` export — when the
engine emits routing decisions, surface them as ``{data, shape, start, dtype}``
base64 raw-byte objects (the form the PD router can merge and the renderers
parse) instead of upstream's single ``.npy`` base64 string. Requests that opt in
via ``extra_args["pack_sampler_head"]`` likewise get their top-k sample logprobs as
a compact ``sampler_head`` (see ``sampler_head.py``).

Everything else (request/response schema, sampling params, error handling)
delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.outputs import RequestOutput

from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture
from prime_rl.inference.vllm.sampler_head import pack_sampler_head, wants_packed_sampler_head


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    # Overrides upstream's base64 ``.npy`` string form with the compact
    # ``{data, shape, start, dtype}`` object the PD router merges and the
    # renderers parse.
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]
    # Compact top-k sample logprobs ``{counts, ids, logprobs, num_positions}`` for
    # requests with ``extra_args["pack_sampler_head"]``.
    sampler_head: dict[str, Any] | None = None


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]


class _GenerateRoutedExpertsCapture(RoutedExpertsCapture):
    def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        choices = [
            PrimeRlGenerateResponseChoice(
                **choice.model_dump(exclude={"routed_experts", "sampler_head"}),
                routed_experts=self.routed_experts.get(choice.index),
                sampler_head=getattr(choice, "sampler_head", None),
            )
            for choice in response.choices
        ]
        return PrimeRlGenerateResponse(**{**dict(response), "choices": choices})


def _sampled_only(logprobs: Any, starts: Any) -> FlatLogprobs | list[dict[int, Logprob]]:
    """Each position's sampled-token entry only, in the container type vLLM produced."""
    if isinstance(logprobs, FlatLogprobs):
        idx = starts.tolist()
        return FlatLogprobs(
            start_indices=list(range(len(idx))),
            end_indices=list(range(1, len(idx) + 1)),
            token_ids=[logprobs.token_ids[i] for i in idx],
            logprobs=[logprobs.logprobs[i] for i in idx],
            ranks=[logprobs.ranks[i] for i in idx],
            decoded_tokens=[logprobs.decoded_tokens[i] for i in idx],
        )
    return [dict([next(iter(position.items()))]) if position else position for position in logprobs]


class _SamplerHeadCapture:
    """Pack each output's sample logprobs into a compact sampler head as outputs stream,
    trimming ``output.logprobs`` to the sampled token so upstream serializes one entry
    per generated token."""

    def __init__(self, generator: AsyncGenerator[RequestOutput, None]):
        self._generator = generator
        self.sampler_heads: dict[int, dict[str, Any]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                if output.logprobs is None:
                    continue
                head, starts = pack_sampler_head(output.logprobs)
                self.sampler_heads[output.index] = head
                output.logprobs = _sampled_only(output.logprobs, starts)
            yield request_output

    def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        choices = [
            PrimeRlGenerateResponseChoice(
                **choice.model_dump(exclude={"routed_experts", "sampler_head"}),
                routed_experts=getattr(choice, "routed_experts", None),
                sampler_head=self.sampler_heads.get(choice.index),
            )
            for choice in response.choices
        ]
        return PrimeRlGenerateResponse(**{**dict(response), "choices": choices})


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens + compact routed experts and sampler heads."""

    async def serve_tokens_full_generator(  # type: ignore[override]
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse:
        # Capture routed_experts as vLLM streams request outputs, then post-process
        # the final response into our GenerateResponse subclass so the encoded
        # experts surface in the JSON.
        capture: _GenerateRoutedExpertsCapture | None = None
        if self.model_config.enable_return_routed_experts:
            capture = _GenerateRoutedExpertsCapture(
                result_generator,
                start=request.sampling_params.routed_experts_prompt_start,
            )
            result_generator = capture
        head_capture: _SamplerHeadCapture | None = None
        if wants_packed_sampler_head(request.sampling_params):
            head_capture = _SamplerHeadCapture(result_generator)
            result_generator = head_capture

        response = await super().serve_tokens_full_generator(
            request, result_generator, request_id, model_name, request_metadata
        )

        if capture is not None and isinstance(response, GenerateResponse):
            response = capture.post_process(response)
        if head_capture is not None and isinstance(response, GenerateResponse):
            response = head_capture.post_process(response)

        return response
