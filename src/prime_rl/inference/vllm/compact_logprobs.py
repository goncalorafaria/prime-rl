"""Compact vLLM logprobs with parallel normalization and sampled-token ranks."""

import os
from functools import wraps

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors


@triton.jit
def _partials(
    X, Sample, Max, Sum, Rank, V: tl.constexpr, STRIDE: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    j = tl.arange(0, BLOCK)
    ids = tile * BLOCK + j
    x = tl.load(X + row * STRIDE + ids, ids < V, other=-float("inf")).to(tl.float32)
    selected = tl.load(Sample + row)
    selected_x = tl.load(X + row * STRIDE + selected).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - tl.maximum(m, -3.402823466e38)), 0)
    rank = tl.sum(((x >= selected_x) & (ids < V)).to(tl.int32), 0)
    offset = row * TILES + tile
    tl.store(Max + offset, m)
    tl.store(Sum + offset, z)
    tl.store(Rank + offset, rank)


@triton.jit
def _finish_dense(
    X,
    Sample,
    Max,
    Sum,
    Rank,
    TopIds,
    OutIds,
    OutLP,
    OutRank,
    STRIDE: tl.constexpr,
    TILES: tl.constexpr,
    K: tl.constexpr,
    REDUCE: tl.constexpr,
    HEAD: tl.constexpr,
):
    row = tl.program_id(0)
    t = tl.arange(0, REDUCE)
    m = tl.load(Max + row * TILES + t, t < TILES, other=-float("inf"))
    z = tl.load(Sum + row * TILES + t, t < TILES, other=0.0)
    r = tl.load(Rank + row * TILES + t, t < TILES, other=0)
    maximum = tl.max(m, 0)
    normalizer = tl.log(tl.sum(z * tl.exp(m - maximum), 0))
    sampled = tl.load(Sample + row)
    selected = tl.load(X + row * STRIDE + sampled).to(tl.float32)
    tl.store(OutIds + row * (K + 1), sampled)
    tl.store(OutLP + row * (K + 1), selected - maximum - normalizer)
    tl.store(OutRank + row, tl.sum(r, 0))
    if K > 0:
        j = tl.arange(0, HEAD)
        token = tl.load(TopIds + row * K + j, j < K, other=0)
        value = tl.load(X + row * STRIDE + token, j < K, other=-float("inf")).to(tl.float32)
        tl.store(OutIds + row * (K + 1) + 1 + j, token, j < K)
        tl.store(OutLP + row * (K + 1) + 1 + j, value - maximum - normalizer, j < K)


def compute_compact_scores(logits, num_logprobs, sampled_token_ids, block=4096):
    batch, vocab = logits.shape
    k = num_logprobs
    tiles = triton.cdiv(vocab, block)
    if k and vocab >= 4096 and logits.is_contiguous():
        from flashinfer import top_k

        # Extraction only: these IDs never restrict the sampling distribution.
        top_ids = top_k(logits, k, sorted=True, deterministic=True)[1]
    else:
        top_ids = torch.topk(logits, k, dim=-1).indices if k else sampled_token_ids
    maxima = torch.empty((batch, tiles), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    ranks = torch.empty((batch, tiles), device=logits.device, dtype=torch.int32)
    _partials[(batch, tiles)](logits, sampled_token_ids, maxima, sums, ranks, vocab, logits.stride(0), tiles, block)
    out_ids = torch.empty((batch, k + 1), device=logits.device, dtype=torch.int64)
    out_lp = torch.empty((batch, k + 1), device=logits.device, dtype=torch.float32)
    out_rank = torch.empty((batch,), device=logits.device, dtype=torch.int64)
    _finish_dense[(batch,)](
        logits,
        sampled_token_ids,
        maxima,
        sums,
        ranks,
        top_ids,
        out_ids,
        out_lp,
        out_rank,
        logits.stride(0),
        tiles,
        k,
        triton.next_power_of_2(tiles),
        triton.next_power_of_2(max(k, 1)),
    )
    return LogprobsTensors(out_ids, out_lp, out_rank)


ENV_VAR = "VLLM_PRIME_COMPACT_LOGPROBS"


def patch_compact_logprobs():
    """Install the opt-in extraction path in the pinned vLLM V2 sampler."""
    if os.environ.get(ENV_VAR) != "1":
        return

    from vllm import envs

    # The fork owns extraction when its native implementation is enabled.
    if getattr(envs, "VLLM_USE_COMPACT_LOGPROBS", False):
        return

    import vllm
    from vllm.v1.worker.gpu.sample import sampler

    original = sampler.compute_topk_scores
    if getattr(original, "_prime_compact_logprobs", False):
        return
    original_init = sampler.Sampler.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        # vLLM catches plugin-load errors; enforce the pin at worker construction.
        if vllm.__version__.split("+")[0] != "0.29.0":
            raise RuntimeError(f"{ENV_VAR} requires vLLM 0.29.0")
        original_init(self, *args, **kwargs)

    @wraps(original)
    def compute(
        logits,
        num_logprobs,
        sampled_token_ids,
        cu_num_logits=None,
        logprob_token_ids_state=None,
        expanded_idx_mapping=None,
        max_per_req_token_ids=0,
        logits_mode=False,
    ):
        if (
            logits.is_cuda
            and logits.ndim == 2
            and logits.shape[0] > 0
            and logits.stride(1) == 1
            and logits.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and sampled_token_ids.ndim == 1
            and sampled_token_ids.is_contiguous()
            and 0 <= num_logprobs <= min(128, logits.shape[1])
            and max_per_req_token_ids == 0
            and not logits_mode
        ):
            result = compute_compact_scores(logits, num_logprobs, sampled_token_ids)
            if isinstance(cu_num_logits, torch.Tensor):
                result = result._replace(cu_num_generated_tokens_tensor=cu_num_logits)
            elif cu_num_logits is not None:
                result = result._replace(cu_num_generated_tokens=cu_num_logits)
            return result
        return original(
            logits,
            num_logprobs,
            sampled_token_ids,
            cu_num_logits,
            logprob_token_ids_state,
            expanded_idx_mapping,
            max_per_req_token_ids,
            logits_mode,
        )

    compute._prime_compact_logprobs = True
    sampler.Sampler.__init__ = init
    sampler.compute_topk_scores = compute
