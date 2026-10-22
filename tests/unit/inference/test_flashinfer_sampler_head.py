"""Exercise the real FlashInfer and Triton kernels; requires a CUDA vLLM install."""

import pytest
import torch

from prime_rl.inference.vllm.flashinfer_sampler_head import sample_with_processed_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA, vLLM and FlashInfer")


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("top_k,top_p", [(None, None), (2, None), (None, 0.5), (3, 0.7)])
@pytest.mark.parametrize("tied", [False, True])
def test_sample_and_sparse_logprobs_share_support(batch_size, top_k, top_p, tied):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

    row = [0.0] * 4 if tied else [1.0, -0.5, 0.2, -2.0]
    logits = torch.tensor([row] * batch_size, device="cuda")
    k = None if top_k is None else torch.full((batch_size,), top_k, device="cuda", dtype=torch.int32)
    p = None if top_p is None else torch.full((batch_size,), top_p, device="cuda")
    expected = apply_top_k_top_p(logits.clone(), k, p)
    sampled, processed = sample_with_processed_logits(logits.clone(), k, p)
    torch.testing.assert_close(processed, expected)
    assert torch.isfinite(processed.gather(1, sampled[:, None])).all()

    reference = expected.log_softmax(-1)
    for recording_k in (0, 1, 4):
        head = compute_topk_scores(processed, recording_k, sampled)
        assert head.logprobs.shape == (batch_size, recording_k + 1)
        torch.testing.assert_close(head.logprob_token_ids[:, 0], sampled)
        torch.testing.assert_close(head.logprobs, reference.gather(1, head.logprob_token_ids))
    # Changing recording width never renormalizes just the recorded head.
    torch.testing.assert_close(reference.exp().sum(-1), torch.ones(batch_size, device="cuda"))


def test_singleton_support_and_sampled_token_outside_recorded_head():
    from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

    logits = torch.tensor([[-float("inf"), 2.0, -float("inf")]], device="cuda")
    sampled, processed = sample_with_processed_logits(logits, None, None)
    assert sampled.item() == 1
    head = compute_topk_scores(processed, 1, sampled)
    torch.testing.assert_close(head.logprobs, torch.zeros_like(head.logprobs))

    # The logprob API must preserve a sampled action outside the recording top-k.
    logits = torch.tensor([[3.0, 2.0, 1.0]], device="cuda")
    sampled = torch.tensor([2], device="cuda")
    head = compute_topk_scores(logits, 1, sampled)
    assert head.logprob_token_ids.tolist() == [[2, 0]]
    torch.testing.assert_close(head.logprobs, logits.log_softmax(-1)[:, [2, 0]])
