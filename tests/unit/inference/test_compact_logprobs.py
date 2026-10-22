"""GPU checks for compact normalization, token selection and rank calculation."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and vLLM")


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("vocab", [129, 4097, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("k", [0, 1, 128])
def test_compact_logprobs(vocab, dtype, tied, k, contiguous):
    from prime_rl.inference.vllm.compact_logprobs import compute_compact_scores

    torch.manual_seed(19)
    logits = torch.randn(16, vocab, device="cuda", dtype=dtype)[::2]
    if contiguous:
        logits = logits.contiguous()
    if tied:
        logits.copy_(logits.round())
    for row, size in enumerate((1, 7, 32, 128, 256, 1000, vocab // 2, vocab)):
        cutoff = logits[row].topk(min(size, vocab)).values[-1]
        logits[row].masked_fill_(logits[row] < cutoff, -torch.inf)
    sampled = torch.multinomial(logits.float().softmax(-1), 1).squeeze(1)
    original_logits = logits.clone()
    result = compute_compact_scores(logits, k, sampled)
    torch.testing.assert_close(logits, original_logits, atol=0, rtol=0)
    reference = logits.float().log_softmax(-1)
    ids = result.logprob_token_ids
    torch.testing.assert_close(ids[:, 0], sampled)
    torch.testing.assert_close(logits.gather(1, ids[:, 1:]), logits.topk(k).values)
    sorted_ids = ids[:, 1:].sort(dim=-1).values
    assert (sorted_ids[:, 1:] > sorted_ids[:, :-1]).all()
    if tied:
        repeated = compute_compact_scores(logits, k, sampled)
        torch.testing.assert_close(ids, repeated.logprob_token_ids)
    torch.testing.assert_close(result.logprobs, reference.gather(1, ids), atol=5e-5, rtol=1e-5)
    rank = (logits >= logits.gather(1, sampled[:, None])).sum(-1)
    torch.testing.assert_close(result.selected_token_ranks, rank)


def test_sampled_token_outside_head():
    from prime_rl.inference.vllm.compact_logprobs import compute_compact_scores

    logits = torch.tensor([[3.0, 2.0, 1.0]], device="cuda")
    sampled = torch.tensor([2], device="cuda")
    result = compute_compact_scores(logits, 1, sampled)
    assert result.logprob_token_ids.tolist() == [[2, 0]]
    torch.testing.assert_close(result.logprobs, logits.log_softmax(-1)[:, [2, 0]])
    assert result.selected_token_ranks.item() == 3
