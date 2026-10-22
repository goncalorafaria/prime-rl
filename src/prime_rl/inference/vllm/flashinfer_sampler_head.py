"""FlashInfer categorical sampling with vLLM's processed-logprob support.

Like Slime #2406, the head and replay mask describe the distribution actually
sampled. Truncation happens once, before sampling; FlashInfer must not apply a
second top-k/top-p filter with different boundary-tie semantics.
"""

import os
from functools import wraps

import torch

ENV_VAR = "VLLM_PRIME_FLASHINFER_SAMPLER_HEAD"


def sample_with_processed_logits(logits, top_k, top_p):
    """Return sampled IDs and the shared filtered logits, entirely on GPU.

    vLLM's compute_topk_scores consumes these logits downstream and its Triton
    kernel emits only top-k plus sampled-token logprobs. The recording width
    does not affect truncation or the normalizer.
    """
    from flashinfer.sampling import sampling_from_probs
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    processed_logits = apply_top_k_top_p(logits.to(torch.float32), top_k, top_p)
    probs = torch.softmax(processed_logits, dim=-1)
    sampled = sampling_from_probs(probs).to(torch.int64)
    return sampled, processed_logits


def patch_flashinfer_sampler_head():
    """Opt-in patch for the pinned vLLM V2, non-speculative sampler."""
    if os.environ.get(ENV_VAR) != "1":
        return

    import vllm
    from vllm.v1.sample.ops.topk_topp_sampler import flashinfer_sampler_supported
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    if getattr(Sampler.sample, "_prime_flashinfer_head", False):
        return

    original_init = Sampler.__init__
    original_sample = Sampler.sample

    @wraps(original_init)
    def init(self, *args, **kwargs):
        # Validate at worker construction: vLLM catches plugin-load exceptions.
        if vllm.__version__.split("+")[0] != "0.29.0":
            raise RuntimeError(f"{ENV_VAR} requires vLLM 0.29.0; review the patch before upgrading")
        original_init(self, *args, **kwargs)
        self._prime_flashinfer_head = flashinfer_sampler_supported()

    @wraps(original_sample)
    def sample(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        return_logprobs=False,
    ):
        use_head_sampler = (
            type(self) is Sampler
            and self._prime_flashinfer_head
            and self.num_speculative_tokens == 1
            and self.logprobs_mode == "processed_logprobs"
            and (return_logprobs or self.return_sampling_mask)
            and not self.sampling_states.any_greedy(idx_mapping_np)
            and not self.sampling_states.any_explicit_seed(idx_mapping_np)
        )
        if not use_head_sampler:
            return original_sample(
                self,
                logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                return_logprobs=return_logprobs,
            )

        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )
        top_k, top_p = self.sampling_states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        return sample_with_processed_logits(processed_logits, top_k, top_p)

    sample._prime_flashinfer_head = True
    Sampler.__init__ = init
    Sampler.sample = sample
