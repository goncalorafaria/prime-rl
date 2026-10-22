# Score centering

IPO and IcePop support an optional score-centering correction:

```toml
[trainer.loss]
type = "icepop"
ratio_low = 0.2
ratio_high = 5.0
aggregation = "group_token_mean" # token_mean is also supported
score_centering = true

[trainer.model]
fused_lm_head_token_chunk_size = "disabled"

[orchestrator.train.sampling]
score_centering_top_k = 128

[inference.vllm]
logprobs_mode = "processed_logprobs"
max_logprobs = 128
```

Replace the entire loss table when switching loss types: IPO-only `eps` and
`kl_tau` do not belong in an IcePop config.

The flag defaults to false. It leaves the selected upstream loss's forward value,
trust-region rule, component weights, group/token normalization, and IPO KL term
unchanged. It subtracts the estimated sampler expectation from the masked,
importance-weighted policy-gradient score. Rejected sampled actions still receive
the expectation correction. This is distinct from centering rewards or advantages.

For a fixed token context, let `p` be the temperature-scaled trainer distribution,
`qhat` the reconstructed behavior distribution, and `M(a)` the configured IPO or
IcePop acceptance mask. The correction's gradient is
`sum_a p(a) * M(a) * 1[qhat(a) > 0] * grad(log p(a))`.
Its coefficients are detached. It enters the loss with the token's detached
advantage and `adv_tau`; `correction - correction.detach()` preserves the reported
loss value. The IPO mask uses absolute probability difference; IcePop uses its
inclusive importance-ratio band.

The renderer records processed sampler top-k logprobs plus the sampled action.
The verifier preserves these through message nodes and branch flattening; the
orchestrator carries them through truncation, packing and transport. The unlogged
sampler tail is approximated proportional to detached trainer probabilities.
This is an approximation unless the recorded head covers the sampler's support.
`score_centering_top_k` controls recording, not sampling truncation.

### Compact logprob extraction

For vLLM 0.29.0 V2 workers, enable the parallel Triton extraction path with:

```bash
export VLLM_PRIME_COMPACT_LOGPROBS=1
export VLLM_USE_V2_MODEL_RUNNER=1
```

Use `logprobs_mode = "processed_logprobs"` for score centering. To retrieve the
most likely 128 tokens without truncating sampling, request `logprobs=128` with
sampling `top_k=-1` and `top_p=1.0`. The no-extraction baseline is `logprobs=None`;
`logprobs=0` still requests the sampled token's logprob.

This patch keeps vLLM's sampling. On contiguous logits with vocabulary size at
least 4096, FlashInfer's deterministic radix lookup finds the logprob entries to
record; smaller vocabularies and strided rows use PyTorch's lookup. This lookup
never filters the sampling distribution. Equal-valued entries may have a different
deterministic order or boundary subset from native extraction. Triton computes partial
normalizers and sampled-token ranks across vocabulary tiles, then combines them
and emits only the requested token IDs and logprobs plus the sampled token.
Normalization includes the entire processed distribution. Recording width does
not change sampling. The existing packed head transport and replay mask continue
to consume these outputs.

The optimized path supports CUDA FP32/FP16/BF16 logits with a contiguous vocabulary
axis and recording widths 0–128. Custom token lists, logits output, larger heads,
and unsupported layouts use native extraction. V1 is not patched. The general
plugin installs the patch in each worker; the launcher rejects an explicit V1
selection with this flag. No extraction runs for `logprobs=None`.

To benchmark the full sampler call on a CUDA worker with the pinned vLLM:

```bash
VLLM_PRIME_COMPACT_LOGPROBS=1 VLLM_PRIME_FLASHINFER_SAMPLER_HEAD=0 \
  uv run tools/benchmark_compact_logprobs.py results.jsonl
```

This compares no logprobs, native top-128, and patched top-128 at batches 1, 8,
64, and 256, without sampling truncation. It checks outputs and records rotated
wall/CUDA timings. It excludes model execution and host transfer.

### FlashInfer sampling with compact logprobs

For the pinned vLLM 0.29.0, set these environment variables on inference workers
before launching inference (including workers launched separately):

```bash
export VLLM_PRIME_FLASHINFER_SAMPLER_HEAD=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_FLASHINFER_SAMPLER=1
```

Keep `logprobs_mode = "processed_logprobs"` and the recording configuration above.
The general plugin installs the patch in spawned vLLM workers. It applies vLLM's
temperature, penalties, min-p and top-k/top-p processing, then passes normalized
probabilities to FlashInfer's categorical sampler. The same filtered logits feed
vLLM's Triton sparse logprob extraction and sampling-mask capture. The renderer
receives packed int32 token IDs/counts and float32 logprobs for up to the requested
top-k plus the sampled token; duplicate and masked entries are removed.

The shared-support contract follows
[Slime #2406](https://github.com/THUDM/slime/pull/2406): report post-truncation
probabilities and retain the sampled action. Slime exports complete top-p support;
our configured recording head can be smaller. Its probabilities are normalized
over the entire retained support, so their sum can be below one. Replay masks
still contain the full support and the trainer still approximates an unrecorded tail.

This path filters **before** FlashInfer samples. It uses vLLM's truncation/tie
semantics, which can differ from FlashInfer's fused top-k/top-p sampler. It avoids
reconstructing a potentially different support after sampling. Greedy or explicitly
seeded batches, speculative sampler subclasses, and unsupported devices retain
the upstream sampler. V1 is not patched; the prime-rl launcher rejects an explicit
V1 selection with this flag. Without the opt-in, upstream behavior is unchanged.

This is separate filtering, categorical sampling, and sparse-logprob kernels,
not a single fused sample-and-head kernel. Logprob output is O(k), but filtering
and normalization still scan the vocabulary and sampling allocates a full FP32
probability tensor. This optional categorical-sampling path has passed GPU
correctness checks, but it is separate from `VLLM_PRIME_COMPACT_LOGPROBS`. The compact extraction path does
not require the FlashInfer sampling flag.

IcePop uses Jasper's head-only residual. For recorded head H, let
`c = max(1-sum_H q, 1e-6) / max(1-sum_H p, 1e-6)` and
`w(r) = r * 1[ratio_low <= r <= ratio_high]`. Then
`alpha = c*w(1/c)` is the tail acceptance indicator, and the correction is
`sum_H stop_gradient(p*keep - alpha*p) * log(p)`.
Only head probabilities and scalar tail masses are constructed. Coefficients
are not normalized by their sum. The 1e-6 tail-mass safeguard follows Jasper's
formula; near zero tail mass this is a numerically regularized approximation.
The sampled-token loss and the correction use the same configured ratio bounds.

IPO remains available with its explicit vocabulary-wide correction because its
absolute-probability acceptance mask can vary across the proportional tail.
It does not have IcePop's head-only memory advantage. TIS is not implemented.

Native top-p/top-k sampling replay remains enabled through upstream's config.
Both the trainer distribution and the reconstructed tail use the replayed support.
The correction uses the same next-token alignment and CP shard as the trainer
logprobs, then is gathered and shifted back before loss reduction.

This implementation requires positive rollout temperature, processed logprobs,
and an unfused output layer. Missing heads on active RL tokens or heads outside
replayed support are errors. With separately launched inference, configure its
logprobs mode and maximum explicitly. Full-vocabulary logits increase memory
usage. IcePop gathers head logits and computes their log-normalizer without
constructing full-vocabulary probabilities, tail tensors, or acceptance masks.
Normalization still scans logits (or replay support); the entire operation is not
O(k). Checkpointing recomputes chunk normalization and avoids retaining copied
logit chunks. The unfused trainer output still materializes full logits; fused
head-logprob extraction is not implemented.
GPU memory and throughput need validation for the chosen model and context length.

With `score_centering = false`, heads are unnecessary and the fused output layer
can be used. For a controlled comparison, keep the unfused output layer and head
recording enabled on both runs and change only the loss flag.

### Chunked trainer projection

Set `trainer.score_centering_token_chunk_size = 128` to checkpoint the LM-head
projection and score correction together in token chunks. This bounds vocabulary
activation memory and recomputes each chunk during backward. The default is
`None`, which keeps the dense path. Custom Qwen3, Qwen3.5, and compatible wrapped
models support this option; sampling must be untruncated. FP32 normalization and
the recorded-head correction are retained.

Sampler-head packing skips empty prefix rows and vectorizes probability checks
within each nonempty row while retaining shape, uniqueness, and mass validation.
