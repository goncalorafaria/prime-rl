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
