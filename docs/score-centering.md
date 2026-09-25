# Score centering

IPO and IcePop support an optional score-centering correction:

```toml
[trainer.loss]
type = "ipo" # or "icepop"
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

Native top-p/top-k sampling replay remains enabled through upstream's config.
Both the trainer distribution and the reconstructed tail use the replayed support.
The correction uses the same next-token alignment and CP shard as the trainer
logprobs, then is gathered and shifted back before loss reduction.

This implementation requires positive rollout temperature, processed logprobs,
and an unfused output layer. Missing heads on active RL tokens or heads outside
replayed support are errors. With separately launched inference, configure its
logprobs mode and maximum explicitly. Full-vocabulary logits increase memory
usage; chunk checkpointing avoids retaining an additional full correction graph.
GPU memory and throughput need validation for the chosen model and context length.

With `score_centering = false`, heads are unnecessary and the fused output layer
can be used. For a controlled comparison, keep the unfused output layer and head
recording enabled on both runs and change only the loss flag.
