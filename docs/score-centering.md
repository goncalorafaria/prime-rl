# Score centering

Score centering subtracts the sampler's expected score at each generated-token prefix. The default DPPO loss supports an optional `score_centering` flag, including `group_token_mean` aggregation. A separate research loss supports unweighted SC and SC applied after token-wise IS, TIS, or MIS. Sampler probabilities and centering coefficients are detached. The default loss retains its differentiable sampled importance ratio; the standalone research surrogate detaches its sampled weights. Reward/advantage streams and component normalization are handled by the existing algorithms and trainer.

The renderer records top-k **processed** sampler log-probabilities during generation. These must describe the actual behavior distribution, including temperature and any sampling filters. The sampled token's own log-probability is retained separately. The top-k head passes through verifier turn tokens, graph nodes, branches, training samples, packing and padding. The trainer reconstructs the unlogged sampler tail by rescaling its own probabilities to the remaining sampler mass, following equations 7–12 of [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/pdf/2609.20807).

Apply this overlay to an otherwise configured v1 RL run:

```toml
[trainer.loss]
type = "default"
aggregation = "group_token_mean"
score_centering = true

[trainer.model]
fused_lm_head_token_chunk_size = "disabled"

[orchestrator.train.sampling]
score_centering_top_k = 128

[inference.vllm_extra]
max_logprobs = 128
logprobs_mode = "processed_logprobs"
```

The flag defaults to `false`. Enabling it preserves the DPPO importance-ratio term, advantage scaling, squared-log-ratio KL regularizer, component weights, and group-token denominators. Reward/advantage normalization is unchanged. With `group_token_mean`, each rollout group contributes its mean active-token loss, then those group means are averaged; this reduction is separate from centering rewards within a group.

For the default loss, define `m(A,v)` as its existing DPPO keep-mask, evaluated at each candidate token with the observed advantage sign held fixed. The ascent direction becomes

```text
A * [m(A,a) * p(a)/q(a) * grad log p(a)
     - E_{v~q_hat}[m(A,v) * p(v)/q_hat(v) * grad log p(v)]]
```

The sampled ratio still uses the sampled token's actual behavior log-probability. The centering expectation uses the logged head plus reconstructed tail. Its correction is added even when the sampled action is masked. All expectation coefficients are detached; the correction surrogate has its detached value subtracted so the reported objective value stays unchanged. KL, CE, and reference-KL gradients receive no SC correction. Since DPPO's mask depends on absolute probability differences, it is evaluated throughout the reconstructed vocabulary, in token chunks, rather than treating every tail token as having the same mask. The expectation holds the advantage sign fixed; it does not establish zero unconditional drift for an action-dependent reward/mask. With no masking and full sampler support, exact IS has zero score drift.

For the standalone research objective, choose `type = "score_centering"` and `weighting = "none"`, `"is"`, `"tis"`, or `"mis"` instead. It replaces the default DPPO+KL objective. `weighting = "tis"` uses `cap = 2.0` by default. `weighting = "mis"` uses `low = 0.2`, `high = 5.0`. `weighting = "is"` uses token-wise ratios; its centering term vanishes when the sampler has full support. Sampling filters that remove support can make this correction nonzero. The weights are detached. This does not implement sequence-wise importance sampling or PPO clipping. Existing CE and reference-KL components remain separate.

The first implementation requires the unfused LM head and materializes trainer logits; account for its memory cost when choosing sequence length and micro-batch size. Top-k tail reconstruction is approximate, so this does not guarantee exactly zero drift under the unknown full sampler distribution. For the standalone research loss, `sc/head_mass` reports retained sampler probability mass. Missing top-k data on an active RL token raises an error. The v0 legacy rollout bridge is not supported.

The companion renderer and verifier changes providing `sampler_head_ids` and `sampler_head_logprobs` response/trace fields are required. External env servers need those same revisions. For an externally managed sampler, configure its processed-logprob mode and maximum logprob count there as well.

Deploy through rexs. Route generation/evaluation requests through the trainer-local LiteRegistry gateway, with shared SQLite head discovery and Redis service registration; this loss does not alter routing, discovery, model replica readiness, or serving capacity. An existing deployment should not be restarted merely to apply these development changes.

The focused mathematical checks cover full-vocabulary centered gradients, TIS/MIS weighting, reconstructed tails, zero-drift exact IS, padding, and equivalence to the logit straight-through construction for a nonlinear policy. A GPU training smoke test is required before treating the implementation as production-validated.
