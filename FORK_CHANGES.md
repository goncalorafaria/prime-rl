# Grouped-rubric integration

Based on upstream PrimeIntellect-ai/prime-rl `f984dee55` (0.9.0), fetched 2026-09-24.
Upstream's dependency pins and GPU requirements are retained. This branch needs
its own current-upstream image; the older Delta training image is not compatible.

## Retained extensions

- `trainer.optim.eps`: configurable AdamW epsilon, positive and finite, default
  `1e-8`. Forwarded to the optimizer; upstream CPU offload reads the same parameter
  group. Existing learning-rate, beta, and weight-decay controls remain available.
- `trainer.loss.aggregation`: `token_mean` (upstream default) or `group_token_mean`.
  Available with upstream IPO, IcePop, and custom RL loss functions. Custom losses
  must return the sum of per-token losses, following upstream's contract.

Group averaging is `sum_g(sum_{t in g} weighted_loss_t / T_g) / G`. `T_g` counts
eligible RL tokens after truncation and component routing, across every branch of
that dispatch group in the current training batch. Trust-region rejection does
not shrink the denominator. Zero component weights exclude tokens; other weights
scale only the numerator. Empty groups are excluded. Partial groups use their
batch-local members. CE and reference-KL retain upstream token normalization.

The orchestrator computes denominators before DP distribution. Packed sequences
carry their denominators through transport; the trainer corrects for CP replication
of full-sequence losses. Missing group metadata is an error in group mode.

## Upstream implementations adopted

Native model implementations, supported runtime fusions, IPO/IcePop loss formulas,
adaptive concurrency, evaluation, native top-p/top-k sampling replay, renderers,
verifiers, and checkpoint/weight transport all come from upstream.

The earlier static inflight/evaluation-capacity changes, top-p forwarding patch,
legacy environment-budget patch, parser override, and old Docker overlay are not
ported. Score centering remains on `codex/score-centering`; this branch does not
expose it. It requires a separate port of its sampler-distribution calculation.

## Configuration fragment

Merge `integration/grouped-rubrics/overrides.toml` into a complete run config.
It selects IPO with optional group averaging and the requested AdamW settings.
It is not a complete launch config: model/checkpoint, native taskset and harness,
resources, output directories, and monitors must be supplied by the deployment.
Preserve the experiment's explicit optimization/reduction dtypes when migrating;
this branch does not change upstream defaults or any active run configuration.

## Validation

65 targeted CPU checks passed: 25 existing packing tests, 20 group-reduction and
optimizer-control checks, and 20 upstream configuration checks. Coverage includes
IPO/IcePop values and gradients, token-mean baseline, truncation, zero-weight and
empty groups, mixed CE, wire serialization, simulated DP/CP partitioning, and
trust-region rejection. The TOML fragment's four configuration tables validate;
Ruff and diff whitespace checks pass.

The CPU runner bypassed eager registration of GPU-only model implementations;
the tested packing, transport types, loss functions, and config classes were the
actual branch sources. This is not GPU/distributed integration validation. Native
Qwen3.5 execution, fused kernels, CPU optimizer offload, sampling replay against
vLLM, and a complete grouped-rubric rollout still require the new runtime image.
