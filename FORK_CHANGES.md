# Grouped-rubric integration with optional score centering

Based on `feat/upstream-group-rubrics` at `d5e78436d`, which integrates upstream
PrimeIntellect-ai/prime-rl `f984dee55` (0.9.0). Renderer and verifier fork pins add
sampler-probability transport on top of those exact dependency versions, retaining
upstream renderer #158 semantics. Upstream GPU requirements are retained. This branch needs
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
ported. Optional score centering is adapted to upstream IPO and IcePop masks and
sampling replay; the legacy DPPO implementation is not restored. The sampler head
includes top-k probabilities plus the sampled action; the missing tail is estimated
from detached trainer probabilities on the replayed support.

## Qwen3.5 reasoning convention

The renderer pin adopts upstream's vLLM 0.26 parity convention, without a local
parser override. A tool-call opener ends an open reasoning region if the sampled
completion contains no explicit reasoning-close marker. If an explicit closing
marker is present later, the preceding tool text remains reasoning. With neither
boundary, the response stays unfinished reasoning. The actual prompt tokens,
not just `enable_thinking`, establish the initial channel.

Upstream's next-turn bridge preserves the sampled prefix and does not insert an
extra thinking close for a tool call that already ended reasoning. This changes
parsing and the tool/reward path; it does not force the model to generate
`</think>`. The version refers to the parsing convention, not a vLLM downgrade:
the branch retains current PrimeRL's newer vLLM requirement for sampling replay.

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

Renderer update validation: 39 upstream Qwen3.5 reasoning-boundary tests passed
using the local 2B SFT checkpoint tokenizer, offline. These cover implicit tool
boundaries, explicit-close precedence, prompt-derived state, unfinished output,
and prefix-preserving next-turn bridges. No live vLLM/GPU validation was performed.

## Score-centering extension

`trainer.loss.score_centering = true` enables the correction; the default is false.
See `docs/score-centering.md` for the formula and assumptions. Apply
`integration/grouped-rubrics/score-centering.toml` by replacing the baseline
loss table (remove IPO-only eps/kl_tau) and merging its other tables. The fragment
selects IcePop with upstream ratio bounds. IcePop uses Jasper's detached head
residual and scalar tail masses; IPO retains the explicit vocabulary correction.
This requires an unfused output layer and recorded processed sampler probabilities.
Both native fusion support and the upstream Qwen3.5 parser convention are retained.
The trainer corrects each token before the selected group/token reduction.

Validation covers explicit small-vocabulary expectations for IPO and IcePop,
full/truncated support, positive/negative advantages, KL preservation, tail
reconstruction, inactive tokens, CP alignment, sampler-to-trace-to-tensor transport,
and configuration rejection. Existing packing/reduction and renderer/verifier
regressions also pass. CPU tests bypass the eager GPU model registry; no live GPU
training, kernel execution, memory/throughput validation, or full verifier service
deployment has been performed for this branch.
