# Unified Rex toy deployment

Status: deployment plan; no jobs submitted. Target: one Qwen3.5-2B rank-8
LoRA training run, one A40 trainer, one L40 inference worker, and one CPU
LiteRegistry head allocation. All new application wrappers belong in this
PrimeRL fork; LiteRegistry remains unchanged.

## Experiment ownership

One Beaker v2 `experiment.yaml`, one Rex experiment ID, three allocations:

| Allocation | Tasks | Initial resource budget | Lifetime |
| --- | --- | --- | --- |
| head | Redis, PrimeBeaker/LiteRegistry gateway, startup coordinator | CPU only, 8 cores, 16 GiB RAM | Until training exits |
| trainer | PrimeRL trainer + orchestrator + LiteCast origin | 1 A40, 8 cores, 64 GiB host RAM | Three optimizer updates |
| inference | vLLM + LiteCast sidecar/cache | 1 L40, 8 cores, 64 GiB host RAM | Until training exits |

These are proposed budgets, not measured capacity. The head is a dedicated
CPU compute allocation, not a service process on the cluster login node.

The experiment's `rexs.allocations` groups head services together, trainer
separately, and inference separately. Set `completion_task: trainer`. Keep
Rex controller refresh running so it observes completion and cleans up owned
allocations. Task wrappers forward signals and reap children. Cleanup must
use this experiment's allocation IDs only.

## Model and workload

- Pin one `Qwen/Qwen3.5-2B` snapshot and use it on trainer and inference.
- LoRA rank 8, alpha 16, existing attention/MLP targets, BF16 model execution.
- Preserve the fork's optimizer/reduction precision settings.
- Three optimizer steps; checkpoint and token/logprob exports every step.
- Start with batch size 4, group size 2, at most 4 simultaneous episodes,
  oversampling 1, 8192-token episode and trainer sequence limits, and 1024
  completion tokens per turn. Keep top-p 0.97 and the existing search harness.
- One vLLM TP1 worker, rank-8 runtime LoRA loading, 8192 context, four maximum
  concurrent sequences, eight CPU adapter slots, and 0.8 GPU memory utilization.
- Keep eight adapter versions and `max_off_policy_steps=2`; require one ready
  replica before advancing to a published version. HTTP LiteCast transport.

Materialize these as a toy-specific config so the existing search smoke
example remains independently usable. A40/L40 runtime support and memory
headroom must be checked in the actual pinned image before training.

## LiteRegistry head and startup

Use LiteRegistry's existing `coop.ports` and `coop.endpoints` APIs. Give every
experiment a unique head namespace and run ID. On a single shared cluster,
use a shared SQLite head registry for bootstrap only. The head publishes
health-checked Redis and gateway endpoints there.

Workers resolve the head's `redis` endpoint to a concrete Redis URL before
starting LiteCast. Pass that resolved URL as `REGISTRY`, including to the
publisher: its atomic owner fencing currently activates for `redis://` or
`rediss://`, so passing a `head+sqlite` URI directly would bypass that path.
Pass the discovered gateway endpoint as `GATEWAY_URL`. Do not share a SQLite
file between unrelated clusters.

Startup order:

1. Head starts Redis, publishes its live endpoint, then starts and advertises
   the gateway. Use dynamically reserved ports and routable advertised hosts.
2. Both GPU allocations resolve the same registry and gateway, stage the
   identical base-model snapshot on node-local storage, and check GPU type.
3. Trainer/orchestrator establishes its publisher lease; the inference worker
   verifies its backend and registers the run's base-model alias. Queue waiting
   is separate from the timeout for a service that has already started.
4. Coordinator checks a gateway completion plus search, terminal and judge
   smoke requests. The training launcher proceeds only when dependencies work.
5. Three optimizer updates publish adapters through LiteCast. Inference aliases
   are advertised only after verification and successful vLLM loading.
6. Trainer archives the small checkpoints, configuration, logs and token exports
   to durable storage, exits, and Rex cleans up the experiment's services.

Inference requests go through the head gateway. Adapter discovery, publication
leases and source heartbeats use LiteRegistry. Adapter bytes travel directly
between discovered LiteCast endpoints. Head-to-worker sidecar access and
trainer/worker LiteCast connectivity are required; registry discovery does not
create network connectivity.

## Search and judge dependency

To stay within exactly two GPUs, the proposed search run reuses existing
PrimeBeaker search, terminal and judge services. Their deployment endpoint and
judge profile must be identified before preparing a runnable config. They are
external dependencies and are never included in this experiment's cleanup.

The new head owns a fresh inference registry. Existing tool services will not
automatically appear there. Extend the PrimeRL launch configuration to accept
a separate tool gateway URL for `/search`, `/terminal`, and `/judge`, while
policy completions continue through the new head gateway. Preserve existing
judge-profile validation. This requires no LiteRegistry changes.

A fully self-contained search experiment would additionally need a local
search index, terminal workers, judge orchestration and judge model capacity.
Do not silently put a 9B judge on the single L40 policy GPU. A toy reward is an
alternative only if explicitly selected; it would not validate the existing
search/judge workload.

## Scheduler placement prerequisite

The local Rex deployment profiles currently identify A40 resources on Delta
and L40 resources on Klone. Rex's current multi-allocation implementation calls
local `sbatch` for every allocation; allocation profiles alone do not dispatch
jobs to remote cluster login nodes.

Preferred simplest deployment: place both requested GPU types under one
scheduler with mutual network connectivity and a shared bootstrap registry.
If the intended pairing is Delta A40 plus Klone L40, first design and validate
remote allocation ownership in Rex, artifact staging, routable Redis/gateway
and LiteCast endpoints, and a network-accessible head bootstrap. That pairing
is not a ready-to-run three-allocation experiment with today's local submitter.
Do not substitute an available GPU type without the user's agreement.

## Implementation and acceptance gates

1. Confirm GPU placement and the reusable tool-service endpoint/profile.
2. Pin and build a compatible LiteCast runtime image from this fork; verify
   Qwen3.5 LoRA imports and actual GPU kernels on A40 and L40. Keep model caches
   and temporary adapters on node-local storage.
3. Add toy config, separate tool-gateway override, cooperative head bootstrap
   wrapper, role wrappers and one unified `experiment.yaml` in this fork.
4. Add site profiles with actual account, partition, image and mount mappings;
   validate and render all allocations with Rex without submitting jobs.
5. Before submission inspect active jobs; leave existing evaluations untouched.
6. Run dependency smokes, then train three updates. Verify that each update is
   loaded under its exact immutable alias and produces successful gateway
   requests. Record reward results separately from infrastructure failures.
7. Report peak GPU memory, adapter size, publish-to-ready latency, rollout
   throughput, checkpoint paths and cleanup status. This is a correctness smoke,
   not an inference scaling benchmark.

In this one-worker experiment, worker loss pauses useful rollouts; it cannot
provide uninterrupted failover. Treat preemption as an explicit failure unless
bounded replacement is configured and tested. Leave replacement stress and
fleet scaling to a later experiment.
