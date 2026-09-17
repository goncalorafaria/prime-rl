# LiteCast-enabled PrimeRL snapshot

This branch preserves the LiteCast training integration and supporting experiment
configs as of September 17, 2026. It is separate from the fork's default branch.

## Included

- LiteRegistry gateways, head discovery, worker admission limits, and rollout
  affinity, including follow-up preference for the replica that served a turn.
- LiteCast weight publication, transfer measurements, policy/version tracking,
  retry/recovery behavior, and multiple publishers joining running middles.
- Inference sidecars that accept additional tenant subscriptions without an
  engine restart while sharing the same replica admission budget.
- Experimental retained-state LoRA updates and associated lineage/cache handling.
- Rex training, inference, middle, judge, and terminal deployment examples;
  toy and signed-rubric configs; transfer, affinity, and GPU smoke harnesses;
  monitoring/plot scripts and operational notes.

All generation goes through LiteRegistry. New deployment examples use shared
SQLite head discovery with Redis as the service registry and a local gateway.
The integration still includes older deployment variants for reproducibility.
No running deployment is migrated by saving this branch.

## Repository boundaries

`packages/litecast` is the vendored transport dependency used by this snapshot.
The standalone extraction is reviewed separately at
https://github.com/goncalorafaria/litecast/pull/1 . This snapshot does not silently
replace the vendored dependency with that unmerged PR.

The PrimeRL-specific gateway, inference sidecars, vLLM patches, and training
integration remain under `src/prime_rl`. Some cluster experiment examples also
require the separate PrimeBeaker environments and LiteRegistry installation at
the paths configured in their Rex manifests. They are cluster-specific examples,
not portable deployments without configuration changes.

## Validation evidence and limits

The late-publisher test used two independent A40 allocations on one physical GPU
host, a CPU coordinator, two middle processes, SQLite head discovery, Redis, and
a local LiteRegistry gateway. Publisher B joined both running replicas in 8.55
seconds while A completed five requests. Both replicas subsequently served both
saved adapters with different log probabilities and no engine/sidecar restart.
The benchmark exited zero and saved a passing result; Rex later classified the
experiment FAILED following cancellation of an inference allocation. This status
discrepancy is separate from the functional assertions.

The test used immutable versions of saved 2B LoRA adapters, not two concurrently
optimizing trainers or concurrent retained-state update boundaries. The supporting
admission/gateway tests passed (13), as did the tenant-isolation tests (2). The
standalone LiteCast extraction has its own 127-test validation recorded in its PR.
This is a saved development snapshot, not a claim that every historical experiment
config or the entire PrimeRL suite has been tested together.

Model weights, datasets, runtime environments, credentials, logs, and experiment
outputs are not part of this branch snapshot. Credentials are supplied at runtime.
