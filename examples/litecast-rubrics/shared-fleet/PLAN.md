# Attach a second trainer to an existing LiteCast fleet

Inspected fleet: Rex f185ab72e55c4e49946b3aca4d5e5304,
`rh-signed-quokka9b-gatewayfix-r8`, September 17, 2026.
Existing trainer reached step 8; all eight middle allocations were running.
This document is a deployment design, not a submitted experiment.

## Deployment

Submit a separate Rex experiment with one trainer task: one nonpreemptible H200,
24 hours, running trainer, orchestrator, and an actual local LiteRegistry gateway.
Reuse the fleet's shared SQLite discovery database and Redis service registry.
Do not create inference, judge, terminal, or head allocations.
Start when one replica advertises the new trainer's adapter namespace.
Default to the current Qwen3.5-4B base checkpoint and training configuration;
allocate distinct output/checkpoint/W&B paths and a new run ID.

## Required attachment work

- Split fleet discovery identity from training identity in role.py. Never use
  LITECAST_PARENT_EXPERIMENT_ID to impersonate the original trainer: RUN currently
  also selects publication keys, adapter names, output paths, and failure markers.
- Keep the new gateway local. Do not overwrite the shared `gateway` discovery
  record used by existing services; publish a trainer-specific record if needed.
- Extend existing inference sidecars to subscribe to both run IDs. TenantPool
  already shares replica identity and admission counts across explicit tenants,
  but subscriptions are read only at process startup. Do not start an independent
  sidecar with an independent capacity budget against the same vLLM process.
- Middle multi-publisher support is implemented: `--publishers` reloads a JSON
  list of run IDs, with isolated caches and publisher retries. See MIDDLES.md.
  Existing CPU allocations still need an initial upgrade and subscription setup;
  the current fleet has not been migrated.
- Keep adapter names, mutable live slots, readiness, transfer acknowledgements,
  lineage, prefix-cache namespaces, and cleanup scoped to each trainer.
- Preserve a single authoritative admission limit per inference replica across
  both local gateways. Gateway counters are local; the shared sidecar must reject
  excess admissions. Evaluate fairness separately from total capacity safety.
- Check concurrent retained-state updates for the two adapters and verify that
  updating A cannot change B's lineage or invalidate B's service registration.
- Scope stopping/failing the new trainer to its own Rex experiment and adapter
  subscriptions. The original experiment currently owns shared services: stopping
  that experiment will still stop the fleet. Full independent fleet ownership is
  a separate lifecycle change, not provided by multi-LoRA support alone.

## Validation before submission

Use Slurm/Rex for model execution. Route generation through LiteRegistry only.
Verify A and B both generate, independently advance weights, respect the common
replica capacity, and remain correctly attributed during overlapping updates.
Stop B and verify A continues and its adapters remain installed. Preserve A's
results throughout. Inspect max_loras/max_cpu_loras for both tenants; the current
inference config has eight slots and retained-state mode uses one live slot per
trainer, but this must be confirmed against the running backend configuration.

Live attachment requires a controlled sidecar upgrade; the current supervisor
exits when a child exits, so killing sidecars directly would also terminate the
inference allocation. Do not treat a plain sidecar kill as a safe hot upgrade.
