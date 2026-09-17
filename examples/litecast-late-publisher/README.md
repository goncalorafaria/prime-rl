# Late publisher on two live inference replicas

Submit `experiment.yaml` with Rex. It allocates two independent A40 GPUs and one
CPU coordinator. Slurm may place the GPU allocations on the same physical host.
The coordinator runs Redis, publishes its endpoint through shared SQLite head
discovery, hosts the local LiteRegistry gateway, and runs two CPU middles.

The test uses the local Sloth Qwen3.5-2B checkpoint and two saved, trained LoRA
adapters. It does not run optimizer steps. All generation goes through the gateway.

1. Both inference sidecars initially subscribe only to publisher A.
2. A begins generating as soon as its first replica is ready.
3. After both replicas are ready, A continues generating while B starts publishing.
4. B is added to the middles' shared publisher list and both sidecar tenant lists.
5. Both replicas must serve both adapters with distinct log probabilities, without
   restarting engines or sidecars. A requests during B's join must succeed.

Results and logs are saved under `outputs/late-publisher-<Rex experiment ID>/`.
`result.json` is written only on success, with replica identities, A requests
throughout joining, join latency, and transfer metrics for each publisher.

## Inference tenant additions

`prime_rl.litecast.worker --tenants tenants.json` reloads the JSON list at its poll
interval. A new entry has `run_id`, `base_model`, `backend_url`, and `shard_port`.
Use atomic file replacement, unique shard ports, and a distinct run ID per trainer.
Entries sharing a backend must use the same base model. Existing entries must stay
unchanged: this mode supports additions, not live removal or backend reassignment.
Malformed/incompatible edits keep existing tenants, with `LITECAST_TENANTS_REJECTED`.
`--max-tenants` defaults to eight. Adapter and cache budgets remain per tenant;
configure the engine's LoRA slot limits for the intended combined workload.

All tenants share the sidecar's replica ID and authoritative admission counter.
A new tenant therefore does not grant extra request capacity against the engine.
This test uses immutable adapter versions; retained-state updates across two
simultaneously training publishers require a separate acceptance test.
