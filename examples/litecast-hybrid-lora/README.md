# Retained-state LoRA updates

This experimental mode preserves active KV/recurrent state while replacing a
logical adapter's weights. Enable the integrated LiteCast path with
`LITECAST_INFLIGHT_UPDATES=1` on trainer and inference roles. The role configures
`PRIME_RL_HYBRID_LORA=1` and `async_scheduling=false` in inference. Requirements:
LoRA enabled, DP=1 and PP=1 per inference replica, no speculative decoding, and
one API server. Validate TP=4 with the supplied GPU check before training.

Each tenant has one stable internal adapter slot. Routing names ending in `-live`
explicitly mean at least the requested retained version; LiteRegistry records
include the actually installed step/digest. Exact publication names acknowledge
readiness and cannot be used for generation in this mode. Other modes keep
immutable adapters. No LiteRegistry source changes are required.

LiteCast stages immutable adapter files, withdraws new admissions while swapping,
then advertises readiness after commit. The engine pauses scheduling with
`mode="keep", clear_cache=False`, reloads the existing adapter ID, and resumes.
Other tenants keep their own weights but share the short scheduling pause.
Failures after preparation fence the replica. Supporting roles can restart.

Prefix sharing remains enabled. Cache namespaces include adapter identity,
version and caller salt. Active requests acquire a history-specific namespace
when switching versions, including computed/output token offsets. Their existing
KV/SSM buffers are retained. Preemption resets the namespace when state is
recomputed under current weights. GPU/cache invariants require live validation.

Non-streaming `/inference/v1/generate` responses contain `hybrid_lora_lineage`:
version segments, output-token switch offsets and scheduler preemption count.
Streaming and n>1 are unsupported. The renderer validates lineage against the
generated token count, preserves sampled behavior logprobs, and stores provenance
in legacy rollout info or native trace info. The orchestrator reports mixed-turn,
transition and preemption metrics. Compact replica records also go to
`outputs/toy-ID/policy-lineage`; these contain no token arrays or KV states.
A mixed response must not be interpreted as ordinary latest-version inference.

Run `preflight-cpu.yaml` and `preflight.yaml` through Rex. The GPU check covers
active A replacement, B isolation, prefix-cache hits, changed generation logprobs,
real LiteCast publication/reload, gateway routing and renderer metadata. The full
run can be queued using `sbatch.dependency=afterok:JOB` in profile copies, with a
monitor cancelling dependent allocations if validation fails. CPU checks alone
do not establish GPU correctness. The concrete launch configuration is
`../litecast-rubrics/inflight-launch/experiment.yaml`.

For isolated engine checks, load an initial adapter through
`/v1/load_lora_adapter`; its initial version is its immutable directory path.
`/litecast/v1/hybrid_status` reports active requests and installed versions.
POST `/litecast/v1/update_lora_inflight` with `lora_name`, `lora_path`, `version`
and `expected_version` to perform a checked update.
