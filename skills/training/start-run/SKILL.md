---
name: start-run
description: How to launch prime-rl training runs — the `rl`, `sft`, and `inference` entrypoints, their config classes, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
---

# Start a run

All entrypoints run via `uv run <command>` and accept TOML configs via `@ path/to.toml` plus CLI overrides.

## Config system at a glance

[`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — Pydantic-based TOML + CLI loader. Highlights (see the `configs` skill for full mechanics):

- Config files via `@ path` (TOML / YAML / JSON); CLI args layer on top, deep-merged with class defaults.
- Nested groups via dotted CLI paths — kebab-case on the CLI, snake_case in TOML.
- Bool toggles: bare `--flag` enables, `--no-flag` disables (nested too).
- Lists: space-separated or JSON literal. Dicts: JSON literal, deep-merged with file values.
- Optional sub-configs (`WandbConfig | None`): bare `--wandb` enables defaults; `--wandb @ wandb.toml` enables from a file; `--no-wandb` disables.
- Discriminated unions are switched by the `type` tag (e.g. `--optimizer.type muon`).
- Validation aliases let renamed fields keep working; legacy keys can be remapped in a `model_validator(mode="before")`.
- Auto-generated `--help` panels from `Field(description=...)` or PEP 224 docstrings.
- Friendly errors: required-field boxes, validator errors point at the offending flag, unknown flags get a "did you mean" hint.

## `rl` — RL training

Launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
uv run rl @ examples/basic/reverse-text/rl.toml --dry-run                                # write scripts, don't run
```

- Config: `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
- Entrypoint: `src/prime_rl/entrypoints/rl.py`
- SLURM: single- and multi-node
- Environment packages: before launching a config with a non-core verifier env id,
  verify the package imports under `uv run` (for example
  `uv run python -c "import importlib.util; print(importlib.util.find_spec('r2e_gym_v1'))"`).
  If a local env exists under `deps/research-environments/environments/` or
  `deps/verifiers/environments/` but does not import, install the env workspace
  members with `uv sync --all-packages` (all) or `uv sync --package prime-rl
  --package <env>` (one) — they're auto-discovered, no `pyproject.toml` edit needed.

## `sft` — SFT training

Launches torchrun internally — never call torchrun directly.

```bash
uv run sft @ examples/basic/reverse-text/sft.toml
uv run sft @ examples/basic/reverse-text/sft.toml --slurm
uv run sft @ examples/basic/reverse-text/sft.toml --dry-run
```

- Config: `SFTConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sft.py`)
- Entrypoint: `src/prime_rl/entrypoints/sft.py`
- SLURM: single- and multi-node

## `inference` — vLLM server

OpenAI-compatible API plus prime-rl custom endpoints (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster`). Always use this entrypoint — never `vllm serve` directly. It starts a `vllm-router` on `server.port` (default 8000, the client-facing URL) fronting the engine on `backend_port` (default 8100); admin endpoints must target the engine port directly.

```bash
uv run inference --model.name Qwen/Qwen3-0.6B
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Smoke checks:

```bash
curl http://<host>:<port>/health
curl http://<host>:<port>/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

- Config: `InferenceConfig` (`packages/prime-rl-configs/src/prime_rl/configs/inference.py`)
- Entrypoint: `src/prime_rl/entrypoints/inference.py`
- SLURM: single-node, multi-node, and disaggregated deployments

## Summary

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `rl` | Full RL pipeline | Production RL training |
| `sft` | Supervised fine-tuning | SFT and hard-distill |
| `inference` | vLLM server | Standalone serving / debugging |

## Key paths

- `src/prime_rl/entrypoints/` — `rl`, `sft`, `inference` (+ `trainer`, `orchestrator` for direct launches)
- `packages/prime-rl-configs/src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs (e.g. `reverse-text/`)

## LiteCast Rex container preflight

The inference sidecar can start before vLLM binds its port. Health/model-list
connection failures log `LITECAST_BACKEND_STARTING` with the endpoint every 30s
(warning after 60s); a previously healthy backend logs `LITECAST_BACKEND_UNAVAILABLE`.
Both withdraw routing registrations until health and the base model validate,
then log `LITECAST_BACKEND_READY`. Check backend engine logs for the underlying
failure if waiting persists. Existing processes need a restart to pick up logging
changes; do not restart healthy training deployments just to change log verbosity.

Run `examples/litecast-toy/preflight.py` using the training runtime inside the
actual clean Apptainer environment before loading models. Host W&B authentication
does not establish container authentication: fakeroot can change HOME. Set NETRC
to the mounted credentials file (or supply WANDB_API_KEY securely); never embed
the key in the experiment YAML or logs. The toy trainer checks this automatically
with a bounded subprocess timeout, so authentication failures fail its allocation.

For the LiteCast toy, run `sbatch examples/litecast-toy/build-runtime.sbatch` before
Rex submission. Task commands stage this immutable archive locally and relocate
Python/console entrypoints. Use a new archive path when dependencies change;
local source remains editable on shared storage. Check RUNTIME_CACHE_HIT/MISS
timing in role logs before attributing startup delays to model downloads.

Runtime copying, compression, compilation and heavy import benchmarks must run
on a Slurm compute allocation, never on the login node. The build script requests
16 CPU cores and 64 GB and checks the resulting cache inside the training image.
Wait for that job to succeed before submitting the training experiment.

PrimeRL renderer clients call `/inference/v1/generate` at the gateway root.
Validate that native token route end to end through both gateway and sidecar,
including queue limits and cancellation, before launching training. Its request
uses `model`, `token_ids` and `sampling_params`; preserve token IDs/logprobs in
the response and select the exact immutable policy version.

### LiteCast shared inference for multiple trainers

Use `prime_rl.litecast.worker --tenants PATH` with the explicit subscription list
in `examples/litecast-multitenant/tenants.json`. Each run has a unique run ID and
shard port. Same-base trainers may share a backend; different base models require
separate inference engines and backend URLs. The sidecar does not launch engines.
Size backend LoRA CPU slots for the sum of retained versions across its tenants.
Keep one sidecar pool per backend so adapter ownership and admission are shared.
Middle processes still subscribe per run. See the adjacent README for readiness,
capacity, trusted-network scope, and startup-only subscription behavior.

For LiteCast model downloads, source `examples/litecast-toy/model-cache.sh` inside
the container before starting inference/training. The toy runtime launcher does
this automatically. It sets HF hub/dataset caches under `/tmp/litecast-hf` after
container-wrapper overrides; customize with `LITECAST_MODEL_CACHE`. Temporary
cache retention across allocations is not guaranteed.

The higher-concurrency LiteCast toy variant is
`examples/litecast-toy/experiment-50-aggressive.yaml`: four inference replicas,
128 rollout episodes (oversampling factor 1), and 16 active requests per replica.
The four-replica variant uses trainer packing length 131072 and orchestrator/vLLM
context 8192, with completion budget 8192 (bounded by remaining model context)
and inference request timeout 1200 seconds. The baseline 50-step config uses packing
length 4096 and rollout context 1024. Keep packing and rollout lengths distinct. Inference sequence and
batched-token limits are sized together with gateway/worker admission. The role
launcher selects configs with `LITECAST_TRAIN_CONFIG`,
`LITECAST_INFERENCE_CONFIG`, and `LITECAST_CAPACITY_CONFIG`; running processes do
not hot-reload them. Preserve active runs when preparing comparisons.


New LiteCast runs use `sqlite://.../outputs/toy-ID/head.sqlite3` for head discovery.
Existing runs with `bootstrap/` retain their file-backed discovery for recovery. Transient endpoint I/O is retried in the PrimeRL bootstrap
wrapper for up to 20 seconds. Restart all roles together when changing bootstrap
locations. Validate relay head replacement using the real Redis binary on Slurm
(`LITECAST_TEST_REDIS_SERVER=/gscratch/ark/graf/redis-stable/src/redis-server`).
Long toy runs keep one checkpoint, six adapter versions, a 12-step rolling window
of consumed rollout artifacts, and no detailed token exports. Do not delete
unconsumed batches or the latest checkpoint when trimming intermediates.

Toy GPU roles inject HF_TOKEN and WANDB_API_KEY from mounted credential files
before spawning processes; never place token values in specs, shell commands,
or logs. Verify the authenticated W&B username separately from the destination
entity: graf's token can otherwise log into another default team. This deployment
explicitly selects entity graf. Reverse-text training must use the staged local
parquet directory from dataset-cache.py, not a Hub dataset name that repeats
metadata discovery on every startup. Verify cache hits and offline loading on
Slurm before relaunching after dataset/authentication changes.

For long-run CPU middle allocations, request 16 CPUs/64 GB without whole-node
`exclusive`; that flag can leave GPUs waiting an hour for idle CPU hosts. When
replacing pending allocations for placement, hold the Rex experiment lock, verify
the scheduler still reports PENDING, preserve script hashes/state, and exclude
the other middle's host to keep the pair separate. Never cancel a running
allocation under a pending-only placement change.

For large subprocess rollout concurrency, remove the fakeroot preload from
runtime child processes after container setup; it serializes metadata operations
and can make concurrent SDK imports take minutes. Preserve other LD_PRELOAD
libraries. Validate import behavior on Slurm. Do not restart a productive run
solely to pick up this startup optimization.

The toy role defaults to middle-only fetching. For HTTP P2P validation use
`examples/litecast-toy/experiment-50-p2p.yaml`, which sets
`LITECAST_REQUIRE_MIDDLE=0` so cached inference peers can serve updates through
LiteRegistry discovery. This does not enable UCXX/RDMA. The Klone rubric experiment in `examples/litecast-rubrics/experiment.yaml` uses
two H200 GPUs with DP and CP=1, 16 TP=4 mixed inference replicas, eight middles, and
16 restricted terminal workers. `LITECAST_MODEL_SOURCE` and
`LITECAST_DATASET_SOURCE` enable node-local staging of the verified SFT checkpoint
and DatasetDict. Supply local PrimeBeaker and tool-client paths on PYTHONPATH.
Run its Rex CPU preflight before GPU submission; do not change to CP without a
new explicit request. Source provenance is recorded in ASSETS.json.

The cached LiteCast runtime has FlashAttention 2, not FlashAttention 3.
For H200 rubric training explicitly set trainer.model.attn="flash_attention_2";
auto selects FA3 on SM90 and fails before model loading with this archive.
A CPU schema check alone does not exercise hardware-dependent attention selection.

The LiteCast role writes outputs/toy-ID/trainer-failed before trainer cleanup
on abnormal trainer exit. Dependent supervisors check it in their watch loop
and on startup, stopping without retrying while Slurm accounting catches up.
Rex remains responsible for queued allocations and the final experiment status.
This file signal is scoped to one experiment; never reuse its output namespace.

The LiteCast runtime launcher explicitly sets TORCHINDUCTOR_CACHE_DIR,
TRITON_CACHE_DIR and TILELANG_CACHE_DIR under its per-user runtime cache.
Container username root otherwise selects /tmp/torchinductor_root, which may
belong to another allocation/user and crash vLLM on its first logprob kernel.
Never chmod/delete another user's cache; use the isolated cache path instead.

For a separate Rex inference trial attached to an existing toy/rubric run, set
`LITECAST_PARENT_EXPERIMENT_ID` to the original Rex experiment ID. Allocate a unique
`BEAKER_REPLICA_RANK` (the TP4 trial uses 1000) to avoid overwriting its inference
config. The trial uses the parent bootstrap, registry, policy and failure marker;
its own Rex record and Slurm allocation remain separate.

Set `LITECAST_PROMPT_AFFINITY=1` on the trainer to enable per-rollout weak
replica affinity (the flag retains its historical name). First requests use
ordinary load balancing. The native token gateway adds litecast_replica_id to
successful responses; renderer clients remember it in rollout state/trace info
and send X-Session-ID: replica:<id> on later turns. Sibling rollouts do not share
state. The gateway only prefers a policy-eligible replica within
`affinity_load_slack` (default 0.125 occupancy) of the least-loaded eligible
replica. Busy, missing, failed or version-ineligible replicas fall back and the
client remembers the actual replacement. Prompt-hash routing remains supported
for the frozen benchmark. Gateway and trainer/env processes need restarting to
adopt changes; source edits do not hot-reload active processes.

For a controlled weak-affinity comparison, submit
`examples/litecast-affinity-bench/experiment.yaml`. It uses four independent TP=1
replicas, the PrimeRL gateway and frozen multi-turn replay. Two replicas would make
its two-preference policy indistinguishable from ordinary balancing. Keep request
corpora fixed, warm kernels first, alternate phase order and isolate cache salts.
Read engine TTFT metrics rather than treating full token-response latency as TTFT.
The benchmark does not change or cancel the training fleet.

`examples/litecast-hybrid-lora/README.md` documents the opt-in inference prototype
for retained-state adapter replacement. `LITECAST_INFLIGHT_UPDATES=1` enables
stable per-tenant slots in the sidecar and `-live` minimum-version routing names
in the pool. The role also sets `PRIME_RL_HYBRID_LORA=1` and disables asynchronous
scheduling per inference replica. Prefix sharing remains enabled. The env-worker
renderer preserves lineage in trace info and requires per-token sampled logprobs;
the orchestrator emits transition/preemption metrics. Compact replica provenance
logs live in outputs/toy-ID/policy-lineage; tokens/states are not saved there.

Before launching this mode, run `examples/litecast-hybrid-lora/preflight-cpu.yaml`
and `preflight.yaml` through Rex. The TP4 check covers A/B isolation, cache hits,
weight-dependent logprobs, LiteCast publication/reload, gateway routing and renderer
metadata. Full-run allocations can be queued with `sbatch.dependency=afterok:JOB`
in copies of their profiles (Rex rejects submission-only options for restartable
tasks). Monitor the preflight and cancel dependent allocations if it fails. Never
claim an enabled training run until its GPU check passes and the trainer starts.
`examples/litecast-rubrics/experiment-inflight-affinity.yaml` is the ungated template;
`inflight-launch/experiment.yaml` records the concrete gated launch.

The vLLM build_app path may eagerly initialize FastAPI.middleware_stack. Install
the hybrid failure fence by wrapping the existing ASGI stack during construction;
app.middleware/add_middleware raises once the stack exists. Run check_fence.py on
a Slurm CPU allocation before a GPU startup retry. It checks both eager and lazy
stack construction and verifies that healthy requests pass and fenced requests
return 503. Do not reset the stack and discard vLLM's existing middleware.

Preserve structured type annotations on patched EngineCore utility methods.
vLLM converts msgspec wire lists using the bound method signature; add_lora must
retain a concrete LoRARequest annotation or its RPC receives a list. Run
check_rpc.py on a CPU Slurm allocation to exercise the real converter before
GPU validation.

For local checkpoints with opaque directory names, configure the renderer explicitly
(e.g. Qwen35RendererConfig for sloth2b) in validation clients as well as training.
AutoRendererConfig cannot infer the family from the local path when applying
chat-template kwargs.
The legacy verifiers ClientConfig field is api_base_url, not base_url; verify
the constructed client URL before a smoke request to avoid default hosted routing.

For the owned single-H200 rubric trainer use experiment-h200-owned.yaml with
train-h200-owned.toml and profile-klone-h200-owned.yaml. The account is
gpu-h200-ark, partition gpu-h200, QoS normal; ckpt-ark/ckpt-all is preemptible.
Check hyakalloc -g ark (the user's machines alias) before sizing the request.
The owned variant requests one GPU, 16 CPUs and 256 GB, retaining batch/context
and numerical settings. Inference remains independently scheduled through Rex
and LiteRegistry. Preserve previous experiment outputs when relaunching.

To add four preemptible TP1 H200 inference replicas to a running rubric experiment,
use Rexs.extend with task=inference, replicas=4,
profile=profile-klone-inference-h200-tp1.yaml and
overrides=extend-inference-h200-tp1.yaml (absolute paths). The extension inherits
the parent registry, run ID, policy subscription and lifecycle, assigning new
replica ranks. It does not restart the trainer or existing TP4 replicas.

LiteCast HTTP fetches group LiteRegistry sources by source role and local version
name, prefer middle groups, and distribute verified shards concurrently across
each group. Peer rebroadcast names can differ and must not be mixed in one
client. One surviving source uses the same verified transfer path. Full-cache middles
without streaming manifests use concurrent size discovery and parallel shard
reads with rotated initial sources; cached_metadata_seconds is the metadata
portion of that download phase. All locations
come from LiteRegistry. LITECAST_TRANSFER_SOURCES lists candidates;
LITECAST_SHARD_TRANSFER records each shard attempt and source; LITECAST_FETCH_TIMING
records discovery, manifest, download and final publication verification times.
HTTP attempt/failure/retry-gap seconds are sums across parallel shard workers,
not additive wall time. Download time includes per-shard checksum verification;
publication_verify_seconds measures the final publication digest/size check.
W&B aggregates available replica measurements at readiness, not eventual fleet
completion. Run preflight-transfer.yaml on Slurm to verify multi-middle use and
a stale registered middle failure. Active worker processes need restart to load
this code; retain running training unless a restart is authorized.

The prepared (blocked for this checkpoint) DCP rubric variant is experiment-h200-owned-dcp4.yaml: owned one-H200
trainer, sixteen mixed TP4+DCP4 replicas, four preemptible H200 TP1 replicas.
DCP reuses TP ranks; it does not request sixteen GPUs per replica. Qwen3.5
hybrid attention shards full-attention KV while replicating recurrent state.
Validate retained-state updates and prefix caching using preflight-dcp4.yaml
before launch. The H200 task uses LITECAST_REPLICA_OFFSET=1000 to avoid
overwriting another inference task's generated config files. g3108 is excluded
from the mixed profile after repeated four-GPU jobs saw only three CUDA devices.

Qwen3.5 sloth2b has two KV heads. The current runtime rejects TP4+DCP4:
DCP must be <= TP / num_kv_heads, hence TP4 supports at most DCP2.
Do not launch the prepared DCP4 rubric variant without resolving this constraint.
Do not bypass the assertion or silently substitute another parallel layout.

Use experiment-h200-owned-dcp2.yaml for the user-approved TP4+DCP2 layout;
inference-tp4-dcp2.toml preserves the 8K batched-token budget and 64K context.
The new inference tasks set LITECAST_RELAY_WAIT_SECONDS=10: when discovery sees
only the origin, wait up to ten seconds for a registered middle/peer before
falling back to the origin. No fixed middle count is required. relay_wait_seconds
is included in fetch timing and W&B. The real Redis/two-middle test exercises
publication before relay registration, parallel fetch and stale-source failover.

TP4+DCP2 passed the retained-state GPU acceptance check (40200229), including
adapter isolation, per-token policy lineage and prefix-cache reuse. Hybrid DCP
requires full cache-block alignment: the 4353-token test reused 4352 tokens;
the 2048-token test did not produce the partial cache hit available without DCP.
Prefix caching remains enabled; do not interpret zero hits on one alignment as
proof that it is disabled. Replica count, GPU layout and inference throughput
still need observation under the full rollout workload.

The TP-only, higher-packing rubric variant is
`experiment-h200-owned-tp8-oversample2.yaml`: eight mixed TP8 replicas, two
H200 TP2 replicas, DCP=1, 8192 batched tokens, and 65536 inference context.
`train-h200-owned-oversample2.toml` uses 262144 trainer packing, no FSDP,
optimizer, or activation offload, and retains activation checkpointing.
Batch128 with oversampling2 requires max_inflight_episodes=256 in the config
validator. GPU acceptance templates are `preflight-tp8.yaml` and
`preflight-h200tp2.yaml` under litecast-hybrid-lora; they validate the trainer
schema and exercise retained-state updates/cache sharing through LiteRegistry.
A schema pass does not establish the 262144 trainer memory fit; check the first
forward/backward on the owned H200. Preserve older configurations/results.
When cancelling a replaced run, check retained allocation attempts too: earlier
attempts can still be live even if Rex tracks a newer replacement. Scope any
additional cancellation strictly to job IDs owned by the requested run.

The user explicitly authorized launching the TP8 replacement without waiting
for its queued GPU check. H200 TP2 acceptance passed on g3131; TP8 remains
unvalidated by that check. Record this distinction when reporting the run and
inspect real inference startup rather than claiming TP8 preflight success.

The TP8 launch sets LITECAST_WAIT_FIRST_REPLICA=1 on the trainer supervisor.
It polls the LiteRegistry gateway model list for this run's base alias before
starting PrimeRL, logging every 30 seconds and starting when the first replica
registers. This keeps scheduler queue time outside the orchestrator's 1800-second
readiness timeout. Waiting remains bounded by the Slurm allocation time limit;
trainer coordination child failures and signals still terminate the supervisor.

The first-replica readiness gate must publish an empty desired-run descriptor
while waiting: workers deliberately withdraw all routes without a live publisher.
Waiting for a base alias before publishing creates a circular startup dependency.
The supervisor now renews a short-lived, unique-owner bootstrap descriptor and
atomically removes only its own descriptor before starting PrimeRL's publisher.
Never overwrite an existing publisher or leave bootstrap renewal running after
handoff. A gateway-only stub test cannot detect this dependency; verify real
worker registration plus the subsequent PrimeRL publisher startup.

The long rubric deployment is `examples/litecast-rubrics/run-24h-512/experiment.yaml`.
It requests 24h for the owned trainer and48h for services to leave startup/recovery
margin. Training batch128 with max_inflight_episodes512 uses oversampling_factor4;
evaluation remains capped at128. It includes eight mixed TP8 replicas with16CPUs,
eight mixed TP4 replicas with32CPUs, and two H200 TP2 replicas. All use64 sequence
slots and8192 batched tokens. Separate inference tasks use unique generated-config
rank offsets (H2001000, TP42000). Equal short lifetimes are unsafe: a head timeout
can finish recovery after the trainer independently exhausts its own allocation.

## Signed RubricHub / Quokka9B LiteCast configuration

`examples/litecast-rubrics/run-signed-quokka9b/experiment-group8.yaml` uses two TP1 policy replicas and one separate Quokka9B judge allocation. The trainer rewrites rubric environments to the gateway `/judge` endpoint; do not pass terminal-only arguments to them. `LITECAST_STAGE_NAMESPACE` isolates dataset/model caches, and `hf://` model sources use the node-local Hugging Face cache. The judge adapter routes model inference through its own local gateway and publishes readiness only after the registered Quokka model, terminal service, and judge API are ready. Quokka uses its existing per-rubric XML terminal workflow in the browsecomp runtime, with a separate CPU terminal allocation; both model and tool requests use the local gateway. Trainer readiness waits for one policy replica and the judge. Signed rewards require explicit pass-means-condition-present wording and empty/truncated output penalties below the signed reward floor. Validate this deployment and perform a small GPU smoke before a full training launch; no smoke is implied by config validation.

Rex container launchers must use `BEAKER_JOB_ID` for allocation identity. The generated Apptainer wrapper maps the Slurm ID into that variable; `SLURM_JOB_ID` is not guaranteed inside the clean container environment.

Publish bare service base URLs for bootstrap endpoints checked with `healthcheck="http"`; the checker appends `/health` itself. Including `/health` in the published URL blocks readiness with `/health/health` 404s.

Signed-rubric runs require the LiteCast gateway `/judge` proxy route. Select the registered judge orchestration service using payload `model_path`; preserve `model` as the Quokka inference model. A healthy judge bootstrap alone does not validate this route: smoke-test a request through the trainer gateway before trusting rollout scoring.

Middle relays can adopt a different publisher while no MiddleNode/cache or source registrations exist. This permits the empty startup descriptor to hand off to the real trainer. Once a MiddleNode exists, retain the publisher-change restart guard to prevent mixing local version identities.

For terminal-enabled Quokka judging, launch `jtc.datadev.literegistry.gateway:create_app` and put the updated JTC root on the judge API PYTHONPATH. JTC must configure `extra_headers_from_state={"X-Session-ID": "trajectory_id"}` in its Verifiers client; sampling `extra_headers` becomes JSON body data in this runtime. Validate actual tool-using judge turns, not just direct chat requests. Run live checks through the Rex Apptainer wrapper on an existing allocation so cluster DNS matches the deployment.

The signed Quokka9B Qwen3.5-4B LoRA run exhausted one H200 during its first backward pass with trainer packing `seq_len=262144` (262091 actual packed tokens, 138.19 GiB process memory). The user-selected retry uses `131072`; memory fit remains unverified until a full backward pass succeeds. Rollout context remains 32768. Changing inflight episode count does not change trainer packing memory. Judge affinity smoke success does not validate training memory or sustained judge throughput.


New LiteCast runs use `head.sqlite3` for endpoint discovery and launch the actual
capacity gateway on the trainer. The head allocation owns Redis and publishes
its current URL; clients use `head+sqlite://...` to discover replacement Redis.
Publisher fencing and startup ownership scripts must execute through head-aware
Redis commands, never fall back to an unconditional KV write for head URIs.
`head_registry(output)` preserves an existing `bootstrap/` directory for old
runs so replacement allocations join their original topology. Do not migrate a
live run piecemeal. Verify Redis replacement and publisher-fencing tests when
changing discovery or lease behavior.

The signed Quokka judge API must not add a whole-request concurrency cap or a
shared workflow thread-pool cap. The user explicitly requested their removal.
Each synchronous request workflow has its own worker; batching within a request
still controls its rubric trajectories. Validate admission with more than 32
simultaneous requests so a default executor pool cannot silently replace the
removed four-request gate. Applying server-code changes requires reloading the
judge API; source edits do not update an existing process.

### Shared LiteCast middle publishers

`prime_rl.litecast.middle --publishers /shared/publishers.json` reloads an explicit
JSON list of run IDs, allowing attach/detach without restarting an upgraded middle.
Each publisher has a separate transfer endpoint/cache and independent retry loop.
Use atomic file replacement; malformed reloads retain the previous subscriptions.
Set `--max-publishers` and budget cache memory per publisher (see
`examples/litecast-rubrics/shared-fleet/MIDDLES.md`). Single `--run-id` remains
supported. This does not make inference sidecars dynamically multi-tenant, and
existing single-run supervisors need an initial upgrade before using this mode.

For a late-publisher inference check, `examples/litecast-late-publisher/experiment.yaml`
uses two A40 replicas and a CPU coordinator with SQLite head discovery, Redis, a
local gateway, and two middle processes. Inference `--tenants` files support live
additions while existing entries stay unchanged; admission remains shared across
tenants. This test publishes saved adapters, not optimizer steps, and checks
immutable versions rather than simultaneous retained-state update boundaries.

The runtime-cache launcher prepends the standalone `/gscratch/ark/graf/litecast`
and shared LiteRegistry source before installed runtime packages. After changing
that source selection, verify PrimeRL protocol, capacity, and middle transfer
compatibility with the same PYTHONPATH in a compute container before relaunch.
Use `--noconftest` for focused tests in shared allocations to avoid global
process-killing fixtures.
