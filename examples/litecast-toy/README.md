# Real LoRA training smoke

One unified Rex experiment runs three Qwen3.5-2B LoRA optimizer steps against
`reverse-text-v1`. The reward compares the model's tagged reversal with the
computed correct reversal; no search service, terminal service or judge model
is needed. This is a deterministic graded reward, not an LLM judge.

Allocations: CPU head (Redis + gateway), two independent CPU middle nodes,
one A40 trainer/orchestrator and one A40 vLLM worker. The worker requires a
registry-advertised middle source for adapter loads. LiteCast itself remains
independent of LiteRegistry; `prime_rl.litecast.middle` owns discovery,
verification and registration. The trainer is the Rex completion task; its exit triggers cleanup of all owned
allocations. Head, middle and inference failures have bounded restart policies. Keep Rex controller refresh active.

The profiles use the local Klone account, pinned Ubuntu CUDA SIF and isolated
training environment at `outputs/toy-runtime-host`. Install that environment
for the image's glibc platform, including the `litecast` and `flash-attn` extras;
the login host's glibc is too old for the pinned Mooncake wheel. Always execute
the training environment inside the image. The existing training environment
and live evaluations are not modified.

Build the runtime archive once, then validate before submitting:

```bash
sbatch examples/litecast-toy/build-runtime.sbatch
rexs validate examples/litecast-toy/experiment.yaml --strict
rexs submit examples/litecast-toy/experiment.yaml --strict --name litecast-toy
```

The run uses its Rex experiment ID for the file-backed bootstrap namespace and
LiteRegistry run ID. Small checkpoints, logs and token exports are under
`outputs/toy-EXPERIMENT_ID`; model caches and temporary adapters use node-local
`/tmp`. The head advertises dynamically selected Redis/gateway ports. All
allocations use routable hostnames on the same cluster.

Success requires actual optimizer steps and checkpoints, rollout rewards,
and `MIDDLE_READY` log records paired with worker adapter loads. A successful
CPU test or successful submission alone is not a completed training smoke.

Offline preflight completed: Rex strict validation (five allocations), both
TOML schemas, head Redis/gateway bootstrap, and real middle supervisor source
registration and withdrawal. GPU outcome is recorded separately after execution.

The toy explicitly enables `[trainer.model.ac_offloading]` with pinned memory
and `trainer.model.fsdp_cpu_offload=true`. FSDP offloads gradients together
with parameters and optimizer state; this fork has no gradient-only offload
flag. `optim_cpu_offload=false` avoids enabling the mutually exclusive
optimizer-only path. Activation checkpointing stays enabled.

Install the selected taskset explicitly in the runtime as well:
`uv pip install --python outputs/toy-runtime-host/bin/python -e deps/verifiers/environments/reverse_text_v1`.
Workspace membership alone does not install this taskset. The inference config
sets `router="None"` to expose the native vLLM LoRA admin endpoints locally;
external requests still pass through the LiteRegistry gateway.

The local trainer/orchestrator rollout handoff uses filesystem transport to
avoid default ZeroMQ port collisions on shared nodes. Before startup, each GPU
role stages Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`;
the trainer also caches the reverse-text dataset. The runtime then uses offline
Hub access so model/tokenizer initialization does not repeatedly resolve remote
endpoints. Adapter distribution still requires LiteCast middle sources.

## Transfer timing in W&B

The trainer and orchestrator share an online W&B run in project `litecast-toy`,
named after the Rex experiment ID. The launch requires configured W&B credentials.
Each completed weight update logs `litecast/*` against `litecast/policy_step`:

- `publish_seconds`: local adapter packaging, sharding and registry publication.
- `ready_wait_seconds`: publication to the configured minimum ready replicas,
  including middle propagation, polling, worker fetch/load and registration.
- `update_seconds`: the total of those two phases.
- `fetch_seconds_mean/max`: worker fetch, including registry discovery, candidate
  retries and integrity verification in the successful fetch call.
- `load_seconds_mean/max`: unpacking, local staging and vLLM adapter loading.
- `payload_bytes` and `measured_replicas`: update size and timing sample count.

Worker timings use local monotonic clocks and travel in LiteRegistry metadata;
LiteRegistry source code is unchanged. Replica statistics cover ready workers at
update completion, not every later joiner. Missing timing samples are counted as
zero measured replicas, never reported as zero transfer latency. CPU middleware
propagation is included in readiness latency, not the worker fetch measurement.
The worker also writes `LITECAST_TRANSFER` to its log. Metrics are emitted on each
update rather than only on the periodic dashboard tick, so short runs keep them.

## CPU and memory reservations

The head requests 4 cores / 16 GiB. Each middle requests 16 cores / 64 GiB
on an exclusive CPU host; independent allocations plus node exclusivity prevent
the two middles from sharing a host. The site-specific middle profile excludes
the GPU hosts in Klone's checkpoint partition. Refresh that list if the cluster
inventory changes. Exclusive scheduling reserves the host's CPUs; the middle
process receives its requested 16-core task allocation and 64 GiB memory limit.
Trainer and inference each request one A40, 8 CPU cores and 96 GiB, with a
one-hour limit for startup and the short training smoke. They may share a GPU
host with separate reserved resources. These requests can wait longer in queue.

The generic launcher warning “LoRA is enabled, but inference is not configured”
refers to its absent local inference block. This experiment launches inference
as a separate Rex task with `enable_lora=true` and `max_lora_rank=8`. Both GPU
roles validate those settings, the base model and target modules before startup,
and print `LORA_CONFIG_VALIDATED`. The warning alone does not indicate LoRA is
disabled on the separate worker.

## Adding inference replicas

Submit with the inference allocation marked `independent_replicas: true`, then:

```bash
rexs extend EXPERIMENT_ID inference --replicas 2 --strict
```

This adds two replicas. Rex preserves `REXS_EXPERIMENT_ID` and assigns new replica
ranks. The same role launcher therefore discovers the existing head, Redis and
gateway, and joins the same LiteCast run. Ports, including vLLM's data-parallel
RPC port, are chosen per process. Workers fetch the newest retained adapter first
through a registered middle, verify the bundle SHA-256, load it into vLLM, then
advertise its immutable model name. The gateway discovers added workers without
changing the trainer's URL. Each worker continues following later publications.
Inference allocation failure triggers a bounded restart, allowing surviving replicas to serve;
if no replica is available, readiness waits remain bounded by the configured
timeout. Keep Rex controller refresh active for cleanup, including added jobs.

A LiteCast rollout group captures both policy step and immutable model name at
creation. All its requests, including later turns and retries, keep that model
name even if training advances. Saved trace info includes `policy_version` and
`inference_model_name`; adapter names contain the run ID, step and full SHA-256.
Workers reject unavailable versions rather than serving another version under
the requested name. Retired versions can therefore fail a long-running rollout;
size `retain_versions` / `max_versions` for the allowed rollout lifetime.
This is application-level provenance using verified adapter bytes and trusted
vLLM workers, not hardware attestation. The base alias denotes the pinned base
checkpoint before any adapter publication.

Rex uses the experiment's stored submission spec for extension. An experiment
submitted before the inference allocation was marked independent cannot acquire
that capability merely by editing this YAML. Existing allocations have not been
cancelled or resubmitted for this change.

## Failure and head recovery

Trainer exit (success, error or cancellation) terminates the experiment's owned
allocations. The saved Rex policy restarts head, middle and inference failures up
to ten times per allocation; an exhausted budget still fails the experiment.
These policies require the Rex controller to keep refreshing the experiment.
A replaced worker/middle rebuilds its cache from the current publisher.

The head persists Redis AOF under the run's shared `redis/` directory. Redis uses
`appendfsync everysec`; a host crash can lose the most recent second, and live
publisher/worker heartbeats rebuild discovery metadata. The head publishes its
current backend addresses in the shared file-backed bootstrap. Trainer-owned TCP
relays expose stable Redis/gateway URLs and resolve the head on each connection,
so clients reconnect after a head replacement changes hosts or ports. A dropped
request can fail and require retry; recovery is not uninterrupted service.
Adapter byte transfers remain direct through LiteCast, bypassing these relays.
The relay is part of the trainer allocation: its failure is a trainer failure.
Registry publication/readiness retries are bounded by the configured timeouts;
recovery that exceeds them can still fail training.

GPU roles materialize the model name as the downloaded local snapshot path before
enabling offline Hub access. This avoids an unpinned `main` lookup in the RL
launcher's pre-download step. Every role uses the same pinned revision and cache
layout; base-model identity checks reject a mismatched worker path.

## Per-replica request capacity

New submissions set `LITECAST_CAPACITY_ENABLED=1` on head and inference tasks.
The head then starts `prime_rl.litecast.gateway:create_app`, which uses LiteRegistry's
routing extension API. All implementation lives in the PrimeRL fork. The settings
in `capacity.toml` allow 2 active HTTP generation requests per replica, 64 waiting
requests at the gateway, and up to 120 seconds waiting for admission. The existing
orchestrator episode limit remains an additional global bound.

A worker advertises its process identity and request capacity with every ready
model name. The gateway shares one active counter across all versions on that
worker, admits only an exact-model match with a free slot, and chooses among
available replicas by relative occupancy. Waiting requests discover new replicas
within the configured polling interval. Queues are FIFO per model version;
different versions can progress independently. A slot is held through the entire
response, including streaming. Client disconnects cancel queued work and release
active slots. Queue overflow returns 429; admission timeout returns 503.

Workers enforce the same limit locally before contacting vLLM. This remains the
hard bound across gateway processes, gateway restarts, and direct callers.
A capacity rejection causes the gateway to release its reservation and requeue
with a short cooldown. Only pre-admission rejection and connection-establishment
failure are automatically retried; a failure after dispatch is not silently
replayed by this policy. No retry changes the requested policy version.

`GET /litecast/capacity` on the gateway exposes active requests by replica, queue
length, completions, overflow/timeouts and worker admission rejections. These are
process-local counters and reset on gateway replacement. Replica slots count
HTTP requests, not tokens, so tune the limit for the model and sequence lengths.
Deploy the PrimeRL gateway and updated sidecars together; workers without capacity
metadata are not eligible for this routing policy.

Middle supervisors log `MIDDLE_STARTED`, then `MIDDLE_STATUS` on state changes
and every 30 seconds. The status distinguishes waiting for a publisher, waiting
for weights/origin, syncing, and ready; it includes published/cache counts and
whether the LiteCast server has actually started. The server is created lazily
when an origin is discovered, so no library transfer logs are expected before
publication. `MIDDLE_READY` remains the per-adapter verified-cache event.

## Runtime cache and authentication

Before submitting, build the runtime archive once with
`sbatch examples/litecast-toy/build-runtime.sbatch`. The archive contains the
installed environment and its base Python; a small decompressor is copied beside
it because the training image does not include zstd. Each role verifies and unpacks it into
node-local storage under `/tmp/litecast-runtime-graf`, protected by a lock and
atomic completion. Console entrypoints and Python paths are relocated. Subsequent
roles/restarts on the same host reuse the extracted runtime. Source remains in the
shared checkout, so dependency changes require a new archive, while source edits
do not. Set `LITECAST_RUNTIME_ARCHIVE` consistently for the build and all tasks
when replacing dependencies; never overwrite an existing archive.

`RUNTIME_CACHE_MISS` / `RUNTIME_CACHE_HIT` log staging time. New hosts or cleaned
local storage still require extraction; scheduler and GPU initialization time are
not cached. Model downloads use `/tmp/litecast-hf/hub` and datasets use
`/tmp/litecast-hf/datasets`. The launcher sets these paths inside the container,
after wrapper environment overrides. Set `LITECAST_MODEL_CACHE` to choose another
cache root. Files are reused while the node retains them; `/tmp` can be cleaned
between allocations, and each new node downloads its own copy.

The trainer sets `NETRC` to the mounted host credentials file. Its bounded W&B
preflight runs inside the container before model startup and must authenticate
online. No API key is stored in this repository or the experiment specification.
A host-side login check is insufficient when the container changes HOME.

Runtime copying, compression, compilation and heavy import benchmarks must run
on a Slurm compute allocation, never on the login node. The build script requests
16 CPU cores and 64 GB and checks the resulting cache inside the training image.
Wait for that job to succeed before submitting the training experiment.

The gateway and worker also forward `/inference/v1/generate`, used by PrimeRL
training clients for exact token IDs and logprobs. It shares the same per-replica
capacity and cancellation handling as the OpenAI completion routes. The immutable
`model` selects the policy version; token IDs, sampling parameters and responses
pass through unchanged. The base alias is rewritten to the local base-model name
only at the worker. All of this routing lives in the PrimeRL fork.

## 50-step training run

`rexs submit examples/litecast-toy/experiment-50.yaml --strict --name litecast-toy-a40-50`
launches the same five allocations with two-hour limits and `train-50.toml`.
It trains for 50 optimizer steps and saves full checkpoints every 10 steps plus
the final checkpoint. The three-step configuration stays available unchanged.
`LITECAST_TRAIN_CONFIG` selects the training TOML for both LoRA validation and launch.
Keep the Rex controller refreshing this experiment until completion for restarts
and cleanup.

The 50-step config uses batch size 128, group size 8, maximum policy lag 4,
and 64 in-flight episodes (oversampling factor 0.5). AdamW uses betas
(0.95, 0.95), epsilon 1e-10, and zero weight decay. The gateway still admits
only two concurrent requests per inference replica, with up to 64 queued.

## Higher inference concurrency

`experiment-50-aggressive.yaml` uses four A40 inference replicas and the same
50-step optimizer/batch settings. `train-50-aggressive.toml` allows 256 in-flight
episodes (oversampling factor 2). `capacity-aggressive.toml` admits 16 requests per
replica and queues up to 256 requests for at most 300 seconds.
`inference-aggressive.toml` permits 16 sequences and 4096 scheduled tokens per
vLLM iteration. The context limit remains 1024 and generation limit remains 256.

The role launcher accepts `LITECAST_INFERENCE_CONFIG` and
`LITECAST_CAPACITY_CONFIG` alongside `LITECAST_TRAIN_CONFIG`. These settings are
read at startup; editing files does not retune running processes. Compare trainer
`time/wait_for_batch` and `time/forward_backward`, orchestrator
`time/wait_for_policy`, and LiteCast transfer timing to assess the result.


## Bootstrap resilience and bounded retention

The experiment advertises head/relay/middle endpoints through LiteRegistry's
file backend at `outputs/toy-ID/bootstrap`. Each endpoint is atomically replaced;
Redis remains the main registry. Shared bootstrap operations retry transient I/O
errors for up to 20 seconds, while permanent errors propagate. This removes the
shared SQLite database from the experiment's bootstrap path. A backend change
requires all roles to restart together; mixed bootstrap locations cannot discover
each other. LiteRegistry source is unchanged.

Long runs disable detailed token exports and keep only the latest full and weight
checkpoint, with checkpoint saves every 10 steps plus the final step. LiteCast
retains six adapter versions for policy lag four. A supervisor sweep every
30 seconds discards rollout/token-export steps older than a 12-step window and
broadcast artifacts older than six steps, using the latest stable trainer
publication as its watermark. Unconsumed future batches, logs, W&B metrics, and
checkpoints are outside that sweep. Checkpoint managers enforce checkpoint
retention after successful saves. Cleanup I/O failures are logged and retried at
the next sweep instead of killing training.

## Dataset cache and account selection

The trainer stages the pinned reverse-text parquet file into
`/tmp/litecast-hf/dataset-snapshots/reverse-text-REVISION`, then points the taskset
at that local directory. A completed cache hit makes no Hub call. On a cold
node, download retries HTTP 429 and transient server errors for up to ten minutes.
Prepared Arrow data remains under `/tmp/litecast-hf/datasets`. Cache directories
are node-local and can be cleared between allocations. Cache-hit reuse and
loading all 1000 examples with network access disabled are checked on Slurm.

Before starting GPU-role subprocesses, `credentials.py` injects `HF_TOKEN` from
`/gscratch/ark/graf/.cache/huggingface/token` and, on the trainer, `WANDB_API_KEY`
from the mounted `.netrc`. Existing environment tokens take precedence. Optional
`LITECAST_HF_TOKEN_FILE` and `LITECAST_WANDB_TOKEN_FILE` select different token
files. Tokens are not embedded in YAML, TOML, or command arguments. The trainer
sets `WANDB_ENTITY=graf` and verifies that its W&B token authenticates as `graf`
before training. Alternate authorized setups may explicitly set `WANDB_ENTITY`
and `LITECAST_WANDB_USERNAME` together.

Long-run middle profiles reserve 16 CPUs and 64 GB each without requesting an
exclusive whole host. Slurm keeps those CPUs unshared on this partition. This
avoids waiting for an entirely idle CPU node while preserving the requested
resources. The current run places its two middle replicas on separate hosts.

The runtime launcher removes only the fakeroot library from child-process
`LD_PRELOAD` after staging. Hundreds of concurrent SDK imports otherwise contend
on fakeroot's metadata semaphore. Other preload libraries are preserved; image
setup still uses the site wrapper. Existing running processes are unaffected.
