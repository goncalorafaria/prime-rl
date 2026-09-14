# ShardCast LoRA search inference

This experiment runs one PrimeRL training node and independently replaceable
L40/L40S inference workers. It uses the PrimeBeaker BC search/WebTerminal/judge
harness. All new inference and weight-distribution code lives in this PrimeRL
fork. Rex only needs to launch and replace the roles below.

```text
                         stable registry / gateway node
                           LiteRegistry Redis + gateway
                            ↑ readiness     ↑ requests
trainer + orchestrator ─────┘                │
  local adapter files                       │
  ShardCast origin ── adapter shards ──→ L40/L40S worker × N
                                        vLLM + sidecar
                                            ↕
                                      ShardCast peer cache
```

## Source and installation

The branch starts from `goncalorafaria/prime-rl` `fix/training-top-p` at
`dc1972200bb4354ab06007bc8541bfe2cf13435a`. The original fork and checkouts are
untouched. The tested local ShardCast source is vendored in `packages/shardcast`;
PyPI's original 0.3.2 release does not contain these memory transport changes.
LiteRegistry is pinned to `948acd2`; no LiteRegistry source changes are required.

Initialize the fork's submodules, then install the `shardcast` extra in the
training/inference image. Keep the fork's CUDA and vLLM dependencies together.

```bash
git submodule update --init --recursive
uv sync --extra shardcast
```

The trainer and registry/gateway image also need the existing PrimeBeaker search
environments and their dependencies. Use the same search service images,
corpus, judge profile, credentials, and template assets as your current
PrimeBeaker deployment. `Dockerfile.shardcast` overlays this source onto an
explicitly supplied, compatible top-p training image. It is not a CUDA base
image and does not identify an already-built Beaker image.

## Roles for Rex

1. **Registry/gateway:** run the existing LiteRegistry Redis service and
   `uv run python -m primebeaker.gateway --registry "$REGISTRY" --port 1212`.
   Reuse the current `localsearch:bc-rl-v1`, WebTerminal and judge service stack.
   PrimeBeaker's gateway adds the search/judge routes to LiteRegistry inference
   routing. The CPU-service recipe is
   `primebeaker/examples/search-agent-webterminal/services.yaml` in the
   PrimeBeaker repository; point every service at the same registry.
2. **Training node:** trainer + orchestrator, local rollout/checkpoint storage,
   ShardCast origin on TCP 8201. Run `run-trainer.sh` once per experiment.
3. **Inference worker:** one GPU, local vLLM on 8000, sidecar gateway target on
   8100, ShardCast cache on 8101. Run `run-worker.sh` once per GPU. The wrapper
   exits if either child exits, allowing Rex to replace the entire worker.
   For multiple GPUs per node, give each process its own `CUDA_VISIBLE_DEVICES`
   and distinct `BACKEND_PORT`, `WORKER_PORT`, and `SHARD_PORT` values.

No worker needs `/weka`, the trainer's filesystem, or a collective rendezvous.
Every worker needs the identical base model available locally or through HF.
Use a pinned model snapshot for reproducible runs. The existing judge service
still needs access to its configured checkpoint and corpus.

The training node and replicas must be mutually reachable on their advertised
ShardCast ports; the gateway must reach each sidecar port. `ADVERTISE_HOST`
must identify that particular node. The `http` default works across ordinary
TCP networks; select UCXX only for a verified RDMA deployment.

## Three-step Qwen3.5-2B search smoke

`search-2b.toml` uses rank-8 LoRA, top-p 0.97, the existing BC search harness,
8 rollouts per batch, two rollouts per prompt, and shorter 8K episodes. It keeps
the search judge/profile settings and uses a smaller budget for the smoke run.
`inference-l40.toml` runs one tensor-parallel rank per GPU with eight adapter
slots. These are smoke settings, not measured L40 capacity limits.

On every worker:

```bash
export REGISTRY=redis://REGISTRY_HOST:6379/0
export RUN_ID=search-2b-smoke-UNIQUE
export ADVERTISE_HOST=THIS_WORKER_IP
export CUDA_VISIBLE_DEVICES=0
bash examples/shardcast-search/run-worker.sh
```

On the training node, with the same `REGISTRY` and `RUN_ID`:

```bash
export GATEWAY_URL=http://REGISTRY_HOST:1212
export ADVERTISE_HOST=TRAINER_IP
bash examples/shardcast-search/run-trainer.sh
```

The launcher accepts additional PrimeRL overrides, e.g. `--max-steps 20` and
`--deployment.num-train-gpus 8`. `REGISTRY_HOST` and `TRAINER_IP` in the checked-in
TOML are placeholders; the script supplies the actual addresses.
Use a fresh run ID for an independent experiment. On controller restart, wait
for its old lease to expire before resuming with the same ID, or choose a new
ID and resume the local trainer checkpoint. Only one publisher may own a run.

## Version and failure semantics

The trainer writes its normal PEFT adapter locally. The orchestrator bundles
`adapter_config.json` and `adapter_model.safetensors`, publishes through
ShardCast, then advertises the publication descriptor in LiteRegistry with a
renewable lease. The descriptor contains only run, model, step, digest and size;
it contains no node addresses or node-local ShardCast versions.

Both the trainer origin and worker peer caches register each retained bundle
under `shardcast:{run_id}:{digest}` using LiteRegistry's existing server API.
Registrations carry the source address, local ShardCast version and origin/peer
role. Sources renew heartbeats and withdraw evicted bundles. Workers resolve all
weight sources through LiteRegistry, try peers first, and fall back to a
registry-discovered origin. There is no configured-origin discovery bypass.

Inference requests use the existing LiteRegistry gateway. Weight transfers use
direct ShardCast connections to the endpoints discovered through LiteRegistry;
adapter bytes never go through Redis or the inference gateway. Consequently,
workers must be able to reach advertised ShardCast ports across the deployment
network. LiteRegistry supplies discovery and liveness, not a network tunnel.
All registration and transfer integration code stays in this PrimeRL fork;
LiteRegistry itself requires no changes.

Workers load immutable, content-addressed adapter names. They register those
names only after transfer integrity verification and a successful vLLM load.
The orchestrator advances its policy only when `min_replicas` report that exact
version ready; it does not wait for a fixed fleet. Slow/preempted workers can
catch up from peers or the origin. It retains the last `retain_versions`
publications; replicas must have at least that many adapter slots. Training
configuration requires retention of at least `max_off_policy_steps + 2`.

Older rollout requests keep their versioned model names. A worker cannot
silently substitute newer or older weights for that name, including when the
gateway has cached an obsolete registration. Retired adapters stop accepting
new requests; an adapter already serving a request is not unloaded until that
request finishes. Episodes that outlive the retention window can fail on their
next turn. Size the window for long search/evaluation episodes.

On loss of the publisher lease or backend health, sidecars refuse requests and
withdraw readiness. Crashed records age out via LiteRegistry heartbeats, and
the gateway retries other replicas. This is failover between requests, not
migration of an in-flight generation/KV cache. A preempted request may need to
be retried by the rollout environment. The rollout off-policy bound still
applies. Rex should restart a lost backend and sidecar together.

The origin and each peer hold at most `max_adapter_bytes * retain_versions`
bytes of ShardCast payload cache. Adapter extraction uses node-local disk.
The publisher is a single writer per run; use unique run IDs to isolate jobs.

## Validation

The CPU lifecycle test uses real ShardCast HTTP servers, the actual LiteRegistry
gateway and registry, and an adapter-aware CPU engine that loads real
safetensors. It tests two immutable adapter versions, peer-assisted late joins,
replica loss, publisher-lease withdrawal, registry-only origin discovery,
heartbeat re-registration and removal of evicted source records. It does not simulate GPU kernels
or establish training reward/throughput results.

```bash
PYTHONPATH=src uv run --no-sync pytest --confcutdir=tests/unit/shardcast \
  tests/unit/shardcast -q
```

A GPU acceptance run must additionally establish: successful Qwen3.5-2B LoRA
loading on L40 and L40S, three actual optimizer updates on the search harness,
correct token/logprob exports, continued rollouts after removing one worker,
a replacement worker reaching the current adapter, and measured update/rollout
latency at increasing fleet sizes. No army-scale throughput claim is made by
the CPU contract test.

## Validation recorded for this branch

On 2026-09-14, the 12 feature tests passed, including the actual PrimeRL pool
factory, real HTTP/gateway transfers and real Redis publisher fencing. Both
TOMLs validate against the fork's pinned config classes. Shell syntax and
Python lint pass. The launcher can materialize a concrete TOML without
starting training via `run-trainer.sh --write-config /tmp/search.toml`.

A full CUDA environment sync was attempted; flash-attn-4's Git metadata scan
hit the shared filesystem's 40-second build timeout. Lock generation succeeded
with `VCS_VERSIONING_SUBPROCESS_TIMEOUT=300` and
`SETUPTOOLS_SCM_SUBPROCESS_TIMEOUT=300`, preserving the existing workspace
environments and dependency pins except dependencies constrained by the new
LiteRegistry extra. GPU training and image execution have not been validated.
