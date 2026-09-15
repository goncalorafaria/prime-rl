# Multiple trainers behind one LiteCast inference endpoint

The worker accepts an explicit JSON list of trainer subscriptions. Trainers A and B
in `tenants.json` share one loaded base model. Trainer C uses another engine with a
different base model. Start the corresponding PrimeRL inference backends first;
`backend_url` must expose health, models, generation, and dynamic LoRA load/unload
endpoints. The worker does not start or schedule these engines.

```bash
uv run python -m prime_rl.litecast.worker \
  --registry redis://HEAD:6379/0 \
  --tenants examples/litecast-multitenant/tenants.json \
  --advertise-host WORKER_HOST --port 8100 \
  --max-inflight-requests 4 --max-versions 8 --require-middle
```

Every trainer/publisher uses a unique `run_id`, its exact base-model identity, and
the same registry/gateway. Provision middle subscriptions for each run using
`prime_rl.litecast.middle --run-id ...`; each middle process currently follows one
run. Enable LoRA on every backend. Set `max_lora_rank` to cover its largest adapter,
`max_cpu_loras` to cover the sum of retained versions for all subscribed trainers
on that backend, and `max_loras` for the desired number of adapters per batch.
Do not attach another independent sidecar to the same backend: one pool owns its
adapter lifecycle and admission accounting. All tenants sharing an engine must
use the same actual base weights/revision and compatible adapter configuration.
A matching display name alone is not proof of checkpoint compatibility.

Routes retain immutable run/step/digest names, including a separate base alias for
each trainer. Reconciliation, readiness, and withdrawal operate independently per
tenant. A tenant with no publisher cannot prevent the others from serving.
`/health` is ready when at least one tenant is available; `/v1/models` lists the
currently ready routes. The subscriptions file is read at startup; restart the
sidecar to change it. Additional replicas can start with the same subscriptions.

All subscriptions share one replica identity and a conservative total request
limit, even across separate backends. This keeps gateway and worker admission
consistent. There are no per-tenant quotas or fairness guarantees. Independent
backend scaling can instead use separate worker endpoints behind the same gateway.
This is for trusted trainers on the existing private network; run IDs provide
routing isolation, not authentication or authorization.

Different base models require separate backend URLs and separately provisioned
GPU memory. This feature does not multiplex arbitrary base weights inside one
vLLM engine. `max_versions` and adapter cache limits apply per tenant, so memory
requirements grow with the subscription count. Single-trainer CLI usage remains
available with `--run-id` and `--base-model`.

Before launching each inference backend, source the local cache settings inside
its container:

```bash
source examples/litecast-toy/model-cache.sh
uv run inference --model.name Qwen/Qwen3.5-2B --enable-lora --max-lora-rank 8
```

Models cache under `/tmp/litecast-hf/hub`, shared by tenants on that node.
`LITECAST_MODEL_CACHE` overrides the root. Start other engines on separate ports.
The sidecar itself stores temporary adapters under its `--cache-dir` (default
`/tmp`). These node-local caches can disappear on preemption or node cleanup.
