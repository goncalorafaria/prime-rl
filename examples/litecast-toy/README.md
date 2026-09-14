# Real LoRA training smoke

One unified Rex experiment runs three Qwen3.5-2B LoRA optimizer steps against
`reverse-text-v1`. The reward compares the model's tagged reversal with the
computed correct reversal; no search service, terminal service or judge model
is needed. This is a deterministic graded reward, not an LLM judge.

Allocations: CPU head (Redis + gateway), two independent CPU middle nodes,
one A40 trainer/orchestrator and one A40 vLLM worker. The worker requires a
registry-advertised middle source for adapter loads. LiteCast itself remains
independent of LiteRegistry; `prime_rl.litecast.middle` owns discovery,
verification and registration. The trainer is the Rex completion task; a
service allocation failure fails the experiment and cleanup is restricted to
its owned allocations. Keep Rex controller refresh active.

The profiles use the local Klone account, pinned Ubuntu CUDA SIF and isolated
training environment at `outputs/toy-runtime-host`. Install that environment
for the image's glibc platform, including the `litecast` and `flash-attn` extras;
the login host's glibc is too old for the pinned Mooncake wheel. Always execute
the training environment inside the image. The existing training environment
and live evaluations are not modified.

Validate before submitting:

```bash
rexs validate examples/litecast-toy/experiment.yaml --strict
rexs submit examples/litecast-toy/experiment.yaml --strict --name litecast-toy
```

The run uses its Rex experiment ID for the SQLite bootstrap namespace and
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
