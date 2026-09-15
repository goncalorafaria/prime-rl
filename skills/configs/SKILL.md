---
name: configs
description: How the prime-rl config system works — TOML files, CLI overrides, composition, and special patterns. Use when creating configs, debugging config errors, or overriding values via CLI.
---

# Configs

prime-rl uses [`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — a Pydantic-based TOML + CLI config system (no tyro). Every entrypoint accepts TOML files via `@` and CLI overrides.

## Loading and composition

```bash
uv run rl @ examples/basic/reverse-text/rl.toml                                  # single TOML
uv run rl @ examples/basic/reverse-text/rl.toml --max-steps 50                   # CLI override
uv run rl @ base.toml @ overlay.toml                                       # left-to-right merge
uv run rl --model @ model.toml --data @ data.toml                          # nested section files
uv run rl @ base.toml --trainer @ trainer.toml --trainer.lr 1e-3           # mixed
```

Resolution order: CLI > config files (left-to-right) > class defaults. Merging is deep — unset fields in an overlay are preserved from the base.

Naming: CLI uses kebab-case (`--model.max-model-len`); TOML uses snake_case (`max_model_len`).

## Inspect & validate

```bash
uv run rl --help                                  # all fields and defaults
uv run rl @ rl.toml --dry-run --output-dir /tmp/x # write resolved TOML to /tmp/x/configs
```

## Validators

Incompatible combinations (e.g. CP requires flash attention) must raise in a `model_validator` at resolve time, not at runtime. When renaming a field, remove the old spelling: no `validation_alias`, no auto-translating `mode="before"` validator. The old key then fails as an unknown key, which is the signal. An alias that stays forever is worse than a break — it never gets retired, and a key whose *meaning* changed silently misconfigures the run.

## Special syntax

**No inline tables** — checked-in configs use `[section]` headers, never `key = { ... }`. Expand `env.taskset = { id = "..." }` to a full-path header (`[orchestrator.train.source.env.taskset]` — subtable headers after a `[[...]]` entry attach to that entry).

**Booleans** — CLI `--flag` / `--no-flag`; TOML must be explicit (`enforce_eager = true`).

**None** — TOML has no null, use the string `"None"` (`max_model_len = "None"`); CLI: `--model.max-model-len None`.

**Lists** — TOML uses array of tables; later config files replace lists wholesale, so overlays must include the full desired list:

```toml
[[orchestrator.train.source]]
name = "reverse-text"

[orchestrator.train.source.env.taskset]
id = "reverse-text-v1"

[orchestrator.train.source.env.agent.harness]
id = "null"

[orchestrator.train.source.env.agent.runtime]
type = "subprocess"

[[orchestrator.eval.source]]
name = "reverse-text-eval"

[orchestrator.eval.source.env.taskset]
id = "reverse-text-v1"
split = "test"

[orchestrator.eval.source.env.agent.harness]
id = "null"

[orchestrator.eval.source.env.agent.runtime]
type = "subprocess"
```

CLI: `--orchestrator.train.source.0.env.taskset.id reverse-text-v1` or `--orchestrator.eval.source.0.env.taskset.id reverse-text-v1`.

**Dicts** — TOML uses a section; CLI takes a JSON string: `--vllm-extra '{"key1": "value1"}'`. This works for plain `dict` fields only — nested pydantic-model fields (e.g. `algo`) reject JSON strings; use dotted keys (`--orchestrator.algo.type max_rl`) or a TOML overlay file.

**Discriminated unions** — set the `type` field to pick the variant (`[orchestrator.algo] type = "max_rl"`). Omit `type` to keep the default variant.

**Algorithms** — `[orchestrator.algo] type = "grpo" | "max_rl" | "rae" | "hierarchical_grpo" | "opd" | "opsd" | "sft" | "echo"` — the type names the algorithm (credit assignment + loss routing, fused), and each type's class defaults are its vetted setting; any other key you set is your own assembly (e.g. `[orchestrator.algo.roles.user] alpha = 0.1` for echo — setting any echo role replaces the whole role table). `hierarchical_grpo` is only valid with a proposer-solver env: it compares solvers with attempts on the same proposed problem and proposers with other proposals in the group. There is no preset layer, and no config hook that points at user code — a new algorithm is a named class in the repo (subclass `Algorithm`, register it). Per-source override: `[orchestrator.train.source.algo] type = "opd"` (the source assembles its own algorithm). prime-rl only hosts the trainable policy; frozen models are inline external endpoints on the algorithm, named where the model is used — `[orchestrator.algo.teacher]` for opd (the frozen model scored against), `[orchestrator.algo.sampling.source]` for sft (the model it samples from), each with `name` + `base_url`. There is no shared `teacher` slot. opsd declares no model — it self-distills against the live policy. See `docs/algorithms.md`.

**`BaseModel | None` fields** — bare flag enables defaults; nested override enables and sets:

```bash
--model.compile             # enables compile with defaults
--model.compile.fullgraph   # enables and sets fullgraph=true
```

In TOML, an empty section header (`[ckpt]`) does the same.

## RL trainer token exports

For rollout debugging, enable trainer-side token export with `trainer.enable_token_export = true` (or `--enable-token-export` when running the trainer entrypoint directly). It writes one JSONL record per exported sequence. Single-run/fallback exports go under `output_dir/token_exports/step_<step>/rank_<rank>.jsonl`; multi-run trainer exports with packer metadata go under the owning run directory, `output_dir/<run_id>/token_exports/step_<run_step>/rank_<rank>.jsonl`. Each record stores aligned per-token arrays for token ids, loss mask, component weight streams (rl/ce/ref_kl), advantages, entropy, mismatch KL, inference/trainer logprobs, importance ratios, probability deltas, and masking diagnostics. It does not decode token text in the trainer.

```toml
enable_token_export = true
```

Leave it unset for normal training. When enabled, it exports every sequence from each exporting rank.

## Key files

- `packages/prime-rl-configs/src/prime_rl/` — config classes under `configs/`; `utils/config.py` re-exports `BaseConfig` and `cli`
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs

## Training nucleus sampling

Set `top_p` directly under `[orchestrator.train.sampling]`, for example
`top_p = 0.97`. The valid range is `(0, 1]`; the default is `1.0`.
Do not put `top_p` in `sampling.extra_body`: renderer clients prioritize the
explicit sampling fields. Configuration validation rejects that placement.

## LiteCast inference replicas

For externally launched preemptible LoRA workers, configure
`orchestrator.model.client.litecast` and point `model.client.base_url` to the
LiteRegistry/PrimeBeaker gateway. Keep the trainer/orchestrator local handoff
as `weight_broadcast.type="filesystem"`; omit the local `inference` section
and set `deployment.num_infer_gpus=0`. Use LoRA, disable direct engine metrics,
and retain at least `max_off_policy_steps + 2` adapter versions. The example
and separate worker command are in `examples/litecast-search/README.md`.

The pinned CLI rejects numeric source-list overrides such as
`--orchestrator.train.source.0.legacy.args.search_server_url`. For the LiteCast
launcher, materialize those addresses in TOML with
`uv run python -m prime_rl.litecast.launch --write-config /tmp/search.toml`
instead; it preserves the source as an array of tables.

For a self-contained LiteCast training smoke without search/judge services, use
`examples/litecast-toy/train.toml` (`reverse-text-v1`, deterministic reversal
similarity reward). Its unified Rex spec owns a CPU head, two CPU middles,
one A40 trainer and one A40 inference worker. `prime_rl.litecast.middle`
provides registry supervision outside the standalone LiteCast package.
The worker's `--require-middle` option restricts weight downloads to registered
middle sources and fails closed while none are ready.

In the toy config, gradient offloading uses `trainer.model.fsdp_cpu_offload`,
which also offloads parameters and optimizer states. Set
`trainer.model.optim_cpu_offload=false` when enabling it; those two offload
paths are mutually exclusive. Activation offloading uses the
`[trainer.model.ac_offloading]` section.

For LiteCast sidecars, set `router="None"` at the root of the standalone
inference TOML. PrimeRL otherwise starts a local vllm-router by default, while
the sidecar needs the engine's native LoRA admin endpoints. The LiteRegistry
gateway already handles external request routing.

The colocated toy trainer/orchestrator uses `[rollout_transport] type="filesystem"`
to avoid default ZeroMQ port 5555 collisions with unrelated jobs on shared nodes.
This is separate from the LiteCast weight transport to remote inference.

### LiteCast toy telemetry

The unified toy enables shared `[wandb]` in `examples/litecast-toy/train.toml`.
Its role launcher must not set `WANDB_MODE=disabled`; the trainer selects online
mode and a Rex-specific run name. LiteCast update metrics use
`litecast/policy_step` as their W&B axis and are emitted per successful update.
Worker fetch/load timings are carried in readiness registration metadata. Check
`litecast/measured_replicas` before interpreting averages; a missing sample is
not a zero-duration transfer. End-to-end readiness includes middle propagation.

### Elastic LiteCast inference

Rex extension requires `independent_replicas: true` on the inference allocation
at submission; `rexs extend ID inference --replicas N` adds N replicas using the
stored experiment spec and preserves its experiment ID. Keep that ID as the
LiteRegistry/LiteCast run namespace, and choose per-replica vLLM RPC ports as well
as HTTP ports. The toy uses bounded Rex restart policies for non-trainer roles. Trainer exit
triggers experiment cleanup; readiness timeouts and restart budgets still apply.
LiteCast groups must pin the immutable model name alongside the policy step;
saved trace info carries both `policy_version` and `inference_model_name`.

For the offline toy, materialize GPU model configs with the local pinned snapshot
path. Caching a commit alone does not create a cached `main` ref, so leaving the
Hub repo name in the RL model config can break `pre_download_model` offline.
The replaceable head uses shared Redis AOF and advertises backend endpoints;
trainer-owned TCP relays publish stable client URLs via the SQLite bootstrap.

### LiteCast request admission

Use `prime_rl.litecast.gateway:create_app` on the head and set
`LITECAST_CAPACITY_CONFIG` to the toy's `capacity.toml`. Give sidecars the matching
`--max-inflight-requests`. Gateway reservations are shared across model versions
by replica process ID; the worker enforces the authoritative limit. Keep the
orchestrator episode limit as a separate global bound. Slot lifetime includes
streaming and cancellation; changing only readiness/min_replicas does not limit
request concurrency. Deploy gateway and worker changes together because admission
requires capacity metadata on readiness registrations.

The toy stores `LITECAST_CAPACITY_ENABLED=1` in head/inference task environments.
This gates the coordinated admission rollout: older submitted specs keep their
original gateway behavior even if a service restarts from the shared checkout.
