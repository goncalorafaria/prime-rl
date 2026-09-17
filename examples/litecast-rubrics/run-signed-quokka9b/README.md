# Signed RubricHub training with Quokka 9B

Submitted on 2026-09-17 as Rex experiment `81c1b41b41a04778a8a6ae4a78cc1642` (`rh-signed-quokka9b-litecast`). Initial Slurm jobs: `40249463`–`40249470`. Submission does not establish model readiness or training progress.

## Allocation layout

- Head: CPU Redis, advertised through the shared SQLite head registry.
- Middle: eight independent CPU weight-distribution relays.
- Trainer: one owned H200, Qwen/Qwen3.5-4B, LoRA rank 32, and a local LiteCast capacity gateway.
- Policy inference: eight independently scheduled single-GPU L40/L40S/A40 replicas, TP1.
- Judge: four independent single-GPU L40/L40S/A40 allocations running Quokka 9B and a local LiteRegistry gateway.
- Judge API: CPU allocation running the existing terminal-enabled Quokka judge workflow.
- Terminal: sixteen independent CPU allocations registered with LiteRegistry.

Thirteen GPUs across thirty-nine Slurm allocations total. All allocations belong to one rexs experiment; trainer completion owns service cleanup. Policy replicas remain independently addable. Start requires the first policy replica plus the judge, not all policy replicas.

## Model and training settings

Policy initialization is the original Qwen/Qwen3.5-4B model, not the Sloth model in the example or a resumed RL checkpoint. Judge checkpoint is the existing `quokka-sft-qwen35-9b-rltracer-xmlv1-search25-baseformula-ba8d07-step400/weights/step_400` in the gfaria-ai2-transfer S3 bucket. No Lion model is configured.

This configuration follows the supplied LiteCast LoRA example: 400 steps, learning rate 4e-6, batch 128, group size 8, 256 inflight episodes (oversampling factor 2), rank 32/alpha 64. Policy context is 32K and completion limit is 16K. These are distinct from the older full-parameter RH run; this is explicitly a LiteCast LoRA run. Numerical precision settings were not changed.

Dataset: `/gscratch/ark/graf/data/rubrichub-signed-penalties-20260917/hf`, train split, 6,437 questions. The dataset retains originals for failed/empty generation rows. No artificial held-out split was created; evaluation is absent until the original validation split is recovered.

## Signed rewards

Criterion present means pass, including a bad behavior. Signed reward is sum(weight * pass) / sum(abs(weight)). Mixed groups sample up to four negative criteria among eight total, filling shortages with the other sign. Empty and truncated outputs receive -1.25; orphan thinking-tag penalty remains -0.2. The generated penalties are proposals with structural/evidence validation, not independently fact-checked labels.

Quokka judges each sampled criterion using its existing XML terminal template (up to 64 tool calls). The `webterminal` backend exposes the trained `terminal` tool with the response on stdin and URL inspection support. Both model calls and terminal calls go through the judge API's local LiteRegistry gateway. The trainer uses its LiteCast gateway for `/judge`. Judge readiness requires both Quokka and the terminal service to register. The API has no shared request-concurrency gate; each synchronous workflow runs in its own worker thread.

## Launch command

```bash
/gscratch/ark/graf/.local/share/venvs/klone-work/bin/rexs submit \
  /gscratch/ark/graf/prime-rl-shardcast/examples/litecast-rubrics/run-signed-quokka9b/experiment-group8.yaml \
  --name rh-signed-quokka9b-litecast --strict
```

Runtime starts from the existing immutable LiteCast runtime cache. Model and dataset staging are node-local and use a separate cache namespace from the existing Sloth experiments. Judge weights download to node-local storage. Credentials remain mounted and are not embedded in the configs.

Submitted following explicit user launch authorization. No separate GPU smoke was performed for this new combination; model readiness and initial training steps still need live verification.

## Offline validation

The initial Rex strict validation resolved eight allocations without warnings; capacity was subsequently extended to four policy replicas and sixteen terminal allocations. The judge container imports the reusable workflow, loads the Quokka profile, and binds the advertised `terminal` tool to the webterminal backend. Contract tests cover gateway-only model routing, the terminal template, and penalties for empty/truncated completions. GPU inference and training have not been tested for this configuration.

Startup recovery: judge launchers use Rex `BEAKER_JOB_ID`. Judge readiness publishes a base URL because the HTTP probe appends `/health`. The judge API was replaced as job `40249878` within the same experiment; the trainer and GPU services were retained.

Relaunched at user request as `rh-signed-quokka9b-litecast-r2`, Rex ID `b7a30b4c874445c8811da1fc814095b2`, with 31 allocations (four policy GPUs, four Quokka GPUs, one trainer GPU, four middle relays, sixteen terminals, head and judge API). Learning rate is 1e-6. The original experiment was cancelled and its outputs retained.

Current relaunch: `rh-signed-quokka9b-litecast-r3`, Rex ID `9b7ef52c5f7648d9ac0a2e4a031ffd95`. Eight policy GPUs, four Quokka GPUs, four middles and sixteen terminals; LR 1e-6. Includes the LiteCast `/judge` proxy fix. R2 cancelled with outputs preserved.

The judge API starts the JTC DataDev gateway and imports the JTC source with per-rollout chat session headers. Its soft affinity uses a shared 900-second binding and a local load slack of two requests. Live acceptance is `check-live-affinity.py`, executed within an existing judge-API allocation against its parent registry/GPU models. It exercises consecutive chat turns, unavailable-replica handoff, correct/incorrect/negative-criterion judge tasks and terminal use, retaining JSON evidence and affinity logs.

## R4: verified judge affinity, 256 inflight, LR 4e-6

Rex experiment `ff0351ce639449ff90296f3cba6f34c2`, name `rh-signed-quokka9b-affinity-r4`, submitted 2026-09-17. Trainer job `40251588`, head `40251547`, judge API `40251570`. All 39 allocations reached Slurm RUNNING on the initial check; this alone does not establish optimizer progress. The layout above includes eight middle allocations. R3 was cancelled after live acceptance, with its outputs retained.

Live acceptance passed on R3's existing GPUs through its registry and trainer gateway before replacement: three chat turns retained their replica; an unavailable preferred replica handed off successfully; three real two-turn judge sessions retained their replicas across terminal calls and returned the expected positive, incorrect-answer, and negative-criterion judgments. All terminal calls succeeded. Evidence: `/gscratch/ark/graf/prime-rl-shardcast/outputs/toy-9b7ef52c5f7648d9ac0a2e4a031ffd95/affinity-smoke-1511226c/evidence.json` (adjacent `judge-api.log` records affinity decisions).

The live test caught that sampling arguments did not become HTTP headers. The final client uses `extra_headers_from_state={"X-Session-ID": "trajectory_id"}` so each tool-use trajectory carries the same HTTP session header across turns. Seventeen gateway tests and strict Rex/config validation passed. R4 uses learning rate `4e-6`, `max_inflight_episodes=256`, and `oversampling_factor=2.0`.

## R5: halved trainer packing

Submitted at user request as `rh-signed-quokka9b-affinity-r5`, Rex ID `1bab5006199f4666bc176eb82c9b38a9`. Trainer packing is `131072`, LR `4e-6`, and inflight episodes `256`. R4 failed on the first backward pass with CUDA out-of-memory at 262091 packed tokens. R5 memory fit requires a successful backward pass; judge timeout and affinity-persistence issues identified in R4 have not been changed for this retry. Existing outputs are preserved.

New runs use `head.sqlite3` for discovery and start the actual gateway on the trainer. Existing runs with a `bootstrap/` directory keep their prior topology on allocation restart. This code change does not migrate R5.

## Verified Quokka judge lineage

The deployed Quokka9B step-400 checkpoint is RL-trained despite `sft` in its directory name. Its archived trainer `output_dir` exactly matches the deployed run directory. The archived orchestrator specifies GRPO, group size 16, batch size 128, and 400 steps; the trainer uses LR 4e-6. It initializes from `quokka_sft_qwen35_9b_base_rltracer_16k_xmlv1_searchsourcecriteria_v3_32k_lr1e-5_200steps-retry1/weights/step_200`. The step-400 archive contains a STABLE marker and all four weight shards.

Authoritative configs copied from `s3://gfaria-ai2-transfer/prime_sft/outputs/quokka-sft-qwen35-9b-rltracer-xmlv1-search25-baseformula-ba8d07-step400/configs/` are saved in `provenance/judge-rl-trainer.toml` and `provenance/judge-rl-orchestrator.toml`. Do not infer training type from the directory's SFT prefix.

## R6: SQLite discovery, trainer-local gateway, uncapped judge API

Rex ID `ba164d8b4a4d43cdbff86ae32a88854d`, name `rh-signed-quokka9b-localgw-r6`. R5 cancelled at explicit user request; outputs preserved. New run uses SQLite head discovery and no trainer TCP relay. Judge request semaphore and shared executor cap removed (25 judge tests passed, including 40 concurrent requests). Packing 131072, LR 4e-6, inflight 256. Submission alone does not establish training progress.

The trainer-local gateway writes stdout and stderr to `outputs/toy-<Rex ID>/logs/gateway.log`, separately from the trainer allocation log. This includes request logs, warnings, and tracebacks; the file appends across service launches. Existing processes retain their original output descriptors until restarted.

## R7 infrastructure recovery relaunch

Rex ID `90e63a1e344641d59d7e5dfda152a71b`, name
`rh-signed-quokka9b-recovery-r7`. R6 cancelled at explicit user request, with
outputs preserved. All 39 initial allocations reached RUNNING: head `40259888`,
judge API `40259909`, trainer `40259927`. Runtime/model staging is still required
before this establishes training progress. This launch loads the shared registry
source with persistent-loop model heartbeats, read-only SQLite initialization,
explicit SQLite connection closure, and nonfatal transient endpoint refreshes.
Soft-affinity metadata failures preserve successful model responses. Judge
requests use TIMEOUT=600 and MAX_RETRIES=5. Trainer-local gateway output goes to
`outputs/toy-90e63a1e344641d59d7e5dfda152a71b/logs/gateway.log`.

R7 extended through `rexs extend` with four additional judge replicas at user
request: jobs `40260111`–`40260114` (judge ranks 4–7), initially PENDING. These
load the updated model-service launcher, which omits `max_num_seqs` for judges
and retains `max_num_batched_tokens=32768`. The original four running judges
retain their 16-sequence setting until restarted. All eight belong to the same
Rex experiment and its completion cleanup.

Judge GPU launcher: model-service plus vLLM only, registering directly with the
shared head after local model health succeeds. Only the judge API owns the judge
workflow gateway. Existing GPU processes retain their old auxiliary gateway
until their next restart.

## R8 full deployment restart

Rex ID `f185ab72e55c4e49946b3aca4d5e5304`, name `rh-signed-quokka9b-gatewayfix-r8`.
R7 cancelled at user request; outputs preserved. The saved spec now includes
eight judges, eight policy replicas, eight middles, sixteen terminals, one head,
one judge API, and one trainer (43 allocations). Judge GPU launchers no longer
start a gateway; the judge API retains its local gateway. All judges omit the
16-sequence override. LR 4e-6, inflight 256, 600-second timeout and five attempts
remain configured. Submission is not confirmation of optimizer progress.

### Live RLTracer view

R8 viewer: `http://localhost:25238/explorer`, using the run’s `rltracer/`
directory. Splits are `policy` and `judge`. `export-judge-split.py --watch`
extracts embedded `info.jtc_rubrichub_judge.judgments[*].trace.messages` from
completed policy traces every 30 seconds, preserving rubric weights, labels,
policy IDs, and tool conversations. It uses only `train/all` to avoid duplicate
effective-batch traces. The exporter runs in tmux `litecast-recovery:JudgeExport`;
the viewer runs in `litecast-recovery:RLTracer`. Policy traces are symlinks and
remain subject to the source run’s retention. Derived judge files are independent.

Judge question formatting: the environment sends the content of a single user
message as plain text. Multi-message prompts retain every turn with readable
role labels. This is applied when building judge requests, not by rewriting
stored prompts or existing traces. Already-running environment workers need a
restart to load this source change.

## R9 updated-source relaunch

Rex ID `59e7c7bb69534afcb3cc22ef2fe1bd76`, name `rh-signed-quokka9b-updated-r9`.
R8 cancelled at explicit user request; outputs preserved. All 43 allocations
submitted; trainer `40264393`, head `40264327`, judge API `40264358`. Uses the
plain-content judge question formatter, current shared PrimeRL/PrimeBeaker/JTC
code, and standalone LiteCast source (`f81de15`) via runtime PYTHONPATH. Source
revisions, diffs, and selected config snapshots are in the run’s
`launch-provenance/`. Protocol/capacity/middle preflight: 22 passed initially;
the remaining Redis-startup timeout passed with deployment-matched Redis and
preload settings. Optimizer progress still requires live verification.
