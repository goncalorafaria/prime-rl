# Multi-turn prompt affinity benchmark

Submit `experiment.yaml` with Rex. It allocates four independent single-A40 jobs, each with 8 CPUs and 64 GB RAM,
plus an 8-CPU/32-GB benchmark coordinator. Each A40 runs one TP=1 Qwen3.5 SFT
2B engine. Workers can land on different hosts. All generation passes through the
PrimeRL LiteRegistry gateway. The existing training fleet is independent.

Compare no affinity against the production prompt-hash weak affinity (two
preferred replicas, capacity-aware overflow). With only two replicas this policy
would prefer the entire fleet, so four replicas are used.

The workload replays eight distinct document prompts, four siblings per prompt,
and four successive turns containing fixed assistant/tool additions. Generated
responses are measured but not appended: both conditions receive exactly the
same prompt tokens. This is controlled cache-locality replay, not task-quality
or live-agent evaluation. It uses the frozen SFT base model without LoRA updates.

Both modes use greedy sampling, 96 forced output tokens, a 16K context limit,
prefix caching, and concurrency 8 or 32. Each concurrency uses order none/weak,
then weak/none. Every phase has a fresh cache salt; kernels receive warmup traffic
before measurement. Engine scheduler limits and hardware stay constant.

Artifacts under `outputs/affinity-bench-<Rex ID>/` include GPU inventory, engine
configs/logs, the exact token corpus, request timings, token/cache usage, aggregate
throughput, episode durations, and deltas of vLLM TTFT/prefill/cache metrics.
TTFT comes from vLLM metrics, not HTTP response latency: the token endpoint
returns a complete response. `COMPLETE` is written only after all eight phases.

Interpret latency and throughput together. Results do not characterize network failures, TP=4 performance, training quality,
or weight-transfer overhead. Cache salting relies on the engine's cache isolation;
check per-phase cache counters and errors before comparing results.
