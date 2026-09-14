# Unified Rex toy deployment

The runnable smoke is in `../litecast-toy/experiment.yaml` and `../litecast-toy/train.toml`.
Use reverse-text with a deterministic verifiable reward; no search or judge services.

One Rex experiment owns five allocations:

| Role | Resources |
| --- | --- |
| Head | CPU Redis and LiteRegistry gateway |
| Middle 0 | CPU LiteCast middle and registry supervisor |
| Middle 1 | CPU LiteCast middle and registry supervisor |
| Trainer | One A40, Qwen3.5-2B rank-8 LoRA, three optimizer steps |
| Inference client | One A40, vLLM, LiteCast client and peer cache |

Both GPU roles use Klone's A40 profile. Training explicitly enables activation
checkpointing, activation offloading and FSDP CPU offloading (parameters,
gradients and optimizer state); optimizer-only CPU offloading is disabled.

The trainer publishes adapters, both middle nodes discover and cache them,
and inference is restricted to registered middle sources. LiteRegistry owns
discovery and leases; LiteCast transfers bytes directly. Integration remains
in PrimeRL, with no LiteRegistry dependency in the standalone LiteCast package.

The trainer is the completion task. Rex cleans up owned allocations on training
completion or required-service failure. A pass requires actual optimizer steps,
checkpoint output, rollout rewards, and successful middle-mediated adapter loads.
See the runnable smoke README for runtime requirements and execution status.
