# Compact logprob vLLM install

The GPU dependency is pinned to `goncalorafaria/vllm` commit `792b0dd`.
The fork uses the vLLM 0.29.0 / Torch 2.13 / CUDA 13.0 binary stack.

```bash
source scripts/vllm-build-env.sh
uv sync --all-extras
```

The installer and CUDA Dockerfile source the same binary selection script.
A direct `uv sync` without that environment builds vLLM from source and needs
the CUDA build toolchain. The lockfile records the fork source revision.

Prime RL enables compact extraction with the V2 runner. An explicit V1 runner
keeps it disabled; `VLLM_USE_COMPACT_LOGPROBS=0` disables it for comparisons.
Use `processed_logprobs` and request `logprobs=128` to record likely tokens.
Sampling can remain `top_k=-1`, `top_p=1.0`; enabling extraction does not request
logprobs or change these sampling parameters.
