#!/usr/bin/env bash
# Source before uv sync to reuse the binary extensions from the fork's base release.
case "$(uname -m)" in
    x86_64|aarch64) ;;
    *) echo "Unsupported vLLM wheel architecture: $(uname -m)" >&2; return 1 ;;
esac
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0-cp38-abi3-manylinux_2_28_$(uname -m).whl"
