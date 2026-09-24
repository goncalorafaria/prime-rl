#!/usr/bin/env bash
set -euo pipefail
cd /gscratch/ark/graf/prime-rl-shardcast-base
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-prime-sc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/gscratch/ark/graf/.local/bin/uv run --no-project --python /gscratch/ark/graf/miniconda3/envs/verl/bin/python -m pytest --noconftest -q -p no:cacheprovider -c /dev/null tests/unit/train/rl/test_score_centering.py
