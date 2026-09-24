#!/usr/bin/env bash
set -euo pipefail
cd /gscratch/ark/graf/prime-rl-shardcast-base
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-prime-sc
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SC_TEST_DEPS=$(mktemp -d /tmp/prime-sc-deps.XXXXXX)
trap 'rm -rf "$SC_TEST_DEPS"' EXIT
/gscratch/ark/graf/.local/bin/uv pip install --quiet --python /gscratch/ark/graf/rleval/.venv/bin/python --target "$SC_TEST_DEPS" beartype==0.22.9 jaxtyping==0.3.10
export PYTHONPATH="$PWD/src:$PWD/packages/prime-rl-configs/src:$SC_TEST_DEPS:/gscratch/ark/graf/.local/share/venvs/klone-work/lib/python3.12/site-packages"
/gscratch/ark/graf/.local/bin/uv run --no-project --python /gscratch/ark/graf/rleval/.venv/bin/python python - <<'RUNNER'
import contextlib
import io
import sys
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import pytest
        result = pytest.main(["--noconftest", "-q", "--disable-warnings", "-p", "no:cacheprovider", "-c", "/dev/null", "tests/unit/train/rl/test_loss.py", "-k", "optional_sc or group_token_mean"])
    lines = buf.getvalue().splitlines()
    errors = [line for line in lines if line.startswith(("E ", "FAILED", "ERROR"))]
    print(" | ".join(errors or lines[-1:])[:260], flush=True)
    sys.exit(result)
except Exception as exc:
    print(type(exc).__name__ + ": " + str(exc)[:220], flush=True)
    raise SystemExit(1)
RUNNER
