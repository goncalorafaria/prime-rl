#!/usr/bin/env bash
set -euo pipefail
: "${REGISTRY:?Set the shared LiteRegistry URI}"
: "${RUN_ID:?Set the same run ID as the trainer}"
: "${ADVERTISE_HOST:?Set a routable worker hostname or IP}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
WORKER_PORT="${WORKER_PORT:-8100}"
SHARD_PORT="${SHARD_PORT:-8101}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; wait || true; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
uv run --no-sync inference @ examples/litecast-search/inference-l40.toml \
  --model.name "$BASE_MODEL" --server.port "$BACKEND_PORT" &
pids+=("$!")
uv run --no-sync python -m prime_rl.litecast.worker --registry "$REGISTRY" --run-id "$RUN_ID" \
  --base-model "$BASE_MODEL" --advertise-host "$ADVERTISE_HOST" \
  --backend-url "http://127.0.0.1:$BACKEND_PORT" --port "$WORKER_PORT" --shard-port "$SHARD_PORT" &
pids+=("$!")
# Exit on either process failure so Rex can replace the whole worker.
wait -n "${pids[@]}"
exit 1
