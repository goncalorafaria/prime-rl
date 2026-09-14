#!/usr/bin/env bash
set -euo pipefail
: "${REGISTRY:?Set the shared LiteRegistry URI}"
: "${GATEWAY_URL:?Set the LiteRegistry/PrimeBeaker gateway URL without /v1}"
: "${RUN_ID:?Set a unique experiment ID}"
: "${ADVERTISE_HOST:?Set a routable training-node hostname or IP}"
exec uv run --no-sync python -m prime_rl.litecast.launch "$@"
