#!/usr/bin/env bash
#SBATCH --job-name=litecast-runtime-build
#SBATCH --account=ckpt-ark
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/ark/graf/prime-rl-shardcast/outputs/runtime-cache/build-%j.log
set -Eeuo pipefail
cd /gscratch/ark/graf/prime-rl-shardcast
archive=${LITECAST_RUNTIME_ARCHIVE:-$PWD/outputs/runtime-cache/toy-runtime-v1.tar.zst}
if [[ ! -f $archive || ! -f $archive.sha256 ]]; then
    bash examples/litecast-toy/runtime-cache.sh build
fi
/gscratch/ark/graf/rexs/deploy/klone/apptainer-klone exec --cleanenv \
    --bind /gscratch/ark/graf:/gscratch/ark/graf \
    --env LITECAST_RUNTIME_ARCHIVE="$archive" \
    --env NETRC=/gscratch/ark/graf/.netrc \
    /gscratch/ark/graf/vllm-openai_latest-cu129-ubuntu2404.sif \
    bash examples/litecast-toy/runtime-cache.sh check /gscratch/ark/graf/prime-rl-shardcast/examples/litecast-toy/check-runtime.py

/gscratch/ark/graf/rexs/deploy/klone/apptainer-klone exec --cleanenv \
    --bind /gscratch/ark/graf:/gscratch/ark/graf \
    --env LITECAST_RUNTIME_ARCHIVE="$archive" \
    --env NETRC=/gscratch/ark/graf/.netrc \
    /gscratch/ark/graf/vllm-openai_latest-cu129-ubuntu2404.sif \
    bash examples/litecast-toy/runtime-cache.sh check /gscratch/ark/graf/prime-rl-shardcast/examples/litecast-toy/preflight.py
