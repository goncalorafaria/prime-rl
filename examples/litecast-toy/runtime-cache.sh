#!/usr/bin/env bash
# Build once on shared storage; stage an immutable runtime on each compute host.
set -Eeuo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
archive=${LITECAST_RUNTIME_ARCHIVE:-$repo/outputs/runtime-cache/toy-runtime-v1.tar.zst}

if [[ ${1:-} == build ]]; then
    [[ -n ${SLURM_JOB_ID:-} ]] || {
        echo "Build on Slurm: sbatch $here/build-runtime.sbatch" >&2
        exit 1
    }
    runtime=${2:-$repo/outputs/toy-runtime-host}
    mkdir -p "$(dirname "$archive")"
    exec 9>"$archive.lock"
    flock 9
    if [[ -e $archive || -e $archive.sha256 ]]; then
        echo 'Archive already exists; use a new LITECAST_RUNTIME_ARCHIVE path to rebuild.' >&2
        exit 1
    fi
    cp -- "$(command -v zstd)" "$archive.zstd"
    base_python=$(dirname "$(dirname "$(readlink -f "$runtime/bin/python")")")
    build_dir=$(mktemp -d /tmp/litecast-build.XXXXXXXX)
    trap 'rm -f "$archive.partial" "$archive.sha256.partial"; rm -rf -- "$build_dir"' EXIT
    uv run --no-project --python "$runtime/bin/python" python "$here/prepare-runtime.py" "$runtime" "$base_python" "$build_dir"
    runtime=$build_dir/runtime
    base_python=$build_dir/python
    echo "RUNTIME_ARCHIVE_BUILDING source=$runtime path=$archive"
    tar --blocking-factor=2048 -I 'zstd -T4 -1' -cf "$archive.partial" \
        --transform='flags=rh;s,^\./,runtime/,' -C "$runtime" . \
        -C "$(dirname "$base_python")" \
        --transform="flags=rh;s,^$(basename "$base_python"),python," "$(basename "$base_python")"
    sha256sum "$archive.partial" | cut -d ' ' -f 1 > "$archive.sha256.partial"
    mv "$archive.partial" "$archive"
    mv "$archive.sha256.partial" "$archive.sha256"
    echo "RUNTIME_ARCHIVE_BUILT path=$archive"
    exit
fi

[[ -f $archive && -f $archive.sha256 ]] || {
    echo "Build the runtime cache first: sbatch $here/build-runtime.sbatch" >&2
    exit 1
}
read -r digest < "$archive.sha256"
[[ $digest =~ ^[a-f0-9]{64}$ ]] || exit 1
cache=${LITECAST_CACHE_ROOT:-/tmp/litecast-runtime-graf}
mkdir -p -m 700 "$cache"
started=$SECONDS
echo "RUNTIME_CACHE_WAIT digest=$digest"
exec 9>"$cache/$digest.lock"
flock 9
local_root=$cache/$digest
if [[ ! -f $local_root/ready ]]; then
    echo "RUNTIME_CACHE_STAGING archive=$archive"
    staging=$(mktemp -d "$cache/.stage.XXXXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    actual=$(sha256sum "$archive" | cut -d ' ' -f 1)
    [[ $actual == "$digest" ]] || { echo 'Runtime archive checksum mismatch' >&2; exit 1; }
    decoder=$(command -v zstd || true)
    decoder=${decoder:-$archive.zstd}
    "$decoder" -qdc "$archive" | tar --ignore-zeros --no-same-owner -xf - -C "$staging"
    # Console entrypoints contain absolute shebangs; uv must use the local copy too.
    for script in "$staging/runtime/bin/"*; do
        [[ -f $script && ! -L $script ]] || continue
        IFS= read -r -n 256 first < "$script" || true
        if [[ $first == '#!'*"/bin/python"* ]]; then
            sed -i "1c\\#!$local_root/runtime/bin/python" "$script"
        fi
    done
    ln -sfn "$local_root/python/bin/python3.12" "$staging/runtime/bin/python"
    sed -i "s|^home = .*|home = $local_root/python/bin|" "$staging/runtime/pyvenv.cfg"
    touch "$staging/ready"
    mv "$staging" "$local_root"
    trap - EXIT
    echo "RUNTIME_CACHE_MISS seconds=$((SECONDS-started)) path=$local_root"
else
    echo "RUNTIME_CACHE_HIT seconds=$((SECONDS-started)) path=$local_root"
fi
flock -u 9
exec 9>&-
export UV_PROJECT_ENVIRONMENT=$local_root/runtime
export PATH=$local_root/runtime/bin:$PATH
if [[ ${1:-} == check ]]; then
    shift
    exec uv run --no-project --python "$local_root/runtime/bin/python" python "$@"
fi
exec uv run --no-project --python "$local_root/runtime/bin/python" python "$here/role.py" "$@"
