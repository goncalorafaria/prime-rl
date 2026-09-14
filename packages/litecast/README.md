# Litecast

Litecast is a small Python package for distributing large files through a tree
of HTTP servers. An origin splits a file into shards, middle nodes cache and
re-serve those shards, and clients download, reassemble, and verify the file.

This document describes Litecast 0.3.2 as it is currently implemented.

## How it works

```text
                         ┌──────────┐
                         │  Origin  │
                         └────┬─────┘
                              │ HTTP GET
                 ┌────────────┴────────────┐
                 ▼                         ▼
          ┌─────────────┐           ┌─────────────┐
          │ Middle node │           │ Middle node │
          └──────┬──────┘           └──────┬──────┘
                 │ HTTP GET                 │ HTTP GET
            ┌────┴────┐                ┌────┴────┐
            ▼         ▼                ▼         ▼
          Client    Client           Client    Client
```

There are three node roles:

1. **Origin**: splits a source file into versioned shards and serves its data
   directory over HTTP.
2. **Middle node**: polls one or more upstream servers for versions, downloads
   their shards, and serves its local cache over HTTP.
3. **Client**: downloads a selected version from one or more servers,
   concatenates its shards in order, and verifies the result with BLAKE3.

Litecast has no upload REST endpoint, coordinator, or database. HTTP `GET` is
the default protocol and control plane. An optional UCXX data plane can move
complete versions directly between CPU-memory buffers over InfiniBand RDMA.

### Broadcast lifecycle

When the origin broadcasts a file:

1. It computes the BLAKE3 checksum of the complete file.
2. It chooses the next version directory, such as `v1` or `v2`.
3. It splits the file into numbered shards. The default shard size is
   50,000,000 bytes.
4. It writes the version, checksum, and shard count to `distribution.txt`.
5. It deletes the oldest versions when the configured retention limit is
   exceeded. The default limit is five versions.

A middle node polls an upstream `distribution.txt` every 30 seconds by default.
For each version it has not processed, it downloads all listed shards in
parallel. Its entire data directory is also served over HTTP, so another middle
node or client can use it as an upstream.

A client reads `distribution.txt`, downloads the selected version's shards in
parallel, sorts and concatenates them, and compares the resulting BLAKE3
checksum with the manifest. Temporary shards are removed after the operation.

### Replicated shard streaming

When every middle is given the same canonical `--middle-servers` list and its
own `--middle-id`, Litecast enables the RF=2 streaming path:

```text
origin → primary owner → replica owner
             └──────────────→ clients
```

Rendezvous hashing assigns each shard one primary and one replica. Only the
primary fetches that shard from the origin; the replica fetches it from the
primary. A shard becomes servable immediately after its individual BLAKE3
checksum passes, so replication and client downloads overlap and shard order
does not gate progress. Clients preallocate the final buffer, stripe requests
across all middles, retry the alternate owner, and hash newly contiguous
ranges without concatenating temporary shard objects.

Example membership:

```bash
MEMBERS=http://10.0.0.20:8001,http://10.0.0.21:8001

litecast-middle \
  --upstream http://10.0.0.10:8000 \
  --middle-id http://10.0.0.20:8001 \
  --middle-servers "$MEMBERS" \
  --replication-factor 2 \
  --no-disk-mirror
```

The second middle uses the same command with its own URL as `--middle-id`.
Clients pass the identical ordered membership to `--servers`. If membership or
the per-shard manifest is absent, nodes retain the legacy full-version path.

## Requirements and installation

- Python 3.8 or newer
- `wget`
- Network access between nodes on their configured HTTP ports

Install from this checkout:

```bash
python -m pip install -e .
```

The Python dependency `blake3` is installed automatically. The downloader
attempts to install `wget` with `apt` if it cannot find it, but this can fail on
non-Debian systems or without root access. Installing `wget` yourself is safer.

Installation creates three commands:

```text
litecast-origin
litecast-middle
litecast-client
```

## Quick start: origin, middle, and client on one machine

Servers create a small legacy `/data1.bin` probe automatically. Current clients
do not require that probe during startup.

Use three terminals from the repository root.

### 1. Start an origin and broadcast a file

```bash
mkdir -p ./origin_data

python examples/example_usage.py \
  --mode origin \
  --data-dir ./origin_data \
  --port 8000 \
  --file-path /absolute/path/to/file.bin
```

For a self-contained test, the example can generate a 100 MB source file:

```bash
mkdir -p ./origin_data

python examples/example_usage.py \
  --mode origin \
  --data-dir ./origin_data \
  --port 8000 \
  --file-path ./source.bin \
  --create-dummy \
  --dummy-size 100
```

The process prints the assigned version (normally `v1`) and remains running to
serve it.

> `litecast-origin` only starts an HTTP server. It does not accept a file path
> or broadcast a file. Use the example above or the Python API when publishing.

### 2. Start a middle node

```bash
mkdir -p ./middle_data

litecast-middle \
  --upstream http://127.0.0.1:8000 \
  --data-dir ./middle_data \
  --port 8001 \
  --check-interval 5
```

The middle node fetches the manifest and shards from the origin and then serves
its cache on port 8001.

### 3. List and download from a client

List available versions:

```bash
litecast-client \
  --servers http://127.0.0.1:8001,http://127.0.0.1:8000 \
  --list
```

Download and verify `v1`:

```bash
litecast-client \
  --servers http://127.0.0.1:8001,http://127.0.0.1:8000 \
  --version v1 \
  --output-file ./downloaded.bin
```

Server values must be complete base URLs, including `http://` and the port.

## Running nodes on different machines

Assume:

- origin: `10.0.0.10:8000`
- middle: `10.0.0.20:8001`
- client: another machine that can reach both addresses

On the origin, run the origin example:

```bash
mkdir -p /srv/litecast/origin

python examples/example_usage.py \
  --mode origin \
  --data-dir /srv/litecast/origin \
  --port 8000 \
  --file-path /srv/files/model.bin
```

On the middle machine:

```bash
mkdir -p /srv/litecast/middle

litecast-middle \
  --upstream http://10.0.0.10:8000 \
  --data-dir /srv/litecast/middle \
  --port 8001
```

On the client machine:

```bash
litecast-client \
  --servers http://10.0.0.20:8001,http://10.0.0.10:8000 \
  --version v1 \
  --output-file ./model.bin
```

Allow inbound TCP traffic on port 8000 at the origin and port 8001 at the
middle. Servers bind to `0.0.0.0`.

Multiple upstreams are comma-separated:

```bash
litecast-middle \
  --upstream http://10.0.0.10:8000,http://10.0.0.11:8000 \
  --data-dir ./middle_data \
  --port 8001
```

Middle nodes can be chained by pointing a downstream middle node at an upstream
middle node.

## UCXX RDMA and in-memory checkpoints

UCXX is optional. HTTP remains available when UCXX is not installed. UCXX must
be installed separately with a UCX build that includes verbs support; the base
Litecast package deliberately does not install or bundle the cluster's UCX
runtime.

One supported installation route is:

```bash
conda install -c conda-forge -c rapidsai ucxx
```

The resulting UCX build still needs `rc`/verbs support on the target cluster.

Verify the environment on every participating compute node:

```bash
python -c 'import ucxx; print(ucxx.get_ucx_version())'
ucx_info -d
ip -4 addr show ib0
```

The UCX device output must include an InfiniBand transport. Litecast initializes
UCX with `TLS=rc` and refuses to advertise the endpoint unless `rc` appears in
the resulting configuration.

The origin CLI only starts listeners and cannot receive a later publish call
from another process. Start the listener and publish a bytes-like checkpoint in
the same Python process:

```python
import time
import litecast

litecast.initialize(
    data_dir="/dev/shm/litecast-origin",
    port=8000,
    transport="ucxx",
    ucxx_port=9000,
    ucxx_interface="ib0",
    ram_cache_bytes=200_000_000_000,
    disk_mirror=False,
)

version = litecast.broadcast_buffer(checkpoint_bytes)

try:
    while True:
        time.sleep(1)
finally:
    litecast.shutdown()
```

Start an RDMA-capable middle node:

```bash
litecast-middle \
  --upstream http://ORIGIN_IB0_IP:8000 \
  --data-dir /dev/shm/litecast-middle \
  --port 8001 \
  --transport ucxx \
  --ucxx-port 9001 \
  --ucxx-interface ib0 \
  --ram-cache-bytes 200000000000 \
  --no-disk-mirror
```

Download directly into CPU memory:

```python
from litecast import ClientNode

client = ClientNode(
    ["http://MIDDLE_IB0_IP:8001", "http://ORIGIN_IB0_IP:8000"],
    transport="ucxx",
    max_version_bytes=200_000_000_000,
)
checkpoint = client.download_version_buffer("v1")
```

Transport modes behave as follows:

- `http` never attempts UCXX.
- `auto` uses an advertised, RDMA-verified UCXX endpoint and falls back to
  in-memory HTTP after a bounded connection failure.
- `ucxx` requires an advertised, RDMA-verified endpoint and fails instead of
  silently using TCP/HTTP.

HTTP remains the discovery control plane. Each server publishes
`/endpoints.json`; bulk checkpoint chunks move through UCXX into one
preallocated destination buffer. The complete BLAKE3 digest is verified before
the buffer is returned or a middle node advertises it.

Use `LITECAST_DISK_MIRROR=0` or `--no-disk-mirror` to avoid shard files.
Small manifests and endpoint metadata are still stored in the node data
directory. Writable input buffers are copied once into immutable storage;
immutable `bytes` inputs are retained without a copy.

`ram_cache_bytes` bounds buffers currently indexed by the store. An active
HTTP/UCXX response holds a memory view until that transfer ends, so eviction
can temporarily leave a retired shard resident. Size production nodes for the
configured cache plus concurrent in-flight shards.

## Slurm checkpoint topology benchmark

`benchmarks/slurm_checkpoint_benchmark.py` benchmarks the actual Qwen 3.5 4B
checkpoint rather than generated data. It reads these two files:

```text
/gscratch/ark/graf/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/model.safetensors-00001-of-00002.safetensors
/gscratch/ark/graf/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/model.safetensors-00002-of-00002.safetensors
```

The complete transferred checkpoint is 9,319,828,096 bytes (8.68 GiB).
Tensor payload metadata in `model.safetensors.index.json` reports
9,319,737,856 bytes; the small difference is the safetensor file headers.

The batch file requests checkpoint CPU nodes from account `ark-ckpt`, partition
`ckpt-all`, and QoS `ckpt`. Its default topology is one origin, two middle
nodes, and eight clients: 11 nodes and tasks in total. `MIDDLE_COUNT` and
`CLIENT_COUNT` parameterize the topology; the allocation must always contain
`1 + MIDDLE_COUNT + CLIENT_COUNT` nodes with one task per node. Each node
receives 16 CPUs and 64 GiB of RAM. Checkpoint bytes and served shards stay in
`/dev/shm` or Litecast's RAM store; only control files, event records, and the
final summary use the shared filesystem.

Build the small HTTP image on a host that permits image builds:

```bash
apptainer build --fakeroot \
  benchmarks/litecast-http.sif \
  benchmarks/litecast-http.def
```

This image starts from `python:3.12-slim-bookworm` and installs only BLAKE3.
Build the larger RDMA image separately:

```bash
apptainer build --fakeroot \
  benchmarks/litecast-rdma.sif \
  benchmarks/litecast-rdma.def
```

The RDMA image uses the RAPIDS base and installs UCXX at image-build time.
UCXX's PyPI wheels cannot replace it for this benchmark because their bundled
UCX libraries are built without InfiniBand or RoCE support. Compute nodes do
not activate or access a host Conda environment.

Before submitting RDMA, verify its image and the host InfiniBand device:

```bash
apptainer test benchmarks/litecast-rdma.sif
apptainer exec benchmarks/litecast-rdma.sif ucx_info -d
ip -4 addr show ib0
```

Run the IP-over-InfiniBand HTTP baseline:

```bash
sbatch --export=NIL \
  benchmarks/run_slurm_checkpoint_benchmark.sbatch \
  http 2 8 "$PWD/benchmarks/litecast-http.sif" "$(command -v apptainer)"
```

Run the RDMA comparison with the same image:

```bash
sbatch --export=NIL \
  benchmarks/run_slurm_checkpoint_benchmark.sbatch \
  ucxx 2 8 "$PWD/benchmarks/litecast-rdma.sif" "$(command -v apptainer)"
```

Run the smallest RF=2 streaming smoke benchmark:

```bash
bash benchmarks/submit_rf2_smoke.sh \
  ucxx 2 8 "$PWD/benchmarks/litecast-rdma.sif" "$(command -v apptainer)"
```

The seventh positional argument to the batch script enables RF=2 directly:

```bash
sbatch --export=NIL \
  benchmarks/run_slurm_checkpoint_benchmark.sbatch \
  ucxx 2 8 "$PWD/benchmarks/litecast-rdma.sif" \
  "$(command -v apptainer)" 0 1
```

RF=2 summaries add origin shard bytes, peer-replication bytes, first-client-
shard latency, and the usual publication-to-client completion events. Clients
start after middle endpoints exist rather than waiting for complete middle
replicas.

`--export=NIL` avoids Klone's login-environment retrieval path. Transport,
middle count, client count, image path, and Apptainer executable are positional
batch-script arguments so the benchmark does not depend on exported shell
state.

To compare 2, 4, 8, 16, and 32 middle nodes with one origin and eight clients,
first inspect the generated submissions without creating jobs:

```bash
DRY_RUN=1 \
TRANSPORT=http \
APPTAINER_IMAGE=$PWD/benchmarks/litecast-http.sif \
bash benchmarks/submit_middle_sweep.sh
```

Submit the HTTP sweep by removing `DRY_RUN=1`. Set `TRANSPORT=ucxx` for the
equivalent RDMA sweep. The submitter requests 11, 13, 17, 25, and 41 nodes,
respectively, and chains jobs with `afterany` so only one topology runs at a
time. Set `REPEATS` to collect repeated measurements. Set `CHAIN_JOBS=0` only
when intentional concurrent runs and their network/filesystem interference are
acceptable.

Set `REPLICATED_STREAMING=1` to run the same sweep with RF=2. The
`submit_true_benchmark.sh` helper submits legacy HTTP, legacy UCXX, and RF=2
UCXX for each middle count and atomically records every job, mode, and repeat
in `benchmarks/true_benchmark_jobs.json`. The analyzer then reports publication
latency, fan-out throughput speedup, origin bytes, and peer-replication bytes.

Each client is assigned one middle node round-robin, spreading eight clients
across the available middle nodes without random HTTP endpoint selection
obscuring the topology. With more than eight middle nodes, additional replicas
measure origin fan-out and propagation cost but cannot increase the number of
simultaneously serving middles beyond the eight clients.

The batch job binds the repository, checkpoint snapshot, and host `/dev/shm`
into the container. Apptainer's normal `/dev` and `/sys` mounts expose the
host's InfiniBand devices and topology to the container. Override
`CONTAINER_PYTHON` only if a custom image does not expose its interpreter as
`python`.

The benchmark measures:

- sequential GPFS-to-RAM reads of the same two safetensors on each client;
- origin publication after a GPFS-to-RAM read;
- origin-to-middle propagation;
- concurrent middle-to-client HTTP/IPoIB or UCXX transfers;
- end-to-end time from publication to each middle and client receiving a
  checksum-verified checkpoint.

Every process writes its own JSONL file under
`benchmark-results/<job-id>/<transport>-m<middles>-c<clients>/events/`.
Records include wall-clock and monotonic nanosecond timestamps, role, rank,
hostname, transfer mode, checkpoint size, checksum, and measured throughput.
The origin merges these records into `summary.json`.

The GPFS comparison is a warm-cluster measurement, not a guaranteed cold-cache
disk measurement: all nodes share the filesystem and the origin reads the
checkpoint first. Compare both elapsed seconds and GiB/s, and use separate
HTTP and UCXX jobs so they have the same topology and memory pressure.

## HTTP file API

Both origin and middle nodes expose the same static file layout.

### `GET /distribution.txt`

Returns the available versions:

```text
v1: 4d1d960b53356285f45ea2e27c89a1a11d10a9601d3ba2a90851f9f227dd9295|3
v2: 8805f050e856f2b88c9f3898a4f506c44d617b569f2858db20285f09db6c107d|2
```

Each line has this format:

```text
<version>: <blake3 checksum>|<number of shards>
```

### `GET /<version>/shard_<number>.bin`

Returns one binary shard. Shard numbers are one-based and padded to five
digits:

```text
GET /v1/shard_00001.bin
GET /v1/shard_00002.bin
GET /v1/shard_00003.bin
```

Concatenating the shards in filename order reproduces the source file.
An in-progress middle returns HTTP `425 Shard not available` for a known shard
that has not arrived yet; clients retry its alternate RF=2 owner.

### `GET /<version>/manifest.json`

Returns the bounded streaming manifest: total and shard sizes, complete BLAKE3,
and one BLAKE3 checksum per shard. This sidecar enables independent,
out-of-order verification while preserving `distribution.txt` compatibility.

### `GET /data1.bin`

Returns a legacy compatibility probe. Servers generate it automatically.

### `GET /endpoints.json`

Returns schema-1 legacy or schema-2 streaming discovery metadata. An
RDMA-capable streaming middle advertises its UCXX address, stable middle ID,
and `get_shard`/`partial_shards` capabilities. Clients treat this file as
optional and only use indexed UCXX requests when `get_shard` is advertised.

All responses include permissive CORS headers. There is no authentication,
encryption, authorization, upload endpoint, or version deletion endpoint. Use a
trusted network or put an authenticated TLS reverse proxy in front of nodes.

## Python API

### Origin convenience API

```python
import time
import litecast

litecast.initialize(
    data_dir="./origin_data",
    port=8000,
    max_distribution_folders=5,
)

version = litecast.broadcast(
    "/path/to/file.bin",
    shard_size=134_217_728,
)
print(f"Published {version}")

try:
    while True:
        time.sleep(1)
finally:
    litecast.shutdown()
```

- `litecast.initialize(data_dir, port, max_distribution_folders)` creates one
  process-global `OriginServer` and starts its HTTP thread.
- `litecast.broadcast(file_path, shard_size)` creates and publishes the next
  version and returns its name.
- `litecast.broadcast_buffer(buffer, shard_size, disk_mirror)` publishes a
  bytes-like object from CPU memory.
- `litecast.shutdown()` signals the origin HTTP server to stop.

### Class API

```python
from litecast import OriginServer, MiddleNode, ClientNode

origin = OriginServer("./origin_data", port=8000)
version = origin.broadcast("/path/to/file.bin")

middle = MiddleNode(
    upstream_servers=["http://origin.example:8000"],
    data_dir="./middle_data",
    port=8001,
    check_interval=30,
)

client = ClientNode(
    servers=[
        "http://middle.example:8001",
        "http://origin.example:8000",
    ],
    output_dir="./downloads",
)

versions = client.list_available_versions()
output_path = client.download_version("v1", "./downloaded.bin")

middle.shutdown()
origin.shutdown()
```

`list_available_versions()` returns a dictionary whose values are manifest info
strings such as `<checksum>|<shard count>`, despite the method's type
description referring only to checksums.

## Command reference

### Origin server

```text
litecast-origin
  [--data-dir PATH] [--port PORT]
  [--transport auto|http|ucxx]
  [--ucxx-port PORT] [--ucxx-interface NAME]
  [--ucxx-connect-timeout SECONDS]
  [--ram-cache-bytes BYTES] [--disk-mirror | --no-disk-mirror]
  [--log-level LEVEL]
```

This serves an existing data directory. It does not broadcast new files.

### Middle node

```text
litecast-middle
  [--upstream URL[,URL...]]
  [--data-dir PATH]
  [--port PORT]
  [--check-interval SECONDS]
  [--transport auto|http|ucxx]
  [--ucxx-port PORT] [--ucxx-interface NAME]
  [--ucxx-connect-timeout SECONDS]
  [--ram-cache-bytes BYTES] [--disk-mirror | --no-disk-mirror]
  [--log-level LEVEL]
```

If `--upstream` is omitted, URLs are read from `IP_ADDR_LIST`.

### Client

```text
litecast-client
  [--servers URL[,URL...]]
  [--output-dir PATH]
  (--list | --version VERSION)
  [--output-file PATH]
  [--transport auto|http|ucxx]
  [--max-version-bytes BYTES]
  [--connect-timeout SECONDS]
  [--log-level LEVEL]
```

If `--servers` is omitted, URLs are read from `IP_ADDR_LIST`.

### `IP_ADDR_LIST`

The middle and client commands accept a space-separated environment variable:

```bash
export IP_ADDR_LIST="http://10.0.0.10:8000 http://10.0.0.11:8000"
litecast-middle --data-dir ./middle_data --port 8001
```

For clients:

```bash
export IP_ADDR_LIST="http://10.0.0.20:8001 http://10.0.0.10:8000"
litecast-client --list
```

## Configuration

Configuration is read lazily from environment variables in
`litecast/envs.py`:

- `LITECAST_SHARD_SIZE` — shard size in bytes; default `134217728` (128 MiB).
- `LITECAST_MAX_DISTRIBUTION_FOLDERS` — versions retained by an origin;
  default `5`.
- `LITECAST_HTTP_PORT` — default server port; default `8000`.
- `LITECAST_RETRY_ATTEMPTS` — attempts per download operation; default `5`.
- `LITECAST_FAST_RETRY_ATTEMPTS` — retries using the short interval; default
  `3`.
- `LITECAST_FAST_RETRY_INTERVAL` — short retry delay in seconds; default `2`.
- `LITECAST_SLOW_RETRY_INTERVAL` — long retry delay in seconds; default `15`.
- `LITECAST_LOG_LEVEL` — default logging level; default `INFO`.
- `LITECAST_DISTRIBUTION_FILE` — manifest filename; default
  `distribution.txt`.
- `LITECAST_HTTP_TIMEOUT` — `wget` timeout in seconds; default `30`.
- `LITECAST_MAX_CONCURRENT_DOWNLOADS` — shard download worker count; default
  `10`.
- `LITECAST_VERSION_PREFIX` — version directory prefix; default `v`.
- `LITECAST_TRANSPORT` — `auto`, `http`, or `ucxx`; default `auto`.
- `LITECAST_UCXX_PORT` — UCXX listener port; `0` disables it by default.
- `LITECAST_UCXX_INTERFACE` — interface used for UCXX address discovery;
  default `ib0`.
- `LITECAST_UCXX_CONNECT_TIMEOUT` — endpoint connection timeout in seconds;
  default `30`.
- `LITECAST_RAM_CACHE_BYTES` — maximum bytes retained by a node's checkpoint
  store; default `1073741824`.
- `LITECAST_DISK_MIRROR` — whether memory publications also write shard files;
  default `true` for compatibility.
- `LITECAST_ENDPOINTS_FILE` — transport sidecar filename; default
  `endpoints.json`.

Set values before starting a process:

```bash
export LITECAST_SHARD_SIZE=100000000
export LITECAST_MAX_CONCURRENT_DOWNLOADS=20
export LITECAST_LOG_LEVEL=DEBUG
```

Changing `LITECAST_DISTRIBUTION_FILE` or `LITECAST_VERSION_PREFIX` must be
coordinated across every node.

## Data layout

An origin or populated middle data directory looks like:

```text
data/
├── data1.bin
├── distribution.txt
├── endpoints.json
├── v1/
│   ├── shard_00001.bin
│   ├── shard_00002.bin
│   └── shard_00003.bin
└── v2/
    ├── shard_00001.bin
    └── shard_00002.bin
```

Versions identify broadcast order, not source filenames. The original filename
and other metadata are not stored in the protocol, so clients must choose their
own output filename.

## Download selection and retries

Downloaders initialize servers with neutral performance metrics. A failed
server is penalized after an actual request rather than during startup.

For later downloads, Litecast tracks an exponentially weighted bandwidth and
success rate for each server. A server is sampled with weight:

```text
success rate × measured bandwidth
```

Each shard is submitted to a thread pool, so different shards may come from
different origin or middle servers. Failed requests are retried using the
configured fast and slow intervals.

## Troubleshooting

### UCXX endpoint is not advertised

Check `/endpoints.json`, `ip -4 addr show ib0`, and `ucx_info -d`. A node only
advertises UCXX after initialization succeeds with the UCX `rc` transport.

### Invalid or unreachable server URL

Always provide the URL scheme and port:

```text
http://192.168.1.100:8000
```

Do not pass only `192.168.1.100`.

### Client reports no versions

Check the manifest directly:

```bash
curl http://SERVER:PORT/distribution.txt
```

An origin started only with `litecast-origin` has an empty manifest until a
file is broadcast through the Python API.

### Middle node has not received a new version

Wait for its polling interval or restart it with a shorter
`--check-interval`. Confirm the origin's manifest and shards are reachable.

### Checksum verification fails

At least one shard is missing, stale, or corrupt. Remove the partial output file
and retry. Also verify that all configured servers expose identical content for
the same version name.

### `wget` installation fails

Install `wget` with the host operating system's package manager and rerun the
command.

## Current limitations

Litecast 0.3.2 is an alpha implementation with several operational
limitations:

- The HTTP server is based on `HTTPServer` and handles one request at a time.
- There is no authentication, TLS, access control, or path-specific API.
- The origin CLI cannot publish files.
- UCXX is an external optional runtime and must be installed consistently on
  participating compute nodes.
- RDMA currently targets CPU RAM, not GPU VRAM or GPUDirect RDMA.
- HTTP is still required for discovery and fallback.
- A writable publish buffer is copied into immutable storage. Middle nodes also
  briefly hold the received mutable buffer while creating their immutable copy.
- RAM capacity eviction can make older versions unavailable from a memory-only
  node even while an upstream origin still retains them.
- Version names are local counters. Independent origins can both create `v1`
  with different content, so mixing unrelated origins is unsafe.
- Hardware UCXX behavior depends on the cluster's UCX build; unit tests use a
  mocked UCXX endpoint and do not replace a two-node deployment test.
- There are currently no containers, service files, or orchestration manifests
  in this repository.

These limitations matter for production deployment. Run nodes on trusted
networks, monitor their logs and manifests, and validate a deployment under its
expected concurrency before relying on it for critical transfers.

## Selective transfers into reusable CPU buffers

Inspired by [Miles P2P weight transfer](https://miles.radixark.com/docs/advanced/p2p-weight-transfer),
consumers can fetch just their assigned shards directly into application-owned
buffers. This avoids allocating a complete checkpoint on each consumer and lets
applications reuse the same destination memory across updates.

```python
# This consumer needs zero-based shards 1 and 3; sizes come from the publisher's
# checkpoint layout (including the actual length of the final shard).
targets = {1: bytearray(shard_size), 3: bytearray(final_shard_size)}
manifest = client.download_shards_into("v1", targets)
# Only after successful return may the application activate these weights.
# Reuse targets on the next update if the shard sizes and layout are unchanged.
```

The method returns `ShardManifest` after every selected shard has passed its
BLAKE3 check and all transfer workers have finished. It checks the manifest
against the published distribution and validates the Merkle root when present.
Unrequested payloads are never fetched; this verifies selected shards, not the
contents of the entire checkpoint. Empty selections are allowed.

Targets may be bytearrays or writable, C-contiguous CPU buffer views, including
slices of a shared allocation. Each must have exactly its shard's byte length;
overlapping targets are rejected before payload writes. `max_version_bytes`
limits the sum of selected bytes for this API. Failures raise; buffers may then
contain partial data. Keep targets alive and exclusively owned during the call.
For live inference, receive into inactive buffers and switch only after success;
this API does not provide an atomic model swap or a cross-rank barrier.

Transfers use existing UCXX batches, persistent lanes, owner selection, and
replica retries. `auto` permits HTTP fallback; `ucxx` remains strict. Both direct
origin manifests and replicated streaming manifests are supported. Selective
operations emit `shards_download_started`, `shards_download_failed`, and
`shards_download_completed` events, separate from full-checkpoint completion.

Applications supply the model-to-shard mapping. This API operates on serialized
CPU byte shards; it does not implement tensor resharding, GPU memory registration,
one-sided remote writes, or Miles/SGLang integration. No RDMA performance gain is
claimed without a cluster benchmark.
