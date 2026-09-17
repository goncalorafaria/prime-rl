"""Two live GPU replicas accept a second LoRA publisher through LiteRegistry."""

import asyncio
import json
import os
import runpy
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import httpx
import tomli_w
import uvicorn
from literegistry import RegistryClient, get_kvstore

from prime_rl.configs.shared import LitecastPoolConfig
from prime_rl.litecast.bootstrap import refresh_endpoint, wait
from prime_rl.litecast.distribution import Publisher
from prime_rl.litecast.gateway import CapacityConfig, create_app

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / ("late-publisher-" + os.environ["REXS_EXPERIMENT_ID"])
OUT.mkdir(parents=True, exist_ok=True)
HEAD = "sqlite://" + str(OUT / "head.sqlite3")
REGISTRY = "head+" + HEAD
HOST = socket.getfqdn()
A = "late-a-" + os.environ["REXS_EXPERIMENT_ID"]
B = "late-b-" + os.environ["REXS_EXPERIMENT_ID"]
MODEL_SOURCE = "/gscratch/ark/graf/data/batchrubrics/models/sloth2b-sft-step400"
FIXTURES = ROOT / "outputs/toy-a9f058f6c65e43a19ceefd21ff62d26a/training/run_default/broadcasts"
CHILDREN = []


def port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def atomic(path, value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def launch(command, name):
    with (OUT / (name + ".log")).open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append(child)
    return child


def py(*args):
    return ["uv", "run", "--no-project", "--python", sys.executable, "python", *args]


async def until(check, timeout=1200):
    async with asyncio.timeout(timeout):
        while True:
            for child in CHILDREN:
                if child.poll() is not None:
                    raise RuntimeError(f"child exited: {child.args}, status={child.returncode}")
            result = await check()
            if result:
                return result
            await asyncio.sleep(1)


async def worker():
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    await asyncio.to_thread(wait, HEAD, "redis", timeout=1800, healthcheck="redis")
    model = str(
        runpy.run_path(str(ROOT / "examples/litecast-rubrics/stage.py"))["stage"](MODEL_SOURCE, "sloth2b-sft-step400")
    )
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv"], text=True)
    (OUT / f"hardware-{rank}.csv").write_text(hardware)
    backend, sidecar, shard_a, shard_b = port(), port(), port(), port()
    config = tomllib.loads((ROOT / "examples/litecast-rubrics/inference.toml").read_text())
    config["model"].update(name=model, max_model_len=4096)
    config["server"].update(host="127.0.0.1", port=backend)
    config["vllm_extra"].update(max_num_batched_tokens=2048, max_num_seqs=4)
    cfg = OUT / f"inference-{rank}.toml"
    cfg.write_text(tomli_w.dumps(config))
    launch(
        ["uv", "run", "--no-sync", "inference", "@", str(cfg), "--data-parallel-rpc-port", str(port())],
        f"engine-{rank}",
    )
    async with httpx.AsyncClient(timeout=5) as http:

        async def healthy():
            try:
                return (await http.get(f"http://127.0.0.1:{backend}/health")).status_code == 200
            except httpx.TransportError:
                return False

        await until(healthy)
    tenants = OUT / f"tenants-{rank}.json"
    a = dict(run_id=A, base_model=model, backend_url=f"http://127.0.0.1:{backend}", shard_port=shard_a)
    atomic(tenants, [a])
    launch(
        py(
            "-m",
            "prime_rl.litecast.worker",
            "--registry",
            REGISTRY,
            "--tenants",
            str(tenants),
            "--advertise-host",
            HOST,
            "--port",
            str(sidecar),
            "--cache-dir",
            tempfile.mkdtemp(prefix="late-worker-"),
            "--max-inflight-requests",
            "1",
            "--request-timeout",
            "180",
            "--poll-seconds",
            ".5",
            "--require-middle",
        ),
        f"worker-{rank}",
    )
    atomic(
        OUT / f"worker-{rank}.json",
        dict(
            model=model,
            sidecar=f"http://{HOST}:{sidecar}",
            pid=CHILDREN[-1].pid,
            engine_pid=CHILDREN[-2].pid,
            job_id=os.environ.get("SLURM_JOB_ID"),
            node=HOST,
        ),
    )
    joined = False
    while not (OUT / "STOP").exists():
        if (OUT / "JOIN_B").exists() and not joined:
            atomic(tenants, [a, dict(a, run_id=B, shard_port=shard_b)])
            joined = True
        for child in CHILDREN:
            if child.poll() is not None:
                raise RuntimeError(f"worker child exited: {child.returncode}")
        await asyncio.sleep(0.5)


async def coordinator():
    redis_port = port()
    launch(
        [
            "/gscratch/ark/graf/redis-stable/src/redis-server",
            "--bind",
            "0.0.0.0",
            "--port",
            str(redis_port),
            "--protected-mode",
            "no",
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        "redis",
    )

    async def heartbeat():
        while True:
            await asyncio.to_thread(refresh_endpoint, HEAD, "redis", f"redis://{HOST}:{redis_port}/0", publisher_id="late-test", ttl_seconds=30)
            await asyncio.sleep(2)

    heartbeat_task = asyncio.create_task(heartbeat())
    registry = RegistryClient(get_kvstore(REGISTRY, raise_on_error=True), cache_ttl=0)
    gateway = uvicorn.Server(
        uvicorn.Config(
            create_app(
                registry,
                CapacityConfig(max_inflight_per_replica=1, max_queued_requests=16, request_timeout_seconds=180),
            ),
            host="127.0.0.1",
            port=port(),
            log_level="warning",
        )
    )
    gateway_task = asyncio.create_task(gateway.serve())
    publishers = []
    background = None
    finish_a = asyncio.Event()
    observations = []
    try:
        while not gateway.started:
            if heartbeat_task.done():
                await heartbeat_task
            if gateway_task.done():
                await gateway_task
            await asyncio.sleep(0.1)
        atomic(OUT / "publishers.json", [A])
        for i in range(2):
            launch(
                py(
                    "-m",
                    "prime_rl.litecast.middle",
                    "--registry",
                    REGISTRY,
                    "--publishers",
                    str(OUT / "publishers.json"),
                    "--advertise-host",
                    HOST,
                    "--port",
                    "0",
                    "--max-versions",
                    "2",
                ),
                f"middle-{i}",
            )

        async def first_worker():
            records = list(OUT.glob("worker-*.json"))
            return json.loads(records[0].read_text()) if records else None

        first = await until(first_worker, 2400)
        model = first["model"]

        def publisher(run):
            p = Publisher(
                LitecastPoolConfig(
                    registry=REGISTRY,
                    run_id=run,
                    origin_host=HOST,
                    origin_port=port(),
                    poll_seconds=0.5,
                    retain_versions=2,
                ),
                model,
            )
            publishers.append(p)
            return p

        pa = publisher(A)
        await pa.start()
        av = await pa.publish(FIXTURES / "step_2", 1)
        await pa.wait_ready(av.model_name, 180)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{gateway.config.port}", timeout=180) as http:

            async def generate(name):
                response = await http.post(
                    "/inference/v1/generate",
                    json={
                        "model": name,
                        "token_ids": [100] * 128,
                        "sampling_params": {
                            "max_tokens": 32,
                            "min_tokens": 32,
                            "ignore_eos": True,
                            "temperature": 0,
                            "logprobs": 1,
                        },
                    },
                )
                response.raise_for_status()
                result = response.json()
                assert result["usage"]["completion_tokens"] == 32, result
                return result

            first_response = await generate(av.model_name)
            print("A_SERVING_BEFORE_B " + json.dumps({"replica": first_response["litecast_replica_id"]}), flush=True)

            async def both_ready(name):
                records = (await registry.models(force=True)).get(name, [])
                return len({r["metadata"].get("replica_id") for r in records if r["metadata"].get("replica_id")}) == 2

            await until(lambda: both_ready(av.model_name), 2400)

            async def keep_a_running():
                while not finish_a.is_set():
                    result = await generate(av.model_name)
                    observations.append({"time": time.time(), "replica": result["litecast_replica_id"]})
                    await asyncio.sleep(0.1)

            background = asyncio.create_task(keep_a_running())
            before = time.time()
            pb = publisher(B)
            await pb.start()
            bv = await pb.publish(FIXTURES / "step_3", 1)
            atomic(OUT / "publishers.json", [A, B])
            (OUT / "JOIN_B").touch()
            await until(lambda: both_ready(bv.model_name), 240)
            joined_seconds = time.time() - before
            finish_a.set()
            await background

            async def exercise_both(name):
                results = await asyncio.gather(*(generate(name) for _ in range(8)))
                identities = {r["litecast_replica_id"] for r in results}
                assert len(identities) == 2, identities
                return results

            a_results = await exercise_both(av.model_name)
            b_results = await exercise_both(bv.model_name)
            assert a_results[0]["choices"][0]["logprobs"] != b_results[0]["choices"][0]["logprobs"], (
                "adapters had identical logprobs"
            )
            assert observations and observations[-1]["time"] >= before
            result = dict(
                status="passed",
                model=model,
                late_join_seconds=joined_seconds,
                A=av.model_name,
                B=bv.model_name,
                a_during_join=observations,
                a_replicas=sorted({r["litecast_replica_id"] for r in a_results}),
                b_replicas=sorted({r["litecast_replica_id"] for r in b_results}),
                workers=[json.loads(p.read_text()) for p in sorted(OUT.glob("worker-*.json"))],
                transfer_A=await pa.transfer_metrics(av),
                transfer_B=await pb.transfer_metrics(bv),
            )
            atomic(OUT / "result.json", result)
            print("LATE_PUBLISHER_PASSED " + json.dumps(result), flush=True)
    finally:
        finish_a.set()
        if background and not background.done():
            background.cancel()
            await asyncio.gather(background, return_exceptions=True)
        for p in publishers:
            await p.close()
        gateway.should_exit = True
        await gateway_task
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        (OUT / "STOP").touch()


async def main():
    try:
        await (worker() if len(sys.argv) > 1 and sys.argv[1] == "worker" else coordinator())
    finally:
        for child in CHILDREN:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in CHILDREN:
            try:
                await asyncio.to_thread(child.wait, 15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                await asyncio.to_thread(child.wait)


asyncio.run(main())
