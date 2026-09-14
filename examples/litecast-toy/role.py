"""Task supervisor for the single Rex LiteCast training experiment."""

import os
import signal
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import tomli_w
from literegistry.coop.endpoints import endpoint_healthy, publish, wait

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = "toy-" + os.environ["REXS_EXPERIMENT_ID"]
OUTPUT = ROOT / "outputs" / RUN
HEAD = "sqlite://" + str(OUTPUT / "head.sqlite3")
HOST = socket.getfqdn()
children = []


def start(*args):
    process = subprocess.Popen(list(args), start_new_session=True)
    children.append(process)
    return process


def python(*args):
    return start("uv", "run", "--no-project", "--python", sys.executable, "python", *args)


def free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def watch(records):
    while True:
        for process in children:
            if process.poll() is not None:
                raise RuntimeError(f"service exited: {process.args} status={process.returncode}")
        for name, uri in records.items():
            publish(HEAD, name, uri, publisher_id=RUN + "-" + name, ttl_seconds=30)
        time.sleep(2)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    role = sys.argv[1]
    os.environ.update(RUN_ID=RUN, ADVERTISE_HOST=HOST, TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled")
    if role == "head":
        redis_port, gateway_port = free_port(), free_port()
        registry = f"redis://{HOST}:{redis_port}/0"
        os.environ["REGISTRY_PATH"] = registry
        os.environ["REGISTRY"] = registry
        start(
            "/gscratch/ark/graf/redis-stable/src/redis-server",
            "--bind",
            "0.0.0.0",
            "--protected-mode",
            "no",
            "--port",
            str(redis_port),
            "--save",
            "",
            "--appendonly",
            "no",
        )
        deadline = time.monotonic() + 60
        while not endpoint_healthy(registry, "redis", 2):
            if time.monotonic() > deadline:
                raise TimeoutError("Redis startup failed")
            time.sleep(0.2)
        python(
            "-m",
            "uvicorn",
            "literegistry.gateway:create_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            str(gateway_port),
        )
        gateway = f"http://{HOST}:{gateway_port}"
        watch({"redis": registry, "gateway": gateway})
        return
    registry = wait(HEAD, "redis", timeout=3600, healthcheck="redis")
    gateway = wait(HEAD, "gateway", timeout=3600, healthcheck="http")
    os.environ.update(REGISTRY=registry, GATEWAY_URL=gateway)
    if role == "middle":
        port = free_port()
        python(
            "-m",
            "prime_rl.litecast.middle",
            "--registry",
            registry,
            "--run-id",
            RUN,
            "--advertise-host",
            HOST,
            "--port",
            str(port),
        )
        rank = os.environ.get("BEAKER_REPLICA_RANK", "0")
        watch({f"middle-{rank}": f"http://{HOST}:{port}"})
    elif role == "inference":
        backend, sidecar, shards = free_port(), free_port(), free_port()
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        start("uv", "run", "--no-sync", "inference", "@", str(HERE / "inference.toml"), "--server.port", str(backend))
        python(
            "-m",
            "prime_rl.litecast.worker",
            "--registry",
            registry,
            "--run-id",
            RUN,
            "--base-model",
            "Qwen/Qwen3.5-2B",
            "--advertise-host",
            HOST,
            "--backend-url",
            f"http://127.0.0.1:{backend}",
            "--port",
            str(sidecar),
            "--shard-port",
            str(shards),
            "--require-middle",
        )
        watch({})
    elif role == "trainer":
        for rank in range(2):
            wait(HEAD, f"middle-{rank}", timeout=3600, healthcheck="none")
        config = tomllib.loads((HERE / "train.toml").read_text())
        config["output_dir"] = str(OUTPUT / "training")
        client = config["orchestrator"]["model"]["client"]
        client["base_url"] = [gateway + "/v1"]
        client["litecast"].update(registry=registry, run_id=RUN, origin_host=HOST, origin_port=free_port())
        destination = OUTPUT / "train.toml"
        destination.write_text(tomli_w.dumps(config))
        result = start("uv", "run", "--no-sync", "rl", "@", str(destination)).wait()
        if result:
            raise SystemExit(result)
        print("TRAINING_SMOKE_COMPLETED " + str(OUTPUT), flush=True)
    else:
        raise ValueError(role)


if __name__ == "__main__":
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        main()
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
