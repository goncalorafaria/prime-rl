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


def validate_lora_config():
    training = tomllib.loads((HERE / "train.toml").read_text())
    inference = tomllib.loads((HERE / "inference.toml").read_text())
    lora = training["trainer"]["model"]["lora"]
    if not inference.get("enable_lora"):
        raise ValueError("Separate inference must set enable_lora=true")
    if inference.get("max_lora_rank", 0) < lora["rank"]:
        raise ValueError("Separate inference max_lora_rank must cover the trainer LoRA rank")
    if inference["model"]["name"] != training["model"]["name"]:
        raise ValueError("Trainer and separate inference must use the same base model")
    targets = inference.get("lora_target_modules")
    if targets is not None and not set(lora["target_modules"]).issubset(targets):
        raise ValueError("Separate inference must support every trainer LoRA target module")
    print(
        f"LORA_CONFIG_VALIDATED separate_inference=true enable_lora=true "
        f"trainer_rank={lora['rank']} max_lora_rank={inference['max_lora_rank']}",
        flush=True,
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    role = sys.argv[1]
    if role in ("trainer", "inference"):
        validate_lora_config()
    os.environ.update(RUN_ID=RUN, ADVERTISE_HOST=HOST, TOKENIZERS_PARALLELISM="false")
    if role == "head":
        redis_port, gateway_port = free_port(), free_port()
        registry = f"redis://{HOST}:{redis_port}/0"
        os.environ["REGISTRY_PATH"] = registry
        os.environ["LITECAST_CAPACITY_CONFIG"] = str(HERE / "capacity.toml")
        os.environ["REGISTRY"] = registry
        redis_directory = OUTPUT / "redis"
        redis_directory.mkdir(exist_ok=True)
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
            "yes",
            "--appendfsync",
            "everysec",
            "--dir",
            str(redis_directory),
        )
        deadline = time.monotonic() + 60
        while not endpoint_healthy(registry, "redis", 2):
            if time.monotonic() > deadline:
                raise TimeoutError("Redis startup failed")
            time.sleep(0.2)
        python(
            "-m",
            "uvicorn",
            "prime_rl.litecast.gateway:create_app"
            if os.getenv("LITECAST_CAPACITY_ENABLED") == "1"
            else "literegistry.gateway:create_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            str(gateway_port),
        )
        gateway = f"http://{HOST}:{gateway_port}"
        watch({"redis-backend": registry, "gateway-backend": gateway})
        return
    if role == "trainer":
        python(
            "-m",
            "prime_rl.litecast.relay",
            "--bootstrap",
            HEAD,
            "--run-id",
            RUN,
            "--advertise-host",
            HOST,
            "--redis-port",
            str(free_port()),
            "--gateway-port",
            str(free_port()),
        )
    registry = wait(HEAD, "redis", timeout=3600, healthcheck="redis")
    gateway = wait(HEAD, "gateway", timeout=3600, healthcheck="http")
    os.environ.update(REGISTRY=registry, GATEWAY_URL=gateway)
    if role in ("trainer", "inference"):
        from huggingface_hub import snapshot_download

        revision = "15852e8c16360a2fea060d615a32b45270f8a8fc"
        snapshot = snapshot_download("Qwen/Qwen3.5-2B", revision=revision)
        print(f"MODEL_STAGED revision={revision} path={snapshot}", flush=True)
        if role == "trainer":
            from datasets import load_dataset

            load_dataset("PrimeIntellect/Reverse-Text-RL", split="train")
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
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
        inference_config = tomllib.loads((HERE / "inference.toml").read_text())
        inference_config["model"]["name"] = str(snapshot)
        inference_path = OUTPUT / f"inference-{os.environ.get('BEAKER_REPLICA_RANK', '0')}.toml"
        inference_path.write_text(tomli_w.dumps(inference_config))
        start(
            "uv",
            "run",
            "--no-sync",
            "inference",
            "@",
            str(inference_path),
            "--server.port",
            str(backend),
            "--data-parallel-rpc-port",
            str(free_port()),
        )
        python(
            "-m",
            "prime_rl.litecast.worker",
            "--registry",
            registry,
            "--run-id",
            RUN,
            "--base-model",
            str(snapshot),
            "--advertise-host",
            HOST,
            "--backend-url",
            f"http://127.0.0.1:{backend}",
            "--port",
            str(sidecar),
            "--shard-port",
            str(shards),
            "--require-middle",
            "--max-inflight-requests",
            str(tomllib.loads((HERE / "capacity.toml").read_text())["max_inflight_per_replica"])
            if os.getenv("LITECAST_CAPACITY_ENABLED") == "1"
            else "4",
        )
        watch({})
    elif role == "trainer":
        for rank in range(2):
            wait(HEAD, f"middle-{rank}", timeout=3600, healthcheck="none")
        config = tomllib.loads((HERE / "train.toml").read_text())
        config["model"]["name"] = str(snapshot)
        config["output_dir"] = str(OUTPUT / "training")
        config["wandb"]["name"] = RUN
        os.environ["WANDB_MODE"] = "online"
        client = config["orchestrator"]["model"]["client"]
        client["base_url"] = [gateway + "/v1"]
        client["litecast"].update(registry=registry, run_id=RUN, origin_host=HOST, origin_port=free_port())
        destination = OUTPUT / "train.toml"
        destination.write_text(tomli_w.dumps(config))
        training = start("uv", "run", "--no-sync", "rl", "@", str(destination))
        while training.poll() is None:
            for child in children:
                if child is not training and child.poll() is not None:
                    raise RuntimeError("Trainer coordination relay exited")
            time.sleep(1)
        result = training.returncode
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
