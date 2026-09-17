"""Task supervisor for the single Rex LiteCast training experiment."""

import logging
import os
import runpy
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import tomli_w
from literegistry.coop.endpoints import endpoint_healthy

from prime_rl.litecast.bootstrap import head_registry, refresh_endpoint, wait
from prime_rl.litecast.retention import prune_intermediates

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = "toy-" + os.environ.get("LITECAST_PARENT_EXPERIMENT_ID", os.environ["REXS_EXPERIMENT_ID"])
OUTPUT = ROOT / "outputs" / RUN
HEAD = head_registry(OUTPUT)
LOCAL_GATEWAY = HEAD.startswith("sqlite:")
TRAINER_FAILED = OUTPUT / "trainer-failed"
TRAIN_CONFIG = Path(os.environ.get("LITECAST_TRAIN_CONFIG", HERE / "train.toml"))
INFERENCE_CONFIG = Path(os.environ.get("LITECAST_INFERENCE_CONFIG", HERE / "inference.toml"))
CAPACITY_CONFIG = Path(os.environ.get("LITECAST_CAPACITY_CONFIG", HERE / "capacity.toml"))
HOST = socket.getfqdn()
children = []
gateway_heartbeat = None


def start(*args, log_path=None):
    if log_path is None:
        process = subprocess.Popen(list(args), start_new_session=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as output:
            process = subprocess.Popen(list(args), start_new_session=True,
                                       stdout=output, stderr=subprocess.STDOUT)
    children.append(process)
    return process


def python(*args, log_path=None):
    return start("uv", "run", "--no-project", "--python", sys.executable, "python", *args, log_path=log_path)


def free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def watch(records):
    while True:
        if TRAINER_FAILED.exists():
            logging.error("Trainer failure signal received; stopping dependent services")
            raise SystemExit(0)
        for process in children:
            if process.poll() is not None:
                raise RuntimeError(f"service exited: {process.args} status={process.returncode}")
        for name, uri in records.items():
            refresh_endpoint(HEAD, name, uri, publisher_id=RUN + "-" + name, ttl_seconds=30)
        time.sleep(2)


def wait_for_first_replica(gateway):
    import json
    from uuid import uuid4
    import asyncio
    import redis
    from literegistry import get_kvstore
    from prime_rl.litecast.bootstrap import HeadRedisCommands
    from prime_rl.litecast.protocol import desired_key

    config = tomllib.loads((OUTPUT / "train.toml").read_text())
    client = config["orchestrator"]["model"]["client"]["litecast"]
    runner = asyncio.Runner() if client["registry"].startswith("head+") else None
    store = get_kvstore(client["registry"]) if runner else None
    commands = HeadRedisCommands(store) if runner else None
    connection = None if runner else redis.Redis.from_url(client["registry"])

    def evaluate(*args):
        return runner.run(commands.eval(*args)) if runner else connection.eval(*args)

    owner = "startup-" + uuid4().hex
    key = desired_key(RUN)
    payload = json.dumps({"schema": 1, "owner": owner,
                          "base_model": config["model"]["name"], "publications": []})

    def renew():
        claimed = evaluate(
            "local old=redis.call('GET',KEYS[1]); "
            "if old and cjson.decode(old).owner ~= ARGV[1] then return 0 end; "
            "redis.call('SET',KEYS[1],ARGV[2],'EX',30); return 1", 1, key, owner, payload)
        if not claimed:
            raise RuntimeError("Run already has a publisher; refusing to replace its descriptor")

    try:
        _wait_for_first_replica(gateway, renew)
    finally:
        # Hand the empty run descriptor over to PrimeRL without deleting another owner.
        evaluate("local old=redis.call('GET',KEYS[1]); "
                        "if old and cjson.decode(old).owner == ARGV[1] then return redis.call('DEL',KEYS[1]) end; "
                        "return 0", 1, key, owner)
        if runner:
            runner.run(store.close())
            runner.close()
        else:
            connection.close()


def _wait_for_first_replica(gateway, renew):
    import json
    from urllib.error import URLError
    from urllib.request import urlopen
    from prime_rl.litecast.protocol import base_alias

    model = base_alias(RUN)
    started = time.monotonic()
    next_log = 0
    while True:
        renew()
        if gateway_heartbeat is not None and not gateway_heartbeat.is_alive():
            raise RuntimeError("Local gateway endpoint publication stopped")
        for process in children:
            if process.poll() is not None:
                raise RuntimeError("Trainer coordination service exited while waiting for inference")
        try:
            with urlopen(gateway + "/v1/models", timeout=10) as response:
                models = json.load(response)
            if any(item.get("id") == model for item in models.get("data", [])):
                logging.info("FIRST_REPLICA_READY model=%s wait_seconds=%.1f", model, time.monotonic()-started)
                return
        except (URLError, TimeoutError, ConnectionError) as exc:
            if time.monotonic() >= next_log:
                logging.warning("Gateway unavailable while waiting for first inference replica: %s", exc)
        if time.monotonic() >= next_log:
            logging.info("WAITING_FIRST_REPLICA model=%s elapsed_seconds=%.1f; Slurm queue time is outside PrimeRL startup timeout",
                         model, time.monotonic()-started)
            next_log = time.monotonic()+30
        time.sleep(5)


def validate_lora_config():
    training = tomllib.loads(TRAIN_CONFIG.read_text())
    inference = tomllib.loads(INFERENCE_CONFIG.read_text())
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
    global gateway_heartbeat
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    role = sys.argv[1]
    if role != "trainer" and TRAINER_FAILED.exists():
        logging.error("Trainer already failed; dependent allocation will not start")
        return
    if role in ("trainer", "inference"):
        runpy.run_path(str(HERE / "credentials.py"))["configure"](trainer=role == "trainer")
    if role == "trainer":
        subprocess.run(
            ["uv", "run", "--no-project", "--python", sys.executable, "python", str(HERE / "preflight.py")],
            check=True,
            timeout=120,
        )
    if role in ("trainer", "inference"):
        validate_lora_config()
    os.environ.update(RUN_ID=RUN, ADVERTISE_HOST=HOST, TOKENIZERS_PARALLELISM="false")
    if role == "head":
        redis_port, gateway_port = free_port(), free_port()
        registry = f"redis://{HOST}:{redis_port}/0"
        os.environ["REGISTRY_PATH"] = registry
        os.environ["LITECAST_CAPACITY_CONFIG"] = str(CAPACITY_CONFIG)
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
        if LOCAL_GATEWAY:
            watch({"redis": registry})
            return
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
    if role == "trainer" and not LOCAL_GATEWAY:
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
    wait(HEAD, "redis", timeout=3600, healthcheck="redis")
    if LOCAL_GATEWAY:
        registry = "head+" + HEAD
        if role == "trainer":
            gateway_port = free_port()
            os.environ.update(REGISTRY_PATH=registry, LITECAST_CAPACITY_CONFIG=str(CAPACITY_CONFIG))
            python("-m", "uvicorn", "prime_rl.litecast.gateway:create_app", "--factory",
                   "--host", "0.0.0.0", "--port", str(gateway_port),
                   log_path=OUTPUT / "logs" / "gateway.log")
            gateway = f"http://{HOST}:{gateway_port}"
            deadline = time.monotonic() + 120
            while not endpoint_healthy(gateway, "http", 2):
                if any(child.poll() is not None for child in children):
                    raise RuntimeError("Trainer local gateway exited during startup")
                if time.monotonic() > deadline:
                    raise TimeoutError("Trainer local gateway startup timed out")
                time.sleep(0.2)
            def renew_gateway():
                while True:
                    refresh_endpoint(HEAD, "gateway", gateway, publisher_id=RUN + "-gateway", ttl_seconds=30)
                    time.sleep(2)
            gateway_heartbeat = threading.Thread(target=renew_gateway, daemon=True)
            gateway_heartbeat.start()
        else:
            gateway = wait(HEAD, "gateway", timeout=3600, healthcheck="http")
    else:
        registry = wait(HEAD, "redis", timeout=3600, healthcheck="redis")
        gateway = wait(HEAD, "gateway", timeout=3600, healthcheck="http")
    os.environ.update(REGISTRY=registry, GATEWAY_URL=gateway)
    if role in ("trainer", "inference"):
        from huggingface_hub import snapshot_download

        custom_model = os.environ.get("LITECAST_MODEL_SOURCE")
        if custom_model:
            stage = runpy.run_path(str(ROOT / "examples/litecast-rubrics/stage.py"))["stage"]
            namespace = os.environ.get("LITECAST_STAGE_NAMESPACE", "")
            if custom_model.startswith("hf://"):
                snapshot = snapshot_download(custom_model[5:], cache_dir=os.environ["HF_HUB_CACHE"])
            else:
                snapshot = stage(custom_model, namespace + "-model" if namespace else "sloth2b-sft-step400")
        else:
            revision = "15852e8c16360a2fea060d615a32b45270f8a8fc"
            snapshot = snapshot_download("Qwen/Qwen3.5-2B", revision=revision, cache_dir=os.environ["HF_HUB_CACHE"])
        print(f"MODEL_STAGED path={snapshot}", flush=True)
        if role == "trainer":
            from datasets import load_dataset, load_from_disk

            if custom_model:
                dataset_path = stage(os.environ["LITECAST_DATASET_SOURCE"], namespace + "-data" if namespace else "active-rl")
                load_from_disk(str(dataset_path))
            else:
                dataset_path = runpy.run_path(str(HERE / "dataset-cache.py"))["stage"]()
                load_dataset(str(dataset_path), split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    if role == "terminal":
        python("-m", "literegistry.services.terminal_server", "--registry", registry,
               "--host", "0.0.0.0", "--port", str(free_port()))
        watch({})
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
        if os.getenv("LITECAST_INFLIGHT_UPDATES") == "1":
            os.environ["PRIME_RL_HYBRID_LORA"] = "1"
            os.environ["LITECAST_LINEAGE_DIR"] = str(OUTPUT / "policy-lineage")
        inference_config = tomllib.loads(INFERENCE_CONFIG.read_text())
        inference_config["model"]["name"] = str(snapshot)
        if os.getenv("LITECAST_INFLIGHT_UPDATES") == "1":
            inference_config.setdefault("vllm_extra", {})["async_scheduling"] = False
        inference_rank = int(os.environ.get("BEAKER_REPLICA_RANK", "0")) + int(os.environ.get("LITECAST_REPLICA_OFFSET", "0"))
        inference_path = OUTPUT / f"inference-{inference_rank}.toml"
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
            "--request-timeout",
            str(tomllib.loads(CAPACITY_CONFIG.read_text()).get("request_timeout_seconds", 300)),
            *(["--require-middle"] if os.environ.get("LITECAST_REQUIRE_MIDDLE", "1") == "1" else []),
            "--max-inflight-requests",
            str(tomllib.loads(CAPACITY_CONFIG.read_text())["max_inflight_per_replica"])
            if os.getenv("LITECAST_CAPACITY_ENABLED") == "1"
            else "4",
        )
        watch({})
    elif role == "trainer":
        for rank in range(int(os.environ.get("LITECAST_MIN_MIDDLES", "2"))):
            wait(HEAD, f"middle-{rank}", timeout=3600, healthcheck="none")
        config = tomllib.loads(TRAIN_CONFIG.read_text())
        config["model"]["name"] = str(snapshot)
        for section in ("train", "eval"):
            for source in config["orchestrator"].get(section, {}).get("source", []):
                if "legacy" in source:
                    source["legacy"]["args"]["dataset"] = str(dataset_path)
                    if source["legacy"]["id"] == "primebeaker.environments.jtc_rubrichub_judge_env":
                        source["legacy"]["args"]["judge_server_url"] = gateway + "/judge"
                    else:
                        source["legacy"]["args"]["terminal_server_url"] = gateway + "/terminal"
                elif source.get("env", {}).get("taskset", {}).get("id") == "reverse-text-v1":
                    source["env"]["taskset"]["dataset_name"] = str(dataset_path)
        config["output_dir"] = str(OUTPUT / "training")
        config["wandb"]["name"] = RUN
        if os.getenv("LITECAST_INFLIGHT_UPDATES") == "1":
            config["wandb"].setdefault("tags", []).extend(["inflight-lora", "weak-affinity"])
        config["wandb"]["entity"] = os.environ["WANDB_ENTITY"]
        os.environ["WANDB_MODE"] = "online"
        client = config["orchestrator"]["model"]["client"]
        client["base_url"] = [gateway + "/v1"]
        client["litecast"].update(registry=registry, run_id=RUN, origin_host=HOST, origin_port=free_port())
        destination = OUTPUT / "train.toml"
        destination.write_text(tomli_w.dumps(config))
        if os.getenv("LITECAST_WAIT_FIRST_REPLICA") == "1":
            wait_for_first_replica(gateway)
        if os.environ.get("LITECAST_JUDGE_SERVICE"):
            wait(HEAD, os.environ["LITECAST_JUDGE_SERVICE"], timeout=3600, healthcheck="http")
        training = start("uv", "run", "--no-sync", "rl", "@", str(destination))
        next_prune = time.monotonic() + 30
        while training.poll() is None:
            if gateway_heartbeat is not None and not gateway_heartbeat.is_alive():
                raise RuntimeError("Local gateway endpoint publication stopped")
            if time.monotonic() >= next_prune and config["max_steps"] > 3:
                try:
                    prune_intermediates(
                        OUTPUT / "training",
                        keep_rollout_steps=config["ckpt"]["interval"] + 2,
                        keep_adapter_steps=client["litecast"]["retain_versions"],
                    )
                except OSError:
                    logging.exception("Intermediate cleanup failed; retrying on the next sweep")
                next_prune = time.monotonic() + 30
            for child in children:
                if child is not training and child.poll() is not None:
                    raise RuntimeError("Trainer coordination service exited")
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
    except BaseException:
        if sys.argv[1] == "trainer":
            OUTPUT.mkdir(parents=True, exist_ok=True)
            temporary = TRAINER_FAILED.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(f"run={RUN} trainer_pid={os.getpid()} time={time.time()}\n")
            temporary.replace(TRAINER_FAILED)
            logging.error("TRAINER_FAILURE_SIGNAL path=%s", TRAINER_FAILED)
        raise
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
