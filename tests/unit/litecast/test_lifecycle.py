"""CPU contract test: real HTTP, LiteCast and LiteRegistry; a tiny adapter-aware engine."""

import asyncio
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI, Request
from literegistry.gateway import create_app as gateway_app
from safetensors.numpy import load_file, save_file

from prime_rl.litecast.distribution import Publisher
from prime_rl.litecast.protocol import base_alias, desired_key
from prime_rl.litecast.worker import Worker, create_app


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)


def engine():
    app = FastAPI()
    adapters = {}

    @app.get("/health")
    async def health():
        return {}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": name} for name in ["test-2b", *adapters]]}

    @app.post("/v1/load_lora_adapter")
    async def load(request: Request):
        value = await request.json()
        assert not value["load_inplace"]
        tensors = load_file(str(Path(value["lora_path"]) / "adapter_model.safetensors"))
        adapters[value["lora_name"]] = float(tensors["lora_A.weight"][0])
        return {}

    @app.post("/v1/unload_lora_adapter")
    async def unload(request: Request):
        adapters.pop((await request.json())["lora_name"], None)
        return {}

    @app.post("/v1/completions")
    async def completion(request: Request):
        value = await request.json()
        return {"model": value["model"], "choices": [{"text": str(adapters.get(value["model"], 0.0))}]}

    return app


@pytest.mark.asyncio
async def test_publish_gateway_late_join_preemption_and_expiry(tmp_path, monkeypatch):
    registry = f"file://{tmp_path / 'registry'}"
    config = SimpleNamespace(
        registry=registry,
        run_id="cpu-smoke",
        origin_host="127.0.0.1",
        origin_port=free_port(),
        transport="http",
        max_adapter_bytes=1024 * 1024,
        retain_versions=2,
        poll_seconds=0.05,
        lease_seconds=2,
        shard_bytes=64,
        min_replicas=1,
    )
    publisher = Publisher(config, "test-2b")

    def make_worker(backend_port):
        return Worker(
            SimpleNamespace(
                registry=registry,
                run_id=config.run_id,
                base_model="test-2b",
                backend_url=f"http://127.0.0.1:{backend_port}",
                cache_dir=tmp_path,
                max_versions=2,
                max_adapter_bytes=config.max_adapter_bytes,
                shard_bytes=64,
                transport="http",
                advertise_host="127.0.0.1",
                port=free_port(),
                shard_port=free_port(),
                poll_seconds=0.05,
                lease_seconds=2,
                request_timeout=5,
            )
        )

    backend_port, gateway_port = free_port(), free_port()
    monkeypatch.setenv("REGISTRY_PATH", registry)
    monkeypatch.setenv("GATEWAY_REGISTER", "false")
    monkeypatch.setenv("REGISTRY_CACHE_TTL_SECONDS", "0")
    async with serve(engine(), backend_port), serve(gateway_app(enable_registration=False), gateway_port):
        await publisher.start()
        worker = make_worker(backend_port)
        async with serve(create_app(worker), worker.args.port):
            await publisher.wait_ready(base_alias(config.run_id), 10)
            directory = tmp_path / "adapter"
            directory.mkdir()
            (directory / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}')
            save_file(
                {"lora_A.weight": np.array([1.0], dtype=np.float32)}, str(directory / "adapter_model.safetensors")
            )
            first = await publisher.publish(directory, 1)
            await publisher.wait_ready(first.model_name, 10)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{gateway_port}", timeout=10) as client:
                response = await client.post("/v1/completions", json={"model": first.model_name, "prompt": "search"})
                response.raise_for_status()
                assert response.json()["choices"][0]["text"] == "1.0"
                save_file(
                    {"lora_A.weight": np.array([2.0], dtype=np.float32)}, str(directory / "adapter_model.safetensors")
                )
                second = await publisher.publish(directory, 2)
                await publisher.wait_ready(second.model_name, 10)
                for publication, expected in [(first, "1.0"), (second, "2.0")]:
                    response = await client.post(
                        "/v1/completions", json={"model": publication.model_name, "prompt": "search"}
                    )
                    response.raise_for_status()
                    assert response.json()["choices"][0]["text"] == expected
            # A late joiner uses a verified peer even after the origin loses the payload.
            publisher.origin.store.clear()
            late_backend = free_port()
            late_worker = make_worker(late_backend)
            async with serve(engine(), late_backend), serve(create_app(late_worker), late_worker.args.port):
                async with asyncio.timeout(10):
                    while second.model_name not in late_worker.serving:
                        await asyncio.sleep(0.05)
                assert first.model_name in late_worker.loaded
                # Simulate loss of the original replica's advertisement.
                worker.task.cancel()
                await asyncio.gather(worker.task, return_exceptions=True)
                await worker.withdraw()
                await publisher.wait_ready(second.model_name, 10)
                async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{gateway_port}", timeout=10) as client:
                    response = await client.post(
                        "/v1/completions", json={"model": second.model_name, "prompt": "search"}
                    )
                    response.raise_for_status()
                    assert response.json()["choices"][0]["text"] == "2.0"
                # Publisher loss withdraws readiness; a cached gateway cannot
                # trick a surviving worker into serving a stale policy.
                publisher.task.cancel()
                await asyncio.gather(publisher.task, return_exceptions=True)
                await publisher.store.delete(desired_key(config.run_id))
                async with asyncio.timeout(10):
                    while late_worker.serving:
                        await asyncio.sleep(0.05)
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"http://127.0.0.1:{late_worker.args.port}/v1/completions",
                        json={"model": second.model_name, "prompt": "search"},
                    )
                    assert response.status_code == 503
        await publisher.close()


@pytest.mark.asyncio
async def test_prime_rl_pool_publishes_and_keeps_versioned_model_name(tmp_path):
    from collections import defaultdict

    from prime_rl.configs.shared import ClientConfig, LitecastPoolConfig
    from prime_rl.utils.client import setup_inference_pool

    backend_port, worker_port = free_port(), free_port()
    config = ClientConfig(
        base_url=["http://gateway:1212/v1"],
        litecast=LitecastPoolConfig(
            registry=f"file://{tmp_path / 'registry'}",
            run_id="pool-test",
            origin_host="127.0.0.1",
            origin_port=free_port(),
            poll_seconds=0.05,
            lease_seconds=2,
            max_adapter_bytes=1024 * 1024,
        ),
    )
    pool = await setup_inference_pool(config, model_name="test-2b")
    worker = Worker(
        SimpleNamespace(
            registry=config.litecast.registry,
            run_id="pool-test",
            base_model="test-2b",
            backend_url=f"http://127.0.0.1:{backend_port}",
            cache_dir=tmp_path,
            max_versions=8,
            max_adapter_bytes=1024 * 1024,
            shard_bytes=64,
            transport="http",
            advertise_host="127.0.0.1",
            port=worker_port,
            shard_port=free_port(),
            poll_seconds=0.05,
            lease_seconds=2,
            request_timeout=5,
        )
    )
    try:
        async with serve(engine(), backend_port), serve(create_app(worker), worker_port):
            await pool.wait_for_ready("test-2b", timeout=10)
            assert pool.model_name == base_alias("pool-test")
            assert pool.admin_clients == []
            directory = tmp_path / "adapter"
            directory.mkdir()
            (directory / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}')
            save_file(
                {"lora_A.weight": np.array([7.0], dtype=np.float32)}, str(directory / "adapter_model.safetensors")
            )
            await pool.update_weights(directory, lora_name="fixed-alias", step=1)
            model_name = pool.model_name
            assert model_name.startswith("sc-pool-test-s1-")
            pool.update_model_name("fixed-alias")
            assert pool.model_name == model_name
            assert (await pool.select_train_client(defaultdict(int))).base_url == "http://gateway:1212/v1"
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_weight_sources_require_registry_discovery(tmp_path):
    from prime_rl.litecast.distribution import fetch_publication
    from prime_rl.litecast.protocol import peer_service

    config = SimpleNamespace(
        registry=f"file://{tmp_path / 'registry'}",
        run_id="discovery",
        origin_host="127.0.0.1",
        origin_port=free_port(),
        transport="http",
        max_adapter_bytes=1024,
        retain_versions=1,
        poll_seconds=60,
        lease_seconds=120,
        shard_bytes=64,
        min_replicas=1,
    )
    publisher = Publisher(config, "test-2b")
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}')
    (directory / "adapter_model.safetensors").write_bytes(b"adapter-one")
    try:
        await publisher.start()
        first = await publisher.publish(directory, 1)
        assert set(first.to_dict()) == {"run_id", "base_model", "step", "digest", "size"}
        service = peer_service(config.run_id, first.digest)
        records = (await publisher.registry.models(force=True))[service]
        assert len(records) == 1
        assert records[0]["metadata"]["source_role"] == "origin"
        payload = await fetch_publication(first, publisher.registry, tmp_path / "download", "http")
        first.verify(payload)

        # No raw origin fallback: a reachable origin must still be discovered.
        await publisher.sources[first.digest][0].deregister()
        with pytest.raises(RuntimeError, match="no valid LiteCast source"):
            await fetch_publication(first, publisher.registry, tmp_path / "missing", "http")
        await publisher.refresh()
        first.verify(await fetch_publication(first, publisher.registry, tmp_path / "renewed", "http"))

        (directory / "adapter_model.safetensors").write_bytes(b"adapter-two")
        second = await publisher.publish(directory, 2)
        models = await publisher.registry.models(force=True)
        assert service not in models
        assert peer_service(config.run_id, second.digest) in models
    finally:
        await publisher.close()
