"""Real HTTP admission, elastic capacity, and cancellation backpressure."""

import asyncio
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse
from test_lifecycle import free_port, serve

from prime_rl.litecast.distribution import Publisher
from prime_rl.litecast.gateway import CapacityConfig, CapacityRouting, create_app
from prime_rl.litecast.protocol import base_alias
from prime_rl.litecast.worker import Worker
from prime_rl.litecast.worker import create_app as worker_app


async def until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


@asynccontextmanager
async def replica(tmp_path, config, gate):
    app = FastAPI()
    counts = {"active": 0, "peak": 0, "total": 0}

    @app.get("/health")
    async def health():
        return {}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "model"}]}

    @app.post("/inference/v1/generate")
    @app.post("/v1/completions")
    async def complete(request: Request):
        payload = await request.json()
        assert payload["model"] == "model"
        if request.url.path == "/inference/v1/generate":
            assert payload["token_ids"] == [101, 102]
            assert payload["sampling_params"] == {"max_tokens": 1, "logprobs": 1}

        async def body():
            counts["active"] += 1
            counts["total"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            try:
                yield b'{"choices":'
                await gate.wait()
                yield b'[{"text":"ok"}]}'
            finally:
                counts["active"] -= 1

        return StreamingResponse(body(), media_type="application/json")

    backend_port = free_port()
    worker = Worker(
        SimpleNamespace(
            registry=config.registry,
            run_id=config.run_id,
            base_model="model",
            backend_url=f"http://127.0.0.1:{backend_port}",
            cache_dir=tmp_path,
            max_versions=2,
            max_adapter_bytes=1024,
            shard_bytes=64,
            transport="http",
            advertise_host="127.0.0.1",
            port=free_port(),
            shard_port=free_port(),
            poll_seconds=0.05,
            lease_seconds=60,
            request_timeout=10,
            max_inflight_requests=1,
        )
    )
    async with serve(app, backend_port), serve(worker_app(worker), worker.args.port):
        await until(lambda: worker.available(base_alias(config.run_id)))
        yield worker, counts


def generation_payload(config):
    return {
        "model": base_alias(config.run_id),
        "token_ids": [101, 102],
        "sampling_params": {"max_tokens": 1, "logprobs": 1},
    }


def publisher_config(tmp_path):
    return SimpleNamespace(
        registry=f"file://{tmp_path / 'registry'}",
        run_id="capacity",
        origin_host="127.0.0.1",
        origin_port=free_port(),
        transport="http",
        max_adapter_bytes=1024,
        retain_versions=2,
        poll_seconds=0.05,
        lease_seconds=60,
        shard_bytes=64,
        min_replicas=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/completions", "/inference/v1/generate"])
async def test_burst_queues_and_new_replica_adds_capacity(tmp_path, endpoint):
    config = publisher_config(tmp_path)
    publisher = Publisher(config, "model")
    await publisher.start()
    gate = asyncio.Event()
    tasks = []
    app = create_app(publisher.registry, CapacityConfig(max_inflight_per_replica=1, discovery_interval_seconds=0.02))
    routing = app.state.capacity_routing
    port = free_port()
    try:
        async with replica(tmp_path, config, gate) as (_, first), serve(app, port):
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
                tasks = [asyncio.create_task(client.post(endpoint, json=generation_payload(config))) for _ in range(3)]
                await until(lambda: first["active"] == 1 and len(routing.pending) == 2)
                assert first["total"] == 1
                async with replica(tmp_path, config, gate) as (_, second):
                    await until(lambda: second["active"] == 1 and len(routing.pending) == 1)
                    assert first["peak"] == second["peak"] == 1
                    gate.set()
                    results = await asyncio.gather(*tasks)
                    assert all(r.status_code == 200 and r.json()["choices"][0]["text"] == "ok" for r in results)
                    await until(lambda: not routing.active)
                    assert not routing.pending
                    assert first["peak"] == second["peak"] == 1
    finally:
        gate.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await publisher.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/completions", "/inference/v1/generate"])
async def test_queue_overflow_disconnect_and_worker_hard_limit(tmp_path, endpoint):
    config = publisher_config(tmp_path)
    publisher = Publisher(config, "model")
    await publisher.start()
    gate = asyncio.Event()
    app = create_app(
        publisher.registry,
        CapacityConfig(max_inflight_per_replica=1, max_queued_requests=1, discovery_interval_seconds=0.02),
    )
    routing = app.state.capacity_routing
    port = free_port()
    tasks = []
    try:
        async with replica(tmp_path, config, gate) as (worker, counts), serve(app, port):
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
                request = generation_payload(config)
                first = asyncio.create_task(client.post(endpoint, json=request))
                tasks.append(first)
                await until(lambda: counts["active"] == 1)
                queued = asyncio.create_task(client.post(endpoint, json=request))
                tasks.append(queued)
                await until(lambda: len(routing.pending) == 1)
                assert (await client.post(endpoint, json=request)).status_code == 429
                # Bypassing gateway admission still cannot overload the backend.
                direct = await client.post(f"http://127.0.0.1:{worker.args.port}{endpoint}", json=request)
                assert direct.status_code == 429
                assert counts["total"] == 1
                queued.cancel()
                await asyncio.gather(queued, return_exceptions=True)
                await until(lambda: not routing.pending)
                # The gateway and worker hold the slot through the partial response.
                assert sum(routing.active.values()) == sum(worker.inflight.values()) == 1
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
                await until(lambda: not routing.active and sum(worker.inflight.values()) == 0)
                gate.set()
                assert (await client.post(endpoint, json=request)).status_code == 200
                assert counts["peak"] == 1
    finally:
        gate.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await publisher.close()


@pytest.mark.asyncio
async def test_capacity_is_shared_across_versions_and_queue_timeout_releases_ticket():
    class Registry:
        async def models(self, force):
            record = {"uri": "http://worker", "metadata": {"replica_id": "one", "max_inflight_requests": 1}}
            return {"version-1": [record], "version-2": [record]}

    from literegistry.gateway import GatewayRequestError

    routing = CapacityRouting(Registry(), CapacityConfig(discovery_interval_seconds=0.01))
    identity, _ = await routing.acquire("version-1", time.monotonic() + 1)
    with pytest.raises(GatewayRequestError) as error:
        await routing.acquire("version-2", time.monotonic() + 0.05)
    assert error.value.status_code == 503
    assert not routing.pending
    assert routing.active == {identity: 1}
    await routing.release(identity)
    assert (await routing.acquire("version-2", time.monotonic() + 1))[0] == identity
    await routing.release(identity)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/completions", "/inference/v1/generate"])
async def test_worker_rejection_requeues_until_another_replica_joins(tmp_path, endpoint):
    config = publisher_config(tmp_path)
    publisher = Publisher(config, "model")
    await publisher.start()
    gate = asyncio.Event()
    app = create_app(publisher.registry, CapacityConfig(max_inflight_per_replica=1, discovery_interval_seconds=0.02))
    routing = app.state.capacity_routing
    port = free_port()
    task = None
    try:
        async with replica(tmp_path, config, gate) as (worker, first), serve(app, port):
            async with httpx.AsyncClient(timeout=10) as client:
                request = generation_payload(config)
                # Another caller/gateway occupies a slot unknown to this gateway.
                async with client.stream(
                    "POST", f"http://127.0.0.1:{worker.args.port}{endpoint}", json=request
                ) as direct:
                    await until(lambda: first["active"] == 1)
                    task = asyncio.create_task(client.post(f"http://127.0.0.1:{port}{endpoint}", json=request))
                    await until(lambda: routing.worker_rejections > 0)
                    assert first["total"] == 1
                    async with replica(tmp_path, config, gate) as (_, second):
                        await until(lambda: second["active"] == 1)
                        gate.set()
                        assert (await task).status_code == 200
                        await direct.aread()
                        await until(lambda: not routing.active)
                        assert first["peak"] == second["peak"] == 1
    finally:
        gate.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await publisher.close()


@pytest.mark.asyncio
async def test_unreachable_replica_does_not_leak_reservations():
    class Registry:
        async def models(self, force):
            return {
                "version": [
                    {"uri": f"http://127.0.0.1:{port}", "metadata": {"replica_id": "dead", "max_inflight_requests": 1}}
                ]
            }

    from literegistry.gateway import GatewayRequestError

    port = free_port()
    routing = CapacityRouting(Registry(), CapacityConfig(queue_timeout_seconds=0.2, discovery_interval_seconds=0.01))
    with pytest.raises(GatewayRequestError) as error:
        await routing.forward(
            SimpleNamespace(endpoint="v1/completions", service="version", payload={"model": "version"})
        )
    assert error.value.status_code == 503
    assert not routing.active
    assert not routing.pending
