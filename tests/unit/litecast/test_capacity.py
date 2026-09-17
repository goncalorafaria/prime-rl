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
async def replica(tmp_path, config, gate, headers_gate=None):
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
        if headers_gate is not None:
            await headers_gate.wait()
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
                from prime_rl.litecast.affinity import prompt_key
                headers = {"X-Session-ID": prompt_key([{"role": "user", "content": "shared question"}])}
                tasks = [asyncio.create_task(client.post(endpoint, json=generation_payload(config), headers=headers)) for _ in range(3)]
                await until(lambda: first["active"] == 1 and len(routing.pending) == 2)
                assert first["total"] == 1
                async with replica(tmp_path, config, gate) as (_, second):
                    await until(lambda: second["active"] == 1 and len(routing.pending) == 1)
                    assert first["peak"] == second["peak"] == 1
                    gate.set()
                    results = await asyncio.gather(*tasks)
                    assert all(r.status_code == 200 and r.json()["choices"][0]["text"] == "ok" for r in results)
                    if endpoint == "/inference/v1/generate":
                        chosen = results[0].json()["litecast_replica_id"]
                        followup = await client.post(endpoint, json=generation_payload(config),
                                                     headers={"X-Session-ID": "replica:" + chosen})
                        assert followup.json()["litecast_replica_id"] == chosen
                    await until(lambda: not routing.active)
                    assert not routing.pending
                    assert routing.affinity_preferred == (4 if endpoint == "/inference/v1/generate" else 3)
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
            SimpleNamespace(endpoint="v1/completions", service="version", payload={"model": "version"}, headers={})
        )
    assert error.value.status_code == 503
    assert not routing.active
    assert not routing.pending


def test_prompt_affinity_uses_original_messages():
    from prime_rl.litecast.affinity import prompt_key

    prompt = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "question"}]
    key = prompt_key(prompt, initial=True)
    assert key == prompt_key([dict(reversed(list(message.items()))) for message in prompt], initial=True)
    assert key == prompt_key(prompt + [{"role": "assistant", "content": "answer"}, {"role": "tool", "content": "result"}])
    assert key != prompt_key([prompt[0], {"role": "user", "content": "different"}])
    assert key != prompt_key([{**prompt[0], "content": "other system"}, prompt[1]])
    assert prompt_key([prompt[0]]) is None


@pytest.mark.asyncio
async def test_prompt_affinity_overflows_and_stays_with_requested_policy():
    class Registry:
        records = [
            {"uri": f"http://{name}", "metadata": {"replica_id": name, "max_inflight_requests": 1}}
            for name in ("one", "two", "three")
        ]

        async def models(self, force):
            return {"v1": self.records, "v2": [self.records[-1]]}

    registry = Registry()
    routing = CapacityRouting(registry, CapacityConfig(prompt_affinity_replicas=2, discovery_interval_seconds=0.001))
    key = "prompt-sha256:" + "a" * 64
    async def acquire(model="v1"):
        return (await routing.acquire(model, time.monotonic() + 1, key))[0]

    first = await acquire()
    await routing.release(first)
    registry.records.reverse()
    routing.refreshed = float("-inf")
    assert await acquire() == first
    second = await acquire()
    third = await acquire()
    assert len({first, second, third}) == 3
    assert routing.affinity_overflow == 1
    for identity in (first, second, third):
        await routing.release(identity)
    only_v2 = await acquire("v2")
    assert only_v2 == registry.records[-1]["metadata"]["replica_id"]
    await routing.release(only_v2)
    registry.records = [r for r in registry.records if r["metadata"]["replica_id"] != first]
    routing.refreshed = float("-inf")
    replacement = await acquire()
    assert replacement != first
    await routing.release(replacement)


@pytest.mark.asyncio
async def test_renderer_affinity_preserves_key_and_other_headers(monkeypatch):
    from prime_rl.litecast.affinity import HEADER, STATE_KEY, install_prompt_affinity
    from verifiers.clients import renderer_client
    from verifiers.v1.clients import config
    import renderers.client as rendering

    original = renderer_client.RendererClient
    monkeypatch.setattr(renderer_client, "RendererClient", original)
    monkeypatch.setattr(config, "TrainClient", config.TrainClient)
    for module, name in [(rendering, "parse_generate_response"), (rendering, "generate"), (renderer_client, "generate")]:
        monkeypatch.setattr(module, name, getattr(module, name))
    calls = []
    async def capture(self, prompt, model, sampling_args, tools=None, **kwargs):
        calls.append(kwargs["extra_headers"])
        return {"litecast_replica_id": "one" if len(calls) == 1 else "two"}

    monkeypatch.setattr(original, "get_native_response", capture)
    install_prompt_affinity()
    client = renderer_client.RendererClient.__new__(renderer_client.RendererClient)
    initial = [{"role": "user", "content": "task"}]
    state = {"prompt": initial}
    await client.get_native_response(initial, "v1", {}, state=state, extra_headers={"Other": "preserve"})
    await client.get_native_response(initial, "v2", {}, state=state)
    await client.get_native_response(initial, "v2", {}, state=state)
    await client.get_native_response(initial, "v1", {}, state={"prompt": initial})
    assert calls == [{"Other": "preserve"}, {HEADER: "replica:one"}, {HEADER: "replica:two"}, {}]
    assert state[STATE_KEY] == "two"


@pytest.mark.asyncio
async def test_rollout_affinity_falls_back_on_load_and_policy():
    class Registry:
        async def models(self, force):
            records = [{"uri": "http://" + name, "metadata": {"replica_id": name, "max_inflight_requests": 16}}
                       for name in ("one", "two")]
            return {"v1": records, "v2": records[1:]}
    routing = CapacityRouting(Registry(), CapacityConfig(max_inflight_per_replica=16))
    async def select(model="v1", preferred="one"):
        identity, _ = await routing.acquire(model, time.monotonic() + 1, preferred_replica=preferred)
        await routing.release(identity)
        return identity
    routing.active = {"one": 4, "two": 2}
    assert await select() == "one"
    routing.active["one"] = 5
    assert await select() == "two"
    routing.active["one"] = 16
    assert await select() == "two"
    routing.active["one"] = 0
    assert await select("v2") == "two"
    assert await select(preferred="removed") == "one"
    assert routing.affinity_preferred == 1
    assert routing.affinity_overflow == 4


@pytest.mark.asyncio
async def test_backend_header_timeout_releases_gateway_and_worker_slots(tmp_path):
    config = publisher_config(tmp_path)
    publisher = Publisher(config, "model")
    await publisher.start()
    gate = asyncio.Event()
    gate.set()
    headers_gate = asyncio.Event()
    app = create_app(publisher.registry, CapacityConfig(max_inflight_per_replica=1))
    port = free_port()
    try:
        async with replica(tmp_path, config, gate, headers_gate) as (worker, counts), serve(app, port):
            worker.backend.timeout = httpx.Timeout(0.05)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
                response = await client.post("/inference/v1/generate", json=generation_payload(config))
                assert response.status_code == 504
                await until(lambda: not app.state.capacity_routing.active and not sum(worker.inflight.values()))
                headers_gate.set()
                worker.backend.timeout = httpx.Timeout(5)
                response = await client.post("/inference/v1/generate", json=generation_payload(config))
                assert response.status_code == 200
    finally:
        headers_gate.set()
        await publisher.close()
