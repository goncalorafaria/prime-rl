"""Capacity-aware routing installed through LiteRegistry's gateway extension API."""

import asyncio
import os
import random
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
from literegistry import RegistryClient, get_kvstore
from literegistry.gateway import (
    GatewayRequestError,
    LoadBalancedRouting,
    RoutingResponse,
    default_proxy_routes,
)
from literegistry.gateway import (
    create_app as registry_app,
)
from starlette.responses import JSONResponse, Response

from prime_rl.litecast.responses import DisconnectCancellationMiddleware, ManagedStreamingResponse


@dataclass(frozen=True)
class CapacityConfig:
    max_inflight_per_replica: int = 4
    max_queued_requests: int = 256
    queue_timeout_seconds: float = 120
    request_timeout_seconds: float = 300
    discovery_interval_seconds: float = 0.25

    def __post_init__(self):
        if min(self.__dict__.values()) <= 0:
            raise ValueError("capacity limits and timeouts must be positive")

    @classmethod
    def load(cls, path=None):
        return cls(**tomllib.loads(Path(path).read_text())) if path else cls()


class CapacityRouting:
    def __init__(self, registry, config: CapacityConfig):
        self.registry = registry
        self.config = config
        self.fallback = LoadBalancedRouting(registry)
        self.active: dict[str, int] = {}
        self.pending: dict[object, str] = {}
        self.cooldown: dict[str, float] = {}
        self.changed = asyncio.Event()
        self.discovery_lock = asyncio.Lock()
        self.records = {}
        self.refreshed = float("-inf")
        self.completed = 0
        self.queue_full = 0
        self.queue_timeouts = 0
        self.worker_rejections = 0

    async def discover(self):
        async with self.discovery_lock:
            if time.monotonic() - self.refreshed >= self.config.discovery_interval_seconds:
                self.records = await self.registry.models(force=True)
                self.refreshed = time.monotonic()
        return self.records

    async def acquire(self, model: str, deadline: float):
        ticket = object()
        if len(self.pending) >= self.config.max_queued_requests:
            self.queue_full += 1
            raise GatewayRequestError("inference admission queue is full", status_code=429)
        self.pending[ticket] = model
        try:
            async with asyncio.timeout_at(deadline):
                while True:
                    records = (await self.discover()).get(model, [])
                    first = next(t for t, name in self.pending.items() if name == model)
                    candidates = []
                    for record in records:
                        metadata = record.get("metadata", {})
                        identity = metadata.get("replica_id")
                        capacity = metadata.get("max_inflight_requests", 0)
                        if not identity or type(capacity) is not int or capacity <= 0:
                            continue
                        capacity = min(capacity, self.config.max_inflight_per_replica)
                        active = self.active.get(identity, 0)
                        if active < capacity and self.cooldown.get(identity, 0) <= time.monotonic():
                            candidates.append((active / capacity, identity, record["uri"]))
                    # No await between the capacity check and reservation: all
                    # policy names share one atomic counter per replica process.
                    if first is ticket and candidates:
                        random.shuffle(candidates)
                        _, identity, uri = min(candidates, key=lambda item: item[0])
                        self.active[identity] = self.active.get(identity, 0) + 1
                        return identity, uri
                    self.changed.clear()
                    try:
                        await asyncio.wait_for(self.changed.wait(), self.config.discovery_interval_seconds)
                    except TimeoutError:
                        pass
        except TimeoutError:
            self.queue_timeouts += 1
            raise GatewayRequestError("timed out waiting for inference capacity", status_code=503) from None
        finally:
            self.pending.pop(ticket, None)
            self.changed.set()

    async def release(self, identity):
        remaining = self.active[identity] - 1
        if remaining:
            self.active[identity] = remaining
        else:
            del self.active[identity]
        self.changed.set()

    async def forward(self, request):
        if request.endpoint.strip("/") not in ("v1/completions", "v1/chat/completions"):
            return await self.fallback.forward(request)
        deadline = time.monotonic() + self.config.queue_timeout_seconds
        while True:
            identity, uri = await self.acquire(request.service, deadline)
            client = httpx.AsyncClient(timeout=self.config.request_timeout_seconds)
            response = None

            async def cleanup():
                try:
                    if response is not None:
                        await response.aclose()
                finally:
                    try:
                        await client.aclose()
                    finally:
                        await self.release(identity)

            try:
                response = await client.send(
                    client.build_request(
                        "POST", uri.rstrip("/") + "/" + request.endpoint.lstrip("/"), json=request.payload
                    ),
                    stream=True,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout):
                self.cooldown[identity] = time.monotonic() + 1
                await asyncio.shield(cleanup())
                continue
            except BaseException:
                await asyncio.shield(cleanup())
                raise
            if response.headers.get("x-litecast-admission-rejected") == "1":
                self.worker_rejections += 1
                self.cooldown[identity] = time.monotonic() + self.config.discovery_interval_seconds
                await asyncio.shield(cleanup())
                continue

            async def body():
                async for chunk in response.aiter_raw():
                    yield chunk
                self.completed += 1

            return RoutingResponse(
                body=ManagedStreamingResponse(
                    body(),
                    cleanup=cleanup,
                    status_code=response.status_code,
                    headers={
                        key: response.headers[key]
                        for key in ("content-type", "content-encoding")
                        if key in response.headers
                    },
                )
            )

    def snapshot(self):
        return {
            "active_requests": sum(self.active.values()),
            "queued_requests": len(self.pending),
            "active_by_replica": dict(self.active),
            "completed_requests": self.completed,
            "queue_full": self.queue_full,
            "queue_timeouts": self.queue_timeouts,
            "worker_rejections": self.worker_rejections,
        }


def create_app(registry=None, config=None):
    registry = registry or RegistryClient(get_kvstore(os.environ["REGISTRY_PATH"], raise_on_error=True), cache_ttl=0)
    routing = CapacityRouting(registry, config or CapacityConfig.load(os.getenv("LITECAST_CAPACITY_CONFIG")))
    routes = [
        replace(route, response_mapper=lambda body: body if isinstance(body, Response) else JSONResponse(body))
        if route.path in ("/v1/completions", "/v1/chat/completions")
        else route
        for route in default_proxy_routes()
    ]
    app = registry_app(registry=registry, routing=routing, routes=routes, enable_registration=False)
    app.state.capacity_routing = routing
    app.add_middleware(DisconnectCancellationMiddleware)

    async def capacity_stats(request):
        return JSONResponse(routing.snapshot())

    app.add_route("/litecast/capacity", capacity_stats)
    return app
