"""Capacity-aware routing installed through LiteRegistry's gateway extension API."""

import asyncio
import hashlib
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
    ProxyRoute,
    RoutingResponse,
    default_proxy_routes,
    service_from_field,
)
from literegistry.gateway import (
    create_app as registry_app,
)
from starlette.responses import JSONResponse, Response

from prime_rl.litecast.responses import GENERATION_PATHS, DisconnectCancellationMiddleware, ManagedStreamingResponse


@dataclass(frozen=True)
class CapacityConfig:
    max_inflight_per_replica: int = 4
    max_queued_requests: int = 256
    queue_timeout_seconds: float = 120
    request_timeout_seconds: float = 300
    discovery_interval_seconds: float = 0.25

    prompt_affinity_replicas: int = 2
    affinity_load_slack: float = 0.125

    def __post_init__(self):
        if min(self.__dict__.values()) <= 0:
            raise ValueError("capacity limits and timeouts must be positive")
        if self.affinity_load_slack > 1:
            raise ValueError("affinity_load_slack must be at most 1")
        if type(self.prompt_affinity_replicas) is not int:
            raise ValueError("prompt_affinity_replicas must be an integer")

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
        self.affinity_preferred = 0
        self.affinity_overflow = 0

    async def discover(self):
        async with self.discovery_lock:
            if time.monotonic() - self.refreshed >= self.config.discovery_interval_seconds:
                self.records = await self.registry.models(force=True)
                self.refreshed = time.monotonic()
        return self.records

    async def acquire(self, model: str, deadline: float, affinity_id: str | None = None, preferred_replica: str | None = None):
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
                    identities = set()
                    for record in records:
                        metadata = record.get("metadata", {})
                        identity = metadata.get("replica_id")
                        capacity = metadata.get("max_inflight_requests", 0)
                        if not identity or type(capacity) is not int or capacity <= 0:
                            continue
                        identities.add(identity)
                        capacity = min(capacity, self.config.max_inflight_per_replica)
                        active = self.active.get(identity, 0)
                        if active < capacity and self.cooldown.get(identity, 0) <= time.monotonic():
                            candidates.append((active / capacity, identity, record["uri"]))
                    # No await between the capacity check and reservation: all
                    # policy names share one atomic counter per replica process.
                    if first is ticket and candidates:
                        if preferred_replica:
                            lowest_load = min(c[0] for c in candidates)
                            preferred = [c for c in candidates if c[1] == preferred_replica
                                         and c[0] <= lowest_load + self.config.affinity_load_slack]
                            if preferred:
                                candidates = preferred
                                self.affinity_preferred += 1
                            else:
                                self.affinity_overflow += 1
                            random.shuffle(candidates)
                            _, identity, uri = min(candidates, key=lambda item: item[0])
                        elif affinity_id:
                            ranked = sorted(
                                identities,
                                key=lambda identity: hashlib.sha256(f"{affinity_id}:{identity}".encode()).digest(),
                                reverse=True,
                            )
                            preference = {identity: rank for rank, identity in enumerate(ranked)}
                            preferred = [c for c in candidates if preference[c[1]] < self.config.prompt_affinity_replicas]
                            if preferred:
                                candidates = preferred
                                self.affinity_preferred += 1
                            else:
                                self.affinity_overflow += 1
                            _, identity, uri = min(candidates, key=lambda item: (item[0], preference[item[1]]))
                        else:
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
        if "/" + request.endpoint.strip("/") not in GENERATION_PATHS:
            return await self.fallback.forward(request)
        affinity_id = next((v for k, v in request.headers.items() if k.lower() == "x-session-id"), None)
        preferred_replica = None
        if affinity_id and affinity_id.startswith("replica:"):
            preferred_replica = affinity_id.removeprefix("replica:")
            if not preferred_replica or len(preferred_replica) > 128:
                preferred_replica = None
            affinity_id = None
        if affinity_id is not None and (
            not affinity_id.startswith("prompt-sha256:") or len(affinity_id) != len("prompt-sha256:") + 64
            or any(char not in "0123456789abcdef" for char in affinity_id.removeprefix("prompt-sha256:"))
        ):
            affinity_id = None
        deadline = time.monotonic() + self.config.queue_timeout_seconds
        while True:
            identity, uri = await self.acquire(request.service, deadline, affinity_id, preferred_replica)
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

            if (request.endpoint.strip("/") == "inference/v1/generate"
                    and not request.payload.get("stream", False) and response.is_success):
                try:
                    await response.aread()
                    payload = response.json()
                    payload["litecast_replica_id"] = identity
                    self.completed += 1
                    return RoutingResponse(body=JSONResponse(payload, status_code=response.status_code))
                finally:
                    await asyncio.shield(cleanup())

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
            "affinity_preferred": self.affinity_preferred,
            "affinity_overflow": self.affinity_overflow,
        }


def create_app(registry=None, config=None):
    registry = registry or RegistryClient(get_kvstore(os.environ["REGISTRY_PATH"], raise_on_error=True), cache_ttl=0)
    routing = CapacityRouting(registry, config or CapacityConfig.load(os.getenv("LITECAST_CAPACITY_CONFIG")))
    proxy_routes = default_proxy_routes() + [
        ProxyRoute(
            "/judge", "judge", service_from_field("model_path", "input", "output", "rubrics", "model"), name="judge"
        ),
        ProxyRoute(
            "/inference/v1/generate", "inference/v1/generate", service_from_field("model"), name="token_generate"
        )
    ]
    routes = [
        replace(route, response_mapper=lambda body: body if isinstance(body, Response) else JSONResponse(body))
        if route.path in GENERATION_PATHS
        else route
        for route in proxy_routes
    ]
    app = registry_app(registry=registry, routing=routing, routes=routes, enable_registration=False)
    app.state.capacity_routing = routing
    app.add_middleware(DisconnectCancellationMiddleware)

    async def capacity_stats(request):
        return JSONResponse(routing.snapshot())

    app.add_route("/litecast/capacity", capacity_stats)
    return app
