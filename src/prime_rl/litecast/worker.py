"""Standalone inference sidecar and version-checking gateway target."""

import argparse
import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from copy import copy
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from literegistry import RegistryClient, get_kvstore
from literegistry.registry import ServerRegistry
from starlette.responses import JSONResponse

from litecast import OriginServer
from prime_rl.litecast import inflight as live_updates
from prime_rl.litecast.distribution import blocking, fetch_publication
from prime_rl.litecast.protocol import (
    Publication,
    base_alias,
    desired_key,
    peer_service,
    split_adapter,
    validate_run_id,
)
from prime_rl.litecast.responses import GENERATION_PATHS, DisconnectCancellationMiddleware, ManagedStreamingResponse

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, args):
        self.args = args
        self.replica_id = uuid4().hex
        self.max_inflight_requests = getattr(args, "max_inflight_requests", 4)
        if self.max_inflight_requests < 1:
            raise ValueError("max_inflight_requests must be positive")
        validate_run_id(args.run_id)
        self.store = get_kvstore(args.registry, raise_on_error=True)
        self.registry = RegistryClient(self.store, cache_ttl=0, max_heartbeat_interval=args.lease_seconds)
        self.backend = httpx.AsyncClient(base_url=args.backend_url.rstrip("/"), timeout=args.request_timeout)
        self.root = Path(tempfile.mkdtemp(prefix="replica-", dir=args.cache_dir))
        self.origin = OriginServer(
            str(self.root / "shards"),
            port=args.shard_port,
            transport=args.transport,
            ram_cache_bytes=args.max_adapter_bytes * args.max_versions,
            max_distribution_folders=args.max_versions,
        )
        self.loaded: dict[str, Path] = {}
        self.transfer_metrics: dict[str, dict[str, float]] = {}
        self.serving: dict[str, ServerRegistry] = {}
        self.peers: dict[str, tuple[ServerRegistry, str]] = {}
        self.inflight: dict[str, int] = {}
        self.last_healthy = float("-inf")
        self.backend_seen_healthy = False
        self.backend_unavailable_since = None
        self.backend_last_warning = float("-inf")
        self.task = None

    def available(self, model_name):
        return model_name in self.serving and time.monotonic() - self.last_healthy < self.args.lease_seconds

    async def register(self, model_name, port, **metadata):
        registration = ServerRegistry(self.store)
        await registration.register_server(
            f"http://{self.args.advertise_host}",
            port,
            {
                "model_path": model_name,
                "run_id": self.args.run_id,
                "replica_id": self.replica_id,
                "max_inflight_requests": self.max_inflight_requests,
                **metadata,
            },
        )
        return registration

    async def reconcile(self):
        raw = await self.store.get(desired_key(self.args.run_id))
        if raw is None:
            self.last_healthy = float("-inf")
            await self.withdraw()
            return
        desired = json.loads(raw)
        if desired.get("schema") != 1 or desired.get("base_model") != self.args.base_model:
            raise ValueError("publisher and worker base model/schema differ")
        publications = [Publication(**item) for item in desired["publications"]]
        if len(publications) > self.args.max_versions:
            raise ValueError("worker max_versions must cover publisher retain_versions")
        if any(
            p.run_id != self.args.run_id or p.base_model != self.args.base_model or p.size > self.args.max_adapter_bytes
            for p in publications
        ):
            raise ValueError("invalid publication for this worker")
        try:
            response = await self.backend.get("/health")
            response.raise_for_status()
            model_response = await self.backend.get("/v1/models")
            model_response.raise_for_status()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            self.last_healthy = float("-inf")
            now = time.monotonic()
            if self.backend_unavailable_since is None:
                self.backend_unavailable_since = now
                self.backend_last_warning = float("-inf")
            elapsed = now - self.backend_unavailable_since
            if now - self.backend_last_warning >= 30:
                log = logger.warning if self.backend_seen_healthy or elapsed >= 60 else logger.info
                log(
                    "LITECAST_BACKEND_%s run_id=%s endpoint=%s elapsed_seconds=%.1f error=%s; retrying",
                    "UNAVAILABLE" if self.backend_seen_healthy else "STARTING",
                    self.args.run_id,
                    exc.request.url,
                    elapsed,
                    str(exc) or type(exc).__name__,
                )
                self.backend_last_warning = now
            await self.withdraw()
            return
        backend_models = {item["id"] for item in model_response.json()["data"]}
        if self.args.base_model not in backend_models:
            raise ValueError("backend does not serve the configured base model")
        if not self.backend_seen_healthy or self.backend_unavailable_since is not None:
            logger.info(
                "LITECAST_BACKEND_READY run_id=%s endpoint=%s wait_seconds=%.1f",
                self.args.run_id,
                self.args.backend_url,
                0.0 if self.backend_unavailable_since is None else time.monotonic() - self.backend_unavailable_since,
            )
        self.backend_seen_healthy = True
        self.backend_unavailable_since = None
        if live_updates.enabled():
            await live_updates.reconcile(self, publications, backend_models)
            return
        # A vLLM process can restart while the sidecar survives. Forget all
        # readiness before attempting to reload any adapter missing upstream.
        for name in list(self.loaded):
            if name not in backend_models:
                if name in self.serving:
                    await self.serving.pop(name).deregister()
                shutil.rmtree(self.loaded.pop(name))
                self.transfer_metrics.pop(name, None)
        keep = {p.model_name for p in publications} | {base_alias(self.args.run_id)}
        for name in list(self.serving):
            if name not in keep:
                await self.serving.pop(name).deregister()
        for name in list(self.loaded):
            if name not in keep and not self.inflight.get(name, 0):
                response = await self.backend.post("/v1/unload_lora_adapter", json={"lora_name": name})
                response.raise_for_status()
                shutil.rmtree(self.loaded.pop(name))
                self.transfer_metrics.pop(name, None)
        self.last_healthy = time.monotonic()
        base = base_alias(self.args.run_id)
        if base not in self.serving:
            self.serving[base] = await self.register(base, self.args.port, base_model=self.args.base_model)
        # Load newest first so replacement workers contribute to the current
        # policy without replaying every historical training update first.
        for publication in reversed(publications):
            name = publication.model_name
            if name not in self.loaded:
                if len(self.loaded) >= self.args.max_versions:
                    break  # A retired version still has active requests.
                if name in backend_models:
                    # Recover a load that completed upstream after its HTTP
                    # response was lost. This name has not been advertised here.
                    response = await self.backend.post("/v1/unload_lora_adapter", json={"lora_name": name})
                    response.raise_for_status()
                fetch_started = time.perf_counter()
                transfer_details = {}
                payload = await fetch_publication(
                    publication,
                    self.registry,
                    self.root,
                    self.args.transport,
                    source_role="middle" if getattr(self.args, "require_middle", False) else None,
                    metrics=transfer_details,
                )
                fetch_seconds = time.perf_counter() - fetch_started
                load_started = time.perf_counter()
                config, weights = split_adapter(payload)
                staging = Path(tempfile.mkdtemp(prefix="adapter-", dir=self.root))
                (staging / "adapter_config.json").write_bytes(config)
                (staging / "adapter_model.safetensors").write_bytes(weights)
                try:
                    response = await self.backend.post(
                        "/v1/load_lora_adapter",
                        json={
                            "lora_name": name,
                            "lora_path": str(staging),
                            "load_inplace": False,
                        },
                    )
                    response.raise_for_status()
                except BaseException:
                    shutil.rmtree(staging)
                    raise
                self.transfer_metrics[name] = {
                    **transfer_details,
                    "fetch_seconds": fetch_seconds,
                    "load_seconds": time.perf_counter() - load_started,
                    "payload_bytes": publication.size,
                }
                logger.info("LITECAST_TRANSFER model=%s metrics=%s", name, self.transfer_metrics[name])
                self.loaded[name] = staging
                version = await blocking(self.origin.broadcast_buffer, payload, self.args.shard_bytes, False)
                service = peer_service(self.args.run_id, publication.digest)
                if service in self.peers:
                    await self.peers.pop(service)[0].deregister()
                self.peers[service] = (
                    await self.register(
                        service,
                        self.origin.port,
                        litecast_version=version,
                        source_role="peer",
                        digest=publication.digest,
                    ),
                    version,
                )
            if name not in self.serving:
                self.serving[name] = await self.register(
                    name,
                    self.args.port,
                    step=publication.step,
                    digest=publication.digest,
                    litecast_transfer=self.transfer_metrics[name],
                )
        for service, (registration, version) in list(self.peers.items()):
            if version not in self.origin.store:
                await registration.deregister()
                del self.peers[service]

    async def heartbeat(self):
        if not self.available(base_alias(self.args.run_id)):
            return
        for registration in list(self.serving.values()):
            await registration.heartbeat(f"http://{self.args.advertise_host}", self.args.port)
        for registration, _ in list(self.peers.values()):
            await registration.heartbeat(f"http://{self.args.advertise_host}", self.origin.port)

    async def run(self):
        while True:
            try:
                await self.reconcile()
                await self.heartbeat()
            except Exception:
                # Preemption, registry outage and an incomplete download are
                # retryable; do not advertise a worker whose health is unknown.
                self.last_healthy = float("-inf")
                logger.exception("Replica reconciliation failed run_id=%s; retrying", self.args.run_id)
                try:
                    await self.withdraw()
                except Exception:
                    logger.exception("Could not withdraw registry records; leases will expire")
            await asyncio.sleep(self.args.poll_seconds)

    async def withdraw(self):
        for name in list(self.serving):
            await self.serving.pop(name).deregister()
        for service in list(self.peers):
            await self.peers.pop(service)[0].deregister()

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        try:
            await self.withdraw()
        finally:
            await self.backend.aclose()
            await self.registry.close()
            await asyncio.to_thread(self.origin.shutdown)
            shutil.rmtree(self.root)


class TenantPool:
    """Explicit trainer subscriptions sharing one admission budget and HTTP endpoint."""

    def __init__(self, workers, subscriptions=None):
        if not workers or len({w.args.run_id for w in workers}) != len(workers):
            raise ValueError("tenant run IDs must be nonempty and unique")
        backends = {}
        for worker in workers:
            url = worker.args.backend_url.rstrip("/")
            if url in backends and backends[url] != worker.args.base_model:
                raise ValueError("different base models require different backend URLs")
            backends[url] = worker.args.base_model
        self.workers = workers
        self.subscriptions = subscriptions
        self.max_inflight_requests = workers[0].max_inflight_requests
        self.inflight = {}
        self.replica_id = uuid4().hex
        self.task = None
        for worker in workers:
            worker.inflight = self.inflight
            worker.replica_id = self.replica_id
            worker.max_inflight_requests = self.max_inflight_requests

    @property
    def serving(self):
        return {name: registration for worker in self.workers for name, registration in worker.serving.items()}

    def owner(self, name):
        return next((worker for worker in self.workers if worker.available(name)), None)

    def available(self, name):
        return self.owner(name) is not None

    async def run(self):
        for worker in self.workers:
            worker.task = asyncio.create_task(worker.run())
        last_error = None
        try:
            while True:
                if self.subscriptions:
                    try:
                        subscriptions = tenant_args(self.subscriptions)
                        existing = {w.args.run_id: w for w in self.workers}
                        incoming = {a.run_id: a for a in subscriptions}
                        fields = ("base_model", "backend_url", "shard_port")
                        if any(
                            run not in incoming or any(getattr(w.args, f) != getattr(incoming[run], f) for f in fields)
                            for run, w in existing.items()
                        ):
                            raise ValueError(
                                "live tenant reload can only add subscriptions; existing tenants must stay unchanged"
                            )
                        for args in subscriptions:
                            if args.run_id not in existing:
                                worker = Worker(args)
                                worker.inflight = self.inflight
                                worker.replica_id = self.replica_id
                                worker.max_inflight_requests = self.max_inflight_requests
                                self.workers.append(worker)
                                worker.task = asyncio.create_task(worker.run())
                                logger.info("LITECAST_TENANT_ATTACHED run=%s replica=%s", args.run_id, self.replica_id)
                        last_error = None
                    except (OSError, ValueError) as exc:
                        if str(exc) != last_error:
                            logger.warning("LITECAST_TENANTS_REJECTED keeping_previous=true error=%s", exc)
                            last_error = str(exc)
                for worker in self.workers:
                    if worker.task.done():
                        await worker.task
                        raise RuntimeError(f"tenant worker stopped: {worker.args.run_id}")
                await asyncio.sleep(self.workers[0].args.poll_seconds)
        finally:
            for worker in self.workers:
                worker.task.cancel()
            await asyncio.gather(*(w.task for w in self.workers), return_exceptions=True)

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        results = await asyncio.gather(*(worker.close() for worker in self.workers), return_exceptions=True)
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("tenant cleanup failed", errors)


def tenant_args(args):
    """Validate all subscriptions before starting any shard servers."""
    if not args.tenants:
        if not args.run_id or not args.base_model:
            raise ValueError("provide --run-id and --base-model, or --tenants")
        return [args]
    if args.run_id or args.base_model:
        raise ValueError("--tenants cannot be combined with --run-id or --base-model")
    items = json.loads(args.tenants.read_text())
    if not isinstance(items, list) or not items:
        raise ValueError("tenants must be a nonempty JSON list")
    if len(items) > getattr(args, "max_tenants", 8):
        raise ValueError("tenant count exceeds max-tenants")
    subscriptions = []
    runs, ports, backends = set(), set(), {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"run_id", "base_model", "backend_url", "shard_port"}:
            raise ValueError("each tenant requires run_id, base_model, backend_url, shard_port")
        validate_run_id(item["run_id"])
        if not isinstance(item["base_model"], str) or not item["base_model"]:
            raise ValueError("tenant base_model must be nonempty")
        url = item["backend_url"]
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError("tenant backend_url must be an HTTP URL")
        url = url.rstrip("/")
        port = item["shard_port"]
        if type(port) is not int or not 1 <= port <= 65535 or port == args.port or port in ports:
            raise ValueError("tenant shard ports must be valid, unique and different from the serving port")
        if item["run_id"] in runs:
            raise ValueError("tenant run IDs must be unique")
        if url in backends and backends[url] != item["base_model"]:
            raise ValueError("different base models require different backend URLs")
        runs.add(item["run_id"])
        ports.add(port)
        backends[url] = item["base_model"]
        subscription = copy(args)
        for key, value in item.items():
            setattr(subscription, key, value)
        subscriptions.append(subscription)
    return subscriptions


def create_app(worker: Worker | TenantPool):
    @asynccontextmanager
    async def lifespan(app):
        worker.task = asyncio.create_task(worker.run())
        try:
            yield
        finally:
            await worker.close()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(DisconnectCancellationMiddleware)

    @app.get("/health")
    async def health():
        if not any(worker.available(name) for name in worker.serving):
            raise HTTPException(503, "replica is not ready")
        return {"status": "ready", "models": list(worker.serving)}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": name, "object": "model"} for name in worker.serving if worker.available(name)],
        }

    async def forward(endpoint: str, request: Request):
        if endpoint not in GENERATION_PATHS:
            raise HTTPException(404)
        payload = await request.json()
        name = payload.get("model")
        owner = None
        if isinstance(name, str):
            owner = worker.owner(name) if isinstance(worker, TenantPool) else worker if worker.available(name) else None
        if owner is None:
            raise HTTPException(
                503,
                "requested policy version is not ready on this replica",
                headers={"x-litecast-admission-rejected": "1"},
            )
        if sum(worker.inflight.values()) >= worker.max_inflight_requests:
            raise HTTPException(
                429, "replica capacity exhausted", headers={"x-litecast-admission-rejected": "1", "Retry-After": "1"}
            )
        if name == base_alias(owner.args.run_id):
            payload["model"] = owner.args.base_model
        if live_updates.enabled() and name != base_alias(owner.args.run_id):
            if not name.endswith("-live") or endpoint != "/inference/v1/generate" or payload.get("stream"):
                raise HTTPException(
                    400, "Retained-state updates require a live alias and non-streaming token generation"
                )
            payload["model"] = base_alias(owner.args.run_id) + "-live"
        worker.inflight[name] = worker.inflight.get(name, 0) + 1
        try:
            upstream = await owner.backend.send(
                owner.backend.build_request("POST", endpoint, json=payload),
                stream=True,
            )
        except httpx.TimeoutException as exc:
            worker.inflight[name] -= 1
            logger.warning("LITECAST_GENERATION_TIMEOUT replica=%s model=%s", owner.replica_id, name)
            raise HTTPException(504, "inference backend timed out") from exc
        except BaseException:
            worker.inflight[name] -= 1
            raise

        async def cleanup():
            try:
                await upstream.aclose()
            finally:
                worker.inflight[name] -= 1

        if live_updates.enabled() and name != base_alias(owner.args.run_id):
            try:
                await upstream.aread()
                if upstream.is_error:
                    return JSONResponse(upstream.json(), status_code=upstream.status_code)
                return JSONResponse(live_updates.record_response(owner, upstream.json(), name))
            finally:
                await cleanup()

        return ManagedStreamingResponse(
            upstream.aiter_bytes(),
            cleanup=cleanup,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    @app.post("/v1/{endpoint:path}")
    async def proxy(endpoint: str, request: Request):
        return await forward(f"/v1/{endpoint}", request)

    @app.post("/inference/v1/generate")
    async def generate(request: Request):
        return await forward("/inference/v1/generate", request)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=os.getenv("REGISTRY"), required=not os.getenv("REGISTRY"))
    parser.add_argument("--run-id")
    parser.add_argument("--tenants", type=Path, help="Reloadable JSON trainer subscriptions; supports live additions")
    parser.add_argument("--max-tenants", type=int, default=8)
    parser.add_argument("--base-model")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--shard-port", type=int, default=8101)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--max-inflight-requests", type=int, default=4)
    parser.add_argument("--max-versions", type=int, default=8)
    parser.add_argument("--max-adapter-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--shard-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--lease-seconds", type=float, default=30)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--require-middle", action="store_true")
    parser.add_argument("--transport", choices=("http", "auto", "ucxx"), default="http")
    args = parser.parse_args()
    if (
        min(
            args.max_inflight_requests,
            args.max_tenants,
            args.max_versions,
            args.max_adapter_bytes,
            args.shard_bytes,
            args.poll_seconds,
            args.request_timeout,
        )
        <= 0
    ):
        parser.error("sizes, limits, and intervals must be positive")
    if args.lease_seconds <= 2 * args.poll_seconds:
        parser.error("lease-seconds must exceed twice poll-seconds")
    try:
        subscriptions = tenant_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    workers = [Worker(subscription) for subscription in subscriptions]
    worker = TenantPool(workers, args) if args.tenants else workers[0]
    uvicorn.run(create_app(worker), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
