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
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from literegistry import RegistryClient, get_kvstore
from literegistry.registry import ServerRegistry

from litecast import OriginServer
from prime_rl.litecast.distribution import blocking, fetch_publication
from prime_rl.litecast.protocol import (
    Publication,
    base_alias,
    desired_key,
    peer_service,
    split_adapter,
    validate_run_id,
)
from prime_rl.litecast.responses import ManagedStreamingResponse

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
        response = await self.backend.get("/health")
        response.raise_for_status()
        model_response = await self.backend.get("/v1/models")
        model_response.raise_for_status()
        backend_models = {item["id"] for item in model_response.json()["data"]}
        if self.args.base_model not in backend_models:
            raise ValueError("backend does not serve the configured base model")
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
                payload = await fetch_publication(
                    publication,
                    self.registry,
                    self.root,
                    self.args.transport,
                    source_role="middle" if getattr(self.args, "require_middle", False) else None,
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
                logger.exception("Replica reconciliation failed; retrying")
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


def create_app(worker: Worker):
    @asynccontextmanager
    async def lifespan(app):
        worker.task = asyncio.create_task(worker.run())
        try:
            yield
        finally:
            await worker.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not worker.available(base_alias(worker.args.run_id)):
            raise HTTPException(503, "replica is not ready")
        return {"status": "ready", "models": list(worker.serving)}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": name, "object": "model"} for name in worker.serving if worker.available(name)],
        }

    @app.post("/v1/{endpoint:path}")
    async def proxy(endpoint: str, request: Request):
        if endpoint not in ("completions", "chat/completions"):
            raise HTTPException(404)
        payload = await request.json()
        name = payload.get("model")
        if not isinstance(name, str) or not worker.available(name):
            raise HTTPException(
                503,
                "requested policy version is not ready on this replica",
                headers={"x-litecast-admission-rejected": "1"},
            )
        if sum(worker.inflight.values()) >= worker.max_inflight_requests:
            raise HTTPException(
                429, "replica capacity exhausted", headers={"x-litecast-admission-rejected": "1", "Retry-After": "1"}
            )
        if name == base_alias(worker.args.run_id):
            payload["model"] = worker.args.base_model
        worker.inflight[name] = worker.inflight.get(name, 0) + 1
        try:
            upstream = await worker.backend.send(
                worker.backend.build_request("POST", f"/v1/{endpoint}", json=payload),
                stream=True,
            )
        except BaseException:
            worker.inflight[name] -= 1
            raise

        async def cleanup():
            try:
                await upstream.aclose()
            finally:
                worker.inflight[name] -= 1

        return ManagedStreamingResponse(
            upstream.aiter_bytes(),
            cleanup=cleanup,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=os.getenv("REGISTRY"), required=not os.getenv("REGISTRY"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model", required=True)
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
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(Worker(args)), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
