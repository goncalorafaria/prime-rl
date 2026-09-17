"""LiteCast byte transport; LiteRegistry stores discovery metadata only."""

import asyncio
import hashlib
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import redis.asyncio as redis
from literegistry import RegistryClient, get_kvstore
from literegistry.registry import ServerRegistry
from redis.exceptions import RedisError

from litecast import OriginServer
from prime_rl.litecast.transfer import MeasuredClient
from prime_rl.litecast.protocol import Publication, desired_key, pack_adapter, peer_service

logger = logging.getLogger(__name__)


async def blocking(function, *args):
    """Drain a blocking operation before cancellation can release its buffers."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class Publisher:
    def __init__(self, config, base_model: str):
        self.config = config
        self.base_model = base_model
        self.store = get_kvstore(config.registry, raise_on_error=True)
        self.registry = RegistryClient(
            self.store, max_heartbeat_interval=config.lease_seconds, cache_ttl=config.poll_seconds
        )
        self.directory = tempfile.TemporaryDirectory(prefix="prime-litecast-origin-")
        self.origin = OriginServer(
            self.directory.name,
            port=config.origin_port,
            transport=config.transport,
            ram_cache_bytes=config.max_adapter_bytes * config.retain_versions,
            max_distribution_folders=config.retain_versions,
        )
        self.publications: list[Publication] = []
        self.sources: dict[str, tuple[ServerRegistry, str]] = {}
        self.fenced = False
        self.owner = uuid4().hex
        self.redis = redis.from_url(config.registry) if config.registry.startswith(("redis://", "rediss://")) else None
        if config.registry.startswith(("head+", "head://")):
            from prime_rl.litecast.bootstrap import HeadRedisCommands
            self.redis = HeadRedisCommands(self.store)
        self.task = None
        self.refresh_lock = asyncio.Lock()

    async def start(self):
        # Use a unique run_id for each concurrent experiment. A live record is
        # never overwritten by a second publisher, including on resume.
        if await self.store.get(desired_key(self.config.run_id)) is not None:
            raise ValueError("run_id already has a live publisher; use a new run_id or wait for its lease to expire")
        await self.refresh()
        self.task = asyncio.create_task(self.heartbeat())

    async def refresh(self):
        async with self.refresh_lock:
            payload = json.dumps(
                {
                    "schema": 1,
                    "owner": self.owner,
                    "base_model": self.base_model,
                    "publications": [p.to_dict() for p in self.publications],
                }
            )
            key = desired_key(self.config.run_id)
            if self.redis is not None:
                # Claim/renew and replace the descriptor atomically. A paused
                # publisher cannot overwrite a replacement after losing its lease.
                written = await self.redis.eval(
                    "local old = redis.call('GET', KEYS[1]); "
                    "if old and cjson.decode(old).owner ~= ARGV[1] then return 0 end; "
                    "redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3]); return 1",
                    1,
                    key,
                    self.owner,
                    payload,
                    max(1, int(self.config.lease_seconds * 1000)),
                )
                if not written:
                    self.fenced = True
                    raise RuntimeError("publisher lease belongs to another controller")
            else:
                # Filesystem registries are useful for local, single-controller tests.
                written = await self.store.set(key, payload, ttl_seconds=self.config.lease_seconds)
                if not written:
                    raise RuntimeError("could not publish adapter descriptor")

            for digest, (registration, version) in list(self.sources.items()):
                if version not in self.origin.store:
                    await registration.deregister()
                    del self.sources[digest]
                else:
                    await registration.heartbeat(f"http://{self.config.origin_host}", self.origin.port)

    async def heartbeat(self):
        while True:
            await asyncio.sleep(self.config.poll_seconds)
            try:
                await self.refresh()
            except Exception as exc:
                logger.warning("Registry publication lease refresh failed: %s", exc)
                if self.fenced:
                    return

    async def registry_retry(self, operation, *args, **kwargs):
        async with asyncio.timeout(getattr(self.config, "update_timeout", 600)):
            while True:
                try:
                    return await operation(*args, **kwargs)
                except (OSError, RedisError) as exc:
                    logger.warning("Waiting for registry recovery: %s", exc)
                    await asyncio.sleep(self.config.poll_seconds)

    async def publish(self, directory: Path, step: int) -> Publication:
        if self.fenced:
            raise RuntimeError("publisher was replaced by another controller")
        payload = await blocking(pack_adapter, directory, self.config.max_adapter_bytes)
        digest = hashlib.sha256(payload).hexdigest()
        if self.publications and step <= self.publications[-1].step:
            previous = self.publications[-1]
            if step == previous.step and digest == previous.digest:
                return previous
            raise ValueError("adapter steps must increase; use a new run_id when restarting from older weights")
        version = await blocking(self.origin.broadcast_buffer, payload, self.config.shard_bytes, False)
        publication = Publication(
            self.config.run_id,
            self.base_model,
            step,
            digest,
            len(payload),
        )
        async with self.refresh_lock:
            registration = ServerRegistry(self.store, max_heartbeat_interval=self.config.lease_seconds)
            await self.registry_retry(
                registration.register_server,
                f"http://{self.config.origin_host}",
                self.origin.port,
                {
                    "model_path": peer_service(self.config.run_id, digest),
                    "run_id": self.config.run_id,
                    "digest": digest,
                    "source_role": "origin",
                    "litecast_version": version,
                },
            )
            if digest in self.sources:
                await self.sources[digest][0].deregister()
            self.sources[digest] = (registration, version)
            self.publications = (self.publications + [publication])[-self.config.retain_versions :]
        await self.registry_retry(self.refresh)
        return publication

    async def wait_ready(self, model_name: str, timeout: float, force: bool = True):
        async with asyncio.timeout(timeout):
            while True:
                if self.fenced:
                    raise RuntimeError("publisher was replaced by another controller")
                ready = await self.registry_retry(self.registry.get_all, model_name, force=force)
                if len(ready) >= self.config.min_replicas:
                    return
                await asyncio.sleep(self.config.poll_seconds)

    async def transfer_metrics(self, publication: Publication) -> dict[str, float]:
        records = (await self.registry_retry(self.registry.models, force=True)).get(publication.model_name, [])
        samples = [
            r["metadata"]["litecast_transfer"]
            for r in records
            if r.get("metadata", {}).get("digest") == publication.digest
            and "litecast_transfer" in r.get("metadata", {})
        ]
        metrics = {"litecast/measured_replicas": float(len(samples))}
        for field in ("fetch_seconds", "load_seconds", "discovery_seconds", "manifest_seconds",
                      "download_seconds", "relay_wait_seconds", "cached_metadata_seconds", "publication_verify_seconds", "http_attempt_seconds_sum",
                      "http_failed_seconds_sum", "retry_gap_seconds_sum", "http_attempts", "http_retries",
                      "http_failures", "http_bytes", "sources_used", "source_group_size", "source_group_failures"):
            values = [sample[field] for sample in samples if field in sample]
            if values:
                metrics[f"litecast/{field}_mean"] = sum(values) / len(values)
                metrics[f"litecast/{field}_max"] = max(values)
        return metrics

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        results = await asyncio.gather(
            *(registration.deregister() for registration, _ in self.sources.values()), return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Origin deregistration failed; its heartbeat will expire: %s", result)
        self.sources.clear()
        # Let the lease expire rather than deleting a possible replacement's record.
        await asyncio.to_thread(self.origin.shutdown)
        self.directory.cleanup()
        await self.registry.close()
        if self.redis is not None:
            await self.redis.aclose()


async def fetch_publication(
    publication: Publication, registry, output_dir: Path, transport: str, source_role: str | None = None,
    metrics: dict | None = None,
) -> bytes:
    started = time.perf_counter()
    records = (await registry.models(force=True)).get(peer_service(publication.run_id, publication.digest), [])
    discovery_seconds = time.perf_counter() - started
    relay_wait_started = time.perf_counter()
    relay_deadline = relay_wait_started + max(0, float(os.getenv('LITECAST_RELAY_WAIT_SECONDS', '0')))
    while (records and source_role is None
           and not any(r.get('metadata', {}).get('source_role') != 'origin' for r in records)
           and time.perf_counter() < relay_deadline):
        await asyncio.sleep(min(0.25, max(0, relay_deadline-time.perf_counter())))
        records = (await registry.models(force=True)).get(peer_service(publication.run_id, publication.digest), [])
    relay_wait_seconds = time.perf_counter() - relay_wait_started
    if source_role is not None:
        records = [r for r in records if r.get("metadata", {}).get("source_role") == source_role]
    random.shuffle(records)
    groups = {}
    for record in records:
        metadata = record.get('metadata', {})
        server, version = record.get('uri'), metadata.get('litecast_version')
        if server and version:
            # A peer rebroadcast can use a different local version name.
            key = (metadata.get('source_role', 'peer'), version)
            groups.setdefault(key, set()).add(server)
    candidates = sorted(groups.items(), key=lambda item: (
        {'middle': 0, 'peer': 1, 'origin': 2}.get(item[0][0], 1), -len(item[1])))
    totals = {'discovery_seconds': discovery_seconds, 'relay_wait_seconds': relay_wait_seconds, 'source_candidates': len(records),
              'source_group_failures': 0, 'publication_verify_seconds': 0.0}
    for (role, version), servers in candidates:
        client = MeasuredClient(sorted(servers), str(output_dir), transport=transport,
                                max_version_bytes=publication.size)
        logger.info('LITECAST_TRANSFER_SOURCES model=%s role=%s version=%s sources=%s',
                    publication.model_name, role, version, sorted(servers))
        try:
            payload = await blocking(client.download_version_buffer, version)
            if payload is None:
                raise RuntimeError('shard download or checksum verification failed')
            payload = bytes(payload)
            verify_started = time.perf_counter()
            try:
                publication.verify(payload)
            finally:
                totals['publication_verify_seconds'] += time.perf_counter() - verify_started
            if metrics is not None:
                metrics.update(source_group_size=len(servers))
            return payload
        except (OSError, ValueError, RuntimeError) as exc:
            totals['source_group_failures'] += 1
            logger.warning('LiteCast sources %s failed: %s', sorted(servers), exc)
        finally:
            sample = client.measurements()
            for key, value in sample.items():
                if key == 'sources_used':
                    totals[key] = max(totals.get(key, 0), value)
                else:
                    totals[key] = totals.get(key, 0) + value
            client.close()
            totals['fetch_seconds'] = time.perf_counter() - started
            logger.info('LITECAST_FETCH_TIMING model=%s metrics=%s', publication.model_name, totals)
            if metrics is not None:
                metrics.update(totals)
    raise RuntimeError(f"no valid LiteCast source for {publication.model_name}")
