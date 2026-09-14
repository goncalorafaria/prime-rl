"""Real Redis discovery and HTTP transfers through two CPU middle nodes."""

import asyncio
import os
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
import redis.asyncio as redis
from litecast import MiddleNode
from literegistry import RegistryClient, get_kvstore
from literegistry.registry import ServerRegistry
from safetensors.numpy import save_file

from prime_rl.litecast.distribution import Publisher, blocking, fetch_publication
from prime_rl.litecast.protocol import peer_service


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_two_middles_registry_transfer_and_loss(tmp_path):
    binary = os.getenv("LITECAST_TEST_REDIS_SERVER") or shutil.which("redis-server")
    if binary is None:
        pytest.skip("redis-server is required for the two-middle smoke")
    process = subprocess.Popen(
        [binary, "--bind", "127.0.0.1", "--port", str(port := free_port()), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    url = f"redis://127.0.0.1:{port}/0"
    control = redis.from_url(url)
    registry = RegistryClient(get_kvstore(url, raise_on_error=True), cache_ttl=0, max_heartbeat_interval=30)
    publisher = None
    middles = []
    registrations = []
    started = time.monotonic()
    try:
        async with asyncio.timeout(5):
            while True:
                try:
                    await control.ping()
                    break
                except redis.ConnectionError:
                    await asyncio.sleep(0.02)
        config = SimpleNamespace(
            registry=url,
            run_id="two-middle-smoke",
            origin_host="127.0.0.1",
            origin_port=0,
            transport="http",
            max_adapter_bytes=2 * 1024 * 1024,
            retain_versions=2,
            poll_seconds=0.1,
            lease_seconds=30,
            shard_bytes=64 * 1024,
            min_replicas=1,
        )
        publisher = Publisher(config, "cpu-adapter-fixture")
        await publisher.start()
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text('{"peft_type":"LORA","r":8,"lora_alpha":16}')
        save_file({"lora_A.weight": np.arange(262144, dtype=np.float32)}, str(adapter / "adapter_model.safetensors"))
        publication = await publisher.publish(adapter, 1)
        service = peer_service(config.run_id, publication.digest)
        origins = (await registry.models(force=True))[service]
        assert len(origins) == 1 and origins[0]["metadata"]["source_role"] == "origin"
        version = origins[0]["metadata"]["litecast_version"]
        for index in range(2):
            middle = MiddleNode(
                [record["uri"] for record in origins],
                str(tmp_path / f"middle-{index}"),
                port=free_port(),
                check_interval=1,
                transport="http",
                ram_cache_bytes=4 * 1024 * 1024,
                disk_mirror=False,
            )
            middles.append(middle)
        async with asyncio.timeout(20):
            while not all(version in middle.processed_versions for middle in middles):
                await asyncio.sleep(0.05)
        for middle in middles:
            registration = ServerRegistry(registry.store)
            registrations.append(registration)
            await registration.register_server(
                "http://127.0.0.1",
                middle.port,
                {
                    "model_path": service,
                    "source_role": "middle",
                    "litecast_version": version,
                    "run_id": config.run_id,
                    "digest": publication.digest,
                },
            )
            await registration.heartbeat("http://127.0.0.1", middle.port)

        # Shut down the origin, rather than merely relying on source preference.
        await publisher.close()
        publisher = None
        records = (await registry.models(force=True))[service]
        assert len(records) == 2 and all(r["metadata"]["source_role"] == "middle" for r in records)
        first = await fetch_publication(publication, registry, tmp_path / "client-first", "http")
        publication.verify(first)

        # Stop the first middle and withdraw it; the surviving cache must suffice.
        await blocking(middles[0].shutdown)
        middles.pop(0)
        await registrations[0].deregister()
        registrations.pop(0)
        records = (await registry.models(force=True))[service]
        assert len(records) == 1
        second = await fetch_publication(publication, registry, tmp_path / "client-survivor", "http")
        assert second == first
        publication.verify(second)

        await blocking(middles[0].shutdown)
        middles.pop(0)
        await registrations[0].deregister()
        registrations.pop(0)
        with pytest.raises(RuntimeError, match="no valid LiteCast source"):
            await fetch_publication(publication, registry, tmp_path / "client-empty", "http")
        print(
            f"\nTwo-middle smoke: {publication.size} bytes, origin offline, survivor verified, "
            f"empty registry rejected; {time.monotonic() - started:.2f}s"
        )
    finally:
        for middle in middles:
            await blocking(middle.shutdown)
        for registration in registrations:
            await registration.deregister()
        if publisher is not None:
            await publisher.close()
        await registry.close()
        await control.aclose()
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.asyncio
async def test_registry_middle_supervisors_publish_ready_sources(tmp_path):
    import sys

    config = SimpleNamespace(
        registry=f"file://{tmp_path / 'registry'}",
        run_id="supervised",
        origin_host="127.0.0.1",
        origin_port=0,
        transport="http",
        max_adapter_bytes=4096,
        retain_versions=2,
        poll_seconds=0.1,
        lease_seconds=30,
        shard_bytes=64,
        min_replicas=1,
    )
    publisher = Publisher(config, "fixture")
    processes = []
    logs = []
    try:
        await publisher.start()
        for index in range(2):
            log = (tmp_path / f"middle-{index}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        "uv",
                        "run",
                        "--no-project",
                        "--python",
                        sys.executable,
                        "python",
                        "-m",
                        "prime_rl.litecast.middle",
                        "--registry",
                        config.registry,
                        "--run-id",
                        config.run_id,
                        "--advertise-host",
                        "127.0.0.1",
                        "--port",
                        str(free_port()),
                        "--max-adapter-bytes",
                        "4096",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        (tmp_path / "adapter_config.json").write_text('{"peft_type":"LORA","r":8}')
        (tmp_path / "adapter_model.safetensors").write_bytes(b"supervised-adapter")
        publication = await publisher.publish(tmp_path, 1)
        service = peer_service(config.run_id, publication.digest)
        # Supervisor imports may cold-start from a shared filesystem.
        async with asyncio.timeout(120):
            while True:
                assert all(p.poll() is None for p in processes), [p.returncode for p in processes]
                records = (await publisher.registry.models(force=True)).get(service, [])
                if sum(r["metadata"].get("source_role") == "middle" for r in records) == 2:
                    break
                await asyncio.sleep(0.1)
        data = await fetch_publication(
            publication, publisher.registry, tmp_path / "download", "http", source_role="middle"
        )
        publication.verify(data)
        for process in processes:
            process.terminate()
            await blocking(process.wait, 10)
        with pytest.raises(RuntimeError, match="no valid LiteCast source"):
            # The live origin must not satisfy a middle-only request.
            await fetch_publication(
                publication, publisher.registry, tmp_path / "no-middle", "http", source_role="middle"
            )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                await blocking(process.wait, 10)
        for log in logs:
            log.close()
        await publisher.close()
