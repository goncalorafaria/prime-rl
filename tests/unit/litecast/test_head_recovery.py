"""A replacement Redis head preserves state behind an unchanged client endpoint."""

import asyncio
import os
import shutil
import socket
import subprocess

import pytest
import redis.asyncio as redis
from literegistry.coop.endpoints import publish

from prime_rl.litecast.relay import forward


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_head_replacement_preserves_state_and_relay_address(tmp_path):
    binary = os.getenv("LITECAST_TEST_REDIS_SERVER") or shutil.which("redis-server")
    if binary is None:
        pytest.skip("redis-server required")
    bootstrap = f"sqlite://{tmp_path / 'bootstrap.sqlite3'}"
    relay = await asyncio.start_server(lambda r, w: forward(r, w, bootstrap, "redis-backend"), "127.0.0.1", 0)
    relay_port = relay.sockets[0].getsockname()[1]
    backend = None
    client = redis.from_url(f"redis://127.0.0.1:{relay_port}", socket_timeout=2)
    try:
        for attempt in range(2):
            port = free_port()
            backend = subprocess.Popen(
                [
                    binary,
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--dir",
                    str(tmp_path),
                    "--save",
                    "",
                    "--appendonly",
                    "yes",
                    "--appendfsync",
                    "always",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            direct = redis.from_url(f"redis://127.0.0.1:{port}")
            try:
                async with asyncio.timeout(10):
                    while True:
                        try:
                            await direct.ping()
                            break
                        except redis.ConnectionError:
                            await asyncio.sleep(0.05)
            finally:
                await direct.aclose()
            await asyncio.to_thread(
                publish, bootstrap, "redis-backend", f"redis://127.0.0.1:{port}", publisher_id="head", ttl_seconds=30
            )
            if attempt == 0:
                await client.set("adapter-owner", "same-publisher")
                backend.kill()
                backend.wait(timeout=5)
                backend = None
                # Drop the dead TCP connection; the URL and client stay unchanged.
                await client.connection_pool.disconnect()
            else:
                assert await client.get("adapter-owner") == b"same-publisher"
                await client.set("next-adapter", "step-2")
                assert await client.get("next-adapter") == b"step-2"
    finally:
        await client.aclose()
        relay.close()
        await relay.wait_closed()
        if backend is not None:
            backend.terminate()
            backend.wait(timeout=5)
