import asyncio
import os
import shutil
import socket
import subprocess
from types import SimpleNamespace

import pytest
import redis.asyncio as redis

from prime_rl.litecast.distribution import Publisher


@pytest.mark.asyncio
async def test_redis_fences_a_replaced_publisher(tmp_path):
    binary = os.getenv("LITECAST_TEST_REDIS_SERVER") or shutil.which("redis-server")
    if binary is None:
        pytest.skip("redis-server is required for the real Redis lease test")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [binary, "--bind", "127.0.0.1", "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    url = f"redis://127.0.0.1:{port}/0"
    client = redis.from_url(url)
    publishers = []
    try:
        async with asyncio.timeout(5):
            while True:
                try:
                    await client.ping()
                    break
                except redis.ConnectionError:
                    await asyncio.sleep(0.02)
        config = SimpleNamespace(
            registry=url,
            run_id="fenced",
            origin_host="127.0.0.1",
            origin_port=0,
            transport="http",
            max_adapter_bytes=1024,
            retain_versions=2,
            poll_seconds=0.05,
            lease_seconds=0.2,
            shard_bytes=64,
            min_replicas=1,
        )
        first, replacement = Publisher(config, "model"), Publisher(config, "model")
        publishers.extend([first, replacement])
        await first.start()
        with pytest.raises(ValueError, match="live publisher"):
            await replacement.start()
        first.task.cancel()
        await asyncio.gather(first.task, return_exceptions=True)
        await asyncio.sleep(0.25)
        await replacement.start()
        with pytest.raises(RuntimeError, match="another controller"):
            await first.refresh()
        with pytest.raises(RuntimeError, match="replaced"):
            await first.wait_ready("any-model", 1)
    finally:
        for publisher in publishers:
            await publisher.close()
        await client.aclose()
        process.terminate()
        process.wait(timeout=5)
