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
@pytest.mark.parametrize("scheme", ["file", "sqlite"])
async def test_head_replacement_preserves_state_and_relay_address(tmp_path, scheme):
    binary = os.getenv("LITECAST_TEST_REDIS_SERVER") or shutil.which("redis-server")
    if binary is None:
        pytest.skip("redis-server required")
    bootstrap = f"{scheme}://{tmp_path / 'bootstrap'}"
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


@pytest.mark.parametrize("kind", ["file", "sqlite"])
def test_bootstrap_retries_transient_io_and_bounds_failure(kind):
    import errno
    import sqlite3

    from prime_rl.litecast.bootstrap import retry_io

    error = OSError(errno.EIO, "disk I/O error")
    if kind == "sqlite":
        error = sqlite3.OperationalError("disk I/O error")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
    calls = []

    def interrupted_once():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return "endpoint"

    assert retry_io(interrupted_once, retry_interval=0) == "endpoint"
    assert len(calls) == 2

    def broken():
        raise error

    with pytest.raises(type(error)):
        retry_io(broken, retry_seconds=0)

    def invalid():
        raise PermissionError(errno.EACCES, "not permitted")

    with pytest.raises(PermissionError):
        retry_io(invalid)


def test_retention_keeps_recent_pending_and_checkpoint_artifacts(tmp_path):
    from prime_rl.litecast.retention import prune_intermediates

    run = tmp_path / "run_default"
    for folder in ("broadcasts", "rollouts", "token_exports", "checkpoints"):
        for step in (1, 8, 9, 14, 15, 20, 21):
            path = run / folder / f"step_{step}"
            path.mkdir(parents=True)
            if folder == "broadcasts" and step <= 20:
                (path / "STABLE").touch()
    prune_intermediates(tmp_path, keep_rollout_steps=12, keep_adapter_steps=6)
    assert not (run / "rollouts/step_8").exists()
    assert (run / "rollouts/step_9").exists()
    assert not (run / "broadcasts/step_14").exists()
    assert (run / "broadcasts/step_15").exists()
    assert (run / "broadcasts/step_21").exists()
    assert (run / "rollouts/step_21").exists()
    assert (run / "checkpoints/step_1").exists()
