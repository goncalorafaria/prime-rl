"""Head discovery follows replacement Redis without a TCP relay."""

import asyncio
import os
import shutil
import socket
import subprocess

import pytest
import httpx
from literegistry import RegistryClient, ServerRegistry
from prime_rl.litecast.gateway import create_app
import redis.asyncio as redis
from literegistry.coop.endpoints import publish

from literegistry import get_kvstore
from prime_rl.litecast.bootstrap import HeadRedisCommands


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["file", "sqlite"])
async def test_head_replacement_preserves_state_without_relay(tmp_path, scheme):
    binary = os.getenv("LITECAST_TEST_REDIS_SERVER") or shutil.which("redis-server")
    if binary is None:
        pytest.skip("redis-server required")
    bootstrap = f"{scheme}://{tmp_path / 'bootstrap'}"
    backend = None
    client = get_kvstore("head+" + bootstrap)
    client.refresh_interval = 0.01
    commands = HeadRedisCommands(client)
    gateway = create_app(registry=RegistryClient(client, cache_ttl=0))
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
                async with asyncio.timeout(60):
                    while True:
                        try:
                            await direct.ping()
                            break
                        except (redis.ConnectionError, redis.TimeoutError):
                            await asyncio.sleep(0.05)
            finally:
                await direct.aclose()
            await asyncio.to_thread(
                publish, bootstrap, "redis", f"redis://127.0.0.1:{port}", publisher_id="head", ttl_seconds=30
            )
            if attempt == 0:
                await client.set("adapter-owner", "same-publisher")
                registration = ServerRegistry(client)
                await registration.register_server("http://127.0.0.1", 12345, {"model_path": "test-policy"})
                backend.kill()
                backend.wait(timeout=5)
                backend = None
                await asyncio.sleep(0.02)
            else:
                assert await client.get("adapter-owner") == b"same-publisher"
                await commands.eval("return redis.call('SET',KEYS[1],ARGV[1])", 1, "next-adapter", "step-2")
                assert await client.get("next-adapter") == b"step-2"
            if attempt == 1:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway") as http:
                    response = await http.get("/v1/models")
                    assert response.status_code == 200
                    assert "test-policy" in [item["id"] for item in response.json()["data"]]
    finally:
        await client.close()
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


def test_new_runs_use_sqlite_and_existing_runs_keep_discovery(tmp_path):
    from prime_rl.litecast.bootstrap import head_registry

    assert head_registry(tmp_path) == "sqlite://" + str(tmp_path / "head.sqlite3")
    (tmp_path / "bootstrap").mkdir()
    assert head_registry(tmp_path) == (tmp_path / "bootstrap").as_uri()


def test_endpoint_lock_does_not_stop_healthy_service(monkeypatch):
    import sqlite3
    from prime_rl.litecast import bootstrap
    def locked(*args, **kwargs):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(bootstrap, 'publish', locked)
    assert bootstrap.refresh_endpoint('sqlite:///head', 'redis', 'redis://head') is False
    def invalid(*args, **kwargs):
        raise PermissionError('not authorized')
    monkeypatch.setattr(bootstrap, 'publish', invalid)
    with pytest.raises(PermissionError):
        bootstrap.refresh_endpoint('sqlite:///head', 'redis', 'redis://head')


@pytest.mark.asyncio
async def test_model_heartbeat_recovers_on_same_loop_after_redis_replacement(tmp_path):
    import threading
    from literegistry.services.executable_wrapper import ExecutableWrapper

    class TestModel(ExecutableWrapper):
        def get_server_command(self): return []
        def get_model_flag(self): return '--model'
        def get_server_name(self): return 'test'
        def check_health(self): return True

    binary = os.getenv('LITECAST_TEST_REDIS_SERVER') or shutil.which('redis-server')
    if binary is None:
        pytest.skip('redis-server required')
    head = 'sqlite://' + str(tmp_path / 'head.sqlite3')
    model = TestModel(registry='head+' + head, port=12345, heartbeat_interval=0.05)
    model._registration_metadata = {'model_path': 'test-model'}
    backend = None
    try:
        for attempt in range(2):
            port = free_port()
            backend = subprocess.Popen([binary, '--port', str(port), '--bind', '127.0.0.1',
                                        '--save', '', '--appendonly', 'no'],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            direct = redis.from_url(f'redis://127.0.0.1:{port}')
            try:
                async with asyncio.timeout(60):
                    while True:
                        try:
                            await direct.ping()
                            break
                        except (redis.ConnectionError, redis.TimeoutError):
                            await asyncio.sleep(0.05)
                await asyncio.to_thread(publish, head, 'redis', f'redis://127.0.0.1:{port}',
                                        publisher_id='head', ttl_seconds=60)
                if attempt == 0:
                    model.heartbeat_thread = threading.Thread(target=model.heartbeat_loop, daemon=True)
                    model.heartbeat_thread.start()
                async with asyncio.timeout(20):
                    while not await direct.exists('server_' + model.registry.server_id):
                        await asyncio.sleep(0.05)
                assert model.heartbeat_thread.is_alive()
                if attempt == 1:
                    # Redis is fresh: an actual heartbeat must recreate registration.
                    assert await direct.keys('server_*')
                else:
                    backend.kill()
                    backend.wait(timeout=5)
                    backend = None
                    await asyncio.sleep(1.2)
            finally:
                await direct.aclose()
    finally:
        await asyncio.to_thread(model.cleanup)
        if backend is not None:
            backend.terminate()
            backend.wait(timeout=5)
