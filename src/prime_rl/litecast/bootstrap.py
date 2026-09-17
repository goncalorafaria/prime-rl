"""Bounded retries for shared-filesystem endpoint discovery."""

import errno
import logging
import sqlite3
import time

from literegistry.coop import endpoints

logger = logging.getLogger(__name__)


def transient_io(exc):
    if isinstance(exc, sqlite3.OperationalError):
        code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
        return code in (sqlite3.SQLITE_IOERR, sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or str(exc) in (
            "database is locked", "database table is locked", "database is busy",
        )
    return isinstance(exc, OSError) and exc.errno in (
        errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EBUSY, errno.EAGAIN,
    )


def refresh_endpoint(*args, **kwargs):
    """A transient discovery-write failure must not kill its healthy service."""
    try:
        publish(*args, **kwargs)
        return True
    except (OSError, sqlite3.OperationalError) as exc:
        if not transient_io(exc):
            raise
        logger.warning("Endpoint refresh delayed; service stays running: %s", exc)
        return False


def retry_io(operation, *args, retry_seconds=20.0, retry_interval=1.0, **kwargs):
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            return operation(*args, **kwargs)
        except (OSError, sqlite3.OperationalError) as exc:
            transient = transient_io(exc)
            remaining = deadline - time.monotonic()
            if not transient or remaining <= 0:
                raise
            logger.warning("Bootstrap I/O interrupted; retrying within %.1fs: %s", remaining, exc)
            time.sleep(min(retry_interval, remaining))


def publish(*args, **kwargs):
    return retry_io(endpoints.publish, *args, **kwargs)


def wait(*args, **kwargs):
    return retry_io(endpoints.wait, *args, **kwargs)


def head_registry(output):
    """Keep existing runs discoverable; new runs use SQLite endpoint discovery."""
    legacy = output / "bootstrap"
    if legacy.exists():
        return legacy.as_uri()
    return "sqlite://" + str(output / "head.sqlite3")


class HeadRedisCommands:
    """Execute atomic Redis scripts through the store's head failover path."""

    def __init__(self, store):
        self.store = store

    async def eval(self, script, numkeys, *args):
        async def execute(client):
            connection = await client._get_redis()
            return await connection.eval(script, numkeys, *args)

        return await self.store._execute(execute)

    async def aclose(self):
        # The registry owns the shared store and closes it.
        pass
