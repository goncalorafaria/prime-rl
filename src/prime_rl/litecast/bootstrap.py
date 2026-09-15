"""Bounded retries for shared-filesystem endpoint discovery."""

import errno
import logging
import sqlite3
import time

from literegistry.coop import endpoints

logger = logging.getLogger(__name__)


def retry_io(operation, *args, retry_seconds=20.0, retry_interval=1.0, **kwargs):
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            return operation(*args, **kwargs)
        except (OSError, sqlite3.OperationalError) as exc:
            if isinstance(exc, sqlite3.OperationalError):
                code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
                transient = code in (sqlite3.SQLITE_IOERR, sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
            else:
                transient = exc.errno in (errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EBUSY, errno.EAGAIN)
            remaining = deadline - time.monotonic()
            if not transient or remaining <= 0:
                raise
            logger.warning("Bootstrap I/O interrupted; retrying within %.1fs: %s", remaining, exc)
            time.sleep(min(retry_interval, remaining))


def publish(*args, **kwargs):
    return retry_io(endpoints.publish, *args, **kwargs)


def wait(*args, **kwargs):
    return retry_io(endpoints.wait, *args, **kwargs)
