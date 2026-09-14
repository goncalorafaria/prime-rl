"""Optional UCXX transport for complete versions held in CPU RAM.

The ``ucxx`` package is imported only by :func:`preflight` or when a client or
server is used.  Importing this module (and therefore importing ``litecast``)
does not require UCXX.
"""

import asyncio
import importlib
import inspect
import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import blake3

from litecast.protocol import (
    MessageType,
    decode_error,
    decode_frame,
    decode_get_version,
    decode_get_shard,
    decode_get_shards,
    decode_metadata,
    decode_shard_metadata,
    decode_shards_metadata,
    decode_wait_version,
    decode_version_available,
    encode_error,
    encode_get_version,
    encode_get_shard,
    encode_get_shards,
    encode_metadata,
    encode_shard_metadata,
    encode_shards_metadata,
    encode_version_available,
    encode_wait_version,
)
from litecast.store import CheckpointStore, VersionMetadata
from litecast.transport.base import (
    FetchResult,
    TransportIntegrityError,
    TransportProtocolError,
    TransportUnavailableError,
    VersionUnavailableError,
)


DEFAULT_MAX_TOTAL_SIZE = 1024 * 1024 * 1024 * 1024
MAX_METADATA_SIZE = 16 * 1024
MAX_SHARD_SIZE = 1024 * 1024 * 1024
MAX_SHARD_COUNT = 1_000_000
MAX_VERSION_SIZE = 1024
_CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")
_UCXX_INIT_LOCK = threading.RLock()


class UCXXUnavailableError(TransportUnavailableError):
    pass


class UCXXConfigurationError(UCXXUnavailableError):
    pass


@dataclass(frozen=True)
class UCXXPreflight:
    address: str
    interface: str
    config: Mapping[str, Any]
    rc_configured: bool
    version: Optional[str] = None

    @property
    def verified_rdma(self) -> bool:
        return self.rc_configured


@dataclass(frozen=True)
class ShardFetchResult:
    data: Any
    version: str
    index: int
    checksum: str
    transport: str = "ucxx"


@dataclass(frozen=True)
class BatchShardFetchResult:
    completed: Tuple[int, ...]
    unavailable: Tuple[int, ...]
    transport: str = "ucxx"


def preflight(interface: str, ucxx_module: Optional[Any] = None) -> UCXXPreflight:
    """Verify UCXX import, explicit ``rc`` configuration, and address lookup.

    Merely obtaining an IP address is not evidence of RDMA.  This function
    succeeds only when UCX configuration explicitly contains an ``rc``
    transport token.
    """
    if not isinstance(interface, str) or not interface or len(interface) > 64:
        raise ValueError("interface must be a non-empty string")
    ucxx = _load_ucxx() if ucxx_module is None else ucxx_module
    config = _configure_rc(ucxx)
    try:
        address = ucxx.get_address(interface)
    except Exception as exc:
        raise UCXXConfigurationError(
            "unable to discover an address for interface {!r}".format(interface)
        ) from exc
    if not isinstance(address, str) or not address:
        raise UCXXConfigurationError("UCXX returned an invalid interface address")
    return UCXXPreflight(
        address=address,
        interface=interface,
        config=config,
        rc_configured=True,
        version=getattr(ucxx, "__version__", None),
    )


preflight_ucxx = preflight


class UCXXTransport:
    """Synchronous complete-version client backed by async UCXX endpoints."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 30.0,
        max_total_size: int = DEFAULT_MAX_TOTAL_SIZE,
        ucxx_module: Optional[Any] = None,
        executor: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        _validate_host_port(host, port)
        _validate_positive_limit(max_total_size, "max_total_size")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.host = host
        self.port = port
        self.timeout = float(timeout)
        self.max_total_size = max_total_size
        self._ucxx = ucxx_module
        self._executor = executor
        self._closed = False
        self._persistent_endpoint = None
        self._persistent_lock = None

    def fetch_version(
        self, version: str, max_total_size: Optional[int] = None
    ) -> FetchResult:
        if self._closed:
            raise UCXXUnavailableError("transport is closed")
        _validate_version(version)
        maximum = self.max_total_size if max_total_size is None else max_total_size
        _validate_positive_limit(maximum, "max_total_size")
        ucxx = _load_ucxx() if self._ucxx is None else self._ucxx
        awaitable = self._fetch(ucxx, version, maximum)
        if self._executor is not None:
            return self._executor(awaitable)
        return _run_sync(awaitable)

    def fetch_bytes(
        self, version: str, max_total_size: Optional[int] = None
    ) -> Any:
        return self.fetch_version(version, max_total_size).data

    fetch = fetch_bytes

    def warm(self) -> None:
        """Establish the reusable endpoint before a timed transfer."""
        if self._closed:
            raise UCXXUnavailableError("transport is closed")
        if self._executor is None:
            return
        ucxx = _load_ucxx() if self._ucxx is None else self._ucxx
        self._executor(self._warm_persistent(ucxx))

    async def _warm_persistent(self, ucxx: Any) -> None:
        if self._persistent_lock is None:
            self._persistent_lock = asyncio.Lock()
        async with self._persistent_lock:
            if self._persistent_endpoint is None:
                _configure_rc(ucxx)
                self._persistent_endpoint = await ucxx.create_endpoint(
                    self.host, self.port, connect_timeout=self.timeout
                )

    def wait_for_version(self, version: str) -> None:
        """Block until a peer announces that a version is discoverable."""
        if self._closed:
            raise UCXXUnavailableError("transport is closed")
        _validate_version(version)
        if self._executor is None:
            raise UCXXUnavailableError(
                "version notification requires an owned event loop"
            )
        ucxx = _load_ucxx() if self._ucxx is None else self._ucxx
        self._executor(self._wait_for_version_persistent(ucxx, version))

    async def _wait_for_version_persistent(
        self, ucxx: Any, version: str
    ) -> None:
        if self._persistent_lock is None:
            self._persistent_lock = asyncio.Lock()
        async with self._persistent_lock:
            try:
                if self._persistent_endpoint is None:
                    _configure_rc(ucxx)
                    self._persistent_endpoint = await ucxx.create_endpoint(
                        self.host, self.port, connect_timeout=self.timeout
                    )
                await asyncio.wait_for(
                    self._persistent_endpoint.send_obj(
                        encode_wait_version(version)
                    ),
                    timeout=self.timeout,
                )
                frame = decode_frame(
                    await asyncio.wait_for(
                        self._persistent_endpoint.recv_obj(),
                        timeout=max(self.timeout, 120),
                    ),
                    max_payload_size=MAX_METADATA_SIZE,
                )
                if frame.message_type == MessageType.ERROR:
                    raise VersionUnavailableError(decode_error(frame))
                if decode_version_available(frame) != version:
                    raise TransportProtocolError(
                        "version notification mismatch"
                    )
            except VersionUnavailableError:
                raise
            except Exception as exc:
                await self._discard_persistent_endpoint()
                if isinstance(exc, TransportProtocolError):
                    raise
                raise UCXXUnavailableError(
                    "UCXX version notification failed"
                ) from exc

    def fetch_shards_into(
        self,
        version: str,
        targets: Mapping[int, memoryview],
        expected_checksums: Mapping[int, str],
    ) -> BatchShardFetchResult:
        """Fetch several shards in one control exchange."""
        if self._closed:
            raise UCXXUnavailableError("transport is closed")
        _validate_version(version)
        if not targets or set(targets) != set(expected_checksums):
            raise ValueError("targets and checksums must contain identical indices")
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or not isinstance(target, memoryview)
            or target.readonly
            for index, target in targets.items()
        ):
            raise TypeError("batch targets must be writable indexed memoryviews")
        if self._executor is None:
            raise UCXXUnavailableError(
                "batched shard transfer requires an owned event loop"
            )
        ucxx = _load_ucxx() if self._ucxx is None else self._ucxx
        return self._executor(
            self._fetch_shards_persistent(
                ucxx, version, targets, expected_checksums
            )
        )

    async def _fetch_shards_persistent(
        self,
        ucxx: Any,
        version: str,
        targets: Mapping[int, memoryview],
        expected_checksums: Mapping[int, str],
    ) -> BatchShardFetchResult:
        if self._persistent_lock is None:
            self._persistent_lock = asyncio.Lock()
        async with self._persistent_lock:
            try:
                if self._persistent_endpoint is None:
                    _configure_rc(ucxx)
                    self._persistent_endpoint = await ucxx.create_endpoint(
                        self.host, self.port, connect_timeout=self.timeout
                    )
                return await self._exchange_shards(
                    self._persistent_endpoint,
                    version,
                    targets,
                    expected_checksums,
                )
            except VersionUnavailableError:
                raise
            except (
                TransportProtocolError,
                TransportIntegrityError,
            ):
                await self._discard_persistent_endpoint()
                raise
            except Exception as exc:
                await self._discard_persistent_endpoint()
                raise UCXXUnavailableError(
                    "UCXX batched shard transfer failed"
                ) from exc

    def fetch_shard(
        self, version: str, index: int, expected_checksum: Optional[str] = None
    ) -> ShardFetchResult:
        if self._closed:
            raise UCXXUnavailableError("transport is closed")
        _validate_version(version)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("index must be a non-negative integer")
        ucxx = _load_ucxx() if self._ucxx is None else self._ucxx
        fetch = (
            self._fetch_shard_persistent
            if self._executor is not None
            else self._fetch_shard
        )
        awaitable = fetch(ucxx, version, index, expected_checksum, None)
        if self._executor is not None:
            return self._executor(awaitable)
        return _run_sync(awaitable)

    def fetch_shard_into(
        self,
        version: str,
        index: int,
        target: memoryview,
        expected_checksum: Optional[str] = None,
    ) -> ShardFetchResult:
        if self._closed:
            raise UCXXUnavailableError("transport is closed")
        _validate_version(version)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("index must be a non-negative integer")
        if not isinstance(target, memoryview) or target.readonly:
            raise TypeError("target must be a writable memoryview")
        ucxx = _load_ucxx() if self._ucxx is None else self._ucxx
        fetch = (
            self._fetch_shard_persistent
            if self._executor is not None
            else self._fetch_shard
        )
        awaitable = fetch(ucxx, version, index, expected_checksum, target)
        if self._executor is not None:
            return self._executor(awaitable)
        return _run_sync(awaitable)

    async def _fetch_shard(
        self,
        ucxx: Any,
        version: str,
        index: int,
        expected_checksum: Optional[str],
        target: Optional[memoryview],
    ) -> ShardFetchResult:
        endpoint = None
        try:
            _configure_rc(ucxx)
            endpoint = await ucxx.create_endpoint(
                self.host, self.port, connect_timeout=self.timeout
            )
            return await self._exchange_shard(
                endpoint, version, index, expected_checksum, target
            )
        except (
            TransportProtocolError,
            TransportIntegrityError,
            VersionUnavailableError,
        ):
            raise
        except Exception as exc:
            raise UCXXUnavailableError("UCXX shard transfer failed") from exc
        finally:
            if endpoint is not None:
                await _close(endpoint)

    async def _fetch_shard_persistent(
        self,
        ucxx: Any,
        version: str,
        index: int,
        expected_checksum: Optional[str],
        target: Optional[memoryview],
    ) -> ShardFetchResult:
        if self._persistent_lock is None:
            self._persistent_lock = asyncio.Lock()
        async with self._persistent_lock:
            try:
                if self._persistent_endpoint is None:
                    _configure_rc(ucxx)
                    self._persistent_endpoint = await ucxx.create_endpoint(
                        self.host, self.port, connect_timeout=self.timeout
                    )
                return await self._exchange_shard(
                    self._persistent_endpoint,
                    version,
                    index,
                    expected_checksum,
                    target,
                )
            except VersionUnavailableError:
                raise
            except (
                TransportProtocolError,
                TransportIntegrityError,
            ):
                await self._discard_persistent_endpoint()
                raise
            except Exception as exc:
                await self._discard_persistent_endpoint()
                raise UCXXUnavailableError(
                    "UCXX shard transfer failed"
                ) from exc

    async def _exchange_shard(
        self,
        endpoint: Any,
        version: str,
        index: int,
        expected_checksum: Optional[str],
        target: Optional[memoryview],
    ) -> ShardFetchResult:
        try:
            await asyncio.wait_for(
                endpoint.send_obj(encode_get_shard(version, index)),
                timeout=self.timeout,
            )
            frame = decode_frame(
                await asyncio.wait_for(
                    endpoint.recv_obj(), timeout=self.timeout
                ),
                max_payload_size=MAX_METADATA_SIZE,
            )
            if frame.message_type == MessageType.ERROR:
                raise VersionUnavailableError(decode_error(frame))
            value = decode_shard_metadata(frame)
            required = {"status", "version", "index", "size", "checksum"}
            if (
                set(value) != required
                or value["status"] != "ok"
                or value["version"] != version
                or value["index"] != index
                or not isinstance(value["size"], int)
                or isinstance(value["size"], bool)
                or not 0 <= value["size"] <= MAX_SHARD_SIZE
                or not isinstance(value["checksum"], str)
                or _CHECKSUM_RE.fullmatch(value["checksum"]) is None
            ):
                raise TransportProtocolError("malformed shard metadata")
            if target is not None and len(target) != value["size"]:
                raise TransportProtocolError("target size does not match shard")
            data = bytearray(value["size"]) if target is None else target
            if data:
                await asyncio.wait_for(
                    endpoint.recv(
                        memoryview(data)
                        if not isinstance(data, memoryview)
                        else data
                    ),
                    timeout=self.timeout,
                )
            digest = blake3.blake3(data).hexdigest()
            if digest != value["checksum"] or (
                expected_checksum is not None and digest != expected_checksum
            ):
                raise TransportIntegrityError("shard BLAKE3 checksum mismatch")
            return ShardFetchResult(data, version, index, digest)
        except (
            TransportProtocolError,
            TransportIntegrityError,
            VersionUnavailableError,
        ):
            raise

    async def _exchange_shards(
        self,
        endpoint: Any,
        version: str,
        targets: Mapping[int, memoryview],
        expected_checksums: Mapping[int, str],
    ) -> BatchShardFetchResult:
        requested = tuple(targets)
        await asyncio.wait_for(
            endpoint.send_obj(encode_get_shards(version, requested)),
            timeout=self.timeout,
        )
        frame = decode_frame(
            await asyncio.wait_for(endpoint.recv_obj(), timeout=self.timeout),
            max_payload_size=MAX_METADATA_SIZE,
        )
        if frame.message_type == MessageType.ERROR:
            raise VersionUnavailableError(decode_error(frame))
        value = decode_shards_metadata(frame)
        if (
            set(value) != {"version", "shards", "unavailable"}
            or value["version"] != version
            or not isinstance(value["shards"], list)
            or not isinstance(value["unavailable"], list)
        ):
            raise TransportProtocolError("malformed batched shard metadata")
        available = []
        seen = set()
        for item in value["shards"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"index", "size", "checksum"}
                or item["index"] not in targets
                or item["index"] in seen
                or item["size"] != len(targets[item["index"]])
                or not isinstance(item["checksum"], str)
                or _CHECKSUM_RE.fullmatch(item["checksum"]) is None
            ):
                raise TransportProtocolError(
                    "malformed batched shard metadata"
                )
            seen.add(item["index"])
            available.append(item)
        unavailable = tuple(value["unavailable"])
        if (
            any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index not in targets
                for index in unavailable
            )
            or len(set(unavailable)) != len(unavailable)
            or seen | set(unavailable) != set(requested)
            or seen & set(unavailable)
        ):
            raise TransportProtocolError("inconsistent batched shard metadata")
        completed = []
        for item in available:
            index = item["index"]
            target = targets[index]
            if target:
                await asyncio.wait_for(
                    endpoint.recv(target), timeout=self.timeout
                )
            digest = blake3.blake3(target).hexdigest()
            if (
                digest != item["checksum"]
                or digest != expected_checksums[index]
            ):
                raise TransportIntegrityError(
                    "batched shard BLAKE3 checksum mismatch"
                )
            completed.append(index)
        return BatchShardFetchResult(tuple(completed), unavailable)

    async def _discard_persistent_endpoint(self) -> None:
        endpoint = self._persistent_endpoint
        self._persistent_endpoint = None
        if endpoint is not None:
            await _close(endpoint)

    async def _fetch(self, ucxx: Any, version: str, maximum: int) -> FetchResult:
        endpoint = None
        try:
            _configure_rc(ucxx)
            endpoint = await ucxx.create_endpoint(
                self.host, self.port, connect_timeout=self.timeout
            )
            await asyncio.wait_for(
                endpoint.send_obj(encode_get_version(version)),
                timeout=self.timeout,
            )
            response_frame = decode_frame(
                await asyncio.wait_for(
                    endpoint.recv_obj(), timeout=self.timeout
                ),
                max_payload_size=MAX_METADATA_SIZE,
            )
            if response_frame.message_type == MessageType.ERROR:
                raise VersionUnavailableError(decode_error(response_frame))
            response = decode_metadata(response_frame)
            metadata = _validate_metadata(response, version, maximum)

            # This is the sole allocation for received bulk data. UCXX writes
            # each chunk directly into its final slice.
            data = bytearray(metadata.total_size)
            view = memoryview(data)
            for index in range(metadata.shard_count):
                start = index * metadata.shard_size
                end = min(start + metadata.shard_size, metadata.total_size)
                await asyncio.wait_for(
                    endpoint.recv(view[start:end]), timeout=self.timeout
                )
            digest = blake3.blake3(data).hexdigest()
            if digest != metadata.checksum:
                raise TransportIntegrityError("BLAKE3 checksum mismatch")
            return FetchResult(data, metadata, "ucxx")
        except (
            TransportProtocolError,
            TransportIntegrityError,
            VersionUnavailableError,
        ):
            raise
        except Exception as exc:
            raise UCXXUnavailableError("UCXX transfer failed") from exc
        finally:
            if endpoint is not None:
                await _close(endpoint)

    def close(self) -> None:
        if self._executor is not None and self._persistent_endpoint is not None:
            try:
                self._executor(self._discard_persistent_endpoint())
            except Exception:
                pass
        self._closed = True


UCXXClient = UCXXTransport


class UCXXEventLoopExecutor:
    """Own one asyncio loop for all UCXX operations in a client process."""

    def __init__(self, result_timeout: float = 120.0) -> None:
        if result_timeout <= 0:
            raise ValueError("result_timeout must be positive")
        self.result_timeout = float(result_timeout)
        self._loop = None  # type: Optional[asyncio.AbstractEventLoop]
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="litecast-ucxx-client",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(10):
            raise UCXXUnavailableError("timed out starting client UCXX loop")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()
            self._loop = None

    def __call__(self, awaitable: Any) -> Any:
        loop = self._loop
        if loop is None or not loop.is_running():
            raise UCXXUnavailableError("client UCXX event loop is not running")
        future = asyncio.run_coroutine_threadsafe(awaitable, loop)
        try:
            return future.result(timeout=self.result_timeout)
        except BaseException:
            future.cancel()
            raise

    def close(self, timeout: float = 10.0) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise UCXXUnavailableError(
                "timed out stopping client UCXX event loop"
            )

    def __enter__(self) -> "UCXXEventLoopExecutor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class UCXXServer:
    """Background asyncio UCXX listener serving a :class:`CheckpointStore`."""

    def __init__(
        self,
        store: CheckpointStore,
        interface: str,
        port: int = 0,
        max_total_size: int = DEFAULT_MAX_TOTAL_SIZE,
        ucxx_module: Optional[Any] = None,
        operation_timeout: float = 30.0,
    ) -> None:
        if not isinstance(store, CheckpointStore):
            raise TypeError("store must be CheckpointStore")
        if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        _validate_positive_limit(max_total_size, "max_total_size")
        self.store = store
        self.interface = interface
        self.requested_port = port
        self.max_total_size = max_total_size
        self._ucxx = ucxx_module
        if operation_timeout <= 0:
            raise ValueError("operation_timeout must be positive")
        self.operation_timeout = float(operation_timeout)
        self._thread = None  # type: Optional[threading.Thread]
        self._loop = None  # type: Optional[asyncio.AbstractEventLoop]
        self._listener = None
        self._preflight = None  # type: Optional[UCXXPreflight]
        self._startup_error = None  # type: Optional[BaseException]
        self._ready = threading.Event()
        self._handlers = set()  # type: set
        self._available_versions = set()  # type: set
        self._available_versions_lock = threading.Lock()

    @property
    def address(self) -> Optional[str]:
        return None if self._preflight is None else self._preflight.address

    @property
    def port(self) -> Optional[int]:
        if self._listener is None:
            return None
        value = getattr(self._listener, "port", None)
        return value() if callable(value) else value

    @property
    def verified_rdma(self) -> bool:
        return self._preflight is not None and self._preflight.rc_configured

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def notify_version_available(self, version: str) -> None:
        """Mark a version discoverable for UCXX publication waiters."""
        _validate_version(version)
        with self._available_versions_lock:
            self._available_versions.add(version)

    def _version_is_available(self, version: str) -> bool:
        with self._available_versions_lock:
            return version in self._available_versions

    def run_coroutine(self, awaitable: Any) -> Any:
        """Run an outgoing UCXX operation on the listener's owning loop."""
        loop = self._loop
        if loop is None or not loop.is_running():
            raise UCXXUnavailableError("UCXX event loop is not running")
        future = asyncio.run_coroutine_threadsafe(awaitable, loop)
        try:
            return future.result(timeout=self.operation_timeout)
        except BaseException:
            future.cancel()
            raise

    def start(self, timeout: float = 10.0) -> "UCXXServer":
        if self.is_running:
            return self
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run, name="litecast-ucxx", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise UCXXUnavailableError("timed out starting UCXX listener")
        if self._startup_error is not None:
            error = self._startup_error
            self._thread = None
            if isinstance(error, TransportUnavailableError):
                raise error
            raise UCXXUnavailableError("unable to start UCXX listener") from error
        return self

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            async def start_listener() -> None:
                ucxx = _load_ucxx() if self._ucxx is None else self._ucxx
                self._preflight = preflight(self.interface, ucxx)
                self._listener = ucxx.create_listener(
                    self._handle_endpoint, port=self.requested_port
                )

            loop.run_until_complete(start_listener())
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            if self._listener is not None:
                loop.run_until_complete(_close(self._listener))
                self._listener = None
            loop.close()
            self._loop = None

    async def _handle_endpoint(self, endpoint: Any) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        served_request = False
        try:
            while True:
                request = decode_frame(
                    await asyncio.wait_for(
                        endpoint.recv_obj(), timeout=self.operation_timeout
                    ),
                    max_payload_size=MAX_METADATA_SIZE,
                )
                await self._serve_request(endpoint, request)
                served_request = True
        except Exception as exc:
            if not served_request:
                try:
                    message = str(exc)[:1024] or "invalid request"
                    await asyncio.wait_for(
                        endpoint.send_obj(encode_error(message)),
                        timeout=self.operation_timeout,
                    )
                except Exception:
                    pass
        finally:
            await _close(endpoint)
            if task is not None:
                self._handlers.discard(task)

    async def _serve_request(self, endpoint: Any, request: Any) -> None:
        if request.message_type == MessageType.WAIT_VERSION:
            version = decode_wait_version(request)
            deadline = (
                asyncio.get_running_loop().time() + self.operation_timeout
            )
            while not self._version_is_available(version):
                if asyncio.get_running_loop().time() >= deadline:
                    await asyncio.wait_for(
                        endpoint.send_obj(
                            encode_error("version notification timed out")
                        ),
                        timeout=self.operation_timeout,
                    )
                    return
                await asyncio.sleep(0.01)
            await asyncio.wait_for(
                endpoint.send_obj(encode_version_available(version)),
                timeout=self.operation_timeout,
            )
            return
        if request.message_type == MessageType.GET_SHARDS:
            version, indices = decode_get_shards(request)
            available = []
            unavailable = []
            for index in indices:
                try:
                    shard, checksum = self.store.get_shard_with_checksum(
                        version, index
                    )
                except Exception:
                    unavailable.append(index)
                else:
                    available.append((index, shard, checksum))
            await asyncio.wait_for(
                endpoint.send_obj(encode_shards_metadata(
                    {
                        "version": version,
                        "shards": [
                            {
                                "index": index,
                                "size": len(shard),
                                "checksum": checksum,
                            }
                            for index, shard, checksum in available
                        ],
                        "unavailable": unavailable,
                    }
                )),
                timeout=self.operation_timeout,
            )
            for _, shard, _ in available:
                if shard:
                    await asyncio.wait_for(
                        endpoint.send(shard),
                        timeout=self.operation_timeout,
                    )
            return
        if request.message_type == MessageType.GET_SHARD:
            version, index = decode_get_shard(request)
            try:
                shard, checksum = self.store.get_shard_with_checksum(
                    version, index
                )
            except Exception:
                await asyncio.wait_for(
                    endpoint.send_obj(encode_error("shard not available")),
                    timeout=self.operation_timeout,
                )
                return
            await asyncio.wait_for(
                endpoint.send_obj(encode_shard_metadata(
                    {
                        "status": "ok",
                        "version": version,
                        "index": index,
                        "size": len(shard),
                        "checksum": checksum,
                    }
                )),
                timeout=self.operation_timeout,
            )
            if shard:
                await asyncio.wait_for(
                    endpoint.send(shard), timeout=self.operation_timeout
                )
            return
        version = decode_get_version(request)
        metadata = self.store.get_metadata(version)
        data = self.store.get(version)
        # Read metadata twice around the data lookup to obtain a coherent
        # immutable store snapshot despite concurrent replacement.
        if (
            metadata is None
            or data is None
            or self.store.get_metadata(version) != metadata
        ):
            await asyncio.wait_for(
                endpoint.send_obj(encode_error("version not found")),
                timeout=self.operation_timeout,
            )
            return
        if metadata.total_size > self.max_total_size:
            await asyncio.wait_for(
                endpoint.send_obj(encode_error("version exceeds server limit")),
                timeout=self.operation_timeout,
            )
            return
        _validate_metadata(
            _metadata_message(metadata), version, self.max_total_size
        )
        await asyncio.wait_for(
            endpoint.send_obj(encode_metadata(_metadata_message(metadata))),
            timeout=self.operation_timeout,
        )
        for index in range(metadata.shard_count):
            start = index * metadata.shard_size
            end = min(start + metadata.shard_size, metadata.total_size)
            await asyncio.wait_for(
                endpoint.send(data[start:end]),
                timeout=self.operation_timeout,
            )

    async def _shutdown_async(self) -> None:
        if self._listener is not None:
            await _close(self._listener)
            self._listener = None
        current = asyncio.current_task()
        handlers = [
            task for task in self._handlers
            if task is not current and not task.done()
        ]
        for task in handlers:
            task.cancel()
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)

    def close(self, timeout: float = 10.0) -> None:
        loop = self._loop
        thread = self._thread
        if loop is not None and loop.is_running():
            if thread is threading.current_thread():
                loop.create_task(self._shutdown_async())
                loop.call_soon(loop.stop)
            else:
                future = asyncio.run_coroutine_threadsafe(
                    self._shutdown_async(), loop
                )
                try:
                    future.result(timeout=timeout)
                except Exception:
                    future.cancel()
                loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise UCXXUnavailableError(
                    "timed out stopping UCXX listener thread"
                )
        self._thread = None

    stop = close

    def __enter__(self) -> "UCXXServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


UCXXTransportServer = UCXXServer


def _metadata_message(metadata: VersionMetadata) -> Dict[str, Any]:
    value = asdict(metadata)
    value["status"] = "ok"
    return value


def _validate_metadata(
    value: Any, expected_name: str, maximum: int
) -> VersionMetadata:
    required = {
        "status",
        "name",
        "total_size",
        "shard_size",
        "shard_count",
        "checksum",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("status") != "ok":
        raise TransportProtocolError("malformed version metadata")
    if value["name"] != expected_name:
        raise TransportProtocolError("metadata version name mismatch")
    total_size = _plain_int(value["total_size"], "total_size", minimum=0, maximum=maximum)
    shard_size = _plain_int(
        value["shard_size"], "shard_size", minimum=1, maximum=MAX_SHARD_SIZE
    )
    shard_count = _plain_int(
        value["shard_count"], "shard_count", minimum=0, maximum=MAX_SHARD_COUNT
    )
    expected_count = (total_size + shard_size - 1) // shard_size if total_size else 0
    if shard_count != expected_count:
        raise TransportProtocolError("inconsistent shard count")
    checksum = value["checksum"]
    if not isinstance(checksum, str) or _CHECKSUM_RE.fullmatch(checksum) is None:
        raise TransportProtocolError("invalid BLAKE3 checksum")
    return VersionMetadata(expected_name, total_size, shard_size, shard_count, checksum)


def _plain_int(
    value: Any, label: str, minimum: int, maximum: int
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise TransportProtocolError("invalid {}".format(label))
    return value


def _config_has_rc(config: Mapping[str, Any]) -> bool:
    values = []
    for key, value in config.items():
        normalized = str(key).upper().replace("UCX_", "")
        if normalized == "TLS":
            values.append(value)
    for value in values:
        if isinstance(value, (list, tuple, set)):
            tokens = [str(item).strip().lower() for item in value]
        else:
            tokens = [token.strip().lower() for token in str(value).split(",")]
        for token in tokens:
            if token == "rc" or token.startswith("rc_"):
                return True
    return False


def _configure_rc(ucxx: Any) -> Dict[str, Any]:
    with _UCXX_INIT_LOCK:
        try:
            if getattr(ucxx, "_litecast_rc_initialized", False):
                config_value = ucxx.get_config()
            elif hasattr(ucxx, "init"):
                options = {"TLS": "rc"}
                net_devices = os.getenv("UCX_NET_DEVICES")
                sockaddr_priority = os.getenv("UCX_SOCKADDR_TLS_PRIORITY")
                if net_devices:
                    options["NET_DEVICES"] = net_devices
                if sockaddr_priority:
                    options["SOCKADDR_TLS_PRIORITY"] = sockaddr_priority
                try:
                    ucxx.init(
                        options=options,
                        env_takes_precedence=True,
                        progress_mode="thread",
                    )
                except TypeError:
                    try:
                        ucxx.init(options=options, progress_mode="thread")
                    except TypeError:
                        try:
                            ucxx.init(options=options)
                        except TypeError:
                            ucxx.init(options)
                config_value = ucxx.get_config()
            else:
                config_value = ucxx.get_config()
        except Exception as exc:
            raise UCXXConfigurationError(
                "unable to initialize UCXX with TLS=rc"
            ) from exc
        if not isinstance(config_value, Mapping):
            raise UCXXConfigurationError("UCXX configuration is unavailable")
        config = dict(config_value)
        if not _config_has_rc(config):
            raise UCXXConfigurationError(
                "UCX rc transport is not explicitly configured"
            )
        try:
            setattr(ucxx, "_litecast_rc_initialized", True)
        except Exception:
            pass
        return config


def _load_ucxx() -> Any:
    try:
        return importlib.import_module("ucxx")
    except ImportError as exc:
        raise UCXXUnavailableError(
            "UCXX transport requires the optional 'ucxx' package"
        ) from exc


async def _close(resource: Any, timeout: float = 10.0) -> None:
    try:
        result = resource.close()
        if inspect.isawaitable(result):
            await asyncio.wait_for(result, timeout=timeout)
    except Exception:
        pass


def _run_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result = []  # type: list
    error = []  # type: list

    def runner() -> None:
        try:
            result.append(asyncio.run(awaitable))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _validate_version(version: Any) -> None:
    if (
        not isinstance(version, str)
        or not version
        or len(version.encode("utf-8")) > MAX_VERSION_SIZE
        or "\x00" in version
    ):
        raise TransportProtocolError("invalid version name")


def _validate_host_port(host: Any, port: Any) -> None:
    if not isinstance(host, str) or not host or len(host) > 255:
        raise ValueError("host must be a non-empty string")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")


def _validate_positive_limit(value: Any, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("{} must be a non-negative integer".format(label))
