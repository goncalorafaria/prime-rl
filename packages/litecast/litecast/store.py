"""Thread-safe, capacity-bounded storage for complete checkpoint versions."""

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import blake3


class StoreError(Exception):
    """Base class for checkpoint store failures."""


class VersionTooLargeError(StoreError, ValueError):
    """A single checkpoint cannot fit in the configured capacity."""


class VersionNotFoundError(StoreError, KeyError):
    """The requested checkpoint version is not present."""


class InvalidShardIndexError(StoreError, IndexError):
    """The requested shard index is outside the version."""


@dataclass(frozen=True)
class VersionMetadata:
    """Immutable metadata describing one complete checkpoint."""

    name: str
    total_size: int
    shard_size: int
    shard_count: int
    checksum: str


@dataclass
class _StoredVersion:
    metadata: VersionMetadata
    data: Optional[Any] = None
    shards: Dict[int, Any] = field(default_factory=dict)
    shard_checksums: Tuple[str, ...] = ()

    @property
    def stored_bytes(self) -> int:
        if self.data is not None:
            return len(self.data)
        return sum(len(shard) for shard in self.shards.values())


class CheckpointStore:
    """Keep complete immutable checkpoints in CPU memory.

    Capacity eviction is insertion ordered: the oldest inserted version is
    removed first. Accesses do not alter that order, making eviction stable and
    reproducible. Replacing a version makes the replacement the newest entry.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if (
            not isinstance(capacity_bytes, int)
            or isinstance(capacity_bytes, bool)
            or capacity_bytes < 0
        ):
            raise ValueError("capacity_bytes must be a non-negative integer")
        self._capacity = capacity_bytes
        self._size = 0
        self._versions = OrderedDict()  # type: OrderedDict[str, _StoredVersion]
        self._lock = threading.RLock()

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._size

    def __len__(self) -> int:
        with self._lock:
            return len(self._versions)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._versions

    def put(
        self,
        name: str,
        data: Any,
        shard_size: int,
        checksum: Optional[str] = None,
        shard_checksums: Tuple[str, ...] = (),
    ) -> VersionMetadata:
        """Insert or atomically replace a complete checkpoint.

        A ``bytes`` object is retained directly. Every other buffer is copied
        into ``bytes`` so a caller cannot mutate stored data through an alias.
        """
        _validate_name(name)
        _validate_shard_size(shard_size)
        immutable = _make_immutable(data)
        data_size = len(immutable)
        if data_size > self._capacity:
            raise VersionTooLargeError(
                "version {!r} is {} bytes; capacity is {}".format(
                    name, data_size, self._capacity
                )
            )
        actual_checksum = blake3.blake3(immutable).hexdigest()
        if checksum is not None:
            if not isinstance(checksum, str):
                raise TypeError("checksum must be str")
            if checksum.lower() != actual_checksum:
                raise ValueError("checksum does not match checkpoint data")

        metadata = VersionMetadata(
            name=name,
            total_size=data_size,
            shard_size=shard_size,
            shard_count=(
                (data_size + shard_size - 1) // shard_size if data_size else 0
            ),
            checksum=actual_checksum,
        )
        checksums = _validate_shard_checksums(
            shard_checksums, metadata.shard_count
        )
        return self._store_complete(metadata, immutable, checksums)

    def put_verified(
        self,
        name: str,
        data: Any,
        shard_size: int,
        checksum: str,
        shard_checksums: Tuple[str, ...] = (),
    ) -> VersionMetadata:
        """Take ownership of already verified complete data without rehashing."""
        _validate_name(name)
        _validate_shard_size(shard_size)
        if not isinstance(checksum, str):
            raise TypeError("checksum must be str")
        try:
            int(checksum, 16)
        except ValueError:
            raise ValueError("checksum must be a BLAKE3 hex digest")
        if len(checksum) != 64:
            raise ValueError("checksum must be a BLAKE3 hex digest")
        retained = _take_buffer_ownership(data)
        data_size = len(retained)
        if data_size > self._capacity:
            raise VersionTooLargeError(
                "version {!r} is {} bytes; capacity is {}".format(
                    name, data_size, self._capacity
                )
            )
        metadata = VersionMetadata(
            name=name,
            total_size=data_size,
            shard_size=shard_size,
            shard_count=(
                (data_size + shard_size - 1) // shard_size if data_size else 0
            ),
            checksum=checksum.lower(),
        )
        checksums = _validate_shard_checksums(
            shard_checksums, metadata.shard_count
        )
        return self._store_complete(metadata, retained, checksums)

    def _store_complete(
        self,
        metadata: VersionMetadata,
        retained: Any,
        shard_checksums: Tuple[str, ...],
    ) -> VersionMetadata:
        data_size = len(retained)
        stored = _StoredVersion(
            metadata, retained, shard_checksums=shard_checksums
        )
        with self._lock:
            previous = self._versions.pop(metadata.name, None)
            if previous is not None:
                self._size -= previous.stored_bytes
            while self._versions and self._size + data_size > self._capacity:
                _, evicted = self._versions.popitem(last=False)
                self._size -= evicted.stored_bytes
            # The oversize check guarantees this also works for capacity zero
            # when the checkpoint itself is empty.
            self._versions[metadata.name] = stored
            self._size += data_size
        return metadata

    def begin_version(
        self,
        metadata: VersionMetadata,
        shard_checksums: Tuple[str, ...],
    ) -> VersionMetadata:
        """Register an in-progress version without allocating its full size."""
        if not isinstance(metadata, VersionMetadata):
            raise TypeError("metadata must be VersionMetadata")
        _validate_name(metadata.name)
        _validate_shard_size(metadata.shard_size)
        checksums = tuple(shard_checksums)
        if len(checksums) != metadata.shard_count:
            raise ValueError("one checksum is required per shard")
        with self._lock:
            existing = self._versions.get(metadata.name)
            if existing is not None:
                if existing.metadata != metadata or existing.shard_checksums not in (
                    (), checksums
                ):
                    raise ValueError("version metadata changed")
                if not existing.shard_checksums:
                    existing.shard_checksums = checksums
                return existing.metadata
            self._versions[metadata.name] = _StoredVersion(
                metadata=metadata, shard_checksums=checksums
            )
        return metadata

    def put_shard(self, name: str, index: int, data: Any) -> memoryview:
        """Insert one independently verified shard and make it immediately visible."""
        return self._put_shard(name, index, data)

    def put_verified_shard(
        self, name: str, index: int, data: Any, verified_checksum: str
    ) -> memoryview:
        """Take ownership of a transport-verified shard without copying or rehashing."""
        if not isinstance(verified_checksum, str):
            raise TypeError("verified_checksum must be str")
        return self._put_shard(
            name,
            index,
            data,
            verified_checksum=verified_checksum.lower(),
            take_ownership=True,
        )

    def _put_shard(
        self,
        name: str,
        index: int,
        data: Any,
        verified_checksum: Optional[str] = None,
        take_ownership: bool = False,
    ) -> memoryview:
        retained = _take_buffer_ownership(data) if take_ownership else _make_immutable(data)
        with self._lock:
            stored = self._versions.get(name)
            if stored is None:
                raise VersionNotFoundError(name)
            if index < 0 or index >= stored.metadata.shard_count:
                raise InvalidShardIndexError(index)
            expected_size = min(
                stored.metadata.shard_size,
                stored.metadata.total_size - index * stored.metadata.shard_size,
            )
            if len(retained) != expected_size:
                raise ValueError("shard size does not match manifest")
            if stored.shard_checksums:
                digest = (
                    verified_checksum
                    if verified_checksum is not None
                    else blake3.blake3(retained).hexdigest()
                )
                if digest != stored.shard_checksums[index]:
                    raise ValueError("shard checksum does not match manifest")
            previous = stored.shards.get(index)
            additional = len(retained) - (0 if previous is None else len(previous))
            while self._versions and self._size + additional > self._capacity:
                oldest_name = next(iter(self._versions))
                if oldest_name == name:
                    break
                evicted = self._versions.pop(oldest_name)
                self._size -= evicted.stored_bytes
            if self._size + additional > self._capacity:
                raise VersionTooLargeError("shard cannot fit in configured capacity")
            stored.shards[index] = retained
            self._size += additional
            return memoryview(retained).toreadonly()

    def has_shard(self, name: str, index: int) -> bool:
        with self._lock:
            stored = self._versions.get(name)
            if stored is None:
                return False
            if stored.data is not None:
                return 0 <= index < stored.metadata.shard_count
            return index in stored.shards

    def available_shards(self, name: str) -> Tuple[int, ...]:
        with self._lock:
            stored = self._versions.get(name)
            if stored is None:
                return ()
            if stored.data is not None:
                return tuple(range(stored.metadata.shard_count))
            return tuple(sorted(stored.shards))

    def is_complete(self, name: str) -> bool:
        with self._lock:
            stored = self._versions.get(name)
            return stored is not None and (
                stored.data is not None
                or len(stored.shards) == stored.metadata.shard_count
            )

    def get(self, name: str) -> Optional[memoryview]:
        """Return a read-only view of the complete checkpoint, or ``None``."""
        with self._lock:
            stored = self._versions.get(name)
            return (
                None
                if stored is None or stored.data is None
                else memoryview(stored.data).toreadonly()
            )

    def get_metadata(self, name: str) -> Optional[VersionMetadata]:
        with self._lock:
            stored = self._versions.get(name)
            return None if stored is None else stored.metadata

    def get_shard(self, name: str, index: int) -> memoryview:
        """Return a zero-copy, read-only, zero-based shard slice."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index must be int")
        with self._lock:
            stored = self._versions.get(name)
            if stored is None:
                raise VersionNotFoundError(name)
            if index < 0 or index >= stored.metadata.shard_count:
                raise InvalidShardIndexError(index)
            if stored.data is None:
                shard = stored.shards.get(index)
                if shard is None:
                    raise VersionNotFoundError(
                        "{} shard {} is not available".format(name, index)
                    )
                return memoryview(shard).toreadonly()
            start = index * stored.metadata.shard_size
            end = min(start + stored.metadata.shard_size, stored.metadata.total_size)
            return memoryview(stored.data)[start:end].toreadonly()

    def get_shard_checksum(self, name: str, index: int) -> str:
        """Return a manifest checksum without rehashing when one is registered."""
        _, checksum = self.get_shard_with_checksum(name, index)
        return checksum

    def get_shard_with_checksum(
        self, name: str, index: int
    ) -> Tuple[memoryview, str]:
        """Return one coherent shard/checksum snapshot."""
        with self._lock:
            stored = self._versions.get(name)
            if stored is None:
                raise VersionNotFoundError(name)
            if index < 0 or index >= stored.metadata.shard_count:
                raise InvalidShardIndexError(index)
            if stored.data is None:
                retained = stored.shards.get(index)
                if retained is None:
                    raise VersionNotFoundError(
                        "{} shard {} is not available".format(name, index)
                    )
                shard = memoryview(retained).toreadonly()
            else:
                start = index * stored.metadata.shard_size
                end = min(
                    start + stored.metadata.shard_size,
                    stored.metadata.total_size,
                )
                shard = memoryview(stored.data)[start:end].toreadonly()
            checksum = (
                stored.shard_checksums[index]
                if stored.shard_checksums
                else blake3.blake3(shard).hexdigest()
            )
            return shard, checksum

    def remove(self, name: str) -> bool:
        """Remove a version, returning whether it existed."""
        with self._lock:
            stored = self._versions.pop(name, None)
            if stored is None:
                return False
            self._size -= stored.stored_bytes
            return True

    def list(self) -> List[VersionMetadata]:
        """Return metadata in deterministic oldest-to-newest order."""
        with self._lock:
            return [stored.metadata for stored in self._versions.values()]

    def list_versions(self) -> List[VersionMetadata]:
        """Named alias for :meth:`list`."""
        return self.list()

    def clear(self) -> None:
        with self._lock:
            self._versions.clear()
            self._size = 0


# A shorter name is convenient for callers and keeps the abstraction reusable.
VersionStore = CheckpointStore


def _make_immutable(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    try:
        view = memoryview(data)
    except TypeError:
        raise TypeError("data must support the buffer protocol")
    return view.tobytes()


def _take_buffer_ownership(data: Any) -> Any:
    if isinstance(data, (bytes, bytearray)):
        return data
    return _make_immutable(data)


def _validate_shard_checksums(
    shard_checksums: Tuple[str, ...], shard_count: int
) -> Tuple[str, ...]:
    checksums = tuple(shard_checksums)
    if checksums and len(checksums) != shard_count:
        raise ValueError("one checksum is required per shard")
    if any(not isinstance(value, str) or len(value) != 64 for value in checksums):
        raise ValueError("invalid shard checksum")
    return checksums


def _validate_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("name must be str")
    if not name:
        raise ValueError("name must not be empty")


def _validate_shard_size(shard_size: int) -> None:
    if (
        not isinstance(shard_size, int)
        or isinstance(shard_size, bool)
        or shard_size <= 0
    ):
        raise ValueError("shard_size must be a positive integer")
