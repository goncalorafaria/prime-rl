"""Small transport interface for fetching complete checkpoint versions."""

from dataclasses import dataclass
from typing import Optional, Protocol, Union, runtime_checkable

from litecast.store import VersionMetadata


Buffer = Union[bytes, bytearray]


class TransportError(Exception):
    """Base class for data-transport failures."""


class TransportUnavailableError(TransportError):
    """The requested optional transport cannot be used."""


class TransportProtocolError(TransportError, ValueError):
    """A peer sent malformed or inconsistent data."""


class TransportIntegrityError(TransportProtocolError):
    """Transferred bytes do not match their advertised digest."""


class VersionUnavailableError(TransportError, KeyError):
    """The remote endpoint does not have the requested version."""


@dataclass(frozen=True)
class FetchResult:
    """One complete version fetched into CPU memory."""

    data: Buffer
    metadata: VersionMetadata
    transport: str

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray)):
            raise TypeError("data must be bytes or bytearray")
        if len(self.data) != self.metadata.total_size:
            raise ValueError("data size does not match metadata")

    @property
    def checksum(self) -> str:
        return self.metadata.checksum


@runtime_checkable
class VersionTransport(Protocol):
    """Protocol implemented by complete-version CPU transports."""

    def fetch_version(
        self, version: str, max_total_size: Optional[int] = None
    ) -> FetchResult:
        ...

    def fetch_bytes(
        self, version: str, max_total_size: Optional[int] = None
    ) -> Buffer:
        ...

    def close(self) -> None:
        ...


Transport = VersionTransport
