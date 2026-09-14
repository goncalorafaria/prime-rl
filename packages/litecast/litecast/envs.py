"""Constants for the litecast package."""

from typing import TYPE_CHECKING, Any, List
import os

if TYPE_CHECKING:
    LITECAST_SHARD_SIZE: int = 134_217_728
    LITECAST_MAX_DISTRIBUTION_FOLDERS: int = 5
    LITECAST_HTTP_PORT: int = 8000
    LITECAST_RETRY_ATTEMPTS: int = 5
    LITECAST_FAST_RETRY_ATTEMPTS: int = 3
    LITECAST_FAST_RETRY_INTERVAL: int = 2
    LITECAST_SLOW_RETRY_INTERVAL: int = 15
    LITECAST_LOG_LEVEL: str = "INFO"
    LITECAST_DISTRIBUTION_FILE: str = "distribution.txt"
    LITECAST_HTTP_TIMEOUT: int = 30
    LITECAST_MAX_CONCURRENT_DOWNLOADS: int = 10
    LITECAST_VERSION_PREFIX: str = "v"
    LITECAST_TRANSPORT: str = "auto"
    LITECAST_UCXX_PORT: int = 0
    LITECAST_UCXX_INTERFACE: str = "ib0"
    LITECAST_UCXX_CONNECT_TIMEOUT: float = 30.0
    LITECAST_RAM_CACHE_BYTES: int = 1073741824
    LITECAST_DISK_MIRROR: bool = True
    LITECAST_ENDPOINTS_FILE: str = "endpoints.json"
    LITECAST_MIDDLE_ID: str = ""
    LITECAST_MIDDLE_SERVERS: str = ""
    LITECAST_REPLICATION_FACTOR: int = 2
    LITECAST_STREAMING_RETRY_SECONDS: float = 300.0
    LITECAST_UCXX_LANES: int = 4
    LITECAST_SHARD_BATCH_SIZE: int = 8


def _bool_env(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError("{} must be a boolean".format(name))

_env = {
    "LITECAST_SHARD_SIZE": lambda: int(os.getenv("LITECAST_SHARD_SIZE", "134217728")),
    "LITECAST_MAX_DISTRIBUTION_FOLDERS": lambda: int(os.getenv("LITECAST_MAX_DISTRIBUTION_FOLDERS", "5")),
    "LITECAST_HTTP_PORT": lambda: int(os.getenv("LITECAST_HTTP_PORT", "8000")),
    "LITECAST_RETRY_ATTEMPTS": lambda: int(os.getenv("LITECAST_RETRY_ATTEMPTS", "5")),
    "LITECAST_FAST_RETRY_ATTEMPTS": lambda: int(os.getenv("LITECAST_FAST_RETRY_ATTEMPTS", "3")),
    "LITECAST_FAST_RETRY_INTERVAL": lambda: int(os.getenv("LITECAST_FAST_RETRY_INTERVAL", "2")),
    "LITECAST_SLOW_RETRY_INTERVAL": lambda: int(os.getenv("LITECAST_SLOW_RETRY_INTERVAL", "15")),
    "LITECAST_LOG_LEVEL": lambda: os.getenv("LITECAST_LOG_LEVEL", "INFO"),
    "LITECAST_DISTRIBUTION_FILE": lambda: os.getenv("LITECAST_DISTRIBUTION_FILE", "distribution.txt"),
    "LITECAST_HTTP_TIMEOUT": lambda: int(os.getenv("LITECAST_HTTP_TIMEOUT", "30")),
    "LITECAST_MAX_CONCURRENT_DOWNLOADS": lambda: int(os.getenv("LITECAST_MAX_CONCURRENT_DOWNLOADS", "10")),
    "LITECAST_VERSION_PREFIX": lambda: os.getenv("LITECAST_VERSION_PREFIX", "v"),
    "LITECAST_TRANSPORT": lambda: os.getenv("LITECAST_TRANSPORT", "auto").strip().lower(),
    "LITECAST_UCXX_PORT": lambda: int(os.getenv("LITECAST_UCXX_PORT", "0")),
    "LITECAST_UCXX_INTERFACE": lambda: os.getenv("LITECAST_UCXX_INTERFACE", "ib0"),
    "LITECAST_UCXX_CONNECT_TIMEOUT": lambda: float(os.getenv("LITECAST_UCXX_CONNECT_TIMEOUT", "30")),
    "LITECAST_RAM_CACHE_BYTES": lambda: int(os.getenv("LITECAST_RAM_CACHE_BYTES", "1073741824")),
    "LITECAST_DISK_MIRROR": lambda: _bool_env("LITECAST_DISK_MIRROR", "true"),
    "LITECAST_ENDPOINTS_FILE": lambda: os.getenv("LITECAST_ENDPOINTS_FILE", "endpoints.json"),
    "LITECAST_MIDDLE_ID": lambda: os.getenv("LITECAST_MIDDLE_ID", ""),
    "LITECAST_MIDDLE_SERVERS": lambda: os.getenv("LITECAST_MIDDLE_SERVERS", ""),
    "LITECAST_REPLICATION_FACTOR": lambda: int(os.getenv("LITECAST_REPLICATION_FACTOR", "2")),
    "LITECAST_STREAMING_RETRY_SECONDS": lambda: float(os.getenv("LITECAST_STREAMING_RETRY_SECONDS", "300")),
    "LITECAST_UCXX_LANES": lambda: int(os.getenv("LITECAST_UCXX_LANES", "4")),
    "LITECAST_SHARD_BATCH_SIZE": lambda: int(os.getenv("LITECAST_SHARD_BATCH_SIZE", "8")),
}


def __getattr__(name: str) -> Any:
    if name not in _env:
        raise AttributeError(f"Invalid environment variable: {name}")
    return _env[name]()


def __dir__() -> List[str]:
    return list(_env.keys())
