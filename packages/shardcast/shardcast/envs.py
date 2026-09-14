"""Constants for the shardcast package."""

from typing import TYPE_CHECKING, Any, List
import os

if TYPE_CHECKING:
    SHARDCAST_SHARD_SIZE: int = 134_217_728
    SHARDCAST_MAX_DISTRIBUTION_FOLDERS: int = 5
    SHARDCAST_HTTP_PORT: int = 8000
    SHARDCAST_RETRY_ATTEMPTS: int = 5
    SHARDCAST_FAST_RETRY_ATTEMPTS: int = 3
    SHARDCAST_FAST_RETRY_INTERVAL: int = 2
    SHARDCAST_SLOW_RETRY_INTERVAL: int = 15
    SHARDCAST_LOG_LEVEL: str = "INFO"
    SHARDCAST_DISTRIBUTION_FILE: str = "distribution.txt"
    SHARDCAST_HTTP_TIMEOUT: int = 30
    SHARDCAST_MAX_CONCURRENT_DOWNLOADS: int = 10
    SHARDCAST_VERSION_PREFIX: str = "v"
    SHARDCAST_TRANSPORT: str = "auto"
    SHARDCAST_UCXX_PORT: int = 0
    SHARDCAST_UCXX_INTERFACE: str = "ib0"
    SHARDCAST_UCXX_CONNECT_TIMEOUT: float = 30.0
    SHARDCAST_RAM_CACHE_BYTES: int = 1073741824
    SHARDCAST_DISK_MIRROR: bool = True
    SHARDCAST_ENDPOINTS_FILE: str = "endpoints.json"
    SHARDCAST_MIDDLE_ID: str = ""
    SHARDCAST_MIDDLE_SERVERS: str = ""
    SHARDCAST_REPLICATION_FACTOR: int = 2
    SHARDCAST_STREAMING_RETRY_SECONDS: float = 300.0
    SHARDCAST_UCXX_LANES: int = 4
    SHARDCAST_SHARD_BATCH_SIZE: int = 8


def _bool_env(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError("{} must be a boolean".format(name))

_env = {
    "SHARDCAST_SHARD_SIZE": lambda: int(os.getenv("SHARDCAST_SHARD_SIZE", "134217728")),
    "SHARDCAST_MAX_DISTRIBUTION_FOLDERS": lambda: int(os.getenv("SHARDCAST_MAX_DISTRIBUTION_FOLDERS", "5")),
    "SHARDCAST_HTTP_PORT": lambda: int(os.getenv("SHARDCAST_HTTP_PORT", "8000")),
    "SHARDCAST_RETRY_ATTEMPTS": lambda: int(os.getenv("SHARDCAST_RETRY_ATTEMPTS", "5")),
    "SHARDCAST_FAST_RETRY_ATTEMPTS": lambda: int(os.getenv("SHARDCAST_FAST_RETRY_ATTEMPTS", "3")),
    "SHARDCAST_FAST_RETRY_INTERVAL": lambda: int(os.getenv("SHARDCAST_FAST_RETRY_INTERVAL", "2")),
    "SHARDCAST_SLOW_RETRY_INTERVAL": lambda: int(os.getenv("SHARDCAST_SLOW_RETRY_INTERVAL", "15")),
    "SHARDCAST_LOG_LEVEL": lambda: os.getenv("SHARDCAST_LOG_LEVEL", "INFO"),
    "SHARDCAST_DISTRIBUTION_FILE": lambda: os.getenv("SHARDCAST_DISTRIBUTION_FILE", "distribution.txt"),
    "SHARDCAST_HTTP_TIMEOUT": lambda: int(os.getenv("SHARDCAST_HTTP_TIMEOUT", "30")),
    "SHARDCAST_MAX_CONCURRENT_DOWNLOADS": lambda: int(os.getenv("SHARDCAST_MAX_CONCURRENT_DOWNLOADS", "10")),
    "SHARDCAST_VERSION_PREFIX": lambda: os.getenv("SHARDCAST_VERSION_PREFIX", "v"),
    "SHARDCAST_TRANSPORT": lambda: os.getenv("SHARDCAST_TRANSPORT", "auto").strip().lower(),
    "SHARDCAST_UCXX_PORT": lambda: int(os.getenv("SHARDCAST_UCXX_PORT", "0")),
    "SHARDCAST_UCXX_INTERFACE": lambda: os.getenv("SHARDCAST_UCXX_INTERFACE", "ib0"),
    "SHARDCAST_UCXX_CONNECT_TIMEOUT": lambda: float(os.getenv("SHARDCAST_UCXX_CONNECT_TIMEOUT", "30")),
    "SHARDCAST_RAM_CACHE_BYTES": lambda: int(os.getenv("SHARDCAST_RAM_CACHE_BYTES", "1073741824")),
    "SHARDCAST_DISK_MIRROR": lambda: _bool_env("SHARDCAST_DISK_MIRROR", "true"),
    "SHARDCAST_ENDPOINTS_FILE": lambda: os.getenv("SHARDCAST_ENDPOINTS_FILE", "endpoints.json"),
    "SHARDCAST_MIDDLE_ID": lambda: os.getenv("SHARDCAST_MIDDLE_ID", ""),
    "SHARDCAST_MIDDLE_SERVERS": lambda: os.getenv("SHARDCAST_MIDDLE_SERVERS", ""),
    "SHARDCAST_REPLICATION_FACTOR": lambda: int(os.getenv("SHARDCAST_REPLICATION_FACTOR", "2")),
    "SHARDCAST_STREAMING_RETRY_SECONDS": lambda: float(os.getenv("SHARDCAST_STREAMING_RETRY_SECONDS", "300")),
    "SHARDCAST_UCXX_LANES": lambda: int(os.getenv("SHARDCAST_UCXX_LANES", "4")),
    "SHARDCAST_SHARD_BATCH_SIZE": lambda: int(os.getenv("SHARDCAST_SHARD_BATCH_SIZE", "8")),
}


def __getattr__(name: str) -> Any:
    if name not in _env:
        raise AttributeError(f"Invalid environment variable: {name}")
    return _env[name]()


def __dir__() -> List[str]:
    return list(_env.keys())
