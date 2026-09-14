"""Litecast: A package for distributing large files via HTTP."""

from litecast.origin_server import OriginServer
from litecast.middle_node import MiddleNode
from litecast.client_node import ClientNode
from litecast.envs import (
    LITECAST_SHARD_SIZE,
    LITECAST_MAX_DISTRIBUTION_FOLDERS,
    LITECAST_HTTP_PORT,
)
from litecast.utils import logger

# Create origin server instance
_origin_server = None


def initialize(
    data_dir: str = "./data",
    port: int = LITECAST_HTTP_PORT,
    max_distribution_folders: int = LITECAST_MAX_DISTRIBUTION_FOLDERS,
    **origin_options,
) -> None:
    """Initialize the litecast package.

    Args:
        data_dir: Directory to store and serve shards from
        port: HTTP port to listen on
    """
    global _origin_server

    # Initialize the origin server
    if _origin_server is None:
        _origin_server = OriginServer(
            data_dir, port, max_distribution_folders, **origin_options
        )
        logger.info(f"Litecast initialized with data directory: {data_dir}")
    else:
        logger.warning("Litecast already initialized")


def broadcast(file_path: str, shard_size: int = LITECAST_SHARD_SIZE) -> str:
    """Broadcast a file by sharding it and making it available for download.

    Args:
        file_path: Path to the file to broadcast
        shard_size: Size of each shard in bytes

    Returns:
        Version folder name
    """
    global _origin_server

    if _origin_server is None:
        raise RuntimeError("Litecast not initialized. Call initialize() first.")

    return _origin_server.broadcast(file_path, shard_size)


def broadcast_buffer(buffer, shard_size: int = LITECAST_SHARD_SIZE, disk_mirror=None) -> str:
    """Publish one immutable complete version from a bytes-like object."""
    if _origin_server is None:
        raise RuntimeError("Litecast not initialized. Call initialize() first.")
    return _origin_server.broadcast_buffer(buffer, shard_size, disk_mirror)


def shutdown() -> None:
    """Shutdown the litecast package."""
    global _origin_server

    if _origin_server is not None:
        _origin_server.shutdown()
        _origin_server = None
        logger.info("Litecast shutdown complete")


__all__ = [
    "initialize",
    "broadcast",
    "broadcast_buffer",
    "shutdown",
    "OriginServer",
    "MiddleNode",
    "ClientNode",
]

__version__ = "0.3.2"
