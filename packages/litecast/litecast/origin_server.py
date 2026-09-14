"""Origin server for the litecast package."""

import os
import time
import shutil
import argparse
import threading
import re
from pathlib import Path

import litecast.server as server
from litecast.envs import (
    LITECAST_SHARD_SIZE,
    LITECAST_MAX_DISTRIBUTION_FOLDERS,
    LITECAST_HTTP_PORT,
    LITECAST_DISTRIBUTION_FILE,
    LITECAST_TRANSPORT,
    LITECAST_UCXX_PORT,
    LITECAST_UCXX_INTERFACE,
    LITECAST_UCXX_CONNECT_TIMEOUT,
    LITECAST_RAM_CACHE_BYTES,
    LITECAST_DISK_MIRROR,
    LITECAST_ENDPOINTS_FILE,
    LITECAST_VERSION_PREFIX,
)
from litecast.store import CheckpointStore
from litecast.events import emit_event
from litecast.endpoints import (
    EndpointMetadata, UCXXEndpoint, STREAMING_ENDPOINT_SCHEMA,
    write_endpoint_metadata,
)
from litecast.transport.ucxx import UCXXServer
from litecast.streaming import MANIFEST_FILENAME, build_manifest
from litecast.utils import (
    compute_checksum,
    update_distribution_file,
    remove_distribution_versions,
    get_shard_count,
    get_shard_filename,
    ensure_dir,
    logger,
)


def _transport_mode(value: str) -> str:
    mode = value.lower()
    if mode not in ("auto", "http", "ucxx"):
        raise ValueError("transport must be auto, http, or ucxx")
    return mode


class OriginServer:
    """Origin server for sharding and distributing files."""

    def __init__(
        self, data_dir: str, port: int = LITECAST_HTTP_PORT,
        max_distribution_folders: int = LITECAST_MAX_DISTRIBUTION_FOLDERS,
        transport: str = LITECAST_TRANSPORT, ucxx_port: int = LITECAST_UCXX_PORT,
        ucxx_interface: str = LITECAST_UCXX_INTERFACE,
        ucxx_connect_timeout: float = LITECAST_UCXX_CONNECT_TIMEOUT,
        ram_cache_bytes: int = LITECAST_RAM_CACHE_BYTES,
        disk_mirror: bool = LITECAST_DISK_MIRROR,
        endpoints_file: str = LITECAST_ENDPOINTS_FILE,
    ):
        """Initialize the origin server.

        Args:
            data_dir: Directory to store and serve shards from
            port: HTTP port to listen on
        """
        self.data_dir = os.path.abspath(data_dir)
        self.port = port
        self.max_distribution_folders = max_distribution_folders
        self.transport = _transport_mode(transport)
        self.disk_mirror = disk_mirror
        self.store = CheckpointStore(ram_cache_bytes)
        self.shutdown_event = threading.Event()
        self._publication_lock = threading.RLock()

        # Ensure data directory exists
        ensure_dir(self.data_dir)

        # Start HTTP server
        self.http_server, self.server_thread = server.run_server(
            self.data_dir, self.port, self.shutdown_event, self.store
        )
        self.port = self.http_server.server_port
        self.ucxx_server = None
        if self.transport == "ucxx" and ucxx_port == 0:
            self.shutdown()
            raise ValueError("ucxx transport requires a non-zero UCXX port")
        if self.transport != "http" and ucxx_port:
            candidate = UCXXServer(self.store, ucxx_interface, ucxx_port)
            try:
                candidate.start(ucxx_connect_timeout)
                self.ucxx_server = candidate
            except Exception:
                if self.transport == "ucxx":
                    self.shutdown()
                    raise
                logger.warning("UCXX startup failed; continuing with HTTP", exc_info=True)

        # Create distribution file if it doesn't exist
        dist_file = os.path.join(self.data_dir, LITECAST_DISTRIBUTION_FILE)
        if not os.path.exists(dist_file):
            with open(dist_file, "w") as f:  # noqa: F841
                pass
        endpoint_path = endpoints_file
        if not os.path.isabs(endpoint_path):
            endpoint_path = os.path.join(self.data_dir, endpoint_path)
        ucxx_endpoint = None
        advertised_host = server.get_local_ip()
        if self.ucxx_server is not None:
            ucxx_endpoint = UCXXEndpoint(
                self.ucxx_server.address, self.ucxx_server.port, ucxx_interface,
                self.ucxx_server.verified_rdma,
            )
            advertised_host = self.ucxx_server.address
        write_endpoint_metadata(
            endpoint_path,
            EndpointMetadata(
                "http://{}:{}".format(advertised_host, self.port),
                ucxx_endpoint,
                STREAMING_ENDPOINT_SCHEMA,
                None,
                ("get_shard", "get_shards", "wait_version"),
            ),
        )
        emit_event(
            "origin_started",
            http_port=self.port,
            transport=self.transport,
            ucxx_port=None if self.ucxx_server is None else self.ucxx_server.port,
        )

    def _next_version(self) -> str:
        pattern = re.compile(r"^{}(\d+)$".format(re.escape(LITECAST_VERSION_PREFIX)))
        names = list(self.store.list_versions())
        candidates = [item.name for item in names]
        dist = os.path.join(self.data_dir, LITECAST_DISTRIBUTION_FILE)
        if os.path.exists(dist):
            with open(dist) as handle:
                candidates.extend(line.split(":", 1)[0].strip() for line in handle if ":" in line)
        candidates.extend(os.listdir(self.data_dir))
        maximum = max((int(m.group(1)) for name in candidates for m in [pattern.match(name)] if m), default=0)
        return "{}{}".format(LITECAST_VERSION_PREFIX, maximum + 1)

    def next_version_name(self) -> str:
        """Return the next publication name for pre-warmed consumers."""
        with self._publication_lock:
            return self._next_version()

    def broadcast_buffer(
        self, buffer, shard_size: int = LITECAST_SHARD_SIZE, disk_mirror=None
    ) -> str:
        with self._publication_lock:
            return self._broadcast_buffer_unlocked(
                buffer, shard_size, disk_mirror
            )

    def _broadcast_buffer_unlocked(
        self, buffer, shard_size: int, disk_mirror=None
    ) -> str:
        mirror = self.disk_mirror if disk_mirror is None else bool(disk_mirror)
        version = self._next_version()
        emit_event("broadcast_started", version=version, disk_mirror=mirror)
        previous_versions = {
            item.name for item in self.store.list_versions()
        }
        shard_manifest = build_manifest(version, buffer, shard_size)
        metadata = self.store.put_verified(
            version,
            buffer,
            shard_size,
            checksum=shard_manifest.checksum,
            shard_checksums=shard_manifest.shard_checksums,
        )
        current_versions = {
            item.name for item in self.store.list_versions()
        }
        capacity_evictions = previous_versions - current_versions
        version_dir = os.path.join(self.data_dir, version)
        ensure_dir(version_dir)
        manifest_path = Path(version_dir) / MANIFEST_FILENAME
        temporary_manifest = manifest_path.with_suffix(".tmp")
        temporary_manifest.write_bytes(shard_manifest.to_bytes())
        os.replace(str(temporary_manifest), str(manifest_path))
        if mirror:
            for index in range(metadata.shard_count):
                with open(os.path.join(version_dir, get_shard_filename(index)), "wb") as handle:
                    handle.write(self.store.get_shard(version, index))
        removed = update_distribution_file(
            os.path.join(self.data_dir, LITECAST_DISTRIBUTION_FILE), version,
            metadata.checksum, self.max_distribution_folders, metadata.shard_count,
        )
        if not mirror:
            remove_distribution_versions(
                os.path.join(self.data_dir, LITECAST_DISTRIBUTION_FILE),
                list(capacity_evictions),
            )
            for old in capacity_evictions:
                shutil.rmtree(os.path.join(self.data_dir, old), ignore_errors=True)
        for old in removed:
            self.store.remove(old)
            old_dir = os.path.join(self.data_dir, old)
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)
        emit_event(
            "version_published",
            version=version,
            total_size=metadata.total_size,
            shard_size=metadata.shard_size,
            shard_count=metadata.shard_count,
            checksum=metadata.checksum,
            manifest_schema=shard_manifest.schema,
        )
        if self.ucxx_server is not None:
            self.ucxx_server.notify_version_available(version)
        return version

    def broadcast(
        self, file_path: str, shard_size: int = LITECAST_SHARD_SIZE
    ) -> str:
        with self._publication_lock:
            return self._broadcast_unlocked(file_path, shard_size)

    def _broadcast_unlocked(
        self, file_path: str, shard_size: int
    ) -> str:
        """Broadcast a file by sharding it and making it available for download.

        Args:
            file_path: Path to the file to broadcast
            shard_size: Size of each shard in bytes

        Returns:
            Version folder name
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size <= self.store.capacity_bytes:
            with open(file_path, "rb") as handle:
                return self.broadcast_buffer(
                    handle.read(), shard_size, disk_mirror=True
                )

        # Preserve the streaming legacy path for checkpoints larger than the
        # configured RAM cache instead of reading the complete file into RAM.
        version = self._next_version()
        version_dir = os.path.join(self.data_dir, version)
        ensure_dir(version_dir)
        num_shards = get_shard_count(file_size, shard_size)
        with open(file_path, "rb") as source:
            for index in range(num_shards):
                with open(
                    os.path.join(version_dir, get_shard_filename(index)), "wb"
                ) as handle:
                    handle.write(source.read(shard_size))
        checksum = compute_checksum(file_path)
        removed = update_distribution_file(
            os.path.join(self.data_dir, LITECAST_DISTRIBUTION_FILE),
            version,
            checksum,
            self.max_distribution_folders,
            num_shards,
        )
        for old in removed:
            self.store.remove(old)
            old_dir = os.path.join(self.data_dir, old)
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)
        return version

    def shutdown(self) -> None:
        """Shutdown the server."""
        logger.info("Shutting down origin server...")
        self.shutdown_event.set()
        if self.ucxx_server is not None:
            self.ucxx_server.close()
        self.http_server.shutdown()
        self.server_thread.join(5)
        self.http_server.server_close()


def main():
    """Run the origin server as a standalone script."""
    parser = argparse.ArgumentParser(description="Litecast Origin Server")
    parser.add_argument("--data-dir", default="./data", help="Directory to store and serve shards from")
    parser.add_argument(
        "--port",
        type=int,
        default=LITECAST_HTTP_PORT,
        help=f"HTTP port to listen on (default: {LITECAST_HTTP_PORT})",
    )
    parser.add_argument("--transport", choices=["auto", "http", "ucxx"], default=LITECAST_TRANSPORT)
    parser.add_argument("--ucxx-port", type=int, default=LITECAST_UCXX_PORT)
    parser.add_argument("--ucxx-interface", default=LITECAST_UCXX_INTERFACE)
    parser.add_argument(
        "--ucxx-connect-timeout",
        type=float,
        default=LITECAST_UCXX_CONNECT_TIMEOUT,
    )
    parser.add_argument("--ram-cache-bytes", type=int, default=LITECAST_RAM_CACHE_BYTES)
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--disk-mirror", dest="disk_mirror", action="store_true")
    mirror_group.add_argument("--no-disk-mirror", dest="disk_mirror", action="store_false")
    parser.set_defaults(disk_mirror=LITECAST_DISK_MIRROR)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(args.log_level)

    # Start the origin server
    origin = OriginServer(
        args.data_dir, args.port, transport=args.transport, ucxx_port=args.ucxx_port,
        ucxx_interface=args.ucxx_interface,
        ucxx_connect_timeout=args.ucxx_connect_timeout,
        ram_cache_bytes=args.ram_cache_bytes,
        disk_mirror=args.disk_mirror,
    )

    try:
        logger.info(f"Origin server running at http://{server.get_local_ip()}:{args.port}")
        logger.info(f"Serving files from {os.path.abspath(args.data_dir)}")
        logger.info("Press Ctrl+C to exit")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        origin.shutdown()


if __name__ == "__main__":
    main()
