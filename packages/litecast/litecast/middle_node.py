"""Middle node for the litecast package."""

import os
import time
import argparse
import threading
import shutil
import blake3
from dataclasses import replace
from typing import List, Dict, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import litecast.server as server
from litecast.client import ShardDownloader
from litecast.client_node import ClientNode
from litecast.store import (
    CheckpointStore, VersionMetadata, VersionTooLargeError
)
from litecast.transport.ucxx import UCXXServer
from litecast.endpoints import (
    EndpointMetadata, UCXXEndpoint, STREAMING_ENDPOINT_SCHEMA,
    write_endpoint_metadata,
)
from litecast.streaming import (
    MANIFEST_FILENAME, MAX_MANIFEST_BYTES, ShardManifest,
    canonical_members, canonical_member_profiles, shard_owners,
)
from litecast.events import emit_event
from litecast.envs import (
    LITECAST_HTTP_PORT,
    LITECAST_DISTRIBUTION_FILE,
    LITECAST_HTTP_TIMEOUT,
    LITECAST_VERSION_PREFIX,
    LITECAST_TRANSPORT,
    LITECAST_UCXX_PORT,
    LITECAST_UCXX_INTERFACE,
    LITECAST_UCXX_CONNECT_TIMEOUT,
    LITECAST_RAM_CACHE_BYTES,
    LITECAST_DISK_MIRROR,
    LITECAST_ENDPOINTS_FILE,
    LITECAST_MIDDLE_ID,
    LITECAST_MIDDLE_SERVERS,
    LITECAST_REPLICATION_FACTOR,
    LITECAST_MAX_CONCURRENT_DOWNLOADS,
)
from litecast.utils import (
    ensure_dir,
    logger,
    update_distribution_file,
)


class MiddleNode:
    """Middle node for downloading and re-serving shards."""

    def __init__(
        self,
        upstream_servers: List[str],
        data_dir: str,
        port: int = LITECAST_HTTP_PORT,
        check_interval: int = 30,
        transport: str = LITECAST_TRANSPORT,
        ucxx_port: int = LITECAST_UCXX_PORT,
        ucxx_interface: str = LITECAST_UCXX_INTERFACE,
        ucxx_connect_timeout: float = LITECAST_UCXX_CONNECT_TIMEOUT,
        ram_cache_bytes: int = LITECAST_RAM_CACHE_BYTES,
        disk_mirror: bool = LITECAST_DISK_MIRROR,
        endpoints_file: str = LITECAST_ENDPOINTS_FILE,
        middle_id: str = LITECAST_MIDDLE_ID,
        middle_servers: Optional[List[str]] = None,
        replication_factor: int = LITECAST_REPLICATION_FACTOR,
        member_profiles: Optional[Dict[str, Dict[str, object]]] = None,
    ):
        """Initialize the middle node.

        Args:
            upstream_servers: List of upstream server URLs or IP addresses
            data_dir: Directory to store and serve shards from
            port: HTTP port to listen on
            check_interval: Interval in seconds to check for new versions
        """
        self.upstream_servers = upstream_servers
        self.data_dir = os.path.abspath(data_dir)
        self.port = port
        self.check_interval = check_interval
        self.transport = transport.lower()
        if self.transport not in ("auto", "http", "ucxx"):
            raise ValueError("transport must be auto, http, or ucxx")
        self.disk_mirror = disk_mirror
        if middle_servers is None and LITECAST_MIDDLE_SERVERS:
            middle_servers = [
                item.strip() for item in LITECAST_MIDDLE_SERVERS.split(",")
                if item.strip()
            ]
        self.middle_servers = (
            canonical_members(middle_servers) if middle_servers else ()
        )
        self.middle_id = middle_id.rstrip("/") if middle_id else ""
        self.replication_factor = replication_factor
        self.member_profiles = (
            canonical_member_profiles(
                self.middle_servers, member_profiles or {}
            )
            if self.middle_servers
            else ()
        )
        self.member_profile_map = {
            member: {
                "bandwidth": bandwidth,
                "capacity": capacity,
                "load": load,
                "failure_domain": failure_domain,
                "locality": locality,
            }
            for (
                member,
                bandwidth,
                capacity,
                load,
                failure_domain,
                locality,
            ) in self.member_profiles
        }
        if self.middle_servers and self.middle_id not in self.middle_servers:
            raise ValueError("middle_id must be present in middle_servers")
        self.store = CheckpointStore(ram_cache_bytes)

        # Create downloader for fetching from upstream servers
        self.downloader = ShardDownloader(upstream_servers, LITECAST_HTTP_TIMEOUT)
        self.client = ClientNode(
            upstream_servers, data_dir, self.transport, ram_cache_bytes,
            ucxx_connect_timeout, endpoints_file,
        )

        # Track which versions we have already processed
        self.processed_versions: Set[str] = set()

        # Track known shards per version
        self.known_shards: Dict[str, int] = {}

        # Processing lock
        self.lock = threading.Lock()

        # Shutdown event
        self.shutdown_event = threading.Event()

        # Ensure data directory exists
        ensure_dir(self.data_dir)

        # Start HTTP server
        self.http_server, self.server_thread = server.run_server(
            self.data_dir, self.port, self.shutdown_event, self.store
        )
        self.port = self.http_server.server_port
        self.ucxx_server = None
        if self.transport == "ucxx" and ucxx_port == 0:
            self.shutdown_event.set()
            self.http_server.shutdown()
            self.server_thread.join(5)
            self.http_server.server_close()
            raise ValueError("ucxx transport requires a non-zero UCXX port")
        if self.transport != "http" and ucxx_port:
            candidate = UCXXServer(self.store, ucxx_interface, ucxx_port)
            try:
                candidate.start(ucxx_connect_timeout)
                self.ucxx_server = candidate
                self.client.ucxx_executor = candidate.run_coroutine
            except Exception:
                if self.transport == "ucxx":
                    self.shutdown_event.set()
                    self.http_server.shutdown()
                    self.server_thread.join(5)
                    self.http_server.server_close()
                    raise
                logger.warning("UCXX startup failed; continuing with HTTP", exc_info=True)
        endpoint_path = endpoints_file if os.path.isabs(endpoints_file) else os.path.join(self.data_dir, endpoints_file)
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
                self.middle_id or None,
                (
                    "get_shard",
                    "get_shards",
                    "wait_version",
                    "partial_shards",
                ),
            ),
        )

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_upstream)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        emit_event(
            "middle_started",
            upstream_servers=self.upstream_servers,
            http_port=self.port,
            transport=self.transport,
            ucxx_port=None if self.ucxx_server is None else self.ucxx_server.port,
        )

    def _monitor_upstream(self) -> None:
        """Monitor upstream servers for new versions."""
        while not self.shutdown_event.is_set():
            try:
                # Download and parse distribution file
                payload = self.downloader.download_bytes(LITECAST_DISTRIBUTION_FILE, retries=1)
                if payload is not None:
                    distribution_content = payload.decode("utf-8")
                else:
                    logger.warning("Failed to download distribution file; retaining last-good manifest")
                    distribution_content = None

                if distribution_content:
                    emit_event(
                        "upstream_manifest_received",
                        bytes=len(distribution_content.encode("utf-8")),
                    )
                    # Parse the distribution file into a dictionary
                    distribution = {}
                    for line in distribution_content.strip().split("\n"):
                        if line and ":" in line:
                            version, info = line.strip().split(":", 1)
                            distribution[version.strip()] = info.strip()

                    # Process each version
                    for version, info in distribution.items():
                        self._process_version(version, info)

                    manifest_path = os.path.join(self.data_dir, LITECAST_DISTRIBUTION_FILE)
                    temporary = manifest_path + ".tmp"
                    local_versions = {
                        item.name for item in self.store.list_versions()
                    }
                    with open(temporary, "w") as handle:
                        for version, info in distribution.items():
                            if (
                                version in local_versions
                                or os.path.isdir(os.path.join(self.data_dir, version))
                            ):
                                handle.write("{}: {}\n".format(version, info))
                    os.replace(temporary, manifest_path)
                    if self.ucxx_server is not None:
                        for version in local_versions:
                            self.ucxx_server.notify_version_available(version)

                    # Remove old versions if necessary
                    for version in Path(self.data_dir).glob(f"{LITECAST_VERSION_PREFIX}*"):
                        if version.stem not in distribution:
                            shutil.rmtree(os.path.join(self.data_dir, version.stem))
                            logger.info(f"Removed old version: {version.stem}")
                    for metadata in self.store.list_versions():
                        if metadata.name not in distribution:
                            self.store.remove(metadata.name)
                    for version in self.processed_versions - set(distribution):
                        self.processed_versions.discard(version)
                        self.known_shards.pop(version, None)

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Error monitoring upstream servers: {str(e)}")

            # Wait before checking again
            self.shutdown_event.wait(self.check_interval)

    def _process_version(self, version: str, info: str) -> None:
        """Process a version by downloading and serving its shards.

        Args:
            version: Version folder name (e.g., "v1")
            info: Information string with checksum and optionally shard count
        """
        with self.lock:
            if version in self.processed_versions:
                return
            # Extract checksum and expected shard count.
            checksum, _, shard_count = info.partition("|")
            if not shard_count:
                return
            try:
                expected_count = int(shard_count)
            except ValueError:
                logger.error("Invalid shard count for version %s", version)
                return
            if expected_count < 0:
                logger.error("Invalid shard count for version %s", version)
                return
            if getattr(self, "middle_servers", ()):
                manifest_payload = None
                for upstream in self.upstream_servers:
                    manifest_payload = self.downloader.download_bytes_from(
                        upstream,
                        "{}/{}".format(version, MANIFEST_FILENAME),
                        MAX_MANIFEST_BYTES,
                    )
                    if manifest_payload is not None:
                        break
                if manifest_payload is not None:
                    try:
                        manifest = ShardManifest.from_bytes(manifest_payload)
                    except ValueError:
                        logger.warning("Invalid streaming manifest for %s", version)
                    else:
                        if (
                            manifest.name == version
                            and manifest.checksum == checksum
                            and manifest.shard_count == expected_count
                        ):
                            manifest = replace(
                                manifest,
                                members=self.middle_servers,
                                replication_factor=self.replication_factor,
                                member_profiles=self.member_profiles,
                            )
                            self._process_streaming_version(
                                manifest, manifest.to_bytes()
                            )
                            return
            emit_event(
                "middle_download_started",
                version=version,
                shard_count=expected_count,
                checksum=checksum,
            )
            data = self.client.download_version_buffer(version)
            if data is None:
                if self.disk_mirror:
                    version_dir = os.path.join(self.data_dir, version)
                    ensure_dir(version_dir)
                    shards = self.downloader.download_shards(
                        version, expected_count, version_dir
                    )
                    if len(shards) != expected_count:
                        shutil.rmtree(version_dir, ignore_errors=True)
                        logger.warning(
                            "Version %s was incomplete or failed verification",
                            version,
                        )
                        return
                    digest = blake3.blake3()
                    for path in sorted(shards):
                        with open(path, "rb") as handle:
                            while True:
                                chunk = handle.read(1024 * 1024)
                                if not chunk:
                                    break
                                digest.update(chunk)
                    if digest.hexdigest() != checksum:
                        shutil.rmtree(version_dir, ignore_errors=True)
                        logger.error("Checksum mismatch for version %s", version)
                        return
                    self.known_shards[version] = expected_count
                    self.processed_versions.add(version)
                    emit_event(
                        "middle_version_ready",
                        version=version,
                        selected_transport="http-disk",
                        shard_count=expected_count,
                    )
                    return
                logger.warning(
                    "Version %s was incomplete or exceeded the RAM limit",
                    version,
                )
                return
            try:
                inferred_shard_size = max(
                    1, (len(data) + expected_count - 1) // expected_count
                ) if expected_count else 1
                metadata = self.store.put(version, data, inferred_shard_size, checksum)
            except (VersionTooLargeError, ValueError):
                logger.error("Unable to cache verified version %s", version, exc_info=True)
                return
            if metadata.shard_count != expected_count:
                self.store.remove(version)
                return
            if self.disk_mirror:
                version_dir = os.path.join(self.data_dir, version)
                ensure_dir(version_dir)
                for index in range(metadata.shard_count):
                    with open(os.path.join(version_dir, "shard_{:05d}.bin".format(index + 1)), "wb") as handle:
                        handle.write(self.store.get_shard(version, index))
            self.known_shards[version] = expected_count
            self.processed_versions.add(version)
            emit_event(
                "middle_version_ready",
                version=version,
                selected_transport=self.transport,
                shard_count=expected_count,
                total_size=metadata.total_size,
            )
            logger.info(f"Finished downloading all {expected_count} shards for version {version}")
            return

    def _fetch_owned_shard(
        self, manifest: ShardManifest, index: int, source: str, role: str
    ) -> int:
        path = "{}/shard_{:05d}.bin".format(manifest.name, index + 1)
        deadline = time.monotonic() + max(300, self.check_interval * 10)
        delay = 0.05
        while time.monotonic() < deadline and not self.shutdown_event.is_set():
            payload = None
            verified_checksum = None
            endpoint = (
                self.client._discover_endpoint_for(source)
                if self.transport != "http"
                else None
            )
            if (
                endpoint is not None
                and endpoint.ucxx is not None
                and endpoint.verified_rdma
                and "get_shard" in endpoint.capabilities
            ):
                try:
                    result = self.client._ucxx_transport_for(
                        endpoint, self.client.ucxx_executor, index
                    ).fetch_shard(
                        manifest.name,
                        index,
                        manifest.shard_checksums[index],
                    )
                    payload = result.data
                    verified_checksum = result.checksum
                except Exception:
                    payload = None
            if payload is None and self.transport != "ucxx":
                payload = self.downloader.download_bytes_from(
                    source, path, manifest.shard_length(index)
                )
            if payload is not None:
                try:
                    if verified_checksum is None:
                        self.store.put_shard(manifest.name, index, payload)
                    else:
                        self.store.put_verified_shard(
                            manifest.name,
                            index,
                            payload,
                            verified_checksum,
                        )
                except ValueError:
                    payload = None
                else:
                    if self.disk_mirror:
                        version_dir = os.path.join(
                            self.data_dir, manifest.name
                        )
                        ensure_dir(version_dir)
                        with open(
                            os.path.join(
                                version_dir,
                                "shard_{:05d}.bin".format(index + 1),
                            ),
                            "wb",
                        ) as handle:
                            handle.write(payload)
                    emit_event(
                        "middle_shard_available",
                        version=manifest.name,
                        shard_index=index,
                        shard_role=role,
                        source=source,
                        bytes=len(payload),
                    )
                    return index
            time.sleep(delay)
            delay = min(delay * 2, 1.0)
        raise RuntimeError(
            "{} shard {} did not become available".format(role, index)
        )

    def _process_streaming_version(
        self, manifest: ShardManifest, manifest_payload: bytes
    ) -> None:
        metadata = VersionMetadata(
            manifest.name,
            manifest.total_size,
            manifest.shard_size,
            manifest.shard_count,
            manifest.checksum,
        )
        self.store.begin_version(
            metadata, manifest.shard_checksums
        )
        version_dir = os.path.join(self.data_dir, manifest.name)
        ensure_dir(version_dir)
        manifest_path = os.path.join(version_dir, MANIFEST_FILENAME)
        temporary = manifest_path + ".tmp"
        with open(temporary, "wb") as handle:
            handle.write(manifest_payload)
        os.replace(temporary, manifest_path)
        update_distribution_file(
            os.path.join(self.data_dir, LITECAST_DISTRIBUTION_FILE),
            manifest.name,
            manifest.checksum,
            100000,
            manifest.shard_count,
        )
        if self.ucxx_server is not None:
            self.ucxx_server.notify_version_available(manifest.name)
        emit_event(
            "middle_version_discoverable",
            version=manifest.name,
            shard_count=manifest.shard_count,
        )

        primaries = []
        replicas = []
        for index in range(manifest.shard_count):
            owners = shard_owners(
                manifest.name,
                index,
                self.middle_servers,
                self.replication_factor,
                self.member_profile_map,
            )
            if owners[0] == self.middle_id:
                primaries.append(
                    (index, self.upstream_servers[0], "primary")
                )
            elif self.middle_id in owners[1:]:
                replicas.append((index, owners[0], "replica"))
        assignments = primaries + replicas
        emit_event(
            "middle_streaming_started",
            version=manifest.name,
            owned_shards=len(assignments),
            replication_factor=self.replication_factor,
        )
        # Replica requests may wait on another middle's primary. Separate
        # pools prevent those waits from occupying every local primary worker.
        with ThreadPoolExecutor(
            max_workers=LITECAST_MAX_CONCURRENT_DOWNLOADS
        ) as primary_executor, ThreadPoolExecutor(
            max_workers=LITECAST_MAX_CONCURRENT_DOWNLOADS
        ) as replica_executor:
            futures = [
                primary_executor.submit(
                    self._fetch_owned_shard, manifest, index, source, role
                )
                for index, source, role in primaries
            ] + [
                replica_executor.submit(
                    self._fetch_owned_shard, manifest, index, source, role
                )
                for index, source, role in replicas
            ]
            for future in as_completed(futures):
                future.result()
        self.known_shards[manifest.name] = len(assignments)
        self.processed_versions.add(manifest.name)
        emit_event(
            "middle_version_ready",
            version=manifest.name,
            selected_transport="replicated-streaming",
            shard_count=len(assignments),
            total_size=sum(
                len(self.store.get_shard(manifest.name, index))
                for index, _, _ in assignments
            ),
        )

    def shutdown(self) -> None:
        """Shutdown the middle node."""
        logger.info("Shutting down middle node...")
        self.shutdown_event.set()
        self.client.close()
        if self.ucxx_server is not None:
            self.ucxx_server.close()
        self.http_server.shutdown()
        self.monitor_thread.join(max(self.check_interval + 1, 2))
        self.server_thread.join(5)
        self.http_server.server_close()


def main():
    """Run the middle node as a standalone script."""
    parser = argparse.ArgumentParser(description="Litecast Middle Node")
    parser.add_argument(
        "--upstream",
        help="Comma-separated list of upstream server URLs or IP addresses (optional if IP_ADDR_LIST env var is set)",
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
    parser.add_argument(
        "--middle-id",
        default=LITECAST_MIDDLE_ID,
        help="This middle's stable URL in the canonical membership list",
    )
    parser.add_argument(
        "--middle-servers",
        default=LITECAST_MIDDLE_SERVERS,
        help="Comma-separated canonical middle URLs enabling RF=2 streaming",
    )
    parser.add_argument(
        "--replication-factor",
        type=int,
        default=LITECAST_REPLICATION_FACTOR,
    )
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--disk-mirror", dest="disk_mirror", action="store_true")
    mirror_group.add_argument("--no-disk-mirror", dest="disk_mirror", action="store_false")
    parser.set_defaults(disk_mirror=LITECAST_DISK_MIRROR)
    parser.add_argument(
        "--data-dir",
        default="./middle_data",
        help="Directory to store and serve shards from",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=LITECAST_HTTP_PORT,
        help=f"HTTP port to listen on (default: {LITECAST_HTTP_PORT})",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Interval in seconds to check for new versions (default: 30)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(args.log_level)

    # Check if IP_ADDR_LIST environment variable is set
    import os
    import re

    upstream_servers = []

    if args.upstream:
        # Parse from command-line argument
        upstream_servers = [s.strip() for s in args.upstream.split(",") if s.strip()]
    else:
        # Try to get from environment variable
        ip_addr_list = os.environ.get("IP_ADDR_LIST")
        if not ip_addr_list:
            logger.error("IP_ADDR_LIST environment variable not set and --upstream not provided")
            logger.error("Set IP_ADDR_LIST='ip1 ip2 ip3' or use --upstream parameter")
            return 1

        # Parse the environment variable - expected format: ("ip1" "ip2" "ip3")
        # Remove parentheses if present
        ip_addr_list = ip_addr_list.strip()
        if ip_addr_list.startswith("(") and ip_addr_list.endswith(")"):
            ip_addr_list = ip_addr_list[1:-1].strip()

        # Extract IPs within quotes
        quoted_ips = re.findall(r'"([^"]+)"', ip_addr_list)
        if quoted_ips:
            upstream_servers = quoted_ips
        else:
            # If no quoted IPs found, try space-separated format
            upstream_servers = [s.strip() for s in ip_addr_list.split() if s.strip()]

    if not upstream_servers:
        logger.error("No upstream servers specified")
        return 1

    logger.info(f"Using upstream servers: {upstream_servers}")

    # Start the middle node
    node = MiddleNode(
        upstream_servers=upstream_servers,
        data_dir=args.data_dir,
        port=args.port,
        check_interval=args.check_interval,
        transport=args.transport,
        ucxx_port=args.ucxx_port,
        ucxx_interface=args.ucxx_interface,
        ucxx_connect_timeout=args.ucxx_connect_timeout,
        ram_cache_bytes=args.ram_cache_bytes,
        disk_mirror=args.disk_mirror,
        middle_id=args.middle_id,
        middle_servers=[
            item.strip() for item in args.middle_servers.split(",") if item.strip()
        ] if args.middle_servers else None,
        replication_factor=args.replication_factor,
    )

    try:
        logger.info(f"Middle node running at http://{server.get_local_ip()}:{args.port}")
        logger.info(f"Serving files from {os.path.abspath(args.data_dir)}")
        logger.info(f"Monitoring upstream servers: {', '.join(upstream_servers)}")
        logger.info("Press Ctrl+C to exit")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        node.shutdown()

    return 0


if __name__ == "__main__":
    exit(main())
