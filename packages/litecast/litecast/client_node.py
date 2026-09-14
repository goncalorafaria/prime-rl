"""Client node for downloading and reassembling shards."""

import os
import ctypes
import re
import sys
import threading
import time
import argparse
from typing import Any, Callable, List, Dict, Optional, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen

import blake3

from litecast.client import ShardDownloader
from litecast.endpoints import MAX_ENDPOINTS_JSON_SIZE, parse_endpoint_metadata
from litecast.transport.ucxx import UCXXEventLoopExecutor, UCXXTransport
from litecast.streaming import (
    MANIFEST_FILENAME,
    MAX_MANIFEST_BYTES,
    ShardManifest,
    canonical_members,
    merkle_root,
    order_owners_for_client,
    shard_owners,
)
from litecast.events import emit_event
from litecast.envs import (
    LITECAST_TRANSPORT, LITECAST_UCXX_CONNECT_TIMEOUT,
    LITECAST_RAM_CACHE_BYTES, LITECAST_ENDPOINTS_FILE,
    LITECAST_MAX_CONCURRENT_DOWNLOADS,
    LITECAST_REPLICATION_FACTOR,
    LITECAST_RETRY_ATTEMPTS,
    LITECAST_STREAMING_RETRY_SECONDS,
    LITECAST_UCXX_LANES,
    LITECAST_SHARD_BATCH_SIZE,
)
from litecast.utils import (
    ensure_dir,
    get_shard_filename,
    logger,
)

_CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_DISTRIBUTION_BYTES = 1024 * 1024
_MAX_SHARD_COUNT = 1_000_000


class ClientNode:
    """Client node for downloading and reassembling shards."""

    def __init__(
        self, servers: List[str], output_dir: str = "./downloads",
        transport: str = LITECAST_TRANSPORT,
        max_version_bytes: int = LITECAST_RAM_CACHE_BYTES,
        connect_timeout: float = LITECAST_UCXX_CONNECT_TIMEOUT,
        endpoints_file: str = LITECAST_ENDPOINTS_FILE,
        ucxx_executor: Optional[Callable[[Any], Any]] = None,
        replication_factor: int = LITECAST_REPLICATION_FACTOR,
        ucxx_lanes: int = LITECAST_UCXX_LANES,
        shard_batch_size: int = LITECAST_SHARD_BATCH_SIZE,
        client_locality: str = "",
    ):
        """Initialize the client node.

        Args:
            servers: List of server URLs or IP addresses
            output_dir: Directory to save downloaded files
        """
        self.servers = servers
        self.output_dir = os.path.abspath(output_dir)
        self.transport = transport.lower()
        if self.transport not in ("auto", "http", "ucxx"):
            raise ValueError("transport must be auto, http, or ucxx")
        if (
            not isinstance(max_version_bytes, int)
            or isinstance(max_version_bytes, bool)
            or max_version_bytes < 0
        ):
            raise ValueError("max_version_bytes must be non-negative")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        self.max_version_bytes = max_version_bytes
        self.connect_timeout = connect_timeout
        self.endpoints_file = endpoints_file
        self.ucxx_executor = ucxx_executor
        self.replication_factor = replication_factor
        if not 1 <= ucxx_lanes <= 16:
            raise ValueError("ucxx_lanes must be between 1 and 16")
        if not 1 <= shard_batch_size <= 256:
            raise ValueError("shard_batch_size must be between 1 and 256")
        self.ucxx_lanes = ucxx_lanes
        self.shard_batch_size = shard_batch_size
        self.client_locality = client_locality
        self._endpoint_cache = {}
        self._ucxx_transports = {}
        self._ucxx_transports_lock = threading.Lock()

        # Create downloader
        self.downloader = ShardDownloader(servers)

        # Ensure output directory exists
        ensure_dir(self.output_dir)
        emit_event(
            "client_started",
            servers=self.servers,
            transport=self.transport,
        )

    def list_available_versions(
        self, retries: int = LITECAST_RETRY_ATTEMPTS
    ) -> Dict[str, str]:
        """List available versions from the distribution file.

        Returns:
            Dictionary mapping version names to file checksums
        """
        # Download distribution file
        payload = self.downloader.download_bytes(
            "distribution.txt",
            retries=retries,
            max_bytes=_MAX_DISTRIBUTION_BYTES,
        )
        if payload is None:
            logger.debug("Failed to download distribution file. Upstream server isn't ready yet.")
            return {}
        try:
            distribution_content = payload.decode("utf-8")
        except UnicodeDecodeError:
            return {}

        # Parse the distribution file
        distribution = {}
        for line in distribution_content.strip().split("\n"):
            if line and ":" in line:
                version, info = line.strip().split(":", 1)
                distribution[version.strip()] = info.strip()

        return distribution

    def download_version(self, version: str, output_file: Optional[str] = None) -> Optional[str]:
        """Download a specific version and reassemble the original file.

        Args:
            version: Version to download (e.g., "v1")
            output_file: Path to save the reassembled file, or None to use default

        Returns:
            Path to the reassembled file, or None if failed
        """
        data = self.download_version_buffer(version)
        if data is None:
            return None
        if output_file is None:
            output_file = os.path.join(self.output_dir, "download_{}.bin".format(version))
        else:
            output_file = os.path.abspath(output_file)
        parent = os.path.dirname(output_file)
        if parent:
            ensure_dir(parent)
        with open(output_file, "wb") as handle:
            handle.write(data)
        return output_file

    def _discover_endpoint(self):
        fallback = None
        for base in self.servers:
            try:
                url = "{}/{}".format(base.rstrip("/"), self.endpoints_file.lstrip("/"))
                with urlopen(url, timeout=self.connect_timeout) as response:
                    payload = response.read(MAX_ENDPOINTS_JSON_SIZE + 1)
                    endpoint = parse_endpoint_metadata(payload)
                if endpoint.verified_rdma:
                    return endpoint
                if fallback is None:
                    fallback = endpoint
            except Exception:
                continue
        return fallback

    def _discover_endpoint_for(self, base: str):
        if base in self._endpoint_cache:
            return self._endpoint_cache[base]
        try:
            url = "{}/{}".format(
                base.rstrip("/"), self.endpoints_file.lstrip("/")
            )
            with urlopen(url, timeout=self.connect_timeout) as response:
                endpoint = parse_endpoint_metadata(
                    response.read(MAX_ENDPOINTS_JSON_SIZE + 1)
                )
        except Exception:
            endpoint = None
        if endpoint is not None:
            self._endpoint_cache[base] = endpoint
        return endpoint

    def _ucxx_transport_for(self, endpoint, executor, lane_hint=0):
        key = (endpoint.ucxx.host, endpoint.ucxx.port, id(executor))
        with self._ucxx_transports_lock:
            transports = self._ucxx_transports.get(key)
            if transports is None:
                transports = [
                    UCXXTransport(
                        endpoint.ucxx.host,
                        endpoint.ucxx.port,
                        self.connect_timeout,
                        self.max_version_bytes,
                        executor=executor,
                    )
                    for _ in range(self.ucxx_lanes)
                ]
                self._ucxx_transports[key] = transports
            return transports[lane_hint % len(transports)]

    def warm_ucxx_connections(self) -> None:
        """Discover and connect to all RDMA peers before timed work."""
        if self.transport == "http" or self.ucxx_executor is None:
            return
        for server in self.servers:
            endpoint = self._discover_endpoint_for(server)
            if (
                endpoint is not None
                and endpoint.ucxx is not None
                and endpoint.verified_rdma
                and "get_shard" in endpoint.capabilities
            ):
                for lane in range(self.ucxx_lanes):
                    self._ucxx_transport_for(
                        endpoint, self.ucxx_executor, lane
                    ).warm()

    def wait_for_version(self, version: str, server: str) -> None:
        """Wait for an explicit UCXX publication notification from a peer."""
        endpoint = self._discover_endpoint_for(server)
        if (
            endpoint is None
            or endpoint.ucxx is None
            or not endpoint.verified_rdma
            or "wait_version" not in endpoint.capabilities
            or self.ucxx_executor is None
        ):
            raise RuntimeError("peer does not support UCXX publication notification")
        self._ucxx_transport_for(
            endpoint, self.ucxx_executor, 0
        ).wait_for_version(version)

    def close(self) -> None:
        with self._ucxx_transports_lock:
            transports = [
                transport
                for lanes in self._ucxx_transports.values()
                for transport in lanes
            ]
            self._ucxx_transports.clear()
        for transport in transports:
            transport.close()

    def _load_shard_manifest(self, version: str) -> Optional[ShardManifest]:
        path = "{}/{}".format(version, MANIFEST_FILENAME)
        for attempt in range(LITECAST_RETRY_ATTEMPTS):
            for server in self.servers:
                payload = self.downloader.download_bytes_from(
                    server, path, MAX_MANIFEST_BYTES
                )
                if payload is None:
                    continue
                try:
                    manifest = ShardManifest.from_bytes(payload)
                except ValueError:
                    continue
                if manifest.name == version:
                    return manifest
            if attempt + 1 < LITECAST_RETRY_ATTEMPTS:
                time.sleep(min(0.05 * (2 ** attempt), 1.0))
        return None

    def download_shards_into(
        self, version: str, targets: Mapping[int, Any]
    ) -> ShardManifest:
        """Fetch only selected zero-based shards into caller-owned CPU buffers.

        Return the verified manifest after every requested write has completed.
        Buffers must be writable, contiguous, disjoint, and exactly shard-sized.
        On failure they may contain partial data; callers must not activate or
        reuse them until this method returns or raises. No full-version buffer
        is allocated. Model-to-shard mapping belongs to the caller.
        """
        manifest = self._load_shard_manifest(version)
        if manifest is None:
            raise ValueError("version has no valid shard manifest")
        published = self.list_available_versions().get(version)
        if published != "{}|{}".format(manifest.checksum, manifest.shard_count):
            raise ValueError("shard manifest does not match distribution manifest")
        views = {}
        spans = []
        for index, buffer in targets.items():
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("shard index must be int")
            size = manifest.shard_length(index)
            view = memoryview(buffer)
            if view.readonly or not view.c_contiguous:
                raise ValueError("targets must be writable contiguous CPU buffers")
            view = view.cast("B")
            if len(view) != size:
                raise ValueError("target size does not match shard manifest")
            address = ctypes.addressof(ctypes.c_char.from_buffer(view))
            spans.append((address, address + size))
            views[index] = view
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("target buffers must not overlap")
        if sum(len(view) for view in views.values()) > self.max_version_bytes:
            raise ValueError("selected shards exceed max_version_bytes")
        if self._download_streaming(manifest, views) is None:
            raise RuntimeError("selected shard transfer failed")
        return manifest

    def _download_streaming(
        self, manifest: ShardManifest, targets: Optional[Mapping[int, memoryview]] = None
    ):
        if targets is None and manifest.total_size > self.max_version_bytes:
            return None
        if (
            manifest.members and (
                manifest.members != canonical_members(self.servers)
                or manifest.replication_factor != self.replication_factor
            )
        ):
            raise ValueError("client membership does not match shard manifest")
        data = bytearray(manifest.total_size) if targets is None else None
        view = memoryview(data) if data is not None else None
        selected = (
            {index: view[index * manifest.shard_size:
                         index * manifest.shard_size + manifest.shard_length(index)]
             for index in range(manifest.shard_count)}
            if targets is None else dict(targets)
        )
        members = manifest.members or canonical_members(self.servers)
        replication_factor = manifest.replication_factor if manifest.members else len(members)
        member_profiles = manifest.profiles
        local_ucxx_executor = None
        ucxx_executor = self.ucxx_executor
        if self.transport != "http" and ucxx_executor is None:
            local_ucxx_executor = UCXXEventLoopExecutor(
                max(self.connect_timeout * 4, 120)
            )
            ucxx_executor = local_ucxx_executor

        def receive(index: int) -> int:
            target = selected[index]
            owners = shard_owners(
                manifest.name,
                index,
                members,
                replication_factor,
                member_profiles,
            )
            owners = order_owners_for_client(
                owners, member_profiles, self.client_locality
            )
            path = "{}/shard_{:05d}.bin".format(manifest.name, index + 1)
            deadline = time.monotonic() + LITECAST_STREAMING_RETRY_SECONDS
            attempt = 0
            while time.monotonic() < deadline:
                for owner in owners:
                    endpoint = (
                        self._discover_endpoint_for(owner)
                        if self.transport != "http"
                        else None
                    )
                    used_ucxx = (
                        endpoint is not None
                        and endpoint.ucxx is not None
                        and endpoint.verified_rdma
                        and "get_shard" in endpoint.capabilities
                    )
                    if used_ucxx:
                        try:
                            self._ucxx_transport_for(
                                endpoint, ucxx_executor, index
                            ).fetch_shard_into(
                                manifest.name,
                                index,
                                target,
                                manifest.shard_checksums[index],
                            )
                            emit_event(
                                "client_shard_received",
                                version=manifest.name,
                                shard_index=index,
                                owner=owner,
                                selected_transport="ucxx",
                            )
                            return index
                        except Exception:
                            if self.transport == "ucxx":
                                continue
                    if self.transport != "ucxx" and self.downloader.download_into_from(
                        owner, path, target
                    ):
                        if (
                            blake3.blake3(target).hexdigest()
                            == manifest.shard_checksums[index]
                        ):
                            emit_event(
                                "client_shard_received",
                                version=manifest.name,
                                shard_index=index,
                                owner=owner,
                                selected_transport="http",
                            )
                            return index
                time.sleep(min(0.05 * (2 ** min(attempt, 5)), 1.0))
                attempt += 1
            raise RuntimeError(
                "shard {} unavailable from both owners".format(index)
            )

        def receive_batch(
            owner: str, indices: List[int], lane: int
        ) -> List[int]:
            endpoint = (
                self._discover_endpoint_for(owner)
                if self.transport != "http"
                else None
            )
            if (
                endpoint is None
                or endpoint.ucxx is None
                or not endpoint.verified_rdma
                or "get_shards" not in endpoint.capabilities
                or ucxx_executor is None
            ):
                return [receive(index) for index in indices]
            targets = {}
            checksums = {}
            for index in indices:
                targets[index] = selected[index]
                checksums[index] = manifest.shard_checksums[index]
            try:
                result = self._ucxx_transport_for(
                    endpoint, ucxx_executor, lane
                ).fetch_shards_into(
                    manifest.name, targets, checksums
                )
            except Exception:
                return [receive(index) for index in indices]
            completed = list(result.completed)
            for index in completed:
                emit_event(
                    "client_shard_received",
                    version=manifest.name,
                    shard_index=index,
                    owner=owner,
                    selected_transport="ucxx-batch",
                )
            completed.extend(
                receive(index) for index in result.unavailable
            )
            return completed

        emit_event(
            "download_started" if targets is None else "shards_download_started",
            version=manifest.name,
            selected_transport="replicated-streaming",
        )
        ready = set()
        frontier = 0
        digest = blake3.blake3() if targets is None and manifest.merkle_root is None else None
        try:
            with ThreadPoolExecutor(
                max_workers=LITECAST_MAX_CONCURRENT_DOWNLOADS
            ) as executor:
                groups = {}
                for index in selected:
                    owners = shard_owners(
                        manifest.name,
                        index,
                        members,
                        replication_factor,
                        member_profiles,
                    )
                    owners = order_owners_for_client(
                        owners, member_profiles, self.client_locality
                    )
                    groups.setdefault(owners[0], []).append(index)
                futures = []
                lane = 0
                for owner, indices in groups.items():
                    for offset in range(
                        0, len(indices), self.shard_batch_size
                    ):
                        batch = indices[
                            offset:offset + self.shard_batch_size
                        ]
                        futures.append(
                            executor.submit(
                                receive_batch, owner, batch, lane
                            )
                        )
                        lane += 1
                for future in as_completed(futures):
                    ready.update(future.result())
                    while targets is None and frontier in ready:
                        start = frontier * manifest.shard_size
                        end = start + manifest.shard_length(frontier)
                        if digest is not None:
                            digest.update(view[start:end])
                        ready.remove(frontier)
                        frontier += 1
        except Exception:
            if local_ucxx_executor is not None:
                self.close()
                local_ucxx_executor.close()
            emit_event(
                "download_failed" if targets is None else "shards_download_failed",
                version=manifest.name,
                selected_transport="replicated-streaming",
                reason="shard_unavailable",
            )
            if self.transport == "ucxx":
                raise
            return None
        if local_ucxx_executor is not None:
            self.close()
            local_ucxx_executor.close()
        if digest is not None:
            if digest.hexdigest() != manifest.checksum:
                return None
        elif manifest.merkle_root is not None and merkle_root(manifest.shard_checksums) != manifest.merkle_root:
            return None
        emit_event(
            "download_completed" if targets is None else "shards_download_completed",
            version=manifest.name,
            selected_transport="replicated-streaming",
            total_size=sum(len(target) for target in selected.values()),
            checksum=manifest.checksum,
            shard_count=len(selected),
        )
        return data if targets is None else selected

    def download_version_buffer(self, version: str):
        """Download and verify a complete version without shard temp files."""
        versions = self.list_available_versions()
        if version not in versions:
            return None
        checksum, separator, count_text = versions[version].partition("|")
        if not separator or _CHECKSUM_RE.fullmatch(checksum) is None:
            return None
        try:
            shard_count = int(count_text)
        except ValueError:
            return None
        emit_event(
            "manifest_received",
            version=version,
            shard_count=shard_count,
            checksum=checksum,
        )
        streaming_manifest = (
            self._load_shard_manifest(version)
            if len(self.servers) >= 2
            else None
        )
        if streaming_manifest is not None and len(self.servers) >= 2:
            if (
                streaming_manifest.checksum != checksum
                or streaming_manifest.shard_count != shard_count
            ):
                raise ValueError(
                    "streaming manifest does not match distribution manifest"
                )
            data = self._download_streaming(streaming_manifest)
            if data is not None or self.transport == "ucxx":
                return data
        if (
            shard_count < 0
            or shard_count > _MAX_SHARD_COUNT
            or shard_count > self.max_version_bytes
        ):
            return None

        endpoint = self._discover_endpoint() if self.transport != "http" else None
        advertised = endpoint is not None and endpoint.ucxx is not None and endpoint.verified_rdma
        if self.transport == "ucxx" and not advertised:
            raise RuntimeError("UCXX transport requires an advertised verified_rdma endpoint")
        if advertised:
            try:
                emit_event(
                    "download_started",
                    version=version,
                    selected_transport="ucxx",
                    server=endpoint.ucxx.host,
                )
                result = UCXXTransport(
                    endpoint.ucxx.host, endpoint.ucxx.port, self.connect_timeout,
                    self.max_version_bytes, executor=self.ucxx_executor,
                ).fetch_version(version)
                if result.metadata.checksum != checksum or result.metadata.shard_count != shard_count:
                    raise ValueError("UCXX metadata does not match distribution manifest")
                emit_event(
                    "download_completed",
                    version=version,
                    selected_transport="ucxx",
                    total_size=len(result.data),
                    checksum=checksum,
                )
                return result.data
            except Exception:
                if self.transport == "ucxx":
                    raise
                logger.warning("UCXX download failed; falling back to HTTP", exc_info=True)

        emit_event(
            "download_started",
            version=version,
            selected_transport="http",
        )
        with ThreadPoolExecutor(max_workers=LITECAST_MAX_CONCURRENT_DOWNLOADS) as executor:
            sizes = list(
                executor.map(
                    lambda index: self.downloader.get_size(
                        "{}/shard_{:05d}.bin".format(version, index + 1)
                    ),
                    range(shard_count),
                )
            )
        if any(size is None for size in sizes):
            return None
        total = sum(sizes)
        if total > self.max_version_bytes:
            return None
        data = bytearray(total)
        view = memoryview(data)
        offsets = []
        offset = 0
        for size in sizes:
            offsets.append((offset, offset + size))
            offset += size
        with ThreadPoolExecutor(
            max_workers=LITECAST_MAX_CONCURRENT_DOWNLOADS
        ) as executor:
            successes = list(
                executor.map(
                    lambda item: self.downloader.download_into(
                        "{}/shard_{:05d}.bin".format(version, item[0] + 1),
                        view[item[1][0] : item[1][1]],
                    ),
                    enumerate(offsets),
                )
            )
        if not all(successes):
            return None
        if blake3.blake3(data).hexdigest() != checksum:
            emit_event(
                "download_failed",
                version=version,
                selected_transport="http",
                reason="checksum_mismatch",
            )
            return None
        emit_event(
            "download_completed",
            version=version,
            selected_transport="http",
            total_size=len(data),
            checksum=checksum,
        )
        return data

    def _discover_and_download_shards(self, version: str, output_dir: str) -> List[str]:
        """Discover and download all shards for a version.

        Args:
            version: Version to download (e.g., "v1")
            output_dir: Directory to save shards

        Returns:
            List of paths to downloaded shards
        """
        # Start timer for download process
        start_time = time.time()

        # Get version info from distribution file to check for known shard count
        available_versions = self.list_available_versions()
        known_shard_count = None

        if version in available_versions:
            known_shard_count = int(available_versions[version].partition("|")[2])

        if known_shard_count is not None:
            logger.info(f"Distribution file indicates {known_shard_count} shards for version {version}")
            # Download all shards at once since we know how many there are
            shards = self.downloader.download_shards(version, known_shard_count, output_dir)

            # Calculate total downloaded size
            total_size = sum(os.path.getsize(shard) for shard in shards if os.path.exists(shard))

            # Calculate download speed
            download_time = time.time() - start_time
            download_speed_bps = total_size / max(download_time, 0.001)
            download_speed_mbps = download_speed_bps / (1024 * 1024)

            logger.info(f"Downloaded {total_size / (1024 * 1024):.2f} MB in {download_time:.2f} seconds")
            logger.info(f"Average download speed: {download_speed_mbps:.2f} MB/s")
            logger.info(f"Metrics: {self.downloader.server_metrics}")

            return shards

        # If shard count is unknown, use discovery mode
        logger.info(f"Discovering shards for version {version} (count unknown)")

        # Start with a reasonable number of shards to try
        initial_shard_count = 10

        # Download the first batch of shards
        initial_shards = self.downloader.download_shards(version, initial_shard_count, output_dir)

        if not initial_shards:
            return []

        # If we got all the initial shards, try to find more
        if len(initial_shards) == initial_shard_count:
            logger.info(f"Found at least {initial_shard_count} shards, searching for more")

            # Continue looking for more shards until we get a failure
            max_shard_index = initial_shard_count
            consecutive_failures = 0
            max_consecutive_failures = 3

            while consecutive_failures < max_consecutive_failures:
                max_shard_index += 1
                shard_filename = get_shard_filename(max_shard_index - 1)
                shard_path = os.path.join(output_dir, shard_filename)
                url_path = f"{version}/{shard_filename}"

                if self.downloader.download_file(url_path, shard_path):
                    initial_shards.append(shard_path)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.debug(f"Failed to find shard {max_shard_index} (failure {consecutive_failures}/{max_consecutive_failures})")

            # Calculate total downloaded size
            total_size = sum(os.path.getsize(shard) for shard in initial_shards if os.path.exists(shard))

            # Calculate download speed
            download_time = time.time() - start_time
            download_speed_bps = total_size / max(download_time, 0.001)
            download_speed_mbps = download_speed_bps / (1024 * 1024)

            logger.info(f"Downloaded {total_size / (1024 * 1024):.2f} MB in {download_time:.2f} seconds")
            logger.info(f"Average download speed: {download_speed_mbps:.2f} MB/s")

            logger.info(f"Found {len(initial_shards)} total shards")

        return initial_shards


def main():
    """Run the client node as a standalone script."""
    parser = argparse.ArgumentParser(description="Litecast Client Node")
    parser.add_argument(
        "--servers",
        help="Comma-separated list of server URLs or IP addresses (optional if IP_ADDR_LIST env var is set)",
    )
    parser.add_argument("--output-dir", default="./downloads", help="Directory to save downloaded files")
    parser.add_argument("--list", action="store_true", help="List available versions and exit")
    parser.add_argument("--version", help="Version to download (e.g., 'v1')")
    parser.add_argument("--output-file", help="Output file path for the reassembled file")
    parser.add_argument("--transport", choices=["auto", "http", "ucxx"], default=LITECAST_TRANSPORT)
    parser.add_argument("--max-version-bytes", type=int, default=LITECAST_RAM_CACHE_BYTES)
    parser.add_argument(
        "--replication-factor",
        type=int,
        default=LITECAST_REPLICATION_FACTOR,
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=LITECAST_UCXX_CONNECT_TIMEOUT,
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

    servers = []

    if args.servers:
        # Parse from command-line argument
        servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    else:
        # Try to get from environment variable
        ip_addr_list = os.environ.get("IP_ADDR_LIST")
        if not ip_addr_list:
            logger.error("IP_ADDR_LIST environment variable not set and --servers not provided")
            logger.error("Set IP_ADDR_LIST='ip1 ip2 ip3' or use --servers parameter")
            return 1

        # Parse the environment variable - expected format: ("ip1" "ip2" "ip3")
        # Remove parentheses if present
        ip_addr_list = ip_addr_list.strip()
        if ip_addr_list.startswith("(") and ip_addr_list.endswith(")"):
            ip_addr_list = ip_addr_list[1:-1].strip()

        # Extract IPs within quotes
        quoted_ips = re.findall(r'"([^"]+)"', ip_addr_list)
        if quoted_ips:
            servers = quoted_ips
        else:
            # If no quoted IPs found, try space-separated format
            servers = [s.strip() for s in ip_addr_list.split() if s.strip()]

    if not servers:
        logger.error("No servers specified")
        return 1

    logger.info(f"Using servers: {servers}")

    # Create client
    client = ClientNode(
        servers, args.output_dir, transport=args.transport,
        max_version_bytes=args.max_version_bytes,
        connect_timeout=args.connect_timeout,
        replication_factor=args.replication_factor,
    )

    if args.list:
        # List available versions
        versions = client.list_available_versions()
        if versions:
            logger.info("Available versions:")
            for version, checksum in sorted(versions.items()):
                logger.info(f"  {version} - Checksum: {checksum}")
        else:
            logger.error("No versions available or failed to retrieve distribution file")
            return 1
    elif args.version:
        # Download specific version
        output_file = client.download_version(args.version, args.output_file)
        if output_file:
            logger.info(f"Successfully downloaded and reassembled: {output_file}")
        else:
            logger.error(f"Failed to download and reassemble version {args.version}")
            return 1
    else:
        logger.error("Either --list or --version must be specified")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
