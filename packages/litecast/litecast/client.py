"""Client module for downloading shards."""

import os
import time
import threading
import subprocess
from typing import List, Dict, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random

from litecast.envs import (
    LITECAST_DISTRIBUTION_FILE,
    LITECAST_HTTP_TIMEOUT,
    LITECAST_MAX_CONCURRENT_DOWNLOADS,
    LITECAST_RETRY_ATTEMPTS,
    LITECAST_FAST_RETRY_ATTEMPTS,
    LITECAST_FAST_RETRY_INTERVAL,
    LITECAST_SLOW_RETRY_INTERVAL,
)
from litecast.utils import logger, ensure_dir
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass
class ServerMetrics:
    """Metrics for a server's performance."""

    bandwidth_mbps: float
    success_rate: float


def _ensure_wget_installed() -> None:
    """Ensure wget is installed on the system."""
    envs = {
        "DEBIAN_FRONTEND": "noninteractive",
        "NEEDRESTART_MODE": "a",
    }

    try:
        # Check if wget is already installed
        subprocess.run(["which", "wget"], check=True, capture_output=True, env=envs)
    except subprocess.CalledProcessError:
        # Try installing wget without sudo first
        try:
            subprocess.run(["apt", "update"], check=True, env=envs)
            subprocess.run(["apt", "install", "--no-install-recommends", "-y", "wget"], check=True, env=envs)
        except subprocess.CalledProcessError:
            # If that fails, try with sudo
            try:
                subprocess.run(["sudo", "apt", "update"], check=True, env=envs)
                subprocess.run(["sudo", "apt", "install", "--no-install-recommends", "-y", "wget"], check=True, env=envs)
            except subprocess.CalledProcessError as e:
                raise RuntimeError("Failed to install wget. Please install it manually.") from e


class ShardDownloader:
    """Client for downloading shards from servers."""

    def __init__(self, servers: List[str], timeout: int = LITECAST_HTTP_TIMEOUT):
        """Initialize the shard downloader.

        Args:
            servers: List of server URLs or IP addresses
            timeout: Timeout for HTTP requests in seconds
        """
        if not servers:
            raise ValueError("at least one server is required")
        self.servers = [server.rstrip("/") for server in servers]
        self.timeout = timeout
        # Lock for thread-safe access to metrics
        self.metrics_lock = threading.Lock()
        logger.info(f"Initializing shard downloader with servers: {self.servers} and timeout: {self.timeout}")

        # Server performance metrics
        self.server_metrics: Dict[str, ServerMetrics] = {}
        self._init_server_metrics()

    def _init_server_metrics(self) -> Dict[str, ServerMetrics]:
        """Initialize server metrics by downloading the distribution file."""
        for server in self.servers:
            self.server_metrics[server] = ServerMetrics(1.0, 1.0)

        logger.info(f"Initialized server metrics: {self.server_metrics}")

    def _sample_best_server(self) -> str:
        """Sample the best server based on performance metrics.

        Returns:
            Best server URL
        """
        with self.metrics_lock:
            # Sort servers by: success_rate (desc), speed (desc)
            weights = [self.server_metrics[s].success_rate * self.server_metrics[s].bandwidth_mbps for s in self.servers]
            return random.choices(self.servers, weights=weights, k=1)[0]

    def _update_server_metrics(self, server: str, bandwidth_mbps: float, success: bool) -> None:
        """Update performance metrics for a server.

        Args:
            server: Server URL
            bandwidth_mbps: Bandwidth in MB/s
            success: Whether the download was successful
        """
        # TODO: ENV / CONSTANT
        alpha = 0.3
        heal_rate = 0.01
        with self.metrics_lock:
            # Initialize metrics if server is not in metrics
            if server not in self.server_metrics:
                self.server_metrics[server] = ServerMetrics(bandwidth_mbps=bandwidth_mbps, success_rate=1.0 if success else 0.0)
                return

            # Otherwise EMA
            self.server_metrics[server].bandwidth_mbps = alpha * bandwidth_mbps + (1 - alpha) * self.server_metrics[server].bandwidth_mbps
            self.server_metrics[server].success_rate = alpha * success + (1 - alpha) * self.server_metrics[server].success_rate

            # Revive others slowly in case they come back online
            # Note: _server is to avoid shadowing server
            for _server in self.servers:
                self.server_metrics[_server].success_rate = min(1.0, self.server_metrics[_server].success_rate + heal_rate)
            logger.debug(f"Metrics: {self.server_metrics}")

    def _wget(self, url: str, output_path: str, timeout: Optional[int] = None) -> bool:
        """Download a file from the best server.

        Args:
            url: URL to download from
            output_path: Local path to save the file
        """
        subprocess.run(
            [
                "wget",
                "-q",  # quiet mode
                "-O",
                output_path,  # output file
                "--timeout",
                str(timeout or self.timeout),  # timeout
                url,
            ],
            check=True,
        )

    def _download_file(self, server: str, url_path: str, output_path: Union[Path, str], update_metrics: bool = True) -> bool:
        """Download a file from a specific server.

        Args:
            server: Server URL
            url_path: Path component of the URL
            output_path: Local path to save the file
            update_metrics: Whether to update the server metrics
        """
        # TODO: Make sure we always use Path objects
        output_path = Path(output_path)
        try:
            url = f"{server}/{url_path}"
            temp_path = output_path.with_suffix(".tmp")

            logger.debug(f"Downloading {url} from {server} to {temp_path}")
            start_time = time.time()
            self._wget(url, temp_path)
            os.rename(temp_path, output_path)

            time_taken = time.time() - start_time
            file_size = output_path.stat().st_size
            bandwidth_mbps = file_size / time_taken / 1e6

            logger.debug(f"Downloaded {url} updating metrics...")
            if update_metrics:
                self._update_server_metrics(server, bandwidth_mbps, True)

            logger.debug(f"Downloaded {url_path} from {server} (bandwidth: {bandwidth_mbps:.0f} MB/s)")

            return True

        except Exception as e:
            # Update metrics with failure
            self._update_server_metrics(server, self.timeout, False)
            logger.debug(f"Failed to download {url_path} from {server}: {str(e)}", exc_info=True)
            return False

    def download_bytes(
        self,
        url_path: str,
        retries: int = LITECAST_RETRY_ATTEMPTS,
        max_bytes: Optional[int] = None,
    ) -> Optional[bytes]:
        """Download a resource into memory with bounded retries."""
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        for retry in range(retries):
            server = self._sample_best_server()
            started = time.time()
            try:
                with urlopen(
                    "{}/{}".format(server, url_path.lstrip("/")),
                    timeout=self.timeout,
                ) as response:
                    content_length = response.headers.get("Content-Length")
                    if (
                        max_bytes is not None
                        and content_length is not None
                        and int(content_length) > max_bytes
                    ):
                        raise ValueError("response exceeds configured memory limit")
                    data = response.read(
                        -1 if max_bytes is None else max_bytes + 1
                    )
                    if max_bytes is not None and len(data) > max_bytes:
                        raise ValueError("response exceeds configured memory limit")
                elapsed = max(time.time() - started, 0.001)
                self._update_server_metrics(server, len(data) / elapsed / 1e6, True)
                return data
            except Exception:
                self._update_server_metrics(server, 0.0, False)
                if retry + 1 < retries:
                    time.sleep(LITECAST_FAST_RETRY_INTERVAL if retry < LITECAST_FAST_RETRY_ATTEMPTS else LITECAST_SLOW_RETRY_INTERVAL)
        return None

    def download_bytes_from(
        self,
        server: str,
        url_path: str,
        max_bytes: Optional[int] = None,
    ) -> Optional[bytes]:
        """Download from one explicit owner without randomized routing."""
        server = server.rstrip("/")
        started = time.time()
        try:
            with urlopen(
                "{}/{}".format(server, url_path.lstrip("/")),
                timeout=self.timeout,
            ) as response:
                length = response.headers.get("Content-Length")
                if (
                    max_bytes is not None
                    and length is not None
                    and int(length) > max_bytes
                ):
                    raise ValueError("response exceeds configured memory limit")
                data = response.read(-1 if max_bytes is None else max_bytes + 1)
                if max_bytes is not None and len(data) > max_bytes:
                    raise ValueError("response exceeds configured memory limit")
            elapsed = max(time.time() - started, 0.001)
            self._update_server_metrics(server, len(data) / elapsed / 1e6, True)
            return data
        except Exception:
            self._update_server_metrics(server, 0.0, False)
            return None

    def get_size(
        self, url_path: str, retries: int = LITECAST_RETRY_ATTEMPTS
    ) -> Optional[int]:
        """Return a resource's Content-Length using a bounded HEAD request."""
        for retry in range(retries):
            server = self._sample_best_server()
            try:
                request = Request(
                    "{}/{}".format(server, url_path.lstrip("/")), method="HEAD"
                )
                with urlopen(request, timeout=self.timeout) as response:
                    size = int(response.headers["Content-Length"])
                if size < 0:
                    raise ValueError("negative Content-Length")
                return size
            except Exception:
                self._update_server_metrics(server, 0.0, False)
                if retry + 1 < retries:
                    time.sleep(
                        LITECAST_FAST_RETRY_INTERVAL
                        if retry < LITECAST_FAST_RETRY_ATTEMPTS
                        else LITECAST_SLOW_RETRY_INTERVAL
                    )
        return None

    def download_into(
        self,
        url_path: str,
        target: memoryview,
        retries: int = LITECAST_RETRY_ATTEMPTS,
    ) -> bool:
        """Download exactly one resource into a writable preallocated view."""
        if not isinstance(target, memoryview) or target.readonly:
            raise TypeError("target must be a writable memoryview")
        for retry in range(retries):
            server = self._sample_best_server()
            started = time.time()
            try:
                with urlopen(
                    "{}/{}".format(server, url_path.lstrip("/")),
                    timeout=self.timeout,
                ) as response:
                    content_length = response.headers.get("Content-Length")
                    if (
                        content_length is not None
                        and int(content_length) != len(target)
                    ):
                        raise ValueError("response size changed after HEAD")
                    offset = 0
                    while offset < len(target):
                        count = response.readinto(target[offset:])
                        if not count:
                            raise ValueError("truncated response")
                        offset += count
                    if response.read(1):
                        raise ValueError("response exceeds advertised size")
                elapsed = max(time.time() - started, 0.001)
                self._update_server_metrics(
                    server, len(target) / elapsed / 1e6, True
                )
                return True
            except Exception:
                self._update_server_metrics(server, 0.0, False)
                if retry + 1 < retries:
                    time.sleep(
                        LITECAST_FAST_RETRY_INTERVAL
                        if retry < LITECAST_FAST_RETRY_ATTEMPTS
                        else LITECAST_SLOW_RETRY_INTERVAL
                    )
        return False

    def download_into_from(
        self, server: str, url_path: str, target: memoryview
    ) -> bool:
        """Download exactly one shard from an explicit primary or replica."""
        if not isinstance(target, memoryview) or target.readonly:
            raise TypeError("target must be a writable memoryview")
        server = server.rstrip("/")
        started = time.time()
        try:
            with urlopen(
                "{}/{}".format(server, url_path.lstrip("/")),
                timeout=self.timeout,
            ) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != len(target):
                    raise ValueError("response size does not match manifest")
                offset = 0
                while offset < len(target):
                    count = response.readinto(target[offset:])
                    if not count:
                        raise ValueError("truncated response")
                    offset += count
                if response.read(1):
                    raise ValueError("response exceeds manifest size")
            elapsed = max(time.time() - started, 0.001)
            self._update_server_metrics(
                server, len(target) / elapsed / 1e6, True
            )
            return True
        except Exception:
            self._update_server_metrics(server, 0.0, False)
            return False

    def download_file(
        self,
        url_path: str,
        output_path: str,
        update_metrics: bool = True,
        retries: int = LITECAST_RETRY_ATTEMPTS,
        log_error_on_failure: bool = True,
    ) -> bool:
        """Download a file from the best server.

        Args:
            url_path: Path component of the URL (e.g., "v1/shard_001.bin")
            output_path: Local path to save the file
            retries: Number of retries to attempt
            log_error_on_failure: Whether to log an error on failure
        Returns:
            True if successful, False otherwise
        """
        # Ensure output directory exists
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        for retry in range(retries):
            server = self._sample_best_server()
            if self._download_file(server, url_path, output_path, update_metrics):
                return True

            # If we get here, all servers failed
            if retry < retries - 1:
                # Use fast retries for the first LITECAST_FAST_RETRY_ATTEMPTS attempts, then switch to slow retries, but never stop
                if retry < LITECAST_FAST_RETRY_ATTEMPTS:
                    wait_time = LITECAST_FAST_RETRY_INTERVAL
                    logger.debug(f"Fast retrying download of {url_path} (attempt {retry + 1}/{retries}, wait: {wait_time}s)")
                else:
                    wait_time = LITECAST_SLOW_RETRY_INTERVAL
                    logger.debug(f"Slow retrying download of {url_path} (attempt {retry + 1}/{retries}, wait: {wait_time}s)")

                # Wait before retrying
                time.sleep(wait_time)

        if log_error_on_failure:
            logger.error(f"Failed to download {url_path} after {retries} attempts")
        return False

    def download_distribution_file(self) -> Optional[str]:
        """Download the distribution file.

        Returns:
            Content of the distribution file or None if failed
        """
        payload = self.download_bytes(
            LITECAST_DISTRIBUTION_FILE, max_bytes=1024 * 1024
        )
        if payload is None:
            return None
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def download_shards(self, version: str, num_shards: int, output_dir: str) -> List[str]:
        """Download all shards for a version.

        Args:
            version: Version folder name (e.g., "v1")
            num_shards: Number of shards to download
            output_dir: Directory to save the shards

        Returns:
            List of successfully downloaded shard paths
        """
        ensure_dir(output_dir)

        # Track successfully downloaded shards
        successful_shards: List[str] = []
        failed_shards: Set[int] = set()

        # Use a thread pool for concurrent downloads
        with ThreadPoolExecutor(max_workers=LITECAST_MAX_CONCURRENT_DOWNLOADS) as executor:
            futures = []

            for i in range(num_shards):
                shard_filename = f"shard_{i + 1:05d}.bin"
                url_path = f"{version}/{shard_filename}"
                output_path = os.path.join(output_dir, shard_filename)

                futures.append(executor.submit(self._download_shard_with_retry, url_path, output_path, i))

            # Wait for all downloads to complete
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        successful_shards.append(result)
                    else:
                        failed_shards.add(i)
                except Exception as e:
                    logger.error(f"Error downloading shard {i + 1}: {str(e)}")
                    failed_shards.add(i)

        # Report summary
        if failed_shards:
            logger.warning(f"Failed to download {len(failed_shards)} out of {num_shards} shards")
            logger.debug(f"Failed shard indices: {sorted(failed_shards)}")
        else:
            logger.info(f"Successfully downloaded all {num_shards} shards")

        return successful_shards

    def _download_shard_with_retry(self, url_path: str, output_path: str, shard_index: int) -> Optional[str]:
        """Download a shard with retries.

        Args:
            url_path: Path component of the URL
            output_path: Local path to save the shard
            shard_index: Index of the shard (for logging)

        Returns:
            Output path if successful, None otherwise
        """
        if self.download_file(url_path, output_path):
            return output_path
        return None
