"""Measured, parallel HTTP transfers from compatible LiteRegistry sources."""
from concurrent.futures import ThreadPoolExecutor
import blake3
import logging
import threading
import time

from litecast import ClientNode

logger = logging.getLogger(__name__)


class MeasuredClient(ClientNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = []
        self.observation_lock = threading.Lock()
        self.manifest_seconds = 0.0
        self.download_seconds = 0.0
        self.cached_metadata_seconds = 0.0
        download = self.downloader.download_into_from

        def measured(server, path, target):
            started = time.perf_counter()
            success = download(server, path, target)
            ended = time.perf_counter()
            with self.observation_lock:
                previous = [r for r in self.observations if r['shard'] == path]
                record = dict(source=server, shard=path, attempt=len(previous) + 1,
                              bytes=len(target) if success else 0, success=success,
                              seconds=ended-started,
                              retry_gap_seconds=max(0, started-previous[-1]['ended']) if previous else 0,
                              ended=ended)
                self.observations.append(record)
            logger.info('LITECAST_SHARD_TRANSFER %s', {k:v for k,v in record.items() if k != 'ended'})
            return success
        self.downloader.download_into_from = measured

    def download_version_buffer(self, version):
        if self.transport != 'http':
            return super().download_version_buffer(version)
        started = time.perf_counter()
        manifest = self._load_shard_manifest(version)
        self.manifest_seconds = time.perf_counter() - started
        started = time.perf_counter()
        try:
            # Also use the manifest path for one remaining source: it avoids
            # per-shard HEAD requests and preserves per-shard verification.
            return self._download_streaming(manifest) if manifest is not None else self._download_cached(version)
        finally:
            self.download_seconds = time.perf_counter() - started

    def _download_cached(self, version):
        """Older full-cache middles advertise sizes/checksum in distribution.txt."""
        metadata_started = time.perf_counter()
        versions = self.list_available_versions()
        checksum, _, count = versions.get(version, '').partition('|')
        if len(checksum) != 64 or not count.isdigit():
            raise ValueError('invalid distribution manifest')
        count = int(count)
        if not 0 < count <= min(1_000_000, self.max_version_bytes):
            raise ValueError('invalid shard count')
        paths = [f'{version}/shard_{index + 1:05d}.bin' for index in range(count)]
        with ThreadPoolExecutor(max_workers=min(16, count)) as pool:
            sizes = list(pool.map(self.downloader.get_size, paths))
        if any(size is None or size < 0 for size in sizes):
            raise RuntimeError('shard sizes unavailable')
        if sum(sizes) > self.max_version_bytes:
            raise ValueError('transfer exceeds size limit')
        self.cached_metadata_seconds = time.perf_counter() - metadata_started
        data = bytearray(sum(sizes))
        view = memoryview(data)
        targets = []
        offset = 0
        for size in sizes:
            targets.append(view[offset:offset + size])
            offset += size

        def receive(index):
            # Rotate initial owners so concurrent requests use every middle.
            for attempt in range(len(self.servers)):
                owner = self.servers[(index + attempt) % len(self.servers)]
                if self.downloader.download_into_from(owner, paths[index], targets[index]):
                    return
            raise RuntimeError(f'shard {index} unavailable from all sources')

        with ThreadPoolExecutor(max_workers=min(16, count)) as pool:
            list(pool.map(receive, range(count)))
        if blake3.blake3(data).hexdigest() != checksum:
            raise ValueError('download checksum mismatch')
        return data

    def measurements(self):
        with self.observation_lock:
            rows = list(self.observations)
        return {
            'manifest_seconds': self.manifest_seconds,
            'download_seconds': self.download_seconds,
            'cached_metadata_seconds': self.cached_metadata_seconds,
            'http_attempt_seconds_sum': sum(r['seconds'] for r in rows),
            'http_failed_seconds_sum': sum(r['seconds'] for r in rows if not r['success']),
            'retry_gap_seconds_sum': sum(r['retry_gap_seconds'] for r in rows),
            'http_attempts': len(rows),
            'http_retries': sum(r['attempt'] > 1 for r in rows),
            'http_failures': sum(not r['success'] for r in rows),
            'http_bytes': sum(r['bytes'] for r in rows),
            'sources_used': len({r['source'] for r in rows if r['success']}),
        }
