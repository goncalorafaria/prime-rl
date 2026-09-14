"""HTTP server module for serving shards."""

import socket
import threading
import os
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from litecast.envs import LITECAST_HTTP_PORT
from litecast.store import CheckpointStore, InvalidShardIndexError, VersionNotFoundError
from litecast.utils import logger

_SHARD_PATH = re.compile(r"^/([^/]+)/shard_(\d{5})\.bin$")


class LitecastRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for serving shards."""

    def __init__(self, *args, **kwargs):
        # Disable logging of requests to stdout
        self.server = args[2]  # server is passed as the third argument
        super().__init__(*args, **kwargs, directory=self.server.directory)

    def log_message(self, format: str, *args) -> None:
        """Override to use our logger instead of printing to stderr."""
        logger.debug(
            "%s - - [%s] %s",
            self.address_string(),
            self.log_date_time_string(),
            format % args,
        )

    def end_headers(self) -> None:
        """Add CORS headers to all responses."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        if self._serve_store(False):
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self._serve_store(True):
            return
        super().do_HEAD()

    def _serve_store(self, head_only: bool) -> bool:
        match = _SHARD_PATH.match(self.path.split("?", 1)[0])
        store = getattr(self.server, "store", None)
        if match is None or store is None:
            return False
        version, shard_number = match.groups()
        try:
            shard = store.get_shard(version, int(shard_number) - 1)
        except InvalidShardIndexError:
            self.send_error(404, "Shard not found")
            return True
        except VersionNotFoundError:
            if store.get_metadata(version) is None:
                return False
            self.send_error(425, "Shard not available")
            return True
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(shard)))
        self.end_headers()
        if not head_only:
            self.wfile.write(shard)
        return True


class LitecastServer(ThreadingHTTPServer):
    """HTTP server for serving shards."""

    def __init__(
        self,
        server_address: Tuple[str, int],
        directory: str,
        shutdown_event: Optional[threading.Event] = None,
        store: Optional[CheckpointStore] = None,
    ):
        """Initialize the server.

        Args:
            server_address: (host, port) tuple
            directory: Directory to serve files from
            shutdown_event: Event to signal server shutdown
        """
        super().__init__(server_address, LitecastRequestHandler)
        self.directory = directory
        self.shutdown_event = shutdown_event or threading.Event()
        self.store = store
        self.daemon_threads = True


def get_local_ip() -> str:
    """Get the local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def run_server(
    directory: str,
    port: int = LITECAST_HTTP_PORT,
    shutdown_event: Optional[threading.Event] = None,
    store: Optional[CheckpointStore] = None,
) -> Tuple[HTTPServer, threading.Thread]:
    """Run an HTTP server in a background thread.

    Args:
        directory: Directory to serve files from
        port: Port to listen on
        shutdown_event: Event to signal server shutdown

    Returns:
        Tuple of (server, thread)
    """
    shutdown_event = shutdown_event or threading.Event()
    os.makedirs(directory, exist_ok=True)
    dummy = os.path.join(directory, "data1.bin")
    if not os.path.exists(dummy):
        with open(dummy, "wb"):
            pass
    server = LitecastServer(("0.0.0.0", port), directory, shutdown_event, store)

    server_thread = threading.Thread(target=_server_thread, args=(server, shutdown_event))
    server_thread.daemon = True
    server_thread.start()

    local_ip = get_local_ip()
    logger.info(f"Server running at http://{local_ip}:{server.server_port}")

    return server, server_thread


def _server_thread(server: HTTPServer, shutdown_event: threading.Event) -> None:
    """Thread function for running the HTTP server.

    Args:
        server: HTTP server instance
        shutdown_event: Event to signal server shutdown
    """
    try:
        server.serve_forever(poll_interval=0.1)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        server.server_close()
        logger.info("Server stopped")
