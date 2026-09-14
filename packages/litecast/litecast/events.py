"""Optional per-process JSONL event registry for distributed measurements."""

import atexit
import json
import os
import re
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_BUFFER = []  # type: List[Tuple[str, str]]
_BUFFER_LOCK = threading.Lock()


def emit_event(event: str, **fields: Any) -> None:
    """Buffer a timestamped event when ``LITECAST_EVENT_DIR`` is configured."""
    directory = os.getenv("LITECAST_EVENT_DIR")
    if not directory:
        return
    role = os.getenv("LITECAST_ROLE", "unknown")
    node_id = os.getenv(
        "LITECAST_NODE_ID",
        os.getenv("SLURM_PROCID", "{}-{}".format(socket.gethostname(), os.getpid())),
    )
    filename = "{}-{}-{}.jsonl".format(
        _safe(role), _safe(node_id), os.getpid()
    )
    record = {
        "event": event,
        "role": role,
        "node_id": node_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }
    record.update(fields)
    path = os.path.join(directory, filename)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with _BUFFER_LOCK:
        _BUFFER.append((path, line))


def flush_events() -> None:
    """Flush all buffered events with one append per destination file."""
    with _BUFFER_LOCK:
        pending = list(_BUFFER)
        _BUFFER.clear()
    if not pending:
        return
    grouped = {}  # type: Dict[str, List[str]]
    for path, line in pending:
        grouped.setdefault(path, []).append(line)
    destinations = list(grouped.items())
    for position, (path, lines) in enumerate(destinations):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.writelines(lines)
        except Exception:
            unwritten = [
                item
                for remaining_path, remaining_lines in destinations[position:]
                for item in (
                    (remaining_path, remaining_line)
                    for remaining_line in remaining_lines
                )
            ]
            with _BUFFER_LOCK:
                _BUFFER[:0] = unwritten
            raise


def _safe(value: str) -> str:
    return _SAFE_NAME.sub("_", str(value))[:128]


atexit.register(flush_events)
