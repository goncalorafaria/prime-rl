"""Supervise a standalone CPU LiteCast middle through LiteRegistry discovery."""

import argparse
import asyncio
import json
import logging
import signal
import tempfile
import time
from copy import copy
from pathlib import Path

from literegistry import RegistryClient, get_kvstore
from literegistry.registry import ServerRegistry

from litecast import ClientNode, MiddleNode
from prime_rl.litecast.distribution import blocking
from prime_rl.litecast.protocol import Publication, desired_key, peer_service, validate_run_id

logger = logging.getLogger(__name__)


async def serve_publisher(args, stop):
    validate_run_id(args.run_id)
    store = get_kvstore(args.registry, raise_on_error=True)
    registry = RegistryClient(store, cache_ttl=0, max_heartbeat_interval=30)
    registrations = {}
    middle = None
    owner = None
    last_status = None
    last_status_time = float("-inf")
    logger.info("MIDDLE_STARTED run=%s configured_endpoint=http://%s:%s", args.run_id, args.advertise_host, args.port)
    with tempfile.TemporaryDirectory(prefix="litecast-middle-") as directory:
        try:
            while not stop.is_set():
                raw = await store.get(desired_key(args.run_id))
                publications = []
                phase = "waiting_for_publisher"
                if raw is None:
                    for registration in registrations.values():
                        await registration.deregister()
                    registrations.clear()
                else:
                    desired = json.loads(raw)
                    if desired.get("schema") != 1:
                        raise ValueError("unsupported publication schema")
                    # An empty startup descriptor owns no cached version identities.
                    if owner is not None and owner != desired["owner"] and (middle is not None or registrations):
                        raise RuntimeError("publisher changed; restart middle to reset local version identities")
                    owner = desired["owner"]
                    publications = [Publication(**p) for p in desired["publications"]]
                    phase = "waiting_for_weights" if not publications else "waiting_for_origin"
                    records = await registry.models(force=True)
                    for publication in publications:
                        if publication.run_id != args.run_id or publication.size > args.max_adapter_bytes:
                            raise ValueError("publication exceeds run or size limits")
                        service = peer_service(args.run_id, publication.digest)
                        origins = [r for r in records.get(service, []) if r["metadata"].get("source_role") == "origin"]
                        if not origins:
                            continue
                        version = origins[0]["metadata"]["litecast_version"]
                        if middle is None:
                            middle = MiddleNode(
                                [r["uri"] for r in origins],
                                str(Path(directory) / "cache"),
                                port=args.port,
                                check_interval=1,
                                transport="http",
                                ram_cache_bytes=args.max_adapter_bytes * args.max_versions,
                                disk_mirror=False,
                            )
                        phase = "syncing"
                        if version not in middle.processed_versions or version not in middle.store:
                            continue
                        if service not in registrations:
                            # Validate the complete cache before advertising it as a source.
                            client = ClientNode(
                                [f"http://127.0.0.1:{middle.port}"],
                                str(Path(directory) / "verify"),
                                transport="http",
                                max_version_bytes=publication.size,
                            )
                            try:
                                payload = await blocking(client.download_version_buffer, version)
                                publication.verify(bytes(payload))
                            finally:
                                client.close()
                            registration = ServerRegistry(store)
                            await registration.register_server(
                                f"http://{args.advertise_host}",
                                middle.port,
                                {
                                    "model_path": service,
                                    "source_role": "middle",
                                    "run_id": args.run_id,
                                    "digest": publication.digest,
                                    "litecast_version": version,
                                },
                            )
                            registrations[service] = registration
                            logger.info(
                                "MIDDLE_READY step=%s digest=%s bytes=%s run=%s",
                                publication.step,
                                publication.digest,
                                publication.size,
                                args.run_id,
                            )
                    keep = {peer_service(args.run_id, p.digest) for p in publications}
                    for service, registration in list(registrations.items()):
                        version = registration._metadata["litecast_version"]
                        if service not in keep or middle is None or version not in middle.store:
                            await registration.deregister()
                            del registrations[service]
                        else:
                            await registration.heartbeat(f"http://{args.advertise_host}", middle.port)
                if publications and len(registrations) == len(publications):
                    phase = "ready"
                status = (phase, len(publications), len(registrations), middle is not None)
                now = time.monotonic()
                if status != last_status or now - last_status_time >= 30:
                    logger.info(
                        "MIDDLE_STATUS phase=%s published_versions=%s cached_versions=%s server_started=%s run=%s",
                        *status,
                        args.run_id,
                    )
                    last_status, last_status_time = status, now
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1)
                except TimeoutError:
                    pass
        finally:
            await asyncio.gather(*(r.deregister() for r in registrations.values()), return_exceptions=True)
            if middle is not None:
                await blocking(middle.shutdown)
            await registry.close()


def publisher_ids(args):
    """A file is a complete subscription snapshot; an empty list detaches all."""
    runs = json.loads(args.publishers.read_text()) if args.publishers else args.run_id
    if not isinstance(runs, list) or any(not isinstance(run, str) for run in runs):
        raise ValueError("publishers must be a JSON list of run ID strings")
    if len(runs) > args.max_publishers or len(runs) != len(set(runs)):
        raise ValueError("publisher IDs must be unique and within max-publishers")
    for run in runs:
        validate_run_id(run)
    return runs


async def supervise_publisher(args, stop):
    while not stop.is_set():
        try:
            await serve_publisher(args, stop)
        except Exception:
            # A publisher lease/origin can change independently of other trainers.
            logger.exception("MIDDLE_PUBLISHER_RETRY run=%s retry_seconds=5", args.run_id)
            try:
                await asyncio.wait_for(stop.wait(), timeout=5)
            except TimeoutError:
                pass


async def serve(args):
    runs = publisher_ids(args)
    logger.info("MIDDLE_SUPERVISOR_STARTED publishers=%s reloadable=%s", len(runs), bool(args.publishers))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    sessions = {}
    invalid_snapshot = None
    try:
        while not stop.is_set():
            # Validate the whole file before removing any healthy subscriptions.
            try:
                runs = publisher_ids(args)
                invalid_snapshot = None
            except (OSError, ValueError) as exc:
                if str(exc) != invalid_snapshot:
                    logger.warning("MIDDLE_SUBSCRIPTIONS_REJECTED keeping_previous=true error=%s", exc)
                    invalid_snapshot = str(exc)
            for run in set(sessions) - set(runs):
                task, publisher_stop = sessions.pop(run)
                publisher_stop.set()
                await task
                logger.info("MIDDLE_PUBLISHER_DETACHED run=%s", run)
            for run in runs:
                if run not in sessions:
                    subscription = copy(args)
                    subscription.run_id = run
                    # Each origin has its own version numbering. Never combine stores.
                    subscription.port = args.port if not args.publishers and len(runs) == 1 else 0
                    publisher_stop = asyncio.Event()
                    task = asyncio.create_task(supervise_publisher(subscription, publisher_stop), name=run)
                    sessions[run] = (task, publisher_stop)
                    logger.info("MIDDLE_PUBLISHER_ATTACHED run=%s", run)
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except TimeoutError:
                pass
    finally:
        for _, publisher_stop in sessions.values():
            publisher_stop.set()
        await asyncio.gather(*(task for task, _ in sessions.values()))
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    subscriptions = parser.add_mutually_exclusive_group(required=True)
    subscriptions.add_argument("--run-id", action="append", help="Publisher run ID; repeat to subscribe to several")
    subscriptions.add_argument("--publishers", type=Path, help="Reloadable JSON list of publisher run IDs")
    parser.add_argument("--max-publishers", type=int, default=8)
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-adapter-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-versions", type=int, default=8)
    args = parser.parse_args()
    if min(args.max_adapter_bytes, args.max_versions, args.max_publishers) <= 0:
        parser.error("cache limits must be positive")
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    try:
        publisher_ids(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
