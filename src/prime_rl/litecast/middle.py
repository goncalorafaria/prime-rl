"""Supervise a standalone CPU LiteCast middle through LiteRegistry discovery."""

import argparse
import asyncio
import json
import logging
import signal
import tempfile
from pathlib import Path

from literegistry import RegistryClient, get_kvstore
from literegistry.registry import ServerRegistry

from litecast import ClientNode, MiddleNode
from prime_rl.litecast.distribution import blocking
from prime_rl.litecast.protocol import Publication, desired_key, peer_service, validate_run_id

logger = logging.getLogger(__name__)


async def serve(args):
    validate_run_id(args.run_id)
    store = get_kvstore(args.registry, raise_on_error=True)
    registry = RegistryClient(store, cache_ttl=0, max_heartbeat_interval=30)
    registrations = {}
    middle = None
    owner = None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    with tempfile.TemporaryDirectory(prefix="litecast-middle-") as directory:
        try:
            while not stop.is_set():
                raw = await store.get(desired_key(args.run_id))
                if raw is None:
                    for registration in registrations.values():
                        await registration.deregister()
                    registrations.clear()
                else:
                    desired = json.loads(raw)
                    if desired.get("schema") != 1:
                        raise ValueError("unsupported publication schema")
                    if owner is not None and owner != desired["owner"]:
                        raise RuntimeError("publisher changed; restart middle to reset local version identities")
                    owner = desired["owner"]
                    publications = [Publication(**p) for p in desired["publications"]]
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
                                "MIDDLE_READY step=%s digest=%s bytes=%s",
                                publication.step,
                                publication.digest,
                                publication.size,
                            )
                    keep = {peer_service(args.run_id, p.digest) for p in publications}
                    for service, registration in list(registrations.items()):
                        version = registration._metadata["litecast_version"]
                        if service not in keep or middle is None or version not in middle.store:
                            await registration.deregister()
                            del registrations[service]
                        else:
                            await registration.heartbeat(f"http://{args.advertise_host}", middle.port)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1)
                except TimeoutError:
                    pass
        finally:
            await asyncio.gather(*(r.deregister() for r in registrations.values()), return_exceptions=True)
            if middle is not None:
                await blocking(middle.shutdown)
            await registry.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-adapter-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-versions", type=int, default=8)
    args = parser.parse_args()
    if args.max_adapter_bytes <= 0 or args.max_versions <= 0:
        parser.error("cache limits must be positive")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
