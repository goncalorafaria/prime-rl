"""Stable trainer-owned TCP endpoints for a replaceable registry/gateway head."""

import argparse
import asyncio
import logging
from urllib.parse import urlparse

from prime_rl.litecast.bootstrap import publish, wait

logger = logging.getLogger(__name__)


async def forward(reader, writer, bootstrap: str, backend: str):
    upstream = None
    tasks = []
    try:
        uri = await asyncio.to_thread(wait, bootstrap, backend, timeout=10, healthcheck="none")
        target = urlparse(uri)
        remote_reader, upstream = await asyncio.wait_for(asyncio.open_connection(target.hostname, target.port), 10)

        async def copy(source, destination):
            while data := await source.read(65536):
                destination.write(data)
                await destination.drain()

        tasks = [asyncio.create_task(copy(reader, upstream)), asyncio.create_task(copy(remote_reader, writer))]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (OSError, TimeoutError) as exc:
        logger.warning("Head connection %s interrupted: %s", backend, exc)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        if upstream is not None:
            upstream.close()


async def serve(args):
    redis_server = await asyncio.start_server(
        lambda r, w: forward(r, w, args.bootstrap, "redis-backend"), "0.0.0.0", args.redis_port
    )
    gateway_server = await asyncio.start_server(
        lambda r, w: forward(r, w, args.bootstrap, "gateway-backend"), "0.0.0.0", args.gateway_port
    )
    async with redis_server, gateway_server:
        while True:
            for name, uri in {
                "redis": f"redis://{args.advertise_host}:{args.redis_port}/0",
                "gateway": f"http://{args.advertise_host}:{args.gateway_port}",
            }.items():
                await asyncio.to_thread(
                    publish, args.bootstrap, name, uri, publisher_id=args.run_id + "-relay-" + name, ttl_seconds=30
                )
            await asyncio.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--redis-port", required=True, type=int)
    parser.add_argument("--gateway-port", required=True, type=int)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(parser.parse_args()))


if __name__ == "__main__":
    main()
