"""Keep admission slots until the response completes or its client disconnects."""

import asyncio
from collections.abc import Awaitable, Callable

from starlette.responses import StreamingResponse

GENERATION_PATHS = ("/v1/completions", "/v1/chat/completions", "/inference/v1/generate")


class ManagedStreamingResponse(StreamingResponse):
    def __init__(self, *args, cleanup: Callable[[], Awaitable[None]], **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.shield(self.cleanup())


class DisconnectCancellationMiddleware:
    """Cancel queued inference before dispatch when its HTTP client goes away."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in GENERATION_PATHS:
            return await self.app(scope, receive, send)
        messages = asyncio.Queue(maxsize=1)
        application = asyncio.create_task(self.app(scope, messages.get, send))

        async def pump():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    application.cancel()
                    return
                await messages.put(message)

        incoming = asyncio.create_task(pump())
        try:
            await asyncio.wait((application, incoming), return_when=asyncio.FIRST_COMPLETED)
            if incoming.done():
                incoming.result()
            if application.done() and not application.cancelled():
                application.result()
        finally:
            application.cancel()
            incoming.cancel()
            await asyncio.gather(application, incoming, return_exceptions=True)
