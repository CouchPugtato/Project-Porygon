from __future__ import annotations

import asyncio
import json
import os
from typing import Awaitable, Callable, Optional

import websockets


class ShowdownGateway:
    def __init__(self, uri: str, username: str = "", password: str = "") -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.uri)

    async def send(self, text: str) -> None:
        if self._ws is None:
            raise RuntimeError("Showdown websocket not connected")
        await self._ws.send(text)

    async def run(self, on_message: Callable[[str], Awaitable[None]]) -> None:
        if self._ws is None:
            await self.connect()
        assert self._ws is not None
        async for message in self._ws:
            await on_message(message)


def default_showdown_uri() -> str:
    host = os.getenv("PS_SERVER", "sim3.psim.us")
    return f"wss://{host}/showdown/websocket"
