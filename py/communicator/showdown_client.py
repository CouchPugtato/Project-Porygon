from __future__ import annotations

import os
import random
import string
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import websockets


def default_showdown_uri() -> str:
    explicit_uri = (os.getenv("PS_URI", "") or "").strip()
    if explicit_uri:
        return explicit_uri
    host = os.getenv("PS_SERVER", "sim3.psim.us")
    return f"wss://{host}/showdown/websocket"


def default_username() -> str:
    suffix = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"Porygon{suffix}"


def infer_is_doubles(battle_id: str, fmt: str) -> bool:
    token = f"{battle_id}-{fmt}".lower()
    return "double" in token or "vgc" in token


@dataclass
class ShowdownEvent:
    room_id: str
    line: str


class ShowdownGateway:
    def __init__(self, uri: str, username: str = "", fmt: str = "gen9randomdoublesbattle") -> None:
        self.uri = uri
        self.username = username
        self.format = fmt
        self.current_username = ""
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._named = False
        self._search_sent = False
        self._seen_battle_rooms: set[str] = set()

    async def connect(self) -> None:
        print(f"[communicator] connecting to {self.uri}")
        self._ws = await websockets.connect(self.uri)
        print("[communicator] websocket connected")

    async def send(self, text: str) -> None:
        if self._ws is None:
            raise RuntimeError("Showdown websocket not connected")
        await self._ws.send(text)

    async def send_room_command(self, room_id: str, command: str) -> None:
        await self.send(f"{room_id}|{command}")

    async def rename_guest(self) -> None:
        if self._named or not self.username:
            return
        print(f"[communicator] requesting guest rename to {self.username}")
        await self.send(f"|/trn {self.username},0")

    async def search_battle(self) -> None:
        if self._search_sent:
            return
        print(f"[communicator] searching for format {self.format}")
        await self.send("|/utm null")
        await self.send(f"|/search {self.format}")
        self._search_sent = True

    async def search_next_battle(self) -> None:
        self._search_sent = False
        await self.search_battle()

    async def close(self) -> None:
        if self._ws is None:
            return
        await self._ws.close()
        self._ws = None

    async def handle_control_line(self, line: str) -> None:
        if line.startswith("|challstr|"):
            if self.username:
                await self.rename_guest()
            return

        if line.startswith("|updateuser|"):
            parts = line.split("|")
            if len(parts) >= 4:
                username = parts[2].strip()
                named = parts[3] == "1"
                self.current_username = username
                self._named = named
                print(f"[communicator] logged in as {username} (named={named})")
                if self.username:
                    if named:
                        await self.search_battle()
                else:
                    await self.search_battle()
            return

        if line.startswith("|updatesearch|"):
            print(f"[communicator] search update: {line}")
            return

        if line.startswith("|popup|"):
            print(f"[communicator] popup: {line}")
            return

    async def run(self, on_event: Callable[[ShowdownEvent], Awaitable[None]]) -> None:
        if self._ws is None:
            await self.connect()
        assert self._ws is not None

        async for message in self._ws:
            current_room = ""
            for raw_line in message.splitlines():
                if not raw_line:
                    continue
                if raw_line.startswith(">"):
                    current_room = raw_line[1:]
                    if current_room.startswith("battle-") and current_room not in self._seen_battle_rooms:
                        self._seen_battle_rooms.add(current_room)
                        print(f"[communicator] joined battle room {current_room}")
                    continue

                if not current_room:
                    await self.handle_control_line(raw_line)
                    continue

                await on_event(ShowdownEvent(current_room, raw_line))
