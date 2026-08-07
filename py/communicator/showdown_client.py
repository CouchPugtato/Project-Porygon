from __future__ import annotations

import asyncio
import os
import random
import string
import json
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
    def __init__(
        self,
        uri: str,
        username: str = "",
        fmt: str = "gen9randomdoublesbattle",
        challenge_target: str = "",
        accept_challenges: bool = False,
        accept_challenge_from: str = "",
    ) -> None:
        self.uri = uri
        self.username = username
        self.format = fmt
        self.challenge_target = challenge_target.strip()
        self.accept_challenges = accept_challenges
        self.accept_challenge_from = accept_challenge_from.strip()
        self.current_username = ""
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._named = False
        self._search_sent = False
        self._seen_battle_rooms: set[str] = set()
        self._challenge_sent = False

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

    async def challenge_battle(self) -> None:
        if self._challenge_sent or not self.challenge_target:
            return
        print(f"[communicator] challenging {self.challenge_target} in format {self.format}")
        await self.send("|/utm null")
        await self.send(f"|/challenge {self.challenge_target}, {self.format}")
        self._challenge_sent = True

    async def accept_challenge(self, challenger: str) -> None:
        print(f"[communicator] accepting challenge from {challenger}")
        await self.send(f"|/accept {challenger}")

    async def start_next_battle(self) -> None:
        if self.challenge_target:
            await self.challenge_battle()
            return
        if self.accept_challenges:
            return
        await self.search_battle()

    async def search_next_battle(self) -> None:
        self._search_sent = False
        self._challenge_sent = False
        await self.start_next_battle()

    async def cancel_search(self) -> None:
        if not self._search_sent:
            return
        print("[communicator] canceling active search")
        await self.send("|/cancelsearch")
        self._search_sent = False

    async def close(self) -> None:
        if self._ws is None:
            return
        await self._ws.close()
        self._ws = None
        self._search_sent = False

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
                        await self.start_next_battle()
                else:
                    await self.start_next_battle()
            return

        if line.startswith("|updatesearch|"):
            print(f"[communicator] search update: {line}")
            if line.strip() == "|updatesearch|":
                self._search_sent = False
            return

        if line.startswith("|pm|"):
            parts = line.split("|", 4)
            if len(parts) >= 5:
                sender_identity = parts[2].strip()
                message = parts[4].strip()
                sender_name = sender_identity.lstrip("!+%@&#~ ").strip()
                if (
                    self.accept_challenges
                    and message.startswith("/challenge ")
                    and (not self.accept_challenge_from or sender_name == self.accept_challenge_from)
                ):
                    await self.accept_challenge(sender_name)
                    return
            return

        if line.startswith("|updatechallenges|"):
            try:
                payload = json.loads(line.split("|updatechallenges|", 1)[1])
            except Exception:
                return
            challenges_from = payload.get("challengesFrom", {}) if isinstance(payload, dict) else {}
            if not self.accept_challenges or not isinstance(challenges_from, dict):
                return
            for challenger, challenge_format in challenges_from.items():
                if self.accept_challenge_from and challenger != self.accept_challenge_from:
                    continue
                if str(challenge_format).strip() != self.format:
                    continue
                await self.accept_challenge(challenger)
                return
            return

        if line.startswith("|popup|"):
            print(f"[communicator] popup: {line}")
            if self.challenge_target and "was not found" in line:
                self._challenge_sent = False
                await asyncio.sleep(1.0)
                await self.challenge_battle()
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
                        self._challenge_sent = False
                        print(f"[communicator] joined battle room {current_room}")
                    continue

                if not current_room:
                    await self.handle_control_line(raw_line)
                    continue

                await on_event(ShowdownEvent(current_room, raw_line))
