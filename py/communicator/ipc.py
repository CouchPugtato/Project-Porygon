from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class LearnerProcess:
    def __init__(self, command: list[str], replay_path: Optional[Path] = None) -> None:
        self._command = command
        self._replay_path = replay_path
        self._process: Optional[asyncio.subprocess.Process] = None

    async def start(self) -> None:
        env = os.environ.copy()
        if self._replay_path is not None:
            env["PORYGON_REPLAY_PATH"] = str(self._replay_path)
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def send(self, payload: Dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            return
        self._process.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
        await self._process.stdin.drain()

    async def read_message(self) -> Optional[Dict[str, Any]]:
        if not self._process or not self._process.stdout:
            return None
        line = await self._process.stdout.readline()
        if not line:
            return None
        return json.loads(line.decode("utf-8"))

    async def read_stderr_line(self) -> Optional[str]:
        if not self._process or not self._process.stderr:
            return None
        line = await self._process.stderr.readline()
        if not line:
            return None
        return line.decode("utf-8", errors="replace").rstrip()
