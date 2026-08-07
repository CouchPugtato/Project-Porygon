from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class LearnerProcess:
    def __init__(self, command: list[str], replay_path: Optional[Path] = None) -> None:
        self._command = command
        self._replay_path = replay_path
        self._process: Optional[asyncio.subprocess.Process] = None
        self._closed = False
        self._stdout_buffer = bytearray()

    async def start(self) -> None:
        env = os.environ.copy()
        if self._replay_path is not None:
            env["PORYGON_REPLAY_PATH"] = str(self._replay_path)
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )

    async def send(self, payload: Dict[str, Any]) -> None:
        if self._closed or not self._process or not self._process.stdin:
            return
        if self._process.returncode is not None:
            self._closed = True
            return
        try:
            self._process.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            self._closed = True

    async def read_message(self) -> Optional[Dict[str, Any]]:
        if not self._process or not self._process.stdout:
            return None
        while True:
            newline_index = self._stdout_buffer.find(b"\n")
            if newline_index < 0:
                chunk = await self._process.stdout.read(65536)
                if not chunk:
                    if not self._stdout_buffer:
                        return None
                    line = bytes(self._stdout_buffer)
                    self._stdout_buffer.clear()
                else:
                    self._stdout_buffer.extend(chunk)
                    continue
            else:
                line = bytes(self._stdout_buffer[:newline_index])
                del self._stdout_buffer[: newline_index + 1]
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                print(f"[learner-stdout] ignored non-JSON line: {text}", file=sys.stderr)

    async def read_stderr_line(self) -> Optional[str]:
        if not self._process or not self._process.stderr:
            return None
        line = await self._process.stderr.readline()
        if not line:
            return None
        return line.decode("utf-8", errors="replace").rstrip()

    async def terminate(self) -> None:
        if self._process is None:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.returncode is None:
            self._process.terminate()
        await self._process.wait()
