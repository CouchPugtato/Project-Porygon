from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import websockets


DEFAULT_SHOWDOWN_DIR = Path("external/pokemon-showdown")
DEFAULT_CLIENT_DIR = Path("external/pokemon-showdown-client")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_CLIENT_PORT = 8001
DEFAULT_FORMAT = "gen9randomdoublesbattle"
DEFAULT_RECONNECT_SECONDS = 5.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_SHUTDOWN_GRACE_SECONDS = 20.0
DEFAULT_USERNAME_PREFIX = "PoryPool"
DEFAULT_ARGS_PATH = Path("config/selfplay_server.args")

BATTLE_STARTED_RE = re.compile(r"\[(?:live|random)\] battle started (battle-[^\s]+)")
BATTLE_ENDED_RE = re.compile(r"\[(?:live|random)\] battle ended (battle-[^\s]+) result=")


def default_server_uri(host: str, port: int) -> str:
    return f"ws://{host}:{port}/showdown/websocket"


def default_client_url(host: str, port: int, entrypoint: str) -> str:
    return f"http://{host}:{port}/{entrypoint.lstrip('/')}"


def default_watch_url(client_host: str, client_port: int, entrypoint: str, server_host: str, server_port: int) -> str:
    return f"{default_client_url(client_host, client_port, entrypoint)}?~~{server_host}:{server_port}"


def run_dir_for_name(run_name: str) -> Path:
    return Path("matches") / "runs" / run_name


def worker_replay_save_token(run_name: str, worker_token: str) -> str:
    return f"{run_name}/{worker_token}"


def worker_replay_path(run_name: str, worker_token: str) -> Path:
    return run_dir_for_name(run_name) / f"{worker_token}_raw.jsonl"


def parse_stats_file(path: Path) -> dict[str, str]:
    stats: dict[str, str] = {}
    if not path.exists():
        return stats
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        stats[key.strip()] = value.strip()
    return stats


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_bool01(value: str) -> bool:
    if value not in {"0", "1"}:
        raise argparse.ArgumentTypeError("value must be 0 or 1")
    return value == "1"


def load_default_args(path: Path) -> list[str]:
    args: list[str] = []
    if not path.exists():
        return args
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        args.append(line)
    return args


def split_weighted_workers(total_workers: int, model_a_weight: int, model_b_weight: int) -> tuple[int, int]:
    if total_workers <= 0:
        return 0, 0
    total_weight = model_a_weight + model_b_weight
    if total_weight <= 0:
        return total_workers // 2, total_workers - (total_workers // 2)
    model_a_workers = round(total_workers * (model_a_weight / total_weight))
    model_a_workers = max(0, min(total_workers, model_a_workers))
    model_b_workers = total_workers - model_a_workers
    return model_a_workers, model_b_workers


def default_server_start_command(port: int) -> list[str]:
    return ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)]


def parse_server_start_command(command: str, port: int) -> list[str]:
    if not command.strip():
        return default_server_start_command(port)
    return shlex.split(command, posix=os.name != "nt")


def child_creationflags() -> int:
    if os.name != "nt":
        return 0
    flags = 0
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        flags |= subprocess.CREATE_NEW_PROCESS_GROUP
    return flags


async def terminate_process(process: asyncio.subprocess.Process | None, timeout_seconds: float = 5.0) -> None:
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        return
    except asyncio.TimeoutError:
        pass

    if os.name == "nt" and process.pid:
        with contextlib.suppress(Exception):
            completed = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(completed.wait(), timeout=5.0)
    else:
        process.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait(), timeout=5.0)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    model_group: str
    checkpoint_path: str
    mode: str
    username: str
    replay_save_token: str
    replay_path: Path
    stdout_log_path: Path
    shutdown_path: Path

    @property
    def worker_token(self) -> str:
        return f"worker_{self.worker_id:03d}_{self.model_group}"


def worker_mode_for_model_spec(default_mode: str, model_spec: str) -> str:
    if (model_spec or "").strip().lower() == "random":
        return "random"
    return default_mode


def checkpoint_for_model_spec(model_spec: str) -> str:
    if (model_spec or "").strip().lower() == "random":
        return ""
    return model_spec


class ServerProcess:
    def __init__(self, repo_root: Path, showdown_dir: Path, command: list[str], server_uri: str, log_path: Path) -> None:
        self.repo_root = repo_root
        self.showdown_dir = showdown_dir
        self.command = command
        self.server_uri = server_uri
        self.log_path = log_path
        self.process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._log_handle: TextIO | None = None

    async def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._log_handle.write(f"$ {' '.join(self.command)}\n")
        self._log_handle.flush()
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=str(self.showdown_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=child_creationflags(),
        )
        self._stdout_task = asyncio.create_task(self._pipe_output())

    async def _pipe_output(self) -> None:
        if self.process is None or self.process.stdout is None or self._log_handle is None:
            return
        while True:
            line = await self.process.stdout.readline()
            if not line:
                return
            decoded = line.decode("utf-8", errors="replace")
            self._log_handle.write(decoded)
            self._log_handle.flush()

    async def wait_until_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process is not None and self.process.returncode is not None:
                raise RuntimeError(f"showdown server exited early with code {self.process.returncode}")
            try:
                async with websockets.connect(self.server_uri, open_timeout=2, close_timeout=1):
                    return
            except Exception:
                await asyncio.sleep(0.5)
        raise TimeoutError(f"timed out waiting for showdown server at {self.server_uri}")

    async def terminate(self) -> None:
        await terminate_process(self.process)
        if self._stdout_task is not None:
            with contextlib.suppress(Exception):
                await self._stdout_task
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None


class ClientProcess:
    def __init__(self, client_dir: Path, host: str, port: int, log_path: Path) -> None:
        self.client_dir = client_dir
        self.host = host
        self.port = port
        self.log_path = log_path
        self.process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._log_handle: TextIO | None = None

    async def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "http.server",
            str(self.port),
            "--bind",
            self.host,
            "--directory",
            str(self.client_dir),
        ]
        self._log_handle.write(f"$ {' '.join(command)}\n")
        self._log_handle.flush()
        self.process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.client_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=child_creationflags(),
        )
        self._stdout_task = asyncio.create_task(self._pipe_output())

    async def _pipe_output(self) -> None:
        if self.process is None or self.process.stdout is None or self._log_handle is None:
            return
        while True:
            line = await self.process.stdout.readline()
            if not line:
                return
            decoded = line.decode("utf-8", errors="replace")
            self._log_handle.write(decoded)
            self._log_handle.flush()

    async def wait_until_ready(self, entrypoint: str, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        url = default_client_url(self.host, self.port, entrypoint)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.returncode is not None:
                raise RuntimeError(f"client server exited early with code {self.process.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=2.0) as response:
                    if response.status == 200:
                        return
            except Exception:
                await asyncio.sleep(0.5)
        raise TimeoutError(f"timed out waiting for client server at {url}")

    async def terminate(self) -> None:
        await terminate_process(self.process)
        if self._stdout_task is not None:
            with contextlib.suppress(Exception):
                await self._stdout_task
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None


class WorkerProcess:
    def __init__(
        self,
        spec: WorkerSpec,
        repo_root: Path,
        python_exe: str,
        server_uri: str,
        battle_format: str,
        reconnect_seconds: float,
        on_log_line,
        worker_log_stdout: bool,
    ) -> None:
        self.spec = spec
        self.repo_root = repo_root
        self.python_exe = python_exe
        self.server_uri = server_uri
        self.battle_format = battle_format
        self.reconnect_seconds = reconnect_seconds
        self.on_log_line = on_log_line
        self.worker_log_stdout = worker_log_stdout
        self.process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._log_handle: TextIO | None = None
        self.graceful_stop_requested = False
        self.active_battles: set[str] = set()
        self.reconnect_pending = False

    def command(self) -> list[str]:
        command = [
            self.python_exe,
            "-m",
            "py.communicator.main",
            "--mode",
            self.spec.mode,
            "--games",
            "0",
            "--server-uri",
            self.server_uri,
            "--username",
            self.spec.username,
            "--format",
            self.battle_format,
            "--replay-save",
            self.spec.replay_save_token,
            "--shutdown-file",
            str(self.spec.shutdown_path),
            "--guest-refresh-seconds",
            "0",
            "--reconnect-seconds",
            str(self.reconnect_seconds),
        ]
        if self.spec.mode == "live":
            command.append("--battle-agent")
            if self.spec.checkpoint_path:
                command.append(self.spec.checkpoint_path)
        return command

    async def start(self) -> None:
        self.spec.stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.spec.stdout_log_path.open("a", encoding="utf-8")
        command = self.command()
        self._log_handle.write(f"$ {' '.join(command)}\n")
        self._log_handle.flush()
        self.process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=child_creationflags(),
        )
        self.graceful_stop_requested = False
        self.active_battles.clear()
        self.reconnect_pending = False
        self._stdout_task = asyncio.create_task(self._pipe_output())

    async def _pipe_output(self) -> None:
        if self.process is None or self.process.stdout is None or self._log_handle is None:
            return
        while True:
            line = await self.process.stdout.readline()
            if not line:
                return
            decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
            self._log_handle.write(decoded + "\n")
            self._log_handle.flush()
            self._consume_runtime_line(decoded)
            if self.worker_log_stdout:
                print(f"[{self.spec.worker_token}] {decoded}")

    def _consume_runtime_line(self, line: str) -> None:
        started = BATTLE_STARTED_RE.search(line)
        if started:
            self.active_battles.add(started.group(1))
        ended = BATTLE_ENDED_RE.search(line)
        if ended:
            self.active_battles.discard(ended.group(1))
        self.on_log_line(self, line)

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    @property
    def active_battle_count(self) -> int:
        return len(self.active_battles)

    async def request_stop(self) -> None:
        if self.graceful_stop_requested or not self.is_running():
            return
        self.graceful_stop_requested = True
        self.spec.shutdown_path.parent.mkdir(parents=True, exist_ok=True)
        self.spec.shutdown_path.write_text("stop\n", encoding="utf-8")

    async def terminate(self) -> None:
        await terminate_process(self.process)
        self.active_battles.clear()
        self.reconnect_pending = False
        if self._stdout_task is not None:
            with contextlib.suppress(Exception):
                await self._stdout_task
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()


class ReplayTailMonitor:
    def __init__(self) -> None:
        self._offsets: dict[Path, int] = {}
        self._pending: dict[Path, bytes] = {}
        self._seen_terminal_ids: dict[Path, set[str]] = {}
        self._seen_global_battle_ids: set[str] = set()
        self.completed_perspectives_by_worker: dict[str, int] = {}
        self.completed_games = 0
        self.completed_worker_perspectives = 0

    def register_worker(self, worker_token: str, replay_path: Path) -> None:
        self._offsets.setdefault(replay_path, 0)
        self._pending.setdefault(replay_path, b"")
        self._seen_terminal_ids.setdefault(replay_path, set())
        self.completed_perspectives_by_worker.setdefault(worker_token, 0)

    def poll(self, worker_specs: list[WorkerSpec]) -> int:
        newly_completed = 0
        for spec in worker_specs:
            replay_path = spec.replay_path
            if not replay_path.exists():
                continue
            with replay_path.open("rb") as handle:
                handle.seek(self._offsets.get(replay_path, 0))
                chunk = handle.read()
                self._offsets[replay_path] = handle.tell()
            if not chunk:
                continue
            pending = self._pending.get(replay_path, b"") + chunk
            lines = pending.split(b"\n")
            if pending.endswith(b"\n"):
                self._pending[replay_path] = b""
            else:
                self._pending[replay_path] = lines.pop()
            seen_ids = self._seen_terminal_ids.setdefault(replay_path, set())
            for raw_line in lines:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except Exception:
                    continue
                if record.get("type") != "terminal":
                    continue
                battle_id = str(record.get("battle_id", "")).strip()
                if not battle_id or battle_id in seen_ids:
                    continue
                seen_ids.add(battle_id)
                self.completed_worker_perspectives += 1
                self.completed_perspectives_by_worker[spec.worker_token] = (
                    self.completed_perspectives_by_worker.get(spec.worker_token, 0) + 1
                )
                if battle_id not in self._seen_global_battle_ids:
                    self._seen_global_battle_ids.add(battle_id)
                    self.completed_games += 1
                    newly_completed += 1
        return newly_completed


class PoolOrchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = Path.cwd()
        self.run_dir = run_dir_for_name(args.run_name)
        self.summary_path = self.run_dir / f"{args.run_name}_summary.json"
        self.log_path = self.run_dir / "orchestrator.log"
        self.server_log_path = self.run_dir / "showdown_server.log"
        self.client_log_path = self.run_dir / "showdown_client_static.log"
        self.server_uri = args.server_uri or default_server_uri(args.server_host, args.server_port)
        self.client_entrypoint = args.client_entrypoint
        self.client_url = default_client_url(args.client_host, args.client_port, self.client_entrypoint)
        self.watch_url = default_watch_url(
            args.client_host,
            args.client_port,
            self.client_entrypoint,
            args.server_host,
            args.server_port,
        )
        self.total_workers = max(2, args.concurrent_games * 2)
        self.model_a_workers, self.model_b_workers = split_weighted_workers(
            self.total_workers,
            args.model_a_weight,
            args.model_b_weight,
        )
        self.model_b_workers = self.total_workers - self.model_a_workers
        self.worker_specs = self._build_worker_specs()
        self.worker_processes: dict[str, WorkerProcess] = {}
        self.replay_monitor = ReplayTailMonitor()
        self.server: ServerProcess | None = None
        self.client: ClientProcess | None = None
        self._log_handle: TextIO | None = None
        self._start_time = 0.0
        self._draining = False
        self._drain_started_at: float | None = None
        self._last_drain_progress_at: float | None = None
        self._shutdown_requested = False
        self._wait_for_current_matches = False
        self._failed = False
        self._failure_reason = ""

    @property
    def target_games_label(self) -> str:
        return "infinite" if self.args.games == 0 else str(self.args.games)

    def _build_worker_specs(self) -> list[WorkerSpec]:
        specs: list[WorkerSpec] = []
        worker_id = 0
        group_counts = [("a", self.model_a_workers, self.args.model_a), ("b", self.model_b_workers, self.args.model_b)]
        for group_name, count, model_spec in group_counts:
            for group_index in range(count):
                worker_token = f"worker_{worker_id:03d}_{group_name}"
                username = f"{self.args.username_prefix}{group_name.upper()}{group_index:03d}"
                specs.append(
                    WorkerSpec(
                        worker_id=worker_id,
                        model_group=group_name,
                        checkpoint_path=checkpoint_for_model_spec(model_spec),
                        mode=worker_mode_for_model_spec(self.args.worker_think_mode, model_spec),
                        username=username,
                        replay_save_token=worker_replay_save_token(self.args.run_name, worker_token),
                        replay_path=worker_replay_path(self.args.run_name, worker_token),
                        stdout_log_path=self.run_dir / f"{worker_token}.stdout.log",
                        shutdown_path=self.run_dir / f"{worker_token}.shutdown",
                    )
                )
                worker_id += 1
        return specs

    def log(self, message: str) -> None:
        line = f"[selfplay] {message}"
        print(line)
        if self._log_handle is not None:
            self._log_handle.write(line + "\n")
            self._log_handle.flush()

    def _on_worker_log_line(self, worker: WorkerProcess, line: str) -> None:
        if "connection lost:" in line:
            if not worker.reconnect_pending:
                worker.reconnect_pending = True
                self.log(f"{worker.spec.worker_token} reconnecting after disconnect")
            return
        if worker.reconnect_pending and "[communicator] websocket connected" in line:
            worker.reconnect_pending = False
            self.log(f"{worker.spec.worker_token} reconnected successfully")

    async def _start_server(self) -> None:
        showdown_dir = (self.repo_root / self.args.showdown_dir).resolve()
        if not showdown_dir.exists():
            raise FileNotFoundError(f"showdown dir not found: {showdown_dir}")
        command = parse_server_start_command(self.args.server_start_command, self.args.server_port)
        self.server = ServerProcess(self.repo_root, showdown_dir, command, self.server_uri, self.server_log_path)
        self.log(f"starting local showdown server in {showdown_dir}")
        await self.server.start()
        await self.server.wait_until_ready(self.args.startup_timeout_seconds)
        self.log(f"showdown server ready at {self.server_uri}")
        self.log(f"watch client URL: {self.watch_url}")

    async def _start_client(self) -> None:
        if not self.args.serve_client:
            return
        client_dir = (self.repo_root / self.args.client_dir).resolve()
        entrypoint_path = client_dir / self.client_entrypoint
        if not client_dir.exists():
            self.log(f"client dir missing; local spectating disabled: {client_dir}")
            return
        if not entrypoint_path.exists():
            self.log(f"client entrypoint missing; local spectating disabled: {entrypoint_path}")
            return
        self.client = ClientProcess(client_dir, self.args.client_host, self.args.client_port, self.client_log_path)
        await self.client.start()
        await self.client.wait_until_ready(self.client_entrypoint, self.args.startup_timeout_seconds)

    async def _start_worker(self, spec: WorkerSpec) -> None:
        if spec.shutdown_path.exists():
            spec.shutdown_path.unlink()
        worker = WorkerProcess(
            spec=spec,
            repo_root=self.repo_root,
            python_exe=self.args.python_exe,
            server_uri=self.server_uri,
            battle_format=self.args.format,
            reconnect_seconds=self.args.reconnect_seconds,
            on_log_line=self._on_worker_log_line,
            worker_log_stdout=self.args.worker_log_stdout,
        )
        self.replay_monitor.register_worker(spec.worker_token, spec.replay_path)
        await worker.start()
        self.worker_processes[spec.worker_token] = worker
        checkpoint_label = spec.checkpoint_path if spec.checkpoint_path else "random"
        self.log(f"started {spec.worker_token} user={spec.username} mode={spec.mode} agent={checkpoint_label}")

    async def _start_workers(self) -> None:
        for index, spec in enumerate(self.worker_specs):
            await self._start_worker(spec)
            if index == len(self.worker_specs) - 1:
                print()

    async def _respawn_if_needed(self) -> None:
        if self._draining:
            return
        for spec in self.worker_specs:
            worker = self.worker_processes.get(spec.worker_token)
            if worker is None:
                continue
            if worker.is_running():
                continue
            return_code = worker.process.returncode if worker.process is not None else None
            if worker.reconnect_pending:
                worker.reconnect_pending = False
                self.log(f"{spec.worker_token} failed to reconnect before exit")
            await worker.terminate()
            self.log(f"{spec.worker_token} exited unexpectedly with code {return_code}; respawning")
            await self._start_worker(spec)

    async def _request_drain(self, reason: str) -> None:
        if self._draining:
            return
        self._draining = True
        self._drain_started_at = time.monotonic()
        self._last_drain_progress_at = self._drain_started_at
        self._wait_for_current_matches = reason == "ctrl+c"
        self.log(f"draining worker pool: {reason}")
        for worker in list(self.worker_processes.values()):
            await worker.request_stop()

    def _active_battle_count(self) -> int:
        return sum(worker.active_battle_count for worker in self.worker_processes.values())

    async def _terminate_workers(self) -> None:
        for worker in list(self.worker_processes.values()):
            await worker.terminate()

    async def _check_server_health(self) -> None:
        if self.server is None:
            return
        if not self.server.is_running() and not self._draining:
            code = self.server.process.returncode if self.server.process is not None else None
            self._failed = True
            self._failure_reason = f"showdown server exited unexpectedly with code {code}"
            raise RuntimeError(self._failure_reason)

    def _build_group_stats_summary(self) -> dict[str, dict[str, object]]:
        groups: dict[str, dict[str, object]] = {
            "a": {
                "matches_played": 0,
                "wins": 0,
                "earned_wins": 0,
                "losses": 0,
                "draws": 0,
                "total_moves": 0,
                "total_protects": 0,
                "total_passes": 0,
                "total_teras": 0,
                "tera_battles": 0,
                "total_move_slot_1": 0,
                "total_move_slot_2": 0,
                "total_move_slot_3": 0,
                "total_move_slot_4": 0,
                "total_switch_slot_1": 0,
                "total_switch_slot_2": 0,
                "total_switch_slot_3": 0,
                "total_switch_slot_4": 0,
                "total_switch_slot_5": 0,
                "total_switch_slot_6": 0,
                "_weighted_tera_turns": 0.0,
            },
            "b": {
                "matches_played": 0,
                "wins": 0,
                "earned_wins": 0,
                "losses": 0,
                "draws": 0,
                "total_moves": 0,
                "total_protects": 0,
                "total_passes": 0,
                "total_teras": 0,
                "tera_battles": 0,
                "total_move_slot_1": 0,
                "total_move_slot_2": 0,
                "total_move_slot_3": 0,
                "total_move_slot_4": 0,
                "total_switch_slot_1": 0,
                "total_switch_slot_2": 0,
                "total_switch_slot_3": 0,
                "total_switch_slot_4": 0,
                "total_switch_slot_5": 0,
                "total_switch_slot_6": 0,
                "_weighted_tera_turns": 0.0,
            },
        }
        sum_keys = [
            "matches_played",
            "wins",
            "earned_wins",
            "losses",
            "draws",
            "total_moves",
            "total_protects",
            "total_passes",
            "total_teras",
            "tera_battles",
            "total_move_slot_1",
            "total_move_slot_2",
            "total_move_slot_3",
            "total_move_slot_4",
            "total_switch_slot_1",
            "total_switch_slot_2",
            "total_switch_slot_3",
            "total_switch_slot_4",
            "total_switch_slot_5",
            "total_switch_slot_6",
        ]
        for spec in self.worker_specs:
            stats = parse_stats_file(spec.replay_path.with_suffix(".stats.txt"))
            group = groups[spec.model_group]
            for key in sum_keys:
                group[key] = int(group[key]) + int(stats.get(key, "0"))
            tera_battles = int(stats.get("tera_battles", "0"))
            avg_turns = float(stats.get("avg_turns_until_tera", "0"))
            group["_weighted_tera_turns"] = float(group["_weighted_tera_turns"]) + (tera_battles * avg_turns)
        for group in groups.values():
            matches_played = int(group["matches_played"])
            tera_battles = int(group["tera_battles"])
            total_move_slots = (
                int(group["total_move_slot_1"])
                + int(group["total_move_slot_2"])
                + int(group["total_move_slot_3"])
                + int(group["total_move_slot_4"])
            )
            total_switch_slots = (
                int(group["total_switch_slot_1"])
                + int(group["total_switch_slot_2"])
                + int(group["total_switch_slot_3"])
                + int(group["total_switch_slot_4"])
                + int(group["total_switch_slot_5"])
                + int(group["total_switch_slot_6"])
            )
            group["win_rate"] = (float(group["wins"]) / matches_played) if matches_played > 0 else 0.0
            group["earned_win_rate"] = (float(group["earned_wins"]) / matches_played) if matches_played > 0 else 0.0
            group["tera_rate"] = (tera_battles / matches_played) if matches_played > 0 else 0.0
            group["avg_turns_until_tera"] = (
                float(group["_weighted_tera_turns"]) / tera_battles if tera_battles > 0 else 0.0
            )
            group["move_slot_rates"] = {
                "slot_1": (int(group["total_move_slot_1"]) / total_move_slots) if total_move_slots > 0 else 0.0,
                "slot_2": (int(group["total_move_slot_2"]) / total_move_slots) if total_move_slots > 0 else 0.0,
                "slot_3": (int(group["total_move_slot_3"]) / total_move_slots) if total_move_slots > 0 else 0.0,
                "slot_4": (int(group["total_move_slot_4"]) / total_move_slots) if total_move_slots > 0 else 0.0,
            }
            group["switch_slot_rates"] = {
                "slot_1": (int(group["total_switch_slot_1"]) / total_switch_slots) if total_switch_slots > 0 else 0.0,
                "slot_2": (int(group["total_switch_slot_2"]) / total_switch_slots) if total_switch_slots > 0 else 0.0,
                "slot_3": (int(group["total_switch_slot_3"]) / total_switch_slots) if total_switch_slots > 0 else 0.0,
                "slot_4": (int(group["total_switch_slot_4"]) / total_switch_slots) if total_switch_slots > 0 else 0.0,
                "slot_5": (int(group["total_switch_slot_5"]) / total_switch_slots) if total_switch_slots > 0 else 0.0,
                "slot_6": (int(group["total_switch_slot_6"]) / total_switch_slots) if total_switch_slots > 0 else 0.0,
            }
            del group["_weighted_tera_turns"]
        return groups

    def _write_summary(self) -> None:
        group_modes = {
            "a": sorted({spec.mode for spec in self.worker_specs if spec.model_group == "a"}),
            "b": sorted({spec.mode for spec in self.worker_specs if spec.model_group == "b"}),
        }
        summary = {
            "status": "failed" if self._failed else "completed",
            "failure_reason": self._failure_reason,
            "run_name": self.args.run_name,
            "target_games": self.args.games,
            "completed_games": self.replay_monitor.completed_games,
            "completed_worker_perspectives": self.replay_monitor.completed_worker_perspectives,
            "concurrent_games": self.args.concurrent_games,
            "worker_count": self.total_workers,
            "model_a_workers": self.model_a_workers,
            "model_b_workers": self.model_b_workers,
            "server_uri": self.server_uri,
            "format": self.args.format,
            "model_group_modes": group_modes,
            "group_stats": self._build_group_stats_summary(),
            "run_duration_seconds": max(time.monotonic() - self._start_time, 0.0),
            "per_worker_completed_perspectives": dict(sorted(self.replay_monitor.completed_perspectives_by_worker.items())),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    async def run(self) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._start_time = time.monotonic()
        loop = asyncio.get_running_loop()
        previous_sigint_handler = signal.getsignal(signal.SIGINT)

        def handle_sigint(_signum: int, _frame: object) -> None:
            loop.call_soon_threadsafe(lambda: asyncio.create_task(self._request_drain("ctrl+c")))

        signal.signal(signal.SIGINT, handle_sigint)
        try:
            await self._start_server()
            await self._start_client()
            await self._start_workers()
            self.log(
                f"pool ready: target_games={self.target_games_label} concurrent_games={self.args.concurrent_games} "
                f"workers={self.total_workers} model_a_workers={self.model_a_workers} model_b_workers={self.model_b_workers}"
            )
            while True:
                await asyncio.sleep(1.0)
                newly_completed = self.replay_monitor.poll(self.worker_specs)
                if newly_completed:
                    self.log(
                        f"completed_games={self.replay_monitor.completed_games}/{self.target_games_label}"
                    )
                await self._check_server_health()
                if not self._draining and self.args.games > 0 and self.replay_monitor.completed_games >= self.args.games:
                    await self._request_drain("target games reached")
                await self._respawn_if_needed()
                if self._draining:
                    all_exited = all(not worker.is_running() for worker in self.worker_processes.values())
                    if all_exited:
                        break
                    drain_elapsed = time.monotonic() - (self._drain_started_at or time.monotonic())
                    active_battles = self._active_battle_count()
                    now = time.monotonic()
                    if (
                        self._last_drain_progress_at is None
                        or (now - self._last_drain_progress_at) >= 10.0
                    ):
                        self.log(
                            f"drain progress: active_battles={active_battles} "
                            f"completed_games={self.replay_monitor.completed_games}/{self.target_games_label}"
                        )
                        self._last_drain_progress_at = now
                    if active_battles == 0:
                        for worker in self.worker_processes.values():
                            await worker.request_stop()
                    if not self._wait_for_current_matches and drain_elapsed >= self.args.shutdown_grace_seconds:
                        self.log("shutdown grace expired; terminating remaining workers")
                        break
            await self._terminate_workers()
            return 1 if self._failed else 0
        except Exception as exc:
            self._failed = True
            self._failure_reason = str(exc)
            self.log(f"fatal error: {exc}")
            await self._terminate_workers()
            return 1
        finally:
            if self.server is not None:
                await self.server.terminate()
            self._write_summary()
            if self._log_handle is not None:
                self._log_handle.flush()
                self._log_handle.close()
            if self.client is not None:
                await self.client.terminate()
            signal.signal(signal.SIGINT, previous_sigint_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--games", required=True, type=nonnegative_int)
    parser.add_argument("--concurrent-games", required=True, type=positive_int)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--showdown-dir", default=str(DEFAULT_SHOWDOWN_DIR))
    parser.add_argument("--client-dir", default=str(DEFAULT_CLIENT_DIR))
    parser.add_argument("--server-host", default=DEFAULT_HOST)
    parser.add_argument("--server-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--server-uri", default="")
    parser.add_argument("--client-host", default=DEFAULT_HOST)
    parser.add_argument("--client-port", type=int, default=DEFAULT_CLIENT_PORT)
    parser.add_argument("--client-entrypoint", default="play.pokemonshowdown.com/testclient-old.html")
    parser.add_argument("--serve-client", type=parse_bool01, default=True)
    parser.add_argument("--server-start-command", default="")
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--worker-think-mode", choices=["live", "random"], default="live")
    parser.add_argument("--model-a-weight", type=positive_int, default=1)
    parser.add_argument("--model-b-weight", type=positive_int, default=1)
    parser.add_argument("--reconnect-seconds", type=nonnegative_float, default=DEFAULT_RECONNECT_SECONDS)
    parser.add_argument("--startup-timeout-seconds", type=nonnegative_float, default=DEFAULT_STARTUP_TIMEOUT_SECONDS)
    parser.add_argument("--shutdown-grace-seconds", type=nonnegative_float, default=DEFAULT_SHUTDOWN_GRACE_SECONDS)
    parser.add_argument("--username-prefix", default=DEFAULT_USERNAME_PREFIX)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--worker-log-stdout", type=parse_bool01, default=False)
    return parser


async def async_main() -> int:
    parser = build_parser()
    default_argv = load_default_args(DEFAULT_ARGS_PATH)
    argv = default_argv + sys.argv[1:]
    args = parser.parse_args(argv)
    orchestrator = PoolOrchestrator(args)
    return await orchestrator.run()


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
