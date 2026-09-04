from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import TextIO

from artifact_io import write_json_atomically
from rl_defaults import float_default

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
    from rich.rule import Rule
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


DEFAULT_RUNS_ROOT = Path("matches") / "runs"
DEFAULT_TRAINER_EXE = Path("build-fresh") / "showdown_client.exe"
DEFAULT_ARGS_PATH = Path("config/train_batch_selfplay.toml")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
EPISODES_RE = re.compile(r"episodes=(\d+)/(\d+)")
TRAIN_SPLIT_RE = re.compile(r"split train_sessions=(\d+)")
BATCH_FILE_START_RE = re.compile(r"\[train-batch\] file_start index=(\d+)/(\d+).* path=(.+)$")
BATCH_FILE_COMPLETE_RE = re.compile(r"\[train-batch\] file_complete index=(\d+)/(\d+)")


def ema_update(previous: float, sample: float, alpha: float) -> float:
    if previous <= 0.0:
        return sample
    return (alpha * sample) + ((1.0 - alpha) * previous)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_bool01(value: str) -> bool:
    if value not in {"0", "1"}:
        raise argparse.ArgumentTypeError("value must be 0 or 1")
    return value == "1"


def parse_config_entries(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    entries: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise SystemExit(f"invalid config {path}:{line_number}: expected key = value")
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip()
        value_text = raw_value.strip()
        if not key:
            raise SystemExit(f"invalid config {path}:{line_number}: empty key")
        lowered = value_text.lower()
        if lowered in {"true", "false"}:
            value = "1" if lowered == "true" else "0"
        elif len(value_text) >= 2 and value_text[0] == '"' and value_text[-1] == '"':
            value = bytes(value_text[1:-1], "utf-8").decode("unicode_escape")
        else:
            value = value_text
        entries.append((key, value))
    return entries


def load_default_args(path: Path) -> list[str]:
    if not path.exists():
        return []
    args: list[str] = []
    for key, value in parse_config_entries(path):
        if key.startswith("env_"):
            continue
        flag = "--" + key.replace("_", "-")
        args.extend([flag, value])
    return args


def load_default_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in parse_config_entries(path):
        if not key.startswith("env_"):
            continue
        env_name = key[4:].strip()
        if not env_name:
            raise SystemExit(f"invalid config {path}: env_ key must include a variable name")
        env[env_name.upper()] = value
    return env


def has_path_separator(path: str) -> bool:
    return "/" in path or "\\" in path


def resolve_batch_checkpoint(run_name: str, checkpoint_arg: str) -> Path:
    checkpoint_path = Path(checkpoint_arg)
    if has_path_separator(checkpoint_arg):
        return checkpoint_path
    stem = checkpoint_path.stem or checkpoint_path.name
    return Path("models") / "runs" / run_name / stem / checkpoint_path.name


def resolve_optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value)


def coerce_value(value: str) -> int | float | str:
    lowered = value.lower()
    if lowered in {"estimating"}:
        return value
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_key_values(line: str) -> dict[str, int | float | str]:
    return {match.group(1): coerce_value(match.group(2)) for match in KEY_VALUE_RE.finditer(line)}


def validation_hash(battle_id: str, seed: int) -> int:
    value = 14695981039346656037
    for shift in range(0, 32, 8):
        value ^= (seed >> shift) & 0xFF
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    for byte in battle_id.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def is_validation_battle(battle_id: str, seed: int) -> bool:
    return validation_hash(battle_id, seed) % 10 == 0


def batch_stats_dir(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / f"{checkpoint_path.stem}_batch_training_stats"


def batch_log_dir(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / f"{checkpoint_path.stem}_batch_training_logs"


def batch_stats_path(checkpoint_path: Path, epoch: int, shard_index: int, replay_path: Path) -> Path:
    safe_replay = replay_path.stem.replace(" ", "_")
    return batch_stats_dir(checkpoint_path) / f"epoch{epoch:03d}_shard{shard_index:04d}_{safe_replay}_training_stats.json"


def batch_log_path(checkpoint_path: Path, epoch: int, shard_index: int, replay_path: Path) -> Path:
    safe_replay = replay_path.stem.replace(" ", "_")
    return batch_log_dir(checkpoint_path) / f"epoch{epoch:03d}_shard{shard_index:04d}_{safe_replay}.log"


def replay_path_manifest_path(checkpoint_path: Path, epoch: int) -> Path:
    return checkpoint_path.parent / f"{checkpoint_path.stem}_epoch{epoch:03d}_replays.txt"


def epoch_batch_stats_path(checkpoint_path: Path, epoch: int) -> Path:
    return batch_stats_dir(checkpoint_path) / f"epoch{epoch:03d}_batch_training_stats.json"


def epoch_batch_log_path(checkpoint_path: Path, epoch: int) -> Path:
    return batch_log_dir(checkpoint_path) / f"epoch{epoch:03d}_batch.log"


def epoch_work_checkpoint_path(checkpoint_path: Path, epoch: int) -> Path:
    return checkpoint_path.with_name(f".{checkpoint_path.stem}_epoch{epoch:03d}_work{checkpoint_path.suffix}")


def write_replay_path_manifest(path: Path, replay_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for replay_path in replay_paths:
                handle.write(str(replay_path.resolve()))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def collect_training_stats(lines: list[str]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("[train] OpenMP threads="):
            stats["openmp"] = parse_key_values(line)
        elif line.startswith("[train] ingest complete "):
            stats["ingest"] = parse_key_values(line)
        elif line.startswith("[train] accepted_labels "):
            accepted = stats.setdefault("accepted_labels", [])
            if isinstance(accepted, list):
                accepted.append(parse_key_values(line))
        elif line.startswith("[train-batch] file_complete "):
            batch_files = stats.setdefault("batch_files", [])
            if isinstance(batch_files, list):
                result = parse_key_values(line)
                index_value = result.get("index")
                if isinstance(index_value, str) and "/" in index_value:
                    index_text, count_text = index_value.split("/", 1)
                    if index_text.isdigit() and count_text.isdigit():
                        result["index"] = int(index_text)
                        result["file_count"] = int(count_text)
                batch_files.append(result)
        elif line.startswith("[train] split "):
            stats["split"] = parse_key_values(line)
        elif line.startswith("[train] epoch=") and "supervised_profile" in line:
            profiles = stats.setdefault("supervised_profile", [])
            if isinstance(profiles, list):
                profiles.append(parse_key_values(line))
        elif (line.startswith("[train] epoch=") or line.startswith("[eval]")) and " validation action_loss=" in line:
            stats["validation_summary"] = parse_key_values(line)
        elif (line.startswith("[train] epoch=") or line.startswith("[eval]")) and " validation top3_accuracy=" in line:
            stats["validation_breakdown"] = parse_key_values(line)
        elif (line.startswith("[train] epoch=") or line.startswith("[eval]")) and " validation metrics_version=" in line:
            stats["validation_policy_metrics"] = parse_key_values(line)
        elif (line.startswith("[train] epoch=") or line.startswith("[eval]")) and " validation elapsed=" in line:
            stats["validation_timing"] = parse_key_values(line)
        elif line.startswith("trained mode="):
            stats["final_train"] = parse_key_values(line)
        elif line.startswith("[train] saved checkpoint "):
            stats["saved_checkpoint"] = line.removeprefix("[train] saved checkpoint ").strip()
    return stats


def load_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def replay_run_summary_path(run_name: str) -> Path:
    return DEFAULT_RUNS_ROOT / run_name / f"{run_name}_summary.json"


def replay_run_manifest_path(run_name: str) -> Path:
    return DEFAULT_RUNS_ROOT / run_name / f"{run_name}_manifest.json"


def infer_collection_pool_source(run_name: str) -> dict[str, str]:
    summary = load_json_if_exists(replay_run_summary_path(run_name))
    if not summary:
        return {}
    pool_source: dict[str, str] = {}
    model_a_pool = summary.get("model_a_pool_path")
    model_b_pool = summary.get("model_b_pool_path")
    if isinstance(model_a_pool, str) and model_a_pool.strip():
        pool_source["model_a_pool_path"] = model_a_pool
    if isinstance(model_b_pool, str) and model_b_pool.strip():
        pool_source["model_b_pool_path"] = model_b_pool
    return pool_source


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0.0, seconds)
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0.0, seconds)
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if total >= 86400:
        return f"{days} d {hours} hr"
    if total < 600:
        return f"{minutes} min {secs} s"
    if hours > 0:
        return f"{hours} hr {minutes} min"
    return f"{minutes} min"


@dataclass
class ShardDisplayState:
    epoch: int
    shard_index: int
    shard_count: int
    filename: str
    status: str = "queued"
    started_at: float | None = None
    elapsed_seconds: float | None = None
    return_code: int | None = None
    stats_path: str | None = None
    raw_log_path: str | None = None
    progress_current: int = 0
    progress_total: int = 0


@dataclass
class RunDisplayState:
    run: str
    mode: str
    checkpoint: str
    epochs: int
    current_epoch: int = 0
    planned_total_shards: int = 0
    completed_files: int = 0
    started_at: float = field(default_factory=time.monotonic)
    current_file: str | None = None
    configured_env: dict[str, str] = field(default_factory=dict)
    eta_seconds: float | None = None
    shards_per_minute: float | None = None
    current_epoch_shards: int = 0
    current_epoch_completed: int = 0
    last_notice: str | None = None
    dashboard_mode: str = "plain-text"
    epoch_shards: list[ShardDisplayState] = field(default_factory=list)
    current_shard_index: int | None = None


class BaseReporter:
    def __init__(self, args: argparse.Namespace, state: RunDisplayState) -> None:
        self.args = args
        self.state = state

    def trainer_line(self, line: str) -> None:
        return

    def run_started(self) -> None:
        return

    def epoch_started(self) -> None:
        return

    def shard_started(self) -> None:
        return

    def shard_finished(self) -> None:
        return

    def shard_failed(self) -> None:
        return

    def run_finished(self) -> None:
        return

    def tick(self) -> None:
        return

    def close(self) -> None:
        return


class PlainTextReporter(BaseReporter):
    def __init__(self, args: argparse.Namespace, state: RunDisplayState, fallback_notice: str | None = None) -> None:
        super().__init__(args, state)
        self.fallback_notice = fallback_notice

    def trainer_line(self, line: str) -> None:
        print(line, end="")

    def run_started(self) -> None:
        if self.fallback_notice:
            print(self.fallback_notice, flush=True)
        sample_desc = self.args.sample_files if self.args.sample_files > 0 else "all"
        print(
            f"[train_batch_selfplay] run={self.args.run} mode={self.args.mode} epochs={self.args.epochs} "
            f"epochs_per_file={self.args.epochs_per_file} shards={self.state.current_epoch_shards or self.state.planned_total_shards} "
            f"sample_files={sample_desc} checkpoint={self.args.resolved_checkpoint} "
            f"config_env={self.state.configured_env if self.state.configured_env else '{}'}",
            flush=True,
        )

    def epoch_started(self) -> None:
        sample_desc = self.args.sample_files if self.args.sample_files > 0 else "all"
        print(
            f"[train_batch_selfplay] epoch {self.state.current_epoch}/{self.args.epochs} start shards={self.state.current_epoch_shards} "
            f"shuffle={int(self.args.shuffle)} sample_files={sample_desc}",
            flush=True,
        )

    def shard_started(self) -> None:
        print(
            f"[train_batch_selfplay] epoch {self.state.current_epoch}/{self.args.epochs} "
            f"shard {self.state.current_shard_index}/{self.state.current_epoch_shards} "
            f"path={self.state.current_file}",
            flush=True,
        )

    def shard_finished(self) -> None:
        if self.state.last_notice:
            print(self.state.last_notice, flush=True)

    def shard_failed(self) -> None:
        if self.state.last_notice:
            print(self.state.last_notice, flush=True)

    def run_finished(self) -> None:
        elapsed = time.monotonic() - self.state.started_at
        print(
            f"[train_batch_selfplay] complete files={self.state.completed_files} total_elapsed={elapsed:.1f}s "
            f"checkpoint={self.args.resolved_checkpoint}",
            flush=True,
        )


class RichDashboardReporter(BaseReporter):
    def __init__(self, args: argparse.Namespace, state: RunDisplayState) -> None:
        super().__init__(args, state)
        self.console = Console()
        self.live = Live(self._render(), console=self.console, refresh_per_second=12, transient=False)
        self.started = False

    def run_started(self) -> None:
        if not self.started:
            self.live.start()
            self.started = True
            self._refresh()

    def epoch_started(self) -> None:
        self._refresh()

    def shard_started(self) -> None:
        self._refresh()

    def shard_finished(self) -> None:
        self._refresh()

    def shard_failed(self) -> None:
        self._refresh()

    def run_finished(self) -> None:
        self._refresh()

    def close(self) -> None:
        if self.started:
            self.live.stop()

    def trainer_line(self, line: str) -> None:
        batch_start = BATCH_FILE_START_RE.search(line)
        if batch_start and self.state.epoch_shards:
            index = int(batch_start.group(1))
            if 1 <= index <= len(self.state.epoch_shards):
                self.state.current_shard_index = index
                self.state.current_file = Path(batch_start.group(3)).name
                shard = self.state.epoch_shards[index - 1]
                shard.status = "running"
                shard.started_at = time.monotonic()
        batch_complete = BATCH_FILE_COMPLETE_RE.search(line)
        if batch_complete and self.state.epoch_shards:
            index = int(batch_complete.group(1))
            if 1 <= index <= len(self.state.epoch_shards):
                shard = self.state.epoch_shards[index - 1]
                if shard.started_at is not None:
                    shard.elapsed_seconds = time.monotonic() - shard.started_at
                shard.started_at = None
                shard.status = "done"
                shard.progress_current = shard.progress_total or 1
                shard.progress_total = shard.progress_total or 1
                self.state.current_epoch_completed = index
                self.state.completed_files += 1
        if self.state.current_shard_index is not None and self.state.epoch_shards:
            shard = self.state.epoch_shards[self.state.current_shard_index - 1]
            split_match = TRAIN_SPLIT_RE.search(line)
            if split_match and shard.progress_total <= 0:
                shard.progress_total = int(split_match.group(1))
                shard.progress_current = 0
            episode_match = EPISODES_RE.search(line)
            if episode_match:
                shard.progress_current = int(episode_match.group(1))
                shard.progress_total = int(episode_match.group(2))
        self._refresh()

    def tick(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self.started:
            self.live.update(self._render(), refresh=True)

    def _truncate(self, text: str, width: int) -> str:
        if width <= 1:
            return text[:width]
        if len(text) <= width:
            return text
        return text[: max(1, width - 1)] + "…"

    def _bar_string(self, completed: int, total: int, width: int, style: str) -> str:
        width = max(12, width)
        total = max(1, total)
        completed = max(0, min(completed, total))
        filled = int(round((completed / total) * width))
        filled = max(0, min(filled, width))
        return f"[{style}]" + ("━" * filled) + f"[/][grey27]" + ("─" * (width - filled)) + "[/grey27]"

    def _running_bar(self, width: int) -> str:
        width = max(12, width)
        pulse_width = max(8, width // 5)
        travel = max(1, width - pulse_width)
        offset = int(time.monotonic() * 9) % (travel * 2)
        if offset >= travel:
            offset = (travel * 2) - offset
        prefix = "─" * offset
        pulse = "━" * pulse_width
        suffix = "─" * max(0, width - offset - pulse_width)
        return f"[grey27]{prefix}[/grey27][bold cyan]{pulse}[/bold cyan][grey27]{suffix}[/grey27]"

    def _shard_layout_widths(self) -> tuple[int, int]:
        terminal_width = max(80, self.console.size.width)
        reserved = 38
        min_file_width = 14
        max_file_width = 24
        min_bar_width = 28
        available = max(min_file_width + min_bar_width, terminal_width - reserved)
        preferred_file_width = min(max_file_width, max(min_file_width, terminal_width // 5))
        file_width = min(preferred_file_width, max(min_file_width, available - min_bar_width))
        bar_width = max(min_bar_width, available - file_width - 2)
        return file_width, bar_width

    def _top_summary(self):
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")
        omp_threads = self.state.configured_env.get("PORYGON_OMP_THREADS", "default")
        elapsed = format_duration(time.monotonic() - self.state.started_at)
        eta = format_eta(self.state.eta_seconds)
        throughput = f"{self.state.shards_per_minute:.2f} shards/min" if self.state.shards_per_minute else "estimating"
        current_file = self.state.current_file or "-"
        table.add_row(
            f"[bold]Run[/]: {self.state.run}",
            f"[bold]Mode[/]: {self.state.mode}",
            f"[bold]Epoch[/]: {self.state.current_epoch}/{self.state.epochs}",
            f"[bold]Shard[/]: {self.state.completed_files + (1 if self.state.current_shard_index else 0)}/{self.state.planned_total_shards}",
        )
        table.add_row(
            f"[bold]Checkpoint[/]: {Path(self.state.checkpoint).name}",
            f"[bold]Elapsed[/]: {elapsed}",
            f"[bold]ETA[/]: {eta}",
            f"[bold]Throughput[/]: {throughput}",
        )
        table.add_row(
            f"[bold]OMP Threads[/]: {omp_threads}",
            f"[bold]Epochs/File[/]: {self.args.epochs_per_file}",
            f"[bold]Epoch Shards[/]: {self.state.current_epoch_completed}/{self.state.current_epoch_shards}",
            f"[bold]Current File[/]: {current_file}",
        )
        return Panel(table, title="Training Summary", border_style="cyan")

    def _epoch_progress(self):
        total = max(1, self.state.current_epoch_shards)
        completed = min(self.state.current_epoch_completed, total)
        terminal_width = max(80, self.console.size.width)
        bar_width = max(24, terminal_width - 32)
        percent = int(round((completed / total) * 100.0))
        bar = self._bar_string(completed, total, bar_width, "cyan")
        line = Table.grid(expand=True)
        line.add_column(justify="left")
        line.add_column(justify="right", width=24)
        line.add_row(
            f"[bold]Epoch {self.state.current_epoch}/{self.state.epochs}[/]  {bar}",
            f"{completed}/{total} shards  {percent:3d}%",
        )
        return Panel(line, border_style="grey42", padding=(0, 1))

    def _status_icon(self, shard: ShardDisplayState) -> str:
        if shard.status == "done":
            return "[green]●[/green]"
        if shard.status == "failed":
            return "[red]✖[/red]"
        if shard.status == "queued":
            return "[grey50]○[/grey50]"
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        return f"[cyan]{spinner[int(time.monotonic() * 10) % len(spinner)]}[/cyan]"

    def _status_bar(self, shard: ShardDisplayState, width: int) -> str:
        inner_width = max(12, width - 2)
        if shard.status == "done":
            total = shard.progress_total or 1
            completed = shard.progress_current or total
            inner = self._bar_string(completed, total, inner_width, "green")
        elif shard.status == "failed":
            total = shard.progress_total or 1
            completed = shard.progress_current or total
            inner = self._bar_string(completed, total, inner_width, "red")
        elif shard.status == "queued":
            inner = self._bar_string(0, 1, inner_width, "grey50")
        else:
            if shard.progress_total > 0:
                inner = self._bar_string(shard.progress_current, shard.progress_total, inner_width, "cyan")
            else:
                inner = self._running_bar(inner_width)
        return f"[grey58][[/grey58]{inner}[grey58]][/grey58]"

    def _activity_text(self, shard: ShardDisplayState) -> str:
        if shard.status == "done":
            return "[green]done[/green]"
        if shard.status == "failed":
            detail = f" exit={shard.return_code}" if shard.return_code is not None else ""
            return f"[red]failed{detail}[/red]"
        if shard.status == "queued":
            return "[grey50]queued[/grey50]"
        if shard.progress_total > 0:
            percent = int(round((shard.progress_current / max(1, shard.progress_total)) * 100.0))
            return f"[cyan]{shard.progress_current}/{shard.progress_total} {percent:02d}%[/cyan]"
        return "[cyan]active[/cyan]"

    def _visible_shards(self) -> list[ShardDisplayState]:
        shards = self.state.epoch_shards
        if not shards:
            return []
        visible = max(1, self.args.dashboard_visible_shards)
        current = (self.state.current_shard_index or 1) - 1
        before = visible // 2
        after = visible - before - 1
        start = max(0, current - before)
        end = min(len(shards), current + after + 1)
        if end - start < visible:
            if start == 0:
                end = min(len(shards), visible)
            elif end == len(shards):
                start = max(0, len(shards) - visible)
        return shards[start:end]

    def _shard_table(self):
        table = Table(expand=True, box=None, pad_edge=False, show_header=True, header_style="bold white")
        file_width, bar_width = self._shard_layout_widths()
        table.add_column("", width=2)
        table.add_column("Shard", width=10)
        table.add_column("File", width=file_width, no_wrap=True, overflow="ellipsis")
        table.add_column("Progress", width=bar_width, no_wrap=True, overflow="crop")
        table.add_column("State", width=8)
        table.add_column("Elapsed", width=8, justify="right")
        for shard in self._visible_shards():
            running_elapsed = None
            if shard.status == "running" and shard.started_at is not None:
                running_elapsed = time.monotonic() - shard.started_at
            effective_elapsed = running_elapsed if running_elapsed is not None else shard.elapsed_seconds
            elapsed = format_duration(effective_elapsed) if effective_elapsed is not None else "-"
            file_style = "bold cyan" if shard.status == "running" else ("red" if shard.status == "failed" else "")
            filename = self._truncate(shard.filename, file_width)
            table.add_row(
                self._status_icon(shard),
                f"{shard.shard_index}/{shard.shard_count}",
                f"[{file_style}]{filename}[/{file_style}]" if file_style else filename,
                self._status_bar(shard, bar_width),
                self._activity_text(shard),
                elapsed,
            )
        return table

    def _render(self):
        renderables: list[object] = []
        if self.args.dashboard_show_top_stats:
            renderables.append(self._top_summary())
        if self.args.dashboard_show_epoch_bar:
            renderables.append(self._epoch_progress())
        if self.args.dashboard_show_shard_window:
            renderables.append(Rule(style="grey50"))
            renderables.append(self._shard_table())
        return Group(*renderables)


def select_reporter(args: argparse.Namespace, state: RunDisplayState) -> BaseReporter:
    if not args.dashboard:
        state.dashboard_mode = "plain-text"
        return PlainTextReporter(args, state)
    if not RICH_AVAILABLE:
        state.dashboard_mode = "plain-text"
        return PlainTextReporter(args, state, "[train_batch_selfplay] dashboard unavailable; using plain-text output")
    if not sys.stdout.isatty():
        state.dashboard_mode = "plain-text"
        return PlainTextReporter(args, state, "[train_batch_selfplay] live terminal unavailable; using plain-text output")
    state.dashboard_mode = "terminal"
    return RichDashboardReporter(args, state)


def run_trainer_and_capture(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    reporter: BaseReporter,
    raw_log_fp: TextIO | None,
) -> tuple[int, list[str]]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    captured_lines: list[str] = []
    assert process.stdout is not None
    line_queue: queue.Queue[str | None] = queue.Queue()

    def enqueue_output() -> None:
        try:
            for line in process.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)

    reader = threading.Thread(target=enqueue_output, daemon=True)
    reader.start()

    stream_closed = False
    while not stream_closed:
        try:
            line = line_queue.get(timeout=0.1)
        except queue.Empty:
            reporter.tick()
            if process.poll() is not None:
                reporter.tick()
            continue
        if line is None:
            stream_closed = True
            continue
        if raw_log_fp is not None:
            raw_log_fp.write(line)
        reporter.trainer_line(line)
        captured_lines.append(line.rstrip("\r\n"))
    if raw_log_fp is not None:
        raw_log_fp.flush()
    reader.join(timeout=0.5)
    return_code = process.wait()
    return return_code, captured_lines


def evaluate_fixed_holdout(
    args: argparse.Namespace,
    replay_paths: list[Path],
    checkpoint_path: Path,
    repo_root: Path,
    env: dict[str, str],
    reporter: BaseReporter,
    state: RunDisplayState,
) -> dict[str, object]:
    shard_stats: list[dict[str, object]] = []
    state.current_shard_index = None
    for index, replay_path in enumerate(replay_paths, start=1):
        state.current_file = f"validation {index}/{len(replay_paths)}: {replay_path.name}"
        state.last_notice = f"evaluating fixed holdout {index}/{len(replay_paths)}"
        reporter.tick()
        command = [
            str(Path(args.trainer_exe)),
            "--eval-supervised",
            str(replay_path),
            str(checkpoint_path),
            "--validation-seed",
            str(args.validation_seed),
        ]
        return_code, lines = run_trainer_and_capture(command, repo_root, env, reporter, None)
        if return_code != 0:
            raise RuntimeError(
                f"fixed holdout evaluation failed shard={index}/{len(replay_paths)} "
                f"path={replay_path} exit_code={return_code}"
            )
        shard_stats.append(collect_training_stats(lines))
    state.current_file = None
    return aggregate_validation_metrics(
        shard_stats,
        source_shard_count=len(replay_paths),
        validation_seed=args.validation_seed,
    )


def write_batch_training_stats(
    path: Path,
    *,
    args: argparse.Namespace,
    epoch: int,
    shard_index: int,
    shard_count: int,
    replay_path: Path,
    command: list[str],
    elapsed_seconds: float,
    return_code: int,
    parsed_stats: dict[str, object],
    raw_log_path_value: str | None,
    dashboard_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": args.run,
        "mode": args.mode,
        "epoch": epoch,
        "epochs": args.epochs,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "replay_path": str(replay_path),
        "checkpoint": str(args.resolved_checkpoint),
        "command": command,
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
        "dashboard_mode": dashboard_mode,
        "raw_log_path": raw_log_path_value,
        "stats": parsed_stats,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_training_manifest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / f"{checkpoint_path.stem}_training_manifest.json"


def dataset_epoch_snapshot_path(checkpoint_path: Path, epoch: int) -> Path:
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_dataset_epoch{epoch:03d}{checkpoint_path.suffix}"
    )


def dataset_epoch_validation_path(checkpoint_path: Path, epoch: int) -> Path:
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_dataset_epoch{epoch:03d}_validation.json"
    )


def copy_file_atomically(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        # Windows rejects fsync on a read-only descriptor. The temporary copy
        # must be opened writable even though no further bytes are changed.
        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


VALIDATION_METRIC_DENOMINATORS = {
    "action_nll": "turns",
    "target_nll": "target_labels",
    "full_turn_nll": "turns",
    "value_loss": "turns",
    "full_turn_accuracy": "turns",
    "top3_accuracy": "turns",
    "slot0_accuracy": "slot0_labels",
    "slot1_accuracy": "slot1_labels",
    "joint_pair_accuracy": "joint_pairs",
    "kind_accuracy": "kind_labels",
    "move_accuracy": "move_labels",
    "switch_accuracy": "switch_labels",
    "tera_accuracy": "tera_labels",
    "target_accuracy": "target_labels",
}

VALIDATION_COUNT_KEYS = (
    "turns",
    "action_labels",
    "slot0_labels",
    "slot1_labels",
    "joint_pairs",
    "kind_labels",
    "move_labels",
    "switch_labels",
    "tera_labels",
    "target_labels",
    "skipped_turns",
    "illegal_predictions",
    "nonfinite_values",
)


def aggregate_validation_metrics(
    shard_stats: list[dict[str, object]],
    *,
    source_shard_count: int,
    validation_seed: int,
) -> dict[str, object]:
    records = [
        stats["validation_policy_metrics"]
        for stats in shard_stats
        if isinstance(stats.get("validation_policy_metrics"), dict)
    ]
    counts = {
        key: sum(int(record.get(key, 0)) for record in records)
        for key in VALIDATION_COUNT_KEYS
    }
    metrics: dict[str, float] = {}
    for metric, denominator_key in VALIDATION_METRIC_DENOMINATORS.items():
        denominator = counts[denominator_key]
        weighted_sum = sum(
            float(record.get(metric, 0.0)) * int(record.get(denominator_key, 0))
            for record in records
        )
        metrics[metric] = weighted_sum / denominator if denominator > 0 else 0.0
    return {
        "metrics_version": 2,
        "validation_seed": validation_seed,
        "source_shard_count": source_shard_count,
        "evaluated_shard_count": len(shard_stats),
        "shards_with_holdout_labels": len(records),
        "counts": counts,
        "metrics": metrics,
    }


def validate_resume_manifest(
    manifest: dict[str, object],
    *,
    run: str,
    mode: str,
    checkpoint: Path,
    source_files: list[Path],
    validation_seed: int,
    epochs: int,
    epochs_per_file: int,
    sample_files: int,
    shuffle: bool,
) -> None:
    expected_sources = [str(path.resolve()) for path in source_files]
    checks = {
        "run": run,
        "mode": mode,
        "output_checkpoint": str(checkpoint.resolve()),
        "source_shards": expected_sources,
        "validation_seed": validation_seed,
        "epochs": epochs,
        "epochs_per_file": epochs_per_file,
        "sample_files": sample_files,
        "shuffle": shuffle,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise SystemExit(
                f"cannot resume: manifest {key}={manifest.get(key)!r} does not match {expected!r}"
            )
    if not isinstance(manifest.get("epoch_plans"), dict) or not isinstance(
        manifest.get("completed_shards"), list
    ):
        raise SystemExit("cannot resume: manifest predates resumable shard provenance")


def completed_shard_map(manifest: dict[str, object]) -> dict[tuple[int, int, str], dict[str, object]]:
    completed: dict[tuple[int, int, str], dict[str, object]] = {}
    records = manifest.get("completed_shards", [])
    if not isinstance(records, list):
        return completed
    for raw in records:
        if not isinstance(raw, dict):
            continue
        key = (
            int(raw.get("outer_epoch", 0)),
            int(raw.get("shard_index", 0)),
            str(raw.get("path", "")),
        )
        if key[0] > 0 and key[1] > 0 and key[2]:
            completed[key] = raw
    return completed


def shard_identity(epoch: int, shard_index: int, replay_path: Path) -> tuple[int, int, str]:
    return epoch, shard_index, str(replay_path.resolve())


def completed_records_for_epoch(
    completed: dict[tuple[int, int, str], dict[str, object]],
    epoch: int,
    replay_paths: list[Path],
) -> dict[tuple[int, int, str], dict[str, object]]:
    records: dict[tuple[int, int, str], dict[str, object]] = {}
    for shard_index, replay_path in enumerate(replay_paths, start=1):
        key = shard_identity(epoch, shard_index, replay_path)
        if key in completed:
            records[key] = completed[key]
    return records


def write_training_manifest(path: Path, payload: dict[str, object]) -> None:
    write_json_atomically(path, payload)


def batch_label_counts(parsed_stats: dict[str, object], expected_files: int) -> dict[int, int]:
    batch_files = parsed_stats.get("batch_files")
    final_train = parsed_stats.get("final_train")
    if not isinstance(batch_files, list) or len(batch_files) != expected_files:
        reported_files = len(batch_files) if isinstance(batch_files, list) else 0
        raise ValueError(
            f"trainer reported {reported_files} completed files for a {expected_files}-file manifest"
        )
    if not isinstance(final_train, dict) or not isinstance(final_train.get("labels"), int):
        raise ValueError("trainer did not report an exact batch label count")

    labels_by_index: dict[int, int] = {}
    for result in batch_files:
        if not isinstance(result, dict):
            continue
        index_value = result.get("index")
        if isinstance(index_value, str) and "/" in index_value:
            index_value = index_value.split("/", 1)[0]
        try:
            index = int(index_value)
            labels = int(result.get("labels", 0))
        except (TypeError, ValueError):
            continue
        labels_by_index[index] = labels

    if set(labels_by_index) != set(range(1, expected_files + 1)):
        raise ValueError("trainer batch file indices do not match the replay manifest")
    if sum(labels_by_index.values()) != int(final_train["labels"]):
        raise ValueError(
            f"trainer batch label total mismatch files={sum(labels_by_index.values())} "
            f"final={final_train['labels']}"
        )
    return labels_by_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run name under matches/runs/")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path/name to train into")
    parser.add_argument("--init-checkpoint", default="", help="Optional checkpoint path/name to use for warm-start initialization")
    parser.add_argument("--experiment-id", default="", help="Optional experiment grouping token for manifests")
    parser.add_argument("--manifest-path", default="", help="Optional explicit training manifest path")
    parser.add_argument("--mode", choices=["supervised", "rl"], default="supervised")
    parser.add_argument("--epochs", type=positive_int, default=1, help="Passes over all worker files")
    parser.add_argument("--pattern", default="worker_*_raw.jsonl", help="Glob for training shards")
    parser.add_argument("--trainer-exe", default=str(DEFAULT_TRAINER_EXE))
    parser.add_argument("--shuffle", type=parse_bool01, default=True, help="Shuffle shard order each epoch")
    parser.add_argument("--supervised-profile", type=parse_bool01, default=True, help="Enable per-episode supervised profiling output")
    parser.add_argument("--supervised-optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--validation-seed", type=int, default=1337)
    parser.add_argument("--resume", type=parse_bool01, default=True)
    parser.add_argument("--dashboard", type=parse_bool01, default=True, help="Enable Rich dashboard output")
    parser.add_argument("--dashboard-visible-shards", type=positive_int, default=5, help="Number of shard rows to show in the dashboard")
    parser.add_argument("--dashboard-show-top-stats", type=parse_bool01, default=True, help="Show the dashboard top summary")
    parser.add_argument("--dashboard-show-epoch-bar", type=parse_bool01, default=True, help="Show the dashboard epoch progress bar")
    parser.add_argument("--dashboard-show-shard-window", type=parse_bool01, default=True, help="Show the dashboard shard window")
    parser.add_argument("--dashboard-write-raw-logs", type=parse_bool01, default=True, help="Write raw per-shard trainer logs")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N matched files")
    parser.add_argument("--limit-files", type=positive_int, default=0, help="Optional cap on matched files")
    parser.add_argument("--sample-files", type=positive_int, default=0, help="Train on a random subset of up to N shards per epoch")
    parser.add_argument("--epochs-per-file", type=positive_int, default=1, help="Epochs to pass to showdown_client for each shard")
    parser.add_argument("--learning-rate", type=float, default=-1.0)
    parser.add_argument("--gamma", type=float, default=float_default("policy_gradient_gamma"))
    parser.add_argument("--entropy-coef", type=float, default=float_default("policy_gradient_entropy_coef"))
    parser.add_argument("--advantage-norm", choices=["0", "1"], default="1")
    parser.add_argument("--reward-mode", default="terminal")
    return parser


def trainer_command_for_file(args: argparse.Namespace, replay_path: Path) -> list[str]:
    command = [
        str(Path(args.trainer_exe)),
        f"--train-{args.mode}",
        str(replay_path),
        str(args.resolved_checkpoint),
        "--epochs",
        str(args.epochs_per_file),
    ]
    if args.mode == "supervised":
        command.extend(
            [
                "--supervised-profile",
                "1" if args.supervised_profile else "0",
                "--supervised-optimizer",
                args.supervised_optimizer,
                "--validation-seed",
                str(args.validation_seed),
                "--aux-checkpoints",
                "0",
            ]
        )
        if args.learning_rate > 0.0:
            command.extend(["--learning-rate", str(args.learning_rate)])
    if args.mode == "rl":
        command.extend(
            [
                "--gamma",
                str(args.gamma),
                "--learning-rate",
                str(args.learning_rate),
                "--entropy-coef",
                str(args.entropy_coef),
                "--advantage-norm",
                args.advantage_norm,
                "--reward-mode",
                args.reward_mode,
            ]
        )
    return command


def trainer_command_for_manifest(args: argparse.Namespace, manifest_path: Path, checkpoint_path: Path) -> list[str]:
    command = [
        str(Path(args.trainer_exe)),
        "--train-supervised-manifest",
        str(manifest_path),
        str(checkpoint_path),
        "--epochs",
        str(args.epochs_per_file),
        "--supervised-profile",
        "1" if args.supervised_profile else "0",
        "--supervised-optimizer",
        args.supervised_optimizer,
        "--validation-seed",
        str(args.validation_seed),
        "--aux-checkpoints",
        "0",
    ]
    if args.learning_rate > 0.0:
        command.extend(["--learning-rate", str(args.learning_rate)])
    return command


def ensure_initialized_checkpoint(args: argparse.Namespace) -> tuple[bool, str | None]:
    resolved_init = args.resolved_init_checkpoint
    if args.resolved_checkpoint.exists():
        if resolved_init is not None:
            print(
                f"[train_batch_selfplay] output checkpoint already exists; ignoring --init-checkpoint {resolved_init}",
                flush=True,
            )
        return False, None
    if resolved_init is None:
        return False, None
    if not resolved_init.exists():
        raise SystemExit(f"init checkpoint not found: {resolved_init}")
    args.resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_init, args.resolved_checkpoint)
    print(
        f"[train_batch_selfplay] initialized checkpoint {args.resolved_checkpoint} from {resolved_init}",
        flush=True,
    )
    return True, str(resolved_init)


def train_supervised_epoch_batch(
    args: argparse.Namespace,
    epoch: int,
    epoch_files: list[Path],
    repo_root: Path,
    subprocess_env: dict[str, str],
    reporter: BaseReporter,
    state: RunDisplayState,
) -> tuple[Path, Path, dict[int, int], float]:
    path_manifest = replay_path_manifest_path(args.resolved_checkpoint, epoch)
    work_checkpoint = epoch_work_checkpoint_path(args.resolved_checkpoint, epoch)
    stats_path = epoch_batch_stats_path(args.resolved_checkpoint, epoch)
    raw_log = epoch_batch_log_path(args.resolved_checkpoint, epoch) if args.dashboard_write_raw_logs else None

    expected_paths = [str(path.resolve()) for path in epoch_files]
    existing_paths = []
    if path_manifest.exists():
        existing_paths = path_manifest.read_text(encoding="utf-8").splitlines()
    existing_stats = load_json_if_exists(stats_path)
    if (
        work_checkpoint.exists()
        and work_checkpoint.stat().st_size > 0
        and existing_paths == expected_paths
        and isinstance(existing_stats, dict)
        and existing_stats.get("return_code") == 0
        and isinstance(existing_stats.get("stats"), dict)
    ):
        try:
            labels_by_index = batch_label_counts(existing_stats["stats"], len(epoch_files))
        except ValueError:
            pass
        else:
            snapshot_path = dataset_epoch_snapshot_path(args.resolved_checkpoint, epoch)
            copy_file_atomically(work_checkpoint, snapshot_path)
            elapsed = float(existing_stats.get("elapsed_seconds", 0.0))
            print(
                f"[train_batch_selfplay] recovered completed epoch {epoch} batch from {work_checkpoint}",
                flush=True,
            )
            state.last_notice = f"recovered completed epoch {epoch} batch"
            return stats_path, work_checkpoint, labels_by_index, elapsed

    write_replay_path_manifest(path_manifest, epoch_files)
    if args.resolved_checkpoint.exists():
        copy_file_atomically(args.resolved_checkpoint, work_checkpoint)
    else:
        work_checkpoint.unlink(missing_ok=True)

    command = trainer_command_for_manifest(args, path_manifest, work_checkpoint)
    raw_log_fp: TextIO | None = None
    state.current_shard_index = None
    state.current_file = "starting supervised batch"
    state.last_notice = f"training {len(epoch_files)} shards in one optimizer process"
    reporter.shard_started()
    if raw_log is not None:
        raw_log.parent.mkdir(parents=True, exist_ok=True)
        raw_log_fp = raw_log.open("w", encoding="utf-8")
    started_at = time.monotonic()
    try:
        return_code, captured_lines = run_trainer_and_capture(
            command, repo_root, subprocess_env, reporter, raw_log_fp
        )
    finally:
        if raw_log_fp is not None:
            raw_log_fp.close()
    elapsed = time.monotonic() - started_at
    parsed_stats = collect_training_stats(captured_lines)
    write_json_atomically(
        stats_path,
        {
            "run": args.run,
            "mode": args.mode,
            "epoch": epoch,
            "epochs": args.epochs,
            "replay_manifest": str(path_manifest.resolve()),
            "checkpoint": str(args.resolved_checkpoint),
            "work_checkpoint": str(work_checkpoint),
            "command": command,
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "dashboard_mode": state.dashboard_mode,
            "raw_log_path": str(raw_log) if raw_log else None,
            "stats": parsed_stats,
        },
    )
    if return_code != 0:
        state.last_notice = f"supervised batch failed with exit code {return_code}"
        reporter.shard_failed()
        raise SystemExit(state.last_notice)

    try:
        labels_by_index = batch_label_counts(parsed_stats, len(epoch_files))
    except ValueError as error:
        raise SystemExit(str(error)) from error

    snapshot_path = dataset_epoch_snapshot_path(args.resolved_checkpoint, epoch)
    copy_file_atomically(work_checkpoint, snapshot_path)
    return stats_path, work_checkpoint, labels_by_index, elapsed


def main() -> None:
    parser = build_parser()
    argv = load_default_args(DEFAULT_ARGS_PATH) + sys.argv[1:]
    configured_env = load_default_env(DEFAULT_ARGS_PATH)
    args = parser.parse_args(argv)
    if args.validation_seed < 0:
        raise SystemExit("--validation-seed must be >= 0")
    repo_root = Path.cwd()
    run_dir = repo_root / DEFAULT_RUNS_ROOT / args.run
    trainer_exe = repo_root / Path(args.trainer_exe)
    args.resolved_checkpoint = resolve_batch_checkpoint(args.run, args.checkpoint)
    args.resolved_init_checkpoint = (
        resolve_batch_checkpoint(args.run, args.init_checkpoint) if args.init_checkpoint else None
    )
    if args.resolved_init_checkpoint is not None and not args.resolved_init_checkpoint.is_absolute():
        args.resolved_init_checkpoint = (repo_root / args.resolved_init_checkpoint).resolve()
    args.resolved_manifest_path = (
        resolve_optional_path(args.manifest_path) if args.manifest_path else default_training_manifest_path(args.resolved_checkpoint)
    )
    if args.resolved_manifest_path is not None and not args.resolved_manifest_path.is_absolute():
        args.resolved_manifest_path = (repo_root / args.resolved_manifest_path).resolve()

    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")
    if not trainer_exe.exists():
        raise SystemExit(f"trainer executable not found: {trainer_exe}")
    args.resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    initialized_from_init, initialized_checkpoint_path = ensure_initialized_checkpoint(args)
    subprocess_env = os.environ.copy()
    subprocess_env.update(configured_env)

    all_files = sorted(run_dir.glob(args.pattern))
    if args.start_index > 0:
        all_files = all_files[args.start_index:]
    if args.limit_files > 0:
        all_files = all_files[: args.limit_files]
    if not all_files:
        raise SystemExit(f"no input files matched {args.pattern!r} in {run_dir}")

    planned_total_shards = args.epochs * (min(len(all_files), args.sample_files) if args.sample_files > 0 else len(all_files))
    source_replay_summary = replay_run_summary_path(args.run).resolve()
    source_replay_manifest = replay_run_manifest_path(args.run).resolve()
    collection_pool_source = infer_collection_pool_source(args.run)
    existing_manifest = load_json_if_exists(args.resolved_manifest_path) if args.resume else None
    if args.resume and args.resolved_manifest_path.exists() and existing_manifest is None:
        raise SystemExit(f"cannot resume: manifest is not valid JSON: {args.resolved_manifest_path}")
    if existing_manifest is not None:
        validate_resume_manifest(
            existing_manifest,
            run=args.run,
            mode=args.mode,
            checkpoint=args.resolved_checkpoint,
            source_files=all_files,
            validation_seed=args.validation_seed,
            epochs=args.epochs,
            epochs_per_file=args.epochs_per_file,
            sample_files=args.sample_files,
            shuffle=bool(args.shuffle),
        )
        if existing_manifest.get("status") == "completed":
            print(
                f"[train_batch_selfplay] manifest already completed: {args.resolved_manifest_path}",
                flush=True,
            )
            return
        training_manifest = existing_manifest
        training_manifest["status"] = "running"
        training_manifest.pop("failure_reason", None)
        raw_dataset_epochs = training_manifest.get("dataset_epochs", [])
        dataset_epoch_records = (
            [record for record in raw_dataset_epochs if isinstance(record, dict)]
            if isinstance(raw_dataset_epochs, list)
            else []
        )
        training_manifest["dataset_epochs"] = dataset_epoch_records
        print(
            f"[train_batch_selfplay] resuming from {args.resolved_manifest_path}",
            flush=True,
        )
    else:
        dataset_epoch_records: list[dict[str, object]] = []
        training_manifest = {
            "run": args.run,
            "mode": args.mode,
            "experiment_id": args.experiment_id,
            "init_checkpoint": str(args.resolved_init_checkpoint) if args.resolved_init_checkpoint is not None else "",
            "output_checkpoint": str(args.resolved_checkpoint.resolve()),
            "checkpoint_initialized_from": initialized_checkpoint_path or "",
            "checkpoint_preexisting": args.resolved_checkpoint.exists() and not initialized_from_init,
            "source_replay_run": args.run,
            "source_replay_summary_path": str(source_replay_summary) if source_replay_summary.exists() else "",
            "source_replay_manifest_path": str(source_replay_manifest) if source_replay_manifest.exists() else "",
            "collection_pool_source": collection_pool_source,
            "pattern": args.pattern,
            "reward_mode": args.reward_mode,
            "supervised_optimizer": args.supervised_optimizer,
            "supervised_batch_mode": args.mode == "supervised",
            "optimizer_process_scope": "outer_epoch" if args.mode == "supervised" else "per_shard",
            "validation_seed": args.validation_seed,
            "aux_checkpoints": False,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "entropy_coef": args.entropy_coef,
            "advantage_norm": args.advantage_norm,
            "sample_files": args.sample_files,
            "epochs": args.epochs,
            "epochs_per_file": args.epochs_per_file,
            "shuffle": bool(args.shuffle),
            "trainer_exe": str(trainer_exe.resolve()),
            "configured_env": configured_env,
            "status": "running",
            "completed_files": 0,
            "planned_total_shards": planned_total_shards,
            "source_shard_count": len(all_files),
            "source_shards": [str(path.resolve()) for path in all_files],
            "epoch_plans": {},
            "completed_shards": [],
            "dataset_epochs": dataset_epoch_records,
        }

    completed_shards = completed_shard_map(training_manifest)
    completed_files = len(completed_shards)
    training_manifest["completed_files"] = completed_files
    write_training_manifest(args.resolved_manifest_path, training_manifest)
    state = RunDisplayState(
        run=args.run,
        mode=args.mode,
        checkpoint=str(args.resolved_checkpoint),
        epochs=args.epochs,
        planned_total_shards=planned_total_shards,
        completed_files=completed_files,
        started_at=time.monotonic(),
        configured_env=configured_env,
    )
    reporter = select_reporter(args, state)
    reporter.run_started()

    shard_rate_ema = 0.0
    eta_alpha = 0.2
    eta_min_elapsed = 30.0
    eta_min_completed = 3
    failure_message = ""

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_plans = training_manifest["epoch_plans"]
            assert isinstance(epoch_plans, dict)
            plan_key = str(epoch)
            stored_plan = epoch_plans.get(plan_key)
            if isinstance(stored_plan, list):
                epoch_files = [Path(str(path)) for path in stored_plan]
                allowed_sources = {str(path.resolve()) for path in all_files}
                unexpected = [path for path in epoch_files if str(path.resolve()) not in allowed_sources]
                if unexpected:
                    raise SystemExit(f"cannot resume: epoch {epoch} plan contains unknown shard {unexpected[0]}")
            else:
                epoch_files = list(all_files)
                if args.shuffle:
                    random.shuffle(epoch_files)
                if args.sample_files > 0 and len(epoch_files) > args.sample_files:
                    epoch_files = epoch_files[: args.sample_files]
                epoch_plans[plan_key] = [str(path.resolve()) for path in epoch_files]
                write_training_manifest(args.resolved_manifest_path, training_manifest)

            epoch_completed = completed_records_for_epoch(completed_shards, epoch, epoch_files)
            epoch_label_count = sum(int(record.get("label_count", 0)) for record in epoch_completed.values())
            completed_snapshot = dataset_epoch_snapshot_path(args.resolved_checkpoint, epoch)
            if args.mode == "supervised" and len(epoch_completed) == len(epoch_files) and completed_snapshot.exists():
                copy_file_atomically(completed_snapshot, args.resolved_checkpoint)
            epoch_started_at = time.monotonic()
            state.current_epoch = epoch
            state.current_epoch_shards = len(epoch_files)
            state.current_epoch_completed = len(epoch_completed)
            state.current_shard_index = 1 if epoch_files else None
            state.epoch_shards = [
                ShardDisplayState(epoch=epoch, shard_index=index, shard_count=len(epoch_files), filename=replay_path.name)
                for index, replay_path in enumerate(epoch_files, start=1)
            ]
            for index, replay_path in enumerate(epoch_files, start=1):
                completed_key = shard_identity(epoch, index, replay_path)
                completed_record = completed_shards.get(completed_key)
                if completed_record is None:
                    continue
                shard_state = state.epoch_shards[index - 1]
                shard_state.status = "done"
                shard_state.stats_path = str(completed_record.get("stats_path", "")) or None
                shard_state.progress_current = 1
                shard_state.progress_total = 1
            state.last_notice = f"epoch {epoch}/{args.epochs} started"
            reporter.epoch_started()

            training_files = epoch_files
            if args.mode == "supervised":
                if epoch_completed and len(epoch_completed) != len(epoch_files):
                    raise SystemExit(
                        "cannot switch a partially completed legacy supervised epoch to batch mode; "
                        "finish it with the previous executable or start a new output checkpoint"
                    )
                training_files = []
                if not epoch_completed:
                    stats_path, work_checkpoint, labels_by_index, batch_elapsed = train_supervised_epoch_batch(
                        args,
                        epoch,
                        epoch_files,
                        repo_root,
                        subprocess_env,
                        reporter,
                        state,
                    )
                    raw_completed_shards = training_manifest["completed_shards"]
                    assert isinstance(raw_completed_shards, list)
                    for shard_index, replay_path in enumerate(epoch_files, start=1):
                        completed_record = {
                            "outer_epoch": epoch,
                            "shard_index": shard_index,
                            "path": str(replay_path.resolve()),
                            "label_count": labels_by_index[shard_index],
                            "stats_path": str(stats_path.resolve()),
                            "optimizer_process": f"outer_epoch_{epoch}",
                        }
                        raw_completed_shards.append(completed_record)
                        completed_shards[shard_identity(epoch, shard_index, replay_path)] = completed_record
                        shard_state = state.epoch_shards[shard_index - 1]
                        shard_state.status = "done"
                        shard_state.stats_path = str(stats_path)
                        shard_state.progress_current = shard_state.progress_total or 1
                        shard_state.progress_total = shard_state.progress_total or 1
                    epoch_label_count = sum(labels_by_index.values())
                    completed_files = len(completed_shards)
                    state.completed_files = completed_files
                    state.current_epoch_completed = len(epoch_files)
                    training_manifest["completed_files"] = completed_files
                    training_manifest["current_epoch"] = epoch
                    training_manifest["last_completed_shard"] = len(epoch_files)
                    write_training_manifest(args.resolved_manifest_path, training_manifest)
                    copy_file_atomically(work_checkpoint, args.resolved_checkpoint)
                    work_checkpoint.unlink(missing_ok=True)
                    state.last_notice = (
                        f"epoch {epoch} batch trained labels={epoch_label_count} elapsed={batch_elapsed:.1f}s"
                    )
                    reporter.shard_finished()

            for shard_index, replay_path in enumerate(training_files, start=1):
                completed_key = shard_identity(epoch, shard_index, replay_path)
                if completed_key in completed_shards:
                    continue
                shard_started_at = time.monotonic()
                command = trainer_command_for_file(args, replay_path)
                raw_log = batch_log_path(args.resolved_checkpoint, epoch, shard_index, replay_path) if args.dashboard_write_raw_logs else None
                raw_log_fp: TextIO | None = None
                shard_state = state.epoch_shards[shard_index - 1]
                shard_state.status = "running"
                shard_state.started_at = time.monotonic()
                shard_state.raw_log_path = str(raw_log) if raw_log else None
                state.current_shard_index = shard_index
                state.current_file = replay_path.name
                state.last_notice = f"running {replay_path.name}"
                reporter.shard_started()

                if raw_log is not None:
                    raw_log.parent.mkdir(parents=True, exist_ok=True)
                    raw_log_fp = raw_log.open("w", encoding="utf-8")
                try:
                    return_code, captured_lines = run_trainer_and_capture(command, repo_root, subprocess_env, reporter, raw_log_fp)
                finally:
                    if raw_log_fp is not None:
                        raw_log_fp.close()

                shard_elapsed = time.monotonic() - shard_started_at
                parsed_stats = collect_training_stats(captured_lines)
                stats_path = batch_stats_path(args.resolved_checkpoint, epoch, shard_index, replay_path)
                write_batch_training_stats(
                    stats_path,
                    args=args,
                    epoch=epoch,
                    shard_index=shard_index,
                    shard_count=len(epoch_files),
                    replay_path=replay_path,
                    command=command,
                    elapsed_seconds=shard_elapsed,
                    return_code=return_code,
                    parsed_stats=parsed_stats,
                    raw_log_path_value=str(raw_log) if raw_log else None,
                    dashboard_mode=state.dashboard_mode,
                )

                shard_state.elapsed_seconds = shard_elapsed
                shard_state.started_at = None
                shard_state.return_code = return_code
                shard_state.stats_path = str(stats_path)
                state.last_notice = f"wrote stats {stats_path}"
                if return_code != 0:
                    if shard_state.progress_total <= 0:
                        shard_state.progress_total = 1
                        shard_state.progress_current = 1
                    shard_state.status = "failed"
                    reporter.shard_failed()
                    raise SystemExit(
                        f"[train_batch_selfplay] shard failed epoch={epoch} shard={shard_index} "
                        f"path={replay_path} exit_code={return_code}"
                    )
                final_train = parsed_stats.get("final_train")
                if not isinstance(final_train, dict) or not isinstance(final_train.get("labels"), int):
                    raise SystemExit(
                        f"[train_batch_selfplay] trainer did not report an exact label count "
                        f"epoch={epoch} shard={shard_index} path={replay_path}"
                    )
                epoch_label_count += int(final_train["labels"])
                completed_record = {
                    "outer_epoch": epoch,
                    "shard_index": shard_index,
                    "path": str(replay_path.resolve()),
                    "label_count": int(final_train["labels"]),
                    "stats_path": str(stats_path.resolve()),
                }
                raw_completed_shards = training_manifest["completed_shards"]
                assert isinstance(raw_completed_shards, list)
                raw_completed_shards.append(completed_record)
                completed_shards[completed_key] = completed_record

                if shard_state.progress_total <= 0:
                    shard_state.progress_total = 1
                    shard_state.progress_current = 1
                else:
                    shard_state.progress_current = shard_state.progress_total
                shard_state.status = "done"
                completed_files += 1
                state.completed_files = completed_files
                state.current_epoch_completed += 1
                total_elapsed = time.monotonic() - state.started_at
                shard_rate_sample = (completed_files / total_elapsed) if total_elapsed > 0.0 else 0.0
                if shard_rate_sample > 0.0:
                    shard_rate_ema = ema_update(shard_rate_ema, shard_rate_sample, eta_alpha)
                state.shards_per_minute = shard_rate_ema * 60.0 if shard_rate_ema > 0.0 else None
                remaining_shards = planned_total_shards - completed_files
                eta_ready = completed_files >= eta_min_completed or total_elapsed >= eta_min_elapsed
                state.eta_seconds = (remaining_shards / shard_rate_ema) if eta_ready and shard_rate_ema > 0.0 else None
                reporter.shard_finished()
                training_manifest["completed_files"] = completed_files
                training_manifest["current_epoch"] = epoch
                training_manifest["last_completed_shard"] = shard_index
                write_training_manifest(args.resolved_manifest_path, training_manifest)

            epoch_already_published = any(
                int(record.get("outer_epoch", 0)) == epoch for record in dataset_epoch_records
            )
            if args.mode == "supervised" and not epoch_already_published:
                snapshot_path = dataset_epoch_snapshot_path(args.resolved_checkpoint, epoch)
                validation_path = dataset_epoch_validation_path(args.resolved_checkpoint, epoch)
                if not snapshot_path.exists():
                    copy_file_atomically(args.resolved_checkpoint, snapshot_path)
                validation_summary = evaluate_fixed_holdout(
                    args,
                    all_files,
                    snapshot_path,
                    repo_root,
                    subprocess_env,
                    reporter,
                    state,
                )
                epoch_record = {
                    "outer_epoch": epoch,
                    "checkpoint": str(snapshot_path.resolve()),
                    "training_shard_count": len(epoch_files),
                    "source_shard_count": len(all_files),
                    "label_count": epoch_label_count,
                    "validation_seed": args.validation_seed,
                    "source_manifest": str(source_replay_manifest) if source_replay_manifest.exists() else "",
                    "training_shards": [str(path.resolve()) for path in epoch_files],
                    "validation_summary": str(validation_path.resolve()),
                }
                validation_summary.update(epoch_record)
                write_json_atomically(validation_path, validation_summary)
                dataset_epoch_records.append(epoch_record)
                training_manifest["dataset_epochs"] = dataset_epoch_records
                write_training_manifest(args.resolved_manifest_path, training_manifest)

            epoch_elapsed = time.monotonic() - epoch_started_at
            state.last_notice = f"epoch {epoch}/{args.epochs} complete elapsed={epoch_elapsed:.1f}s"
            state.current_file = None
            state.current_shard_index = None
            reporter.shard_finished()

        state.last_notice = "training complete"
        reporter.run_finished()
        training_manifest["status"] = "completed"
    finally:
        if training_manifest.get("status") != "completed":
            training_manifest["status"] = "failed"
            training_manifest["failure_reason"] = state.last_notice or failure_message
        training_manifest["completed_files"] = completed_files
        training_manifest["checkpoint_exists"] = args.resolved_checkpoint.exists()
        training_manifest["manifest_written_at_epoch_count"] = args.epochs
        write_training_manifest(args.resolved_manifest_path, training_manifest)
        reporter.close()


if __name__ == "__main__":
    main()
