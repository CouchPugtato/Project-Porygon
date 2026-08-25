from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from opponent_sampling import is_adaptive_pool, refresh_adaptive_pool
from rl_defaults import bool_default, float_default, int_default, reward_float_default

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


DEFAULT_ARGS_PATH = Path("config/live_rl_orchestrator.toml")
DEFAULT_TRAINER_EXE = Path("build-fresh") / "showdown_client.exe"
DEFAULT_RUNS_ROOT = Path("matches") / "runs"
DEFAULT_MODELS_ROOT = Path("models") / "runs"
SELFPLAY_PROGRESS_RE = re.compile(r"^\[selfplay\].*\bcompleted_games=(\d+)/(\d+)")
TRAIN_PROGRESS_RE = re.compile(r"^\[train-(?:ppo|rl)\].*\bepisodes=(\d+)/(\d+)")
TRAIN_ETA_RE = re.compile(r"^\[train\].*\beta=([^\s]+)")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
LIVE_PHASE_RE = re.compile(
    r"^\[live-rl\] dashboard phase=(collection|training) round=(\d+)/(\d+)(?: total=(\d+))?"
)
LIVE_ROUND_COMPLETE_RE = re.compile(r"^\[live-rl\] dashboard round_completed=(\d+)/(\d+)")
LIVE_COLLAPSE_RE = re.compile(r"^\[live-rl\] dashboard collapse_flags=(\[.*\])$")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def parse_key_values(line: str) -> dict[str, int | float | str]:
    values: dict[str, int | float | str] = {}
    for match in KEY_VALUE_RE.finditer(line):
        raw = match.group(2)
        try:
            values[match.group(1)] = float(raw) if any(token in raw for token in (".", "e", "E")) else int(raw)
        except ValueError:
            values[match.group(1)] = raw
    return values


@dataclass
class WorkflowDashboardState:
    run_name: str
    rounds_total: int
    games_per_round: int
    evaluation_games_per_side: int = 0
    title: str = "Live PPO"
    started_at: float = field(default_factory=time.monotonic)
    phase: str = "starting"
    round_index: int = 0
    rounds_completed: int = 0
    collection_current: int = 0
    collection_total: int = 0
    training_current: int = 0
    training_total: int = 0
    evaluation_side: str = ""
    evaluation_valid: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0})
    evaluation_invalid: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0})
    evaluation_attempt_current: int = 0
    evaluation_attempt_total: int = 0
    active_started_at: float | None = None
    active_eta_seconds: float | None = None
    metrics: dict[str, int | float | str] = field(default_factory=dict)
    collapse_flags: list[str] = field(default_factory=list)
    promotion_status: str = "pending"
    valid_win_rate: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    last_notice: str = ""
    dashboard_mode: str = "plain-text"

    def begin_collection(self, round_index: int, total: int) -> None:
        self.phase = "collection"
        self.round_index = round_index
        self.collection_current = 0
        self.collection_total = total
        self.training_current = 0
        self.training_total = 0
        self.active_started_at = time.monotonic()
        self.active_eta_seconds = None

    def begin_training(self, round_index: int, total: int) -> None:
        self.phase = "training"
        self.round_index = round_index
        self.training_current = 0
        self.training_total = total
        self.active_started_at = time.monotonic()
        self.active_eta_seconds = None
        for key in (
            "policy_loss", "value_loss", "entropy", "approx_kl", "anchor_kl_mean",
            "anchor_kl_max", "clip_fraction", "hard_kl_breaches", "labels",
        ):
            self.metrics.pop(key, None)

    def complete_round(self, round_index: int, metrics: dict[str, object] | None = None, collapse_flags: list[str] | None = None) -> None:
        self.rounds_completed = max(self.rounds_completed, round_index)
        self.round_index = round_index
        self.phase = "round complete"
        self.active_eta_seconds = 0.0
        if metrics:
            self.metrics.update({key: value for key, value in metrics.items() if isinstance(value, (int, float, str))})
        if collapse_flags is not None:
            self.collapse_flags = list(collapse_flags)

    def begin_evaluation(self, side: str, valid: int, invalid: int, attempt_total: int) -> None:
        self.phase = "evaluation"
        self.evaluation_side = side
        self.evaluation_valid[side] = valid
        self.evaluation_invalid[side] = invalid
        self.evaluation_attempt_current = 0
        self.evaluation_attempt_total = attempt_total
        self.active_started_at = time.monotonic()
        self.active_eta_seconds = None

    def update_evaluation(self, side: str, valid: int, invalid: int) -> None:
        self.evaluation_valid[side] = valid
        self.evaluation_invalid[side] = invalid

    def finish_evaluation(self, summary: dict[str, object]) -> None:
        self.phase = "promotion decision"
        self.valid_win_rate = float(summary.get("valid_win_rate", 0.0) or 0.0)
        self.confidence_low = float(summary.get("confidence_low", 0.0) or 0.0)
        self.confidence_high = float(summary.get("confidence_high", 1.0) or 1.0)
        self.active_eta_seconds = 0.0

    def set_promotion(self, assessment: dict[str, object] | None) -> None:
        if assessment is None:
            return
        self.promotion_status = str(assessment.get("status", "pending"))
        self.phase = "completed"

    def active_progress(self) -> tuple[int, int]:
        if self.phase == "collection":
            return self.collection_current, self.collection_total
        if self.phase == "training":
            return self.training_current, self.training_total
        if self.phase == "evaluation":
            return self.evaluation_attempt_current, self.evaluation_attempt_total
        return 0, 0

    def update_rate_eta(self, current: int, total: int) -> None:
        if self.active_started_at is None or current <= 0 or total <= current:
            return
        elapsed = time.monotonic() - self.active_started_at
        if elapsed > 0.0:
            self.active_eta_seconds = elapsed / current * (total - current)

    def progress_payload(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "dashboard_mode": self.dashboard_mode,
            "elapsed_seconds": time.monotonic() - self.started_at,
            "active_eta_seconds": self.active_eta_seconds,
            "round": self.round_index,
            "rounds_completed": self.rounds_completed,
            "rounds_total": self.rounds_total,
            "collection": {"current": self.collection_current, "total": self.collection_total},
            "training": {"current": self.training_current, "total": self.training_total},
            "evaluation": {
                "side": self.evaluation_side,
                "valid": dict(self.evaluation_valid),
                "invalid": dict(self.evaluation_invalid),
                "valid_target_per_side": self.evaluation_games_per_side,
                "attempt_current": self.evaluation_attempt_current,
                "attempt_total": self.evaluation_attempt_total,
            },
            "metrics": dict(self.metrics),
            "collapse_flags": list(self.collapse_flags),
            "promotion_status": self.promotion_status,
            "valid_win_rate": self.valid_win_rate,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
        }


class DashboardProgressWriter:
    def __init__(self, path: Path, manifest: dict[str, object], state: WorkflowDashboardState) -> None:
        self.path = path
        self.manifest = manifest
        self.state = state
        self.last_write = 0.0

    def update(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_write < 1.0:
            return
        self.manifest["progress"] = self.state.progress_payload()
        write_json(self.path, self.manifest)
        self.last_write = now


class BaseWorkflowReporter:
    def __init__(self, state: WorkflowDashboardState, progress_writer: DashboardProgressWriter | None = None) -> None:
        self.state = state
        self.progress_writer = progress_writer

    def start(self) -> None:
        return

    def close(self) -> None:
        if self.progress_writer:
            self.progress_writer.update(force=True)

    def refresh(self) -> None:
        if self.progress_writer:
            self.progress_writer.update()

    def notice(self, message: str) -> None:
        self.state.last_notice = message
        self.refresh()

    def command_started(self, command: list[str]) -> None:
        self.notice(f"running {Path(command[0]).name}")

    def child_line(self, line: str) -> None:
        phase_match = LIVE_PHASE_RE.search(line)
        if phase_match:
            phase, current_round, total_rounds, raw_total = phase_match.groups()
            self.state.rounds_total = int(total_rounds)
            total = int(raw_total or 0)
            if phase == "collection":
                self.state.begin_collection(int(current_round), total)
            else:
                self.state.begin_training(int(current_round), total)
        completed_match = LIVE_ROUND_COMPLETE_RE.search(line)
        if completed_match:
            self.state.complete_round(int(completed_match.group(1)))
        collapse_match = LIVE_COLLAPSE_RE.search(line.strip())
        if collapse_match:
            try:
                parsed_flags = json.loads(collapse_match.group(1))
                if isinstance(parsed_flags, list):
                    self.state.collapse_flags = [str(flag) for flag in parsed_flags]
            except json.JSONDecodeError:
                pass

        selfplay_match = SELFPLAY_PROGRESS_RE.search(line)
        if selfplay_match:
            current, total = int(selfplay_match.group(1)), int(selfplay_match.group(2))
            if self.state.phase == "evaluation":
                self.state.evaluation_attempt_current = current
                self.state.evaluation_attempt_total = total
            else:
                self.state.collection_current = current
                self.state.collection_total = total
            self.state.update_rate_eta(current, total)

        train_match = TRAIN_PROGRESS_RE.search(line)
        if train_match:
            self.state.training_current = int(train_match.group(1))
            self.state.training_total = int(train_match.group(2))
            values = parse_key_values(line)
            for key in (
                "policy_loss", "value_loss", "entropy", "approx_kl", "anchor_kl_mean",
                "anchor_kl_max", "clip_fraction", "hard_kl_breaches", "labels",
            ):
                if key in values:
                    self.state.metrics[key] = values[key]
            self.state.update_rate_eta(self.state.training_current, self.state.training_total)

        eta_match = TRAIN_ETA_RE.search(line)
        if eta_match and eta_match.group(1) != "estimating":
            try:
                self.state.active_eta_seconds = float(eta_match.group(1).removesuffix("s"))
            except ValueError:
                pass
        self.refresh()


class PlainTextWorkflowReporter(BaseWorkflowReporter):
    def __init__(
        self,
        state: WorkflowDashboardState,
        progress_writer: DashboardProgressWriter | None = None,
        fallback_notice: str = "",
    ) -> None:
        super().__init__(state, progress_writer)
        self.fallback_notice = fallback_notice

    def start(self) -> None:
        if self.fallback_notice:
            log(self.fallback_notice)

    def notice(self, message: str) -> None:
        super().notice(message)
        log(message)

    def command_started(self, command: list[str]) -> None:
        self.notice("exec: " + " ".join(f'\"{part}\"' if " " in part else part for part in command))

    def child_line(self, line: str) -> None:
        super().child_line(line)
        print(line, end="")


class RichWorkflowReporter(BaseWorkflowReporter):
    def __init__(
        self,
        state: WorkflowDashboardState,
        refresh_per_second: float,
        progress_writer: DashboardProgressWriter | None = None,
    ) -> None:
        super().__init__(state, progress_writer)
        self.console = Console()
        self.live = Live(self._render(), console=self.console, refresh_per_second=refresh_per_second, transient=False)
        self.started = False

    def start(self) -> None:
        self.live.start()
        self.started = True
        atexit.register(self.close)
        self.refresh()

    def close(self) -> None:
        super().close()
        if self.started:
            self.live.stop()
            self.started = False
            try:
                atexit.unregister(self.close)
            except Exception:
                pass

    def refresh(self) -> None:
        super().refresh()
        if self.started:
            self.live.update(self._render(), refresh=True)

    @staticmethod
    def _bar(current: int, total: int, width: int = 28, style: str = "cyan") -> str:
        if total <= 0:
            return "[grey50]waiting[/grey50]"
        bounded = min(max(0, current), total)
        filled = min(width, int(round(width * bounded / total)))
        return f"[{style}]" + "=" * filled + f"[/{style}][grey27]" + "-" * (width - filled) + "[/grey27]"

    def _summary(self) -> Panel:
        state = self.state
        table = Table.grid(expand=True)
        for _ in range(4):
            table.add_column()
        round_text = f"{state.round_index}/{state.rounds_total}" if state.rounds_total else "-"
        table.add_row(
            f"[bold]Run[/]: {state.run_name}", f"[bold]Phase[/]: {state.phase}",
            f"[bold]Round[/]: {round_text}", f"[bold]Elapsed[/]: {format_duration(time.monotonic() - state.started_at)}",
        )
        table.add_row(
            f"[bold]Rounds done[/]: {state.rounds_completed}/{state.rounds_total}",
            f"[bold]Active ETA[/]: {format_duration(state.active_eta_seconds)}",
            f"[bold]Promotion[/]: {state.promotion_status}", "",
        )
        return Panel(table, title=state.title, border_style="cyan")

    def _progress(self) -> Panel:
        state = self.state
        table = Table.grid(expand=True)
        table.add_column(width=18)
        table.add_column(ratio=1)
        table.add_column(width=30, justify="right")
        rows: list[tuple[str, int, int, str]] = [
            ("Rounds", state.rounds_completed, state.rounds_total, f"{state.rounds_completed}/{state.rounds_total}"),
            ("Collection", state.collection_current, state.collection_total, f"{state.collection_current}/{state.collection_total} games"),
            ("PPO training", state.training_current, state.training_total, f"{state.training_current}/{state.training_total} episodes"),
        ]
        if state.evaluation_games_per_side:
            for side in ("a", "b"):
                valid = state.evaluation_valid[side]
                invalid = state.evaluation_invalid[side]
                detail = f"{valid}/{state.evaluation_games_per_side} valid; {invalid} invalid"
                if state.phase == "evaluation" and state.evaluation_side == side and state.evaluation_attempt_total:
                    detail += f"; attempt {state.evaluation_attempt_current}/{state.evaluation_attempt_total} raw"
                rows.append((f"Evaluation {side.upper()}", valid, state.evaluation_games_per_side, detail))
        for label, current, total, detail in rows:
            style = "green" if total > 0 and current >= total else "cyan"
            table.add_row(f"[bold]{label}[/]", self._bar(current, total, style=style), detail)
        return Panel(table, border_style="grey42", padding=(0, 1))

    def _metrics(self) -> Panel:
        state = self.state
        table = Table.grid(expand=True)
        for _ in range(4):
            table.add_column()
        metrics = state.metrics
        def metric(key: str, digits: int = 4) -> str:
            value = metrics.get(key)
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)
        table.add_row(
            f"[bold]LR[/]: {metric('learning_rate')}", f"[bold]Entropy coef[/]: {metric('entropy_coef')}",
            f"[bold]Anchor coef[/]: {metric('anchor_kl_coef')}", f"[bold]Clip[/]: {metric('ppo_clip_epsilon')}",
        )
        table.add_row(
            f"[bold]Policy loss[/]: {metric('policy_loss')}", f"[bold]Value loss[/]: {metric('value_loss')}",
            f"[bold]Entropy[/]: {metric('entropy')}", f"[bold]Labels[/]: {metric('labels', 0)}",
        )
        table.add_row(
            f"[bold]Approx KL[/]: {metric('approx_kl')}", f"[bold]Anchor KL[/]: {metric('anchor_kl_mean')}",
            f"[bold]Clip frac[/]: {metric('clip_fraction', 3)}", f"[bold]KL guard[/]: {metric('hard_kl_breaches', 0)}",
        )
        if state.valid_win_rate is not None:
            table.add_row(
                f"[bold]Valid score[/]: {state.valid_win_rate:.2%}",
                f"[bold]Lower 95%[/]: {state.confidence_low:.2%}" if state.confidence_low is not None else "",
                f"[bold]Upper 95%[/]: {state.confidence_high:.2%}" if state.confidence_high is not None else "",
                "",
            )
        if state.collapse_flags:
            table.add_row("[red]Safety[/]: " + ", ".join(state.collapse_flags[:2]), "", "", "")
        return Panel(table, title="Current metrics", border_style="grey42")

    def _render(self) -> Group:
        items: list[object] = [self._summary(), self._progress(), self._metrics()]
        if self.state.last_notice:
            items.append(Panel(escape(self.state.last_notice), border_style="grey42"))
        return Group(*items)


def select_workflow_reporter(
    dashboard: bool,
    refresh_per_second: float,
    state: WorkflowDashboardState,
    progress_writer: DashboardProgressWriter | None = None,
) -> BaseWorkflowReporter:
    if not dashboard:
        return PlainTextWorkflowReporter(state, progress_writer)
    if not RICH_AVAILABLE:
        return PlainTextWorkflowReporter(state, progress_writer, "dashboard unavailable; using plain-text output")
    if not sys.stdout.isatty():
        return PlainTextWorkflowReporter(state, progress_writer, "live terminal unavailable; using plain-text output")
    state.dashboard_mode = "terminal"
    return RichWorkflowReporter(state, refresh_per_second, progress_writer)


def run_reported_command(
    command: list[str],
    cwd: Path,
    reporter: BaseWorkflowReporter,
    raw_log_path: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    reporter.command_started(command)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if raw_log_path:
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log: TextIO | None = raw_log_path.open("w", encoding="utf-8") if raw_log_path else None
    try:
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
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                if raw_log:
                    raw_log.write(line)
                    raw_log.flush()
                reporter.child_line(line)
        return_code = process.wait()
    finally:
        if raw_log:
            raw_log.close()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}")


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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true"}:
        return True
    if lowered in {"0", "false"}:
        return False
    raise argparse.ArgumentTypeError("value must be true or false")


def load_default_args(path: Path) -> list[str]:
    if not path.exists():
        return []
    args: list[str] = []
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
        flag = "--" + key.replace("_", "-")
        lowered = value_text.lower()
        if lowered in {"true", "false"}:
            args.extend([flag, "1" if lowered == "true" else "0"])
            continue
        if len(value_text) >= 2 and value_text[0] == '"' and value_text[-1] == '"':
            value = bytes(value_text[1:-1], "utf-8").decode("unicode_escape")
        else:
            value = value_text
        args.extend([flag, value])
    return args


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def run_dir_for_name(repo_root: Path, run_name: str) -> Path:
    return (repo_root / DEFAULT_RUNS_ROOT / run_name).resolve()


def worker_glob_for_side(side: str) -> str:
    return f"worker_*_{side}_raw.jsonl"


def log(message: str) -> None:
    print(f"[live-rl] {message}", flush=True)


def extract_episode_batch(run_dir: Path, side: str, output_path: Path) -> dict[str, int]:
    worker_paths = sorted(run_dir.glob(worker_glob_for_side(side)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    scanned = 0
    source_files = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        for worker_path in worker_paths:
            source_files += 1
            with worker_path.open("r", encoding="utf-8") as in_handle:
                for raw_line in in_handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    scanned += 1
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict) or payload.get("type") != "episode_complete":
                        continue
                    out_handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                    written += 1
    return {
        "source_files": source_files,
        "scanned_lines": scanned,
        "written_episodes": written,
    }


def copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_selfplay_command(
    args: argparse.Namespace,
    repo_root: Path,
    run_name: str,
    checkpoint_path: Path,
    model_b_pool_path: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str((repo_root / "py" / "tools" / "selfplay_server.py").resolve()),
        "--run-name",
        run_name,
        "--games",
        str(args.games),
        "--concurrent-games",
        str(args.concurrent_games),
        "--worker-pairs",
        str(args.worker_pairs),
        "--worker-games",
        str(args.worker_games),
        "--ensure-shard-count",
        "true" if args.ensure_shard_count else "false",
        "--model-a-pool",
        "",
        "--model-a",
        str(checkpoint_path),
        "--pool-seed",
        str(args.pool_seed),
        "--format",
        args.format,
        "--worker-think-mode",
        args.worker_think_mode,
        "--serve-client",
        "1" if args.serve_client else "0",
        "--worker-log-stdout",
        "1" if args.worker_log_stdout else "0",
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
        "--resource-check-seconds",
        str(args.resource_check_seconds),
        "--min-available-memory-gb",
        str(args.min_available_memory_gb),
        "--min-available-pagefile-gb",
        str(args.min_available_pagefile_gb),
    ]
    resolved_pool = str(model_b_pool_path) if model_b_pool_path is not None else args.model_b_pool
    if resolved_pool:
        command.extend(["--model-b", "", "--model-b-pool", resolved_pool])
    else:
        command.extend(["--model-b-pool", "", "--model-b", args.model_b])
    return command


def build_train_command(args: argparse.Namespace, trainer_exe: Path, episode_batch_path: Path, output_checkpoint: Path) -> list[str]:
    return [
        str(trainer_exe),
        "--train-live-ppo" if args.training_mode == "ppo" else "--train-live-rl",
        str(episode_batch_path),
        str(output_checkpoint),
        "--policy-tag-expected",
        str(args.current_policy_tag),
        "--parent-checkpoint",
        str(args.current_policy_tag),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--gamma",
        str(args.gamma),
        "--entropy-coef",
        str(args.entropy_coef),
        "--advantage-norm",
        "1" if args.advantage_norm else "0",
        "--gae-lambda",
        str(args.gae_lambda),
        "--ppo-clip-epsilon",
        str(args.ppo_clip_epsilon),
        "--ppo-value-clip-epsilon",
        str(args.ppo_value_clip_epsilon),
        "--target-kl",
        str(args.target_kl),
        "--target-kl-min-episodes",
        str(args.target_kl_min_episodes),
        "--target-kl-min-labels",
        str(args.target_kl_min_labels),
        "--target-kl-hard-multiplier",
        str(args.target_kl_hard_multiplier),
        "--target-kl-hard-consecutive-updates",
        str(args.target_kl_hard_consecutive_updates),
        "--shuffle-seed",
        str(args.shuffle_seed),
        "--ppo-minibatch-episodes",
        str(args.ppo_minibatch_episodes),
        "--adam-beta1",
        str(args.adam_beta1),
        "--adam-beta2",
        str(args.adam_beta2),
        "--adam-epsilon",
        str(args.adam_epsilon),
        "--reward-mode",
        args.reward_mode,
        "--training-summary-path",
        str(args.training_summary_path),
        "--anchor-checkpoint",
        args.anchor_checkpoint,
        "--anchor-kl-coef",
        str(args.anchor_kl_coef),
        "--dense-additive-hp-swing-weight",
        str(args.dense_additive_hp_swing_weight),
        "--dense-additive-faint-swing-weight",
        str(args.dense_additive_faint_swing_weight),
        "--dense-additive-reward-clip",
        str(args.dense_additive_reward_clip),
    ]


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def round_manifest_completed(round_manifest_path: Path) -> bool:
    if not round_manifest_path.exists():
        return False
    try:
        payload = load_json(round_manifest_path)
    except Exception:
        return False
    if payload.get("status") != "completed":
        return False
    output_checkpoint = Path(str(payload.get("output_checkpoint", "")))
    training_summary = Path(str(payload.get("training_round_stats_path", "")))
    if not output_checkpoint.exists() or not training_summary.exists():
        return False
    for key in ("opponent_pool_path", "next_opponent_pool_path"):
        artifact = str(payload.get(key, "")).strip()
        if artifact and not Path(artifact).exists():
            return False
    return True


def trainer_env(args: argparse.Namespace) -> dict[str, str] | None:
    if args.omp_threads <= 0:
        return None
    return {"PORYGON_OMP_THREADS": str(args.omp_threads)}


def collapse_flags_from_training_summary(summary: dict[str, object], baseline_tera_action_rate: float, min_episodes_warn: int, anchor_kl_warn_threshold: float) -> list[str]:
    flags: list[str] = []
    episode_count = int(summary.get("episode_count", 0) or 0)
    tera_rate = float(summary.get("tera_action_rate", summary.get("tera_rate", 0.0)) or 0.0)
    move_slot_rates = summary.get("move_slot_rates", {}) or {}
    anchor_kl_mean = float(summary.get("anchor_kl_mean", 0.0) or 0.0)
    if episode_count < min_episodes_warn:
        flags.append(f"warn_low_episode_count:{episode_count}")
    if baseline_tera_action_rate > 0.0 and tera_rate < (float_default("warn_tera_baseline_ratio") * baseline_tera_action_rate):
        flags.append(f"warn_tera_action_rate_low:{tera_rate:.3f}:{baseline_tera_action_rate:.3f}")
    for key, value in move_slot_rates.items():
        rate = float(value or 0.0)
        if rate > 0.70:
            flags.append(f"hard_move_slot_collapse:{key}:{rate:.3f}")
        elif rate > 0.55:
            flags.append(f"warn_move_slot_concentration:{key}:{rate:.3f}")
    if anchor_kl_warn_threshold > 0.0 and anchor_kl_mean > anchor_kl_warn_threshold:
        flags.append(f"warn_anchor_kl_high:{anchor_kl_mean:.3f}")
    return flags


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--rounds", type=positive_int, default=1)
    parser.add_argument("--games", type=nonnegative_int, required=True)
    parser.add_argument("--concurrent-games", type=positive_int, required=True)
    parser.add_argument("--worker-pairs", type=positive_int, default=200)
    parser.add_argument("--worker-games", type=nonnegative_int, default=0)
    parser.add_argument("--ensure-shard-count", type=parse_bool, default=True)
    parser.add_argument("--model-b", default="random")
    parser.add_argument("--model-b-pool", default="")
    parser.add_argument("--pool-seed", type=int, default=1)
    parser.add_argument("--format", default="gen9randomdoublesbattle")
    parser.add_argument("--worker-think-mode", choices=["live", "random"], default="live")
    parser.add_argument("--serve-client", type=parse_bool, default=True)
    parser.add_argument("--worker-log-stdout", type=parse_bool, default=False)
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.25)
    parser.add_argument("--resource-check-seconds", type=float, default=2.0)
    parser.add_argument("--min-available-memory-gb", type=float, default=2.0)
    parser.add_argument("--min-available-pagefile-gb", type=float, default=4.0)
    parser.add_argument("--trainer-exe", default=str(DEFAULT_TRAINER_EXE))
    parser.add_argument("--training-mode", choices=["rl", "ppo"], default="ppo")
    parser.add_argument("--checkpoint-prefix", default="live_rl")
    parser.add_argument("--episode-side", choices=["a", "b"], default="a")
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--learning-rate", type=float, default=float_default("live_ppo_learning_rate"))
    parser.add_argument("--gamma", type=float, default=float_default("ppo_gamma"))
    parser.add_argument("--entropy-coef", type=float, default=float_default("ppo_entropy_coef"))
    parser.add_argument("--advantage-norm", type=parse_bool, default=bool_default("advantage_norm"))
    parser.add_argument("--gae-lambda", type=float, default=float_default("gae_lambda"))
    parser.add_argument("--ppo-clip-epsilon", type=float, default=float_default("ppo_clip_epsilon"))
    parser.add_argument("--ppo-value-clip-epsilon", type=float, default=float_default("ppo_value_clip_epsilon"))
    parser.add_argument("--target-kl", type=float, default=float_default("ppo_target_kl"))
    parser.add_argument("--target-kl-min-episodes", type=int, default=int_default("ppo_target_kl_min_episodes"))
    parser.add_argument("--target-kl-min-labels", type=int, default=int_default("ppo_target_kl_min_labels"))
    parser.add_argument("--target-kl-hard-multiplier", type=float, default=float_default("ppo_target_kl_hard_multiplier"))
    parser.add_argument("--target-kl-hard-consecutive-updates", type=int, default=int_default("ppo_target_kl_hard_consecutive_updates"))
    parser.add_argument("--shuffle-seed", type=int, default=int_default("ppo_shuffle_seed"))
    parser.add_argument("--ppo-minibatch-episodes", type=int, default=int_default("ppo_minibatch_episodes"))
    parser.add_argument("--adam-beta1", type=float, default=float_default("adam_beta1"))
    parser.add_argument("--adam-beta2", type=float, default=float_default("adam_beta2"))
    parser.add_argument("--adam-epsilon", type=float, default=float_default("adam_epsilon"))
    parser.add_argument("--reward-mode", choices=["terminal", "dense_additive"], default="terminal")
    parser.add_argument("--dense-additive-hp-swing-weight", type=float, default=reward_float_default("dense_additive_hp_swing_weight"))
    parser.add_argument("--dense-additive-faint-swing-weight", type=float, default=reward_float_default("dense_additive_faint_swing_weight"))
    parser.add_argument("--dense-additive-reward-clip", type=float, default=reward_float_default("dense_additive_reward_clip"))
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--anchor-kl-coef", type=float, default=float_default("anchor_kl_coef"))
    parser.add_argument(
        "--baseline-tera-action-rate",
        "--baseline-tera-rate",
        dest="baseline_tera_action_rate",
        type=float,
        default=float_default("baseline_tera_action_rate"),
    )
    parser.add_argument("--anchor-kl-warn-threshold", type=float, default=float_default("anchor_kl_warn_threshold"))
    parser.add_argument("--min-episodes-warn", type=int, default=int_default("min_training_episodes_warn"))
    parser.add_argument("--stop-on-collapse", type=parse_bool, default=True)
    parser.add_argument("--omp-threads", type=int, default=0)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--dashboard", type=parse_bool, default=True)
    parser.add_argument("--dashboard-refresh-per-second", type=positive_float, default=8.0)
    parser.add_argument("--dashboard-write-raw-logs", type=parse_bool, default=True)
    return parser


def main() -> None:
    parser = build_parser()
    argv = load_default_args(DEFAULT_ARGS_PATH) + sys.argv[1:]
    args = parser.parse_args(argv)
    repo_root = resolve_repo_root()
    trainer_exe = resolve_path(repo_root, args.trainer_exe)
    if not trainer_exe.exists():
        raise SystemExit(f"trainer exe not found: {trainer_exe}")
    if not args.model_b and not args.model_b_pool:
        raise SystemExit("provide either --model-b or --model-b-pool")
    if args.model_b and args.model_b_pool:
        raise SystemExit("--model-b and --model-b-pool are mutually exclusive")

    initial_pool_path: Path | None = None
    current_pool_payload: dict[str, object] | None = None
    if args.model_b_pool:
        initial_pool_path = resolve_path(repo_root, args.model_b_pool)
        if not initial_pool_path.exists():
            raise SystemExit(f"model B pool not found: {initial_pool_path}")
        loaded_pool = load_json(initial_pool_path)
        if not isinstance(loaded_pool, dict):
            raise SystemExit(f"invalid model B pool: expected object in {initial_pool_path}")
        current_pool_payload = loaded_pool

    init_checkpoint = resolve_path(repo_root, args.init_checkpoint)
    if not init_checkpoint.exists():
        raise SystemExit(f"init checkpoint not found: {init_checkpoint}")

    workflow_dir = (repo_root / DEFAULT_MODELS_ROOT / args.run_name).resolve()
    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_manifest_path = workflow_dir / f"{args.run_name}_live_rl_manifest.json"

    if args.resume and workflow_manifest_path.exists():
        workflow_manifest = load_json(workflow_manifest_path)
        workflow_manifest["status"] = "running"
        workflow_manifest.pop("failure_reason", None)
        workflow_manifest["resumed_at_unix"] = time.time()
        workflow_manifest.setdefault("round_manifests", [])
        workflow_manifest.setdefault("opponent_pool_history", [])
        workflow_manifest["opponent"] = {
            "model_b": args.model_b,
            "model_b_pool": str(initial_pool_path) if initial_pool_path else "",
            "adaptive": bool(current_pool_payload and is_adaptive_pool(current_pool_payload)),
        }
        log(f"resuming workflow from {workflow_manifest_path}")
    else:
        workflow_manifest = {
            "run_name": args.run_name,
            "status": "running",
            "started_at_unix": time.time(),
            "init_checkpoint": str(init_checkpoint),
            "trainer_exe": str(trainer_exe),
            "rounds": args.rounds,
            "games": args.games,
            "concurrent_games": args.concurrent_games,
            "worker_pairs": args.worker_pairs,
            "worker_games": args.worker_games,
            "launch_stagger_seconds": args.launch_stagger_seconds,
            "resource_check_seconds": args.resource_check_seconds,
            "min_available_memory_gb": args.min_available_memory_gb,
            "min_available_pagefile_gb": args.min_available_pagefile_gb,
            "ensure_shard_count": args.ensure_shard_count,
            "reward_mode": args.reward_mode,
            "dense_additive_hp_swing_weight": args.dense_additive_hp_swing_weight,
            "dense_additive_faint_swing_weight": args.dense_additive_faint_swing_weight,
            "dense_additive_reward_clip": args.dense_additive_reward_clip,
            "anchor_checkpoint": args.anchor_checkpoint,
            "anchor_kl_coef": args.anchor_kl_coef,
            "gamma": args.gamma,
            "entropy_coef": args.entropy_coef,
            "advantage_norm": args.advantage_norm,
            "baseline_tera_action_rate": args.baseline_tera_action_rate,
            "gae_lambda": args.gae_lambda,
            "ppo_clip_epsilon": args.ppo_clip_epsilon,
            "ppo_value_clip_epsilon": args.ppo_value_clip_epsilon,
            "target_kl": args.target_kl,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
            "episode_side": args.episode_side,
            "omp_threads": args.omp_threads,
            "dashboard": args.dashboard,
            "dashboard_write_raw_logs": args.dashboard_write_raw_logs,
            "opponent": {
                "model_b": args.model_b,
                "model_b_pool": str(initial_pool_path) if initial_pool_path else "",
                "adaptive": bool(current_pool_payload and is_adaptive_pool(current_pool_payload)),
            },
            "round_manifests": [],
            "opponent_pool_history": [],
        }
    write_json(workflow_manifest_path, workflow_manifest)

    completed_round_paths = [Path(path) for path in workflow_manifest.get("round_manifests", []) if round_manifest_completed(Path(path))]
    workflow_manifest["round_manifests"] = [str(path) for path in completed_round_paths]
    current_checkpoint = Path(str(workflow_manifest.get("latest_checkpoint", init_checkpoint))).resolve()
    if not completed_round_paths or not current_checkpoint.exists():
        current_checkpoint = init_checkpoint
    dashboard_state = WorkflowDashboardState(
        run_name=args.run_name,
        rounds_total=args.rounds,
        games_per_round=args.games,
        title="Live PPO" if args.training_mode == "ppo" else "Live RL",
    )
    dashboard_state.rounds_completed = len(completed_round_paths)
    dashboard_state.metrics.update({
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "anchor_kl_coef": args.anchor_kl_coef,
        "ppo_clip_epsilon": args.ppo_clip_epsilon,
    })
    progress_writer = DashboardProgressWriter(workflow_manifest_path, workflow_manifest, dashboard_state)
    reporter = select_workflow_reporter(
        args.dashboard, args.dashboard_refresh_per_second, dashboard_state, progress_writer,
    )
    logs_dir = workflow_dir / "logs"
    reporter.start()
    try:
        for round_index in range(1, args.rounds + 1):
            round_token = f"round{round_index:02d}"
            collect_run_name = f"{args.run_name}_{round_token}_collect"
            round_dir = workflow_dir / round_token
            round_dir.mkdir(parents=True, exist_ok=True)
            output_checkpoint = round_dir / f"{args.checkpoint_prefix}_{round_token}.chk"
            training_summary_path = round_dir / f"{args.run_name}_{round_token}_training_summary.json"
            episode_batch_path = run_dir_for_name(repo_root, collect_run_name) / f"episode_batch_{args.episode_side}.jsonl"
            round_manifest_path = round_dir / f"{args.run_name}_{round_token}_manifest.json"

            if args.resume and round_manifest_completed(round_manifest_path):
                round_manifest = load_json(round_manifest_path)
                current_checkpoint = Path(str(round_manifest.get("output_checkpoint", current_checkpoint))).resolve()
                next_pool_path_text = str(round_manifest.get("next_opponent_pool_path", "")).strip()
                if next_pool_path_text:
                    next_pool_path = Path(next_pool_path_text).resolve()
                    current_pool_payload = load_json(next_pool_path)
                    workflow_manifest["latest_opponent_pool_path"] = str(next_pool_path)
                if str(round_manifest_path) not in workflow_manifest["round_manifests"]:
                    workflow_manifest["round_manifests"].append(str(round_manifest_path))
                workflow_manifest["latest_checkpoint"] = str(current_checkpoint)
                write_json(workflow_manifest_path, workflow_manifest)
                reporter.child_line(f"[live-rl] dashboard round_completed={round_index}/{args.rounds}\n")
                reporter.notice(f"skipping completed {round_token}")
                continue

            round_pool_path: Path | None = None
            if current_pool_payload is not None:
                round_pool_path = round_dir / f"{args.run_name}_{round_token}_opponent_pool_used.json"
                write_json(round_pool_path, current_pool_payload)

            round_manifest: dict[str, object] = {
                "round": round_index,
                "status": "running",
                "started_at_unix": time.time(),
                "input_checkpoint": str(current_checkpoint),
                "output_checkpoint": str(output_checkpoint),
                "collection_run": collect_run_name,
                "episode_batch_path": str(episode_batch_path),
                "training_round_stats_path": str(training_summary_path),
                "trainer_exe": str(trainer_exe),
                "opponent_pool_path": str(round_pool_path) if round_pool_path else "",
                "opponent_pool_adaptive": bool(current_pool_payload and is_adaptive_pool(current_pool_payload)),
            }
            write_json(round_manifest_path, round_manifest)

            selfplay_command = build_selfplay_command(
                args,
                repo_root,
                collect_run_name,
                current_checkpoint,
                round_pool_path,
            )
            round_manifest["selfplay_command"] = selfplay_command
            write_json(round_manifest_path, round_manifest)
            dashboard_state.begin_collection(round_index, args.games)
            reporter.notice(f"collecting round {round_index}/{args.rounds}: {args.games} games")
            reporter.child_line(
                f"[live-rl] dashboard phase=collection round={round_index}/{args.rounds} total={args.games}\n"
            )
            run_reported_command(
                selfplay_command,
                repo_root,
                reporter,
                logs_dir / f"{collect_run_name}.log" if args.dashboard_write_raw_logs else None,
            )

            collection_summary_path = (
                run_dir_for_name(repo_root, collect_run_name)
                / f"{collect_run_name}_summary.json"
            )
            round_manifest["collection_summary_path"] = str(collection_summary_path)
            if current_pool_payload is not None and is_adaptive_pool(current_pool_payload):
                collection_summary = load_json(collection_summary_path)
                group_member_stats = collection_summary.get("group_member_stats", {}) or {}
                opponent_member_stats = (
                    group_member_stats.get("b", {})
                    if isinstance(group_member_stats, dict)
                    else {}
                )
                if not isinstance(opponent_member_stats, dict):
                    opponent_member_stats = {}
                refreshed_pool, refresh_results = refresh_adaptive_pool(
                    current_pool_payload,
                    opponent_member_stats,
                    collect_run_name,
                )
                next_pool_path = round_dir / f"{args.run_name}_{round_token}_opponent_pool_next.json"
                write_json(next_pool_path, refreshed_pool)
                current_pool_payload = refreshed_pool
                round_manifest["next_opponent_pool_path"] = str(next_pool_path)
                round_manifest["opponent_pool_refresh_results"] = refresh_results
                workflow_manifest["latest_opponent_pool_path"] = str(next_pool_path)
                pool_history = workflow_manifest.setdefault("opponent_pool_history", [])
                if isinstance(pool_history, list):
                    pool_history[:] = [
                        entry
                        for entry in pool_history
                        if not isinstance(entry, dict) or int(entry.get("round", -1)) != round_index
                    ]
                    pool_history.append({
                        "round": round_index,
                        "used": str(round_pool_path),
                        "next": str(next_pool_path),
                    })
                reporter.notice(f"refreshed adaptive opponent pool for {round_token}: {next_pool_path}")
            write_json(round_manifest_path, round_manifest)

            extract_stats = extract_episode_batch(run_dir_for_name(repo_root, collect_run_name), args.episode_side, episode_batch_path)
            round_manifest["episode_extract"] = extract_stats
            if not extract_stats["written_episodes"]:
                raise SystemExit(f"no episode_complete records found for {collect_run_name} side {args.episode_side}")

            copy_checkpoint(current_checkpoint, output_checkpoint)
            round_manifest["checkpoint_initialized_from"] = str(current_checkpoint)
            round_manifest["checkpoint_initialized_to"] = str(output_checkpoint)

            args.current_policy_tag = str(current_checkpoint)
            args.training_summary_path = training_summary_path
            train_command = build_train_command(args, trainer_exe, episode_batch_path, output_checkpoint)
            round_manifest["train_command"] = train_command
            write_json(round_manifest_path, round_manifest)
            dashboard_state.begin_training(round_index, int(extract_stats["written_episodes"]))
            reporter.notice(
                f"training round {round_index}/{args.rounds}: {int(extract_stats['written_episodes'])} episodes"
            )
            reporter.child_line(
                f"[live-rl] dashboard phase=training round={round_index}/{args.rounds} "
                f"total={int(extract_stats['written_episodes'])}\n"
            )
            run_reported_command(
                train_command,
                repo_root,
                reporter,
                logs_dir / f"{args.run_name}_{round_token}_training.log" if args.dashboard_write_raw_logs else None,
                extra_env=trainer_env(args),
            )

            training_summary = load_json(training_summary_path)
            collapse_flags = collapse_flags_from_training_summary(
                training_summary,
                args.baseline_tera_action_rate,
                args.min_episodes_warn,
                args.anchor_kl_warn_threshold,
            )
            round_manifest["training_round_summary"] = str(training_summary_path)
            round_manifest["training_round_collapse_flags"] = collapse_flags
            round_manifest["parent_checkpoint"] = str(current_checkpoint)
            round_manifest["anchor_checkpoint"] = args.anchor_checkpoint
            round_manifest["anchor_kl_coef"] = args.anchor_kl_coef
            round_manifest["reward_mode"] = args.reward_mode
            round_manifest["dense_additive_hp_swing_weight"] = args.dense_additive_hp_swing_weight
            round_manifest["dense_additive_faint_swing_weight"] = args.dense_additive_faint_swing_weight
            round_manifest["dense_additive_reward_clip"] = args.dense_additive_reward_clip

            current_checkpoint = output_checkpoint
            round_manifest["status"] = "completed"
            round_manifest["completed_at_unix"] = time.time()
            write_json(round_manifest_path, round_manifest)
            workflow_manifest["round_manifests"].append(str(round_manifest_path))
            workflow_manifest["latest_checkpoint"] = str(current_checkpoint)
            write_json(workflow_manifest_path, workflow_manifest)
            dashboard_state.complete_round(round_index, training_summary, collapse_flags)
            reporter.child_line(
                "[live-rl] dashboard collapse_flags="
                + json.dumps(collapse_flags, separators=(",", ":"))
                + "\n"
            )
            reporter.child_line(f"[live-rl] dashboard round_completed={round_index}/{args.rounds}\n")
            reporter.notice(f"completed round {round_index}/{args.rounds}")
            if args.stop_on_collapse and any(flag.startswith("hard_move_slot_collapse") for flag in collapse_flags):
                workflow_manifest["status"] = "stopped_on_collapse"
                workflow_manifest["completed_at_unix"] = time.time()
                workflow_manifest["latest_checkpoint"] = str(current_checkpoint)
                workflow_manifest["stop_reason"] = collapse_flags
                write_json(workflow_manifest_path, workflow_manifest)
                dashboard_state.phase = "stopped on collapse"
                reporter.notice("stopped on hard policy collapse")
                return

        workflow_manifest["status"] = "completed"
        workflow_manifest["completed_at_unix"] = time.time()
        workflow_manifest["latest_checkpoint"] = str(current_checkpoint)
        write_json(workflow_manifest_path, workflow_manifest)
        dashboard_state.phase = "completed"
        reporter.notice("live training completed")
    except Exception as exc:
        workflow_manifest["status"] = "failed"
        workflow_manifest["failure_reason"] = str(exc)
        workflow_manifest["completed_at_unix"] = time.time()
        write_json(workflow_manifest_path, workflow_manifest)
        dashboard_state.phase = "failed"
        reporter.notice(str(exc))
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
