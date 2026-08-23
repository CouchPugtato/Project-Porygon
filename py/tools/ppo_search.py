from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

from rl_defaults import bool_default, float_default, int_default
from eval_collapse_check import collapse_flags_for_group

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


DEFAULT_RUNS_ROOT = Path("models") / "runs"
DEFAULT_MATCH_RUNS_ROOT = Path("matches") / "runs"
DEFAULT_SEARCH_ROOT = Path("models") / "search"
DEFAULT_TRAINER_EXE = Path("build-fresh") / "showdown_client.exe"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ppo_search.toml"
TRAIN_PROGRESS_RE = re.compile(r"^\[train-ppo\].*\bepisodes=(\d+)/(\d+)")
TRAIN_ETA_RE = re.compile(r"^\[train\].*\bepisodes_per_sec=([^\s]+)\s+eta=([^\s]+)")
SELFPLAY_PROGRESS_RE = re.compile(r"^\[selfplay\].*\bcompleted_games=(\d+)/(\d+)")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
FIXED_CONFIG_KEYS = {"trainer_exe"}


def load_config_args(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"PPO search config not found: {path}")
    args: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise RuntimeError(f"invalid PPO search config {path}:{line_number}: expected key = value")
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip()
        value_text = raw_value.strip()
        if not key or not value_text:
            raise RuntimeError(f"invalid PPO search config {path}:{line_number}: empty key or value")
        if key in FIXED_CONFIG_KEYS:
            raise RuntimeError(
                f"invalid PPO search config {path}:{line_number}: {key} is fixed internally and cannot be configured"
            )
        if len(value_text) >= 2 and value_text[0] == '"' and value_text[-1] == '"':
            value = bytes(value_text[1:-1], "utf-8").decode("unicode_escape")
        else:
            value = value_text
        args.extend(["--" + key.replace("_", "-"), value])
    return args


def config_path_from_args(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    known, _ = parser.parse_known_args(argv)
    path = Path(known.config)
    if not path.is_absolute():
        path = (resolve_repo_root() / path).resolve()
    return path


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true"}:
        return True
    if lowered in {"0", "false"}:
        return False
    raise argparse.ArgumentTypeError("value must be true or false")


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


def csv_floats(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one float")
    return values


def csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return values


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def log(message: str) -> None:
    print(f"[ppo-search] {message}", flush=True)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 1.0)
    p = wins / games
    denominator = 1.0 + (z * z / games)
    center = (p + z * z / (2.0 * games)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) / games) + (z * z / (4.0 * games * games))) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    entropy_coef: float
    anchor_kl_coef: float
    ppo_clip_epsilon: float
    shuffle_seed: int


@dataclass
class EvaluationResult:
    stage: str
    games: int
    wins: int
    earned_wins: int
    win_rate: float
    earned_win_rate: float
    confidence_low: float
    confidence_high: float
    collapse_flags: list[str]
    run_names: list[str]
    candidate_stats: dict[str, object] = field(default_factory=dict)
    baseline_stats: dict[str, object] = field(default_factory=dict)


GROUP_COUNT_KEYS = (
    "matches_played", "wins", "earned_wins", "losses", "draws",
    "total_moves", "total_protects", "total_passes", "total_teras", "tera_battles",
    "total_move_slot_1", "total_move_slot_2", "total_move_slot_3", "total_move_slot_4",
    "total_switch_slot_1", "total_switch_slot_2", "total_switch_slot_3",
    "total_switch_slot_4", "total_switch_slot_5", "total_switch_slot_6",
)


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def aggregate_group_stats(
    existing: dict[str, object],
    addition: dict[str, object],
) -> dict[str, object]:
    counts = {
        key: int(existing.get(key, 0) or 0) + int(addition.get(key, 0) or 0)
        for key in GROUP_COUNT_KEYS
    }
    move_total = sum(counts[f"total_move_slot_{slot}"] for slot in range(1, 5))
    switch_total = sum(counts[f"total_switch_slot_{slot}"] for slot in range(1, 7))
    matches = counts["matches_played"]
    counts.update({
        "win_rate": safe_rate(counts["wins"], matches),
        "earned_win_rate": safe_rate(counts["earned_wins"], matches),
        "tera_battle_rate": safe_rate(counts["tera_battles"], matches),
        "tera_rate": safe_rate(counts["tera_battles"], matches),
        "move_slot_rates": {
            f"slot_{slot}": safe_rate(counts[f"total_move_slot_{slot}"], move_total)
            for slot in range(1, 5)
        },
        "switch_slot_rates": {
            f"slot_{slot}": safe_rate(counts[f"total_switch_slot_{slot}"], switch_total)
            for slot in range(1, 7)
        },
    })
    return counts


def aggregate_collapse_flags(
    candidate_stats: dict[str, object],
    baseline_stats: dict[str, object],
) -> list[str]:
    return collapse_flags_for_group(
        candidate_stats,
        baseline_stats,
        warn_move_slot_concentration=float_default("warn_move_slot_concentration"),
        hard_move_slot_collapse=float_default("hard_move_slot_collapse"),
        warn_switch_slot6_concentration=float_default("warn_switch_slot_6_concentration"),
        warn_tera_baseline_ratio=float_default("warn_tera_baseline_ratio"),
        fail_fast_earned_win_rate=float_default("fail_fast_earned_win_rate"),
        fail_fast_min_games=int_default("fail_fast_min_games"),
    )


def merge_evaluation_results(
    existing: EvaluationResult | None,
    addition: EvaluationResult,
) -> EvaluationResult:
    if existing is None:
        return addition
    if existing.stage != addition.stage:
        raise ValueError(f"cannot merge {existing.stage} and {addition.stage} evaluations")
    games = existing.games + addition.games
    wins = existing.wins + addition.wins
    earned_wins = existing.earned_wins + addition.earned_wins
    low, high = wilson_interval(wins, games)
    candidate_stats = aggregate_group_stats(existing.candidate_stats, addition.candidate_stats)
    baseline_stats = aggregate_group_stats(existing.baseline_stats, addition.baseline_stats)
    collapse_flags = (
        aggregate_collapse_flags(candidate_stats, baseline_stats)
        if candidate_stats.get("matches_played")
        else sorted(set(existing.collapse_flags + addition.collapse_flags))
    )
    return EvaluationResult(
        stage=existing.stage,
        games=games,
        wins=wins,
        earned_wins=earned_wins,
        win_rate=(wins / games) if games else 0.0,
        earned_win_rate=(earned_wins / games) if games else 0.0,
        confidence_low=low,
        confidence_high=high,
        collapse_flags=collapse_flags,
        run_names=existing.run_names + addition.run_names,
        candidate_stats=candidate_stats,
        baseline_stats=baseline_stats,
    )


@dataclass
class TrialResult:
    run_name: str
    hyperparameters: Hyperparameters
    checkpoint_path: str
    training_summary_path: str
    safety_flags: list[str]
    approx_kl: float
    target_kl_trigger: float
    target_kl_exceeded: bool
    target_kl_hard_stop: bool
    target_kl_hard_breach_count: int
    target_kl_hard_consecutive_updates: int
    anchor_kl_mean: float
    anchor_kl_max: float
    clip_fraction: float
    labels: int
    episode_count: int
    available_episode_count: int
    training_reused: bool
    screen_evaluation: EvaluationResult | None = None
    final_evaluation: EvaluationResult | None = None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def parse_key_values(line: str) -> dict[str, int | float | str]:
    values: dict[str, int | float | str] = {}
    for match in KEY_VALUE_RE.finditer(line):
        raw = match.group(2)
        try:
            values[match.group(1)] = float(raw) if any(token in raw for token in (".", "e", "E")) else int(raw)
        except ValueError:
            values[match.group(1)] = raw
    return values


def parse_training_progress(line: str) -> dict[str, int | float | str] | None:
    match = TRAIN_PROGRESS_RE.search(line)
    if not match:
        return None
    values = parse_key_values(line)
    values["current"] = int(match.group(1))
    values["total"] = int(match.group(2))
    return values


def parse_evaluation_progress(line: str) -> tuple[int, int] | None:
    match = SELFPLAY_PROGRESS_RE.search(line)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


@dataclass
class SearchDisplayState:
    run_prefix: str
    max_trials: int
    configured_screen_candidates: int
    configured_finalists: int
    screen_games_per_side: int
    final_games_per_side: int
    started_at: float = field(default_factory=time.monotonic)
    phase: str = "starting"
    trained_count: int = 0
    safe_count: int = 0
    rejected_count: int = 0
    screened_count: int = 0
    finalized_count: int = 0
    screen_games_completed: int = 0
    final_games_completed: int = 0
    screen_games_planned: int = 0
    final_games_planned: int = 0
    screen_planned_candidates: int = 0
    final_planned_candidates: int = 0
    screen_plan_finalized: bool = False
    final_plan_finalized: bool = False
    active_trial_index: int = 0
    active_run_name: str = ""
    active_params: Hyperparameters | None = None
    active_kind: str = ""
    active_stage: str = ""
    active_side: str = ""
    active_current: int = 0
    active_total: int = 0
    active_started_at: float | None = None
    active_eta_seconds: float | None = None
    active_metrics: dict[str, int | float | str] = field(default_factory=dict)
    trials: list[TrialResult] = field(default_factory=list)
    trial_status: dict[str, str] = field(default_factory=dict)
    training_durations: list[float] = field(default_factory=list)
    evaluation_seconds: float = 0.0
    evaluation_games: int = 0
    last_notice: str = ""
    dashboard_mode: str = "plain-text"

    def begin_operation(
        self,
        kind: str,
        run_name: str,
        index: int,
        params: Hyperparameters,
        total: int,
        stage: str = "",
        side: str = "",
    ) -> None:
        self.active_kind = kind
        self.active_run_name = run_name
        self.active_trial_index = index
        self.active_params = params
        self.active_stage = stage
        self.active_side = side
        self.active_current = 0
        self.active_total = total
        self.active_started_at = time.monotonic()
        self.active_eta_seconds = None
        self.active_metrics = {}
        self.trial_status[run_name] = "training" if kind == "training" else stage

    def finish_operation(self, completed: int | None = None, record_duration: bool = True) -> None:
        elapsed = time.monotonic() - self.active_started_at if self.active_started_at is not None else 0.0
        if record_duration and self.active_kind == "training" and elapsed > 0.0:
            self.training_durations.append(elapsed)
        elif record_duration and self.active_kind == "evaluation" and elapsed > 0.0:
            games = completed if completed is not None else self.active_current
            self.evaluation_seconds += elapsed
            self.evaluation_games += max(0, games)
        self.active_current = self.active_total if completed is None else completed
        self.active_eta_seconds = 0.0

    def clear_operation(self) -> None:
        self.active_kind = ""
        self.active_stage = ""
        self.active_side = ""
        self.active_run_name = ""
        self.active_trial_index = 0
        self.active_params = None
        self.active_current = 0
        self.active_total = 0
        self.active_started_at = None
        self.active_eta_seconds = None
        self.active_metrics = {}

    def estimated_remaining_seconds(self) -> float | None:
        estimate = self.active_eta_seconds or 0.0
        has_estimate = self.active_eta_seconds is not None
        if self.training_durations:
            remaining_trials = max(0, self.max_trials - self.trained_count - (1 if self.active_kind == "training" else 0))
            estimate += remaining_trials * (sum(self.training_durations) / len(self.training_durations))
            has_estimate = True
        if self.evaluation_games > 0:
            seconds_per_game = self.evaluation_seconds / self.evaluation_games
            screen_candidates = self.screen_planned_candidates if self.screen_plan_finalized else self.configured_screen_candidates
            final_candidates = self.final_planned_candidates if self.final_plan_finalized else self.configured_finalists
            screen_total = self.screen_games_planned or (2 * screen_candidates * self.screen_games_per_side)
            final_total = self.final_games_planned or (2 * final_candidates * self.final_games_per_side)
            active_screen = self.active_current if self.active_kind == "evaluation" and self.active_stage == "screen" else 0
            active_final = self.active_current if self.active_kind == "evaluation" and self.active_stage == "final" else 0
            remaining_games = max(0, screen_total - self.screen_games_completed - active_screen)
            remaining_games += max(0, final_total - self.final_games_completed - active_final)
            estimate += remaining_games * seconds_per_game
            has_estimate = True
        return estimate if has_estimate else None

    def progress_payload(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "dashboard_mode": self.dashboard_mode,
            "elapsed_seconds": time.monotonic() - self.started_at,
            "eta_seconds": self.estimated_remaining_seconds(),
            "trained_candidates": self.trained_count,
            "safe_candidates": self.safe_count,
            "rejected_candidates": self.rejected_count,
            "screened_candidates": self.screened_count,
            "finalized_candidates": self.finalized_count,
            "screen_games_completed": self.screen_games_completed,
            "screen_games_planned": self.screen_games_planned,
            "final_games_completed": self.final_games_completed,
            "final_games_planned": self.final_games_planned,
            "active": {
                "kind": self.active_kind,
                "stage": self.active_stage,
                "side": self.active_side,
                "trial_index": self.active_trial_index,
                "run_name": self.active_run_name,
                "current": self.active_current,
                "total": self.active_total,
                "eta_seconds": self.active_eta_seconds,
                "metrics": self.active_metrics,
            },
        }


class ManifestProgressWriter:
    def __init__(self, path: Path, manifest: dict[str, object], state: SearchDisplayState) -> None:
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


class BaseSearchReporter:
    def __init__(self, args: argparse.Namespace, state: SearchDisplayState, progress_writer: ManifestProgressWriter) -> None:
        self.args = args
        self.state = state
        self.progress_writer = progress_writer

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def refresh(self) -> None:
        self.progress_writer.update()

    def notice(self, message: str) -> None:
        self.state.last_notice = message
        self.refresh()

    def command_started(self, command: list[str]) -> None:
        self.notice(f"running {Path(command[0]).name}")

    def child_line(self, line: str) -> None:
        if self.state.active_kind == "training":
            progress = parse_training_progress(line)
            if progress:
                self.state.active_current = int(progress.pop("current"))
                self.state.active_total = int(progress.pop("total"))
                self.state.active_metrics.update(progress)
            eta_match = TRAIN_ETA_RE.search(line)
            if eta_match and eta_match.group(2) != "estimating":
                try:
                    self.state.active_eta_seconds = float(eta_match.group(2).removesuffix("s"))
                except ValueError:
                    pass
        elif self.state.active_kind == "evaluation":
            progress = parse_evaluation_progress(line)
            if progress:
                self.state.active_current, self.state.active_total = progress
                if self.state.active_started_at is not None and self.state.active_current > 0:
                    elapsed = time.monotonic() - self.state.active_started_at
                    self.state.active_eta_seconds = max(0.0, elapsed / self.state.active_current * (self.state.active_total - self.state.active_current))
        self.refresh()


class PlainTextSearchReporter(BaseSearchReporter):
    def __init__(
        self,
        args: argparse.Namespace,
        state: SearchDisplayState,
        progress_writer: ManifestProgressWriter,
        fallback_notice: str = "",
    ) -> None:
        super().__init__(args, state, progress_writer)
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


class RichSearchReporter(BaseSearchReporter):
    def __init__(self, args: argparse.Namespace, state: SearchDisplayState, progress_writer: ManifestProgressWriter) -> None:
        super().__init__(args, state, progress_writer)
        self.console = Console()
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=args.dashboard_refresh_per_second,
            transient=False,
        )
        self.started = False

    def start(self) -> None:
        self.live.start()
        self.started = True
        self.refresh()

    def close(self) -> None:
        if self.started:
            self.live.stop()
            self.started = False

    def refresh(self) -> None:
        super().refresh()
        if self.started:
            self.live.update(self._render(), refresh=True)

    @staticmethod
    def _bar(current: int, total: int, width: int = 28, style: str = "cyan") -> str:
        if total <= 0:
            return "[grey50]estimating[/grey50]"
        bounded = min(max(0, current), total)
        filled = min(width, int(round(width * bounded / total)))
        return f"[{style}]" + "=" * filled + f"[/{style}][grey27]" + "-" * (width - filled) + "[/grey27]"

    def _summary(self) -> Panel:
        table = Table.grid(expand=True)
        for _ in range(4):
            table.add_column()
        table.add_row(
            f"[bold]Search[/]: {self.state.run_prefix}",
            f"[bold]Phase[/]: {self.state.phase}",
            f"[bold]Elapsed[/]: {format_duration(time.monotonic() - self.state.started_at)}",
            f"[bold]ETA[/]: {format_duration(self.state.estimated_remaining_seconds())}",
        )
        table.add_row(
            f"[bold]Trained[/]: {self.state.trained_count}/{self.state.max_trials}",
            f"[green]Safe[/]: {self.state.safe_count}",
            f"[red]Rejected[/]: {self.state.rejected_count}",
            f"[bold]Screened/Final[/]: {self.state.screened_count}/{self.state.finalized_count}",
        )
        active = self.state.active_run_name or "-"
        detail = self.state.active_kind
        if self.state.active_side:
            detail += f" {self.state.active_stage} side {self.state.active_side.upper()}"
        table.add_row(f"[bold]Active[/]: {active}", f"[bold]Operation[/]: {detail or '-'}", "", "")
        return Panel(table, title="PPO Search", border_style="cyan")

    def _progress(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(width=18)
        table.add_column(ratio=1)
        table.add_column(width=20, justify="right")
        screen_candidates = self.state.screen_planned_candidates if self.state.screen_plan_finalized else self.state.configured_screen_candidates
        final_candidates = self.state.final_planned_candidates if self.state.final_plan_finalized else self.state.configured_finalists
        screen_total = self.state.screen_games_planned or (2 * screen_candidates * self.state.screen_games_per_side)
        final_total = self.state.final_games_planned or (2 * final_candidates * self.state.final_games_per_side)
        screen_current = self.state.screen_games_completed
        final_current = self.state.final_games_completed
        if self.state.active_kind == "evaluation" and self.state.active_stage == "screen":
            screen_current += self.state.active_current
        elif self.state.active_kind == "evaluation" and self.state.active_stage == "final":
            final_current += self.state.active_current
        active_label = "idle"
        if self.state.active_kind:
            unit = "episodes" if self.state.active_kind == "training" else "games"
            active_label = f"{self.state.active_current}/{self.state.active_total} {unit}"
        rows = (
            ("Training sweep", self.state.trained_count, self.state.max_trials, f"{self.state.trained_count}/{self.state.max_trials}"),
            ("Screening", screen_current, screen_total, f"{screen_current}/{screen_total} games"),
            ("Final evaluation", final_current, final_total, f"{final_current}/{final_total} games"),
            ("Active operation", self.state.active_current, self.state.active_total, active_label),
        )
        for label, current, total, text_value in rows:
            table.add_row(f"[bold]{label}[/]", self._bar(current, total), text_value)
        return Panel(table, border_style="grey42", padding=(0, 1))

    @staticmethod
    def _result_text(trial: TrialResult) -> str:
        result = trial.final_evaluation or trial.screen_evaluation
        if not result:
            return "-"
        return f"{result.win_rate:.1%} [{result.confidence_low:.1%}, {result.confidence_high:.1%}]"

    def _candidate_table(self) -> Table:
        table = Table(expand=True, box=None, pad_edge=False, header_style="bold white")
        for name, width in (("#", 3), ("Status", 11), ("LR", 9), ("Entropy", 9), ("Anchor", 8), ("Clip", 6), ("KL", 8), ("Anchor KL", 10), ("Clip frac", 10), ("KL guard", 9)):
            table.add_column(name, width=width)
        table.add_column("Safety", ratio=1)
        table.add_column("Evaluation", ratio=1)
        indexed = list(enumerate(self.state.trials, start=1))[-self.args.dashboard_visible_trials :]
        for index, trial in indexed:
            status = self.state.trial_status.get(trial.run_name, "safe" if not trial.safety_flags else "rejected")
            style = "red" if trial.safety_flags else ("green" if status in {"safe", "screened", "finalized"} else "cyan")
            table.add_row(
                str(index), f"[{style}]{status}[/{style}]", f"{trial.hyperparameters.learning_rate:g}",
                f"{trial.hyperparameters.entropy_coef:g}", f"{trial.hyperparameters.anchor_kl_coef:g}",
                f"{trial.hyperparameters.ppo_clip_epsilon:g}", f"{trial.approx_kl:.4f}",
                f"{trial.anchor_kl_mean:.4f}", f"{trial.clip_fraction:.3f}",
                f"{trial.target_kl_hard_breach_count}/{trial.target_kl_hard_consecutive_updates}",
                trial.safety_flags[0] if trial.safety_flags else "ok", self._result_text(trial),
            )
        if self.state.active_run_name and not any(t.run_name == self.state.active_run_name for t in self.state.trials):
            params = self.state.active_params
            metrics = self.state.active_metrics
            if params:
                table.add_row(
                    str(self.state.active_trial_index), "[cyan]training[/cyan]", f"{params.learning_rate:g}",
                    f"{params.entropy_coef:g}", f"{params.anchor_kl_coef:g}", f"{params.ppo_clip_epsilon:g}",
                    f"{float(metrics.get('approx_kl', 0.0)):.4f}", f"{float(metrics.get('anchor_kl_mean', 0.0)):.4f}",
                    f"{float(metrics.get('clip_fraction', 0.0)):.3f}",
                    str(metrics.get("hard_kl_breaches", "0/0")), "pending",
                    f"{self.state.active_current}/{self.state.active_total} episodes",
                )
        return table

    def _leaderboard(self) -> Table | None:
        ranked = sorted(
            (trial for trial in self.state.trials if trial.final_evaluation or trial.screen_evaluation),
            key=lambda trial: evaluation_rank_key(trial, final=trial.final_evaluation is not None),
            reverse=True,
        )[: self.args.dashboard_leaderboard_size]
        if not ranked:
            return None
        table = Table(expand=True, box=None, title="Current leaderboard", header_style="bold white")
        table.add_column("Rank", width=5)
        table.add_column("Candidate", ratio=1)
        table.add_column("Stage", width=8)
        table.add_column("Win rate", width=10)
        table.add_column("Lower 95%", width=11)
        for index, trial in enumerate(ranked, start=1):
            result = trial.final_evaluation or trial.screen_evaluation
            assert result is not None
            table.add_row(str(index), trial.run_name, result.stage, f"{result.win_rate:.1%}", f"{result.confidence_low:.1%}")
        return table

    def _render(self) -> Group:
        divider = "[grey50]" + ("-" * 72) + "[/grey50]"
        items: list[object] = [self._summary(), self._progress(), divider, self._candidate_table()]
        leaderboard = self._leaderboard()
        if leaderboard is not None:
            items.extend((divider, leaderboard))
        if self.state.last_notice:
            items.append(Panel(self.state.last_notice, border_style="grey42"))
        return Group(*items)


def select_search_reporter(
    args: argparse.Namespace,
    state: SearchDisplayState,
    progress_writer: ManifestProgressWriter,
) -> BaseSearchReporter:
    if not args.dashboard:
        return PlainTextSearchReporter(args, state, progress_writer)
    if not RICH_AVAILABLE:
        return PlainTextSearchReporter(args, state, progress_writer, "dashboard unavailable; using plain-text output")
    if not sys.stdout.isatty():
        return PlainTextSearchReporter(args, state, progress_writer, "live terminal unavailable; using plain-text output")
    state.dashboard_mode = "terminal"
    return RichSearchReporter(args, state, progress_writer)


def run_command(
    command: list[str],
    cwd: Path,
    reporter: BaseSearchReporter,
    raw_log_path: Path | None,
    env_updates: dict[str, str] | None = None,
) -> None:
    reporter.command_started(command)
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
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


def safe_float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def trial_run_name(prefix: str, index: int, params: Hyperparameters) -> str:
    return (
        f"{prefix}_{index:02d}_lr{safe_float_token(params.learning_rate)}"
        f"_ent{safe_float_token(params.entropy_coef)}"
        f"_anchor{safe_float_token(params.anchor_kl_coef)}"
        f"_clip{safe_float_token(params.ppo_clip_epsilon)}"
        f"_seed{params.shuffle_seed}"
    )


def build_train_command(
    args: argparse.Namespace,
    trainer_exe: Path,
    episode_batch: Path,
    init_checkpoint: Path,
    anchor_checkpoint: Path,
    output_checkpoint: Path,
    summary_path: Path,
    params: Hyperparameters,
) -> list[str]:
    expected_tag = args.policy_tag_expected or str(init_checkpoint)
    return [
        str(trainer_exe), "--train-live-ppo", str(episode_batch), str(output_checkpoint),
        "--policy-tag-expected", expected_tag,
        "--parent-checkpoint", str(init_checkpoint),
        "--epochs", str(args.epochs),
        "--learning-rate", str(params.learning_rate),
        "--gamma", str(args.gamma),
        "--entropy-coef", str(params.entropy_coef),
        "--advantage-norm", "1" if args.advantage_norm else "0",
        "--gae-lambda", str(args.gae_lambda),
        "--ppo-clip-epsilon", str(params.ppo_clip_epsilon),
        "--ppo-value-clip-epsilon", str(args.ppo_value_clip_epsilon),
        "--target-kl", str(args.target_kl),
        "--target-kl-min-episodes", str(args.target_kl_min_episodes),
        "--target-kl-min-labels", str(args.target_kl_min_labels),
        "--target-kl-hard-multiplier", str(args.target_kl_hard_multiplier),
        "--target-kl-hard-consecutive-updates", str(args.target_kl_hard_consecutive_updates),
        "--shuffle-seed", str(params.shuffle_seed),
        "--ppo-minibatch-episodes", str(args.ppo_minibatch_episodes),
        "--adam-beta1", str(args.adam_beta1),
        "--adam-beta2", str(args.adam_beta2),
        "--adam-epsilon", str(args.adam_epsilon),
        "--reward-mode", args.reward_mode,
        "--training-summary-path", str(summary_path),
        "--anchor-checkpoint", str(anchor_checkpoint),
        "--anchor-kl-coef", str(params.anchor_kl_coef),
        "--dense-additive-hp-swing-weight", str(args.dense_additive_hp_swing_weight),
        "--dense-additive-faint-swing-weight", str(args.dense_additive_faint_swing_weight),
        "--dense-additive-reward-clip", str(args.dense_additive_reward_clip),
    ]


def build_eval_command(
    args: argparse.Namespace,
    repo_root: Path,
    run_name: str,
    candidate_checkpoint: Path,
    parent_checkpoint: Path,
    candidate_side: str,
    games: int,
    pool_seed: int,
) -> list[str]:
    model_a = candidate_checkpoint if candidate_side == "a" else parent_checkpoint
    model_b = parent_checkpoint if candidate_side == "a" else candidate_checkpoint
    return [
        sys.executable, str((repo_root / "py" / "tools" / "selfplay_server.py").resolve()),
        "--run-name", run_name,
        "--games", str(games),
        "--concurrent-games", str(args.eval_concurrent_games),
        "--worker-pairs", str(args.eval_worker_pairs),
        "--worker-games", "0",
        "--ensure-shard-count", "true",
        "--model-a-pool", "",
        "--model-a", str(model_a),
        "--model-b-pool", "",
        "--model-b", str(model_b),
        "--pool-seed", str(pool_seed),
        "--format", args.format,
        "--worker-think-mode", "live",
        "--serve-client", "1",
        "--worker-log-stdout", "0",
        "--launch-stagger-seconds", str(args.launch_stagger_seconds),
        "--resource-check-seconds", str(args.resource_check_seconds),
        "--min-available-memory-gb", str(args.min_available_memory_gb),
        "--min-available-pagefile-gb", str(args.min_available_pagefile_gb),
        "--startup-timeout-seconds", str(args.startup_timeout_seconds),
    ]


def training_safety_flags(summary: dict[str, object], args: argparse.Namespace, anchor_coef: float) -> list[str]:
    flags: list[str] = []
    episodes = int(summary.get("episode_count", 0) or 0)
    labels = int(summary.get("labels", 0) or 0)
    approx_kl = float(summary.get("approx_kl", 0.0) or 0.0)
    anchor_mean = float(summary.get("anchor_kl_mean", 0.0) or 0.0)
    anchor_max = float(summary.get("anchor_kl_max", 0.0) or 0.0)
    clip_fraction = float(summary.get("clip_fraction", 0.0) or 0.0)
    target_exceeded = bool(summary.get("target_kl_exceeded", False))
    if labels <= 0 or episodes <= 0:
        flags.append("no_training_data")
    if bool(summary.get("target_kl_hard_stop", False)):
        flags.append("target_kl_hard_stop")
    if target_exceeded and episodes < args.target_kl_min_episodes:
        flags.append(f"target_kl_before_min_episodes:{episodes}")
    if target_exceeded and labels < args.target_kl_min_labels:
        flags.append(f"target_kl_before_min_labels:{labels}")
    if approx_kl > args.max_approx_kl:
        flags.append(f"approx_kl_high:{approx_kl:.6f}")
    if anchor_mean > args.max_anchor_kl_mean:
        flags.append(f"anchor_kl_mean_high:{anchor_mean:.6f}")
    if anchor_max > args.max_anchor_kl_max:
        flags.append(f"anchor_kl_max_high:{anchor_max:.6f}")
    if anchor_coef > 0.0 and episodes > 1 and anchor_mean <= 0.0:
        flags.append("anchor_inactive")
    if clip_fraction > args.max_clip_fraction:
        flags.append(f"clip_fraction_high:{clip_fraction:.6f}")
    return flags


def collect_training_result(
    run_name: str,
    params: Hyperparameters,
    checkpoint_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
    training_reused: bool = False,
) -> TrialResult:
    summary = load_json(summary_path)
    return TrialResult(
        run_name=run_name,
        hyperparameters=params,
        checkpoint_path=str(checkpoint_path),
        training_summary_path=str(summary_path),
        safety_flags=training_safety_flags(summary, args, params.anchor_kl_coef),
        approx_kl=float(summary.get("approx_kl", 0.0) or 0.0),
        target_kl_trigger=float(summary.get("target_kl_trigger", 0.0) or 0.0),
        target_kl_exceeded=bool(summary.get("target_kl_exceeded", False)),
        target_kl_hard_stop=bool(summary.get("target_kl_hard_stop", False)),
        target_kl_hard_breach_count=int(summary.get("target_kl_hard_breach_count", 0) or 0),
        target_kl_hard_consecutive_updates=int(summary.get("target_kl_hard_consecutive_updates", 1) or 1),
        anchor_kl_mean=float(summary.get("anchor_kl_mean", 0.0) or 0.0),
        anchor_kl_max=float(summary.get("anchor_kl_max", 0.0) or 0.0),
        clip_fraction=float(summary.get("clip_fraction", 0.0) or 0.0),
        labels=int(summary.get("labels", 0) or 0),
        episode_count=int(summary.get("episode_count", 0) or 0),
        available_episode_count=int(summary.get("available_episode_count", 0) or 0),
        training_reused=training_reused,
    )


def training_artifacts_match(
    summary_path: Path,
    episode_batch: Path,
    init_checkpoint: Path,
    anchor_checkpoint: Path,
    output_checkpoint: Path,
    params: Hyperparameters,
    args: argparse.Namespace,
) -> bool:
    try:
        summary = load_json(summary_path)
        expected_paths = {
            "input_episode_batch": episode_batch,
            "parent_checkpoint": init_checkpoint,
            "anchor_checkpoint": anchor_checkpoint,
            "output_checkpoint": output_checkpoint,
        }
        for key, expected in expected_paths.items():
            recorded = str(summary.get(key, "")).strip()
            if not recorded or resolve_path(resolve_repo_root(), recorded) != expected.resolve():
                return False
        expected_numbers = {
            "learning_rate": params.learning_rate,
            "entropy_coef": params.entropy_coef,
            "anchor_kl_coef": params.anchor_kl_coef,
            "ppo_clip_epsilon": params.ppo_clip_epsilon,
        }
        for key, expected in expected_numbers.items():
            if not math.isclose(float(summary.get(key, math.nan)), expected, rel_tol=1e-6, abs_tol=1e-12):
                return False
        return (
            int(summary.get("shuffle_seed", -1)) == params.shuffle_seed
            and int(summary.get("minibatch_episodes", -1)) == args.ppo_minibatch_episodes
            and int(summary.get("target_kl_hard_consecutive_updates", -1))
                == args.target_kl_hard_consecutive_updates
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def training_screen_key(trial: TrialResult, target_kl: float) -> tuple[int, float, int, float]:
    safe = 1 if not trial.safety_flags else 0
    return (safe, -abs(trial.approx_kl - target_kl * 0.5), trial.labels, -trial.anchor_kl_mean)


def evaluation_summary_path(repo_root: Path, run_name: str) -> Path:
    return repo_root / DEFAULT_MATCH_RUNS_ROOT / run_name / f"{run_name}_summary.json"


def evaluation_artifacts_match(
    summary_path: Path,
    model_a: Path,
    model_b: Path,
    requested_games: int,
) -> bool:
    try:
        summary = load_json(summary_path)
        specs = summary.get("model_specs", {}) or {}
        recorded_a = str((specs.get("a", {}) or {}).get("path", "")).strip()
        recorded_b = str((specs.get("b", {}) or {}).get("path", "")).strip()
        return (
            summary.get("status") == "completed"
            and int(summary.get("target_games", 0) or 0) == requested_games
            and bool(recorded_a)
            and bool(recorded_b)
            and resolve_path(resolve_repo_root(), recorded_a) == model_a.resolve()
            and resolve_path(resolve_repo_root(), recorded_b) == model_b.resolve()
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_balanced_evaluation(
    args: argparse.Namespace,
    repo_root: Path,
    trial: TrialResult,
    parent_checkpoint: Path,
    stage: str,
    games_per_side: int,
    seed_base: int,
    trial_index: int,
    reporter: BaseSearchReporter,
    state: SearchDisplayState,
    logs_dir: Path,
    block_index: int = 0,
) -> EvaluationResult:
    total_games = 0
    total_wins = 0
    total_earned_wins = 0
    candidate_stats: dict[str, object] = {}
    baseline_stats: dict[str, object] = {}
    run_names: list[str] = []
    candidate_checkpoint = Path(trial.checkpoint_path)
    for side_index, side in enumerate(("a", "b")):
        block_token = f"_block_{block_index:02d}" if block_index > 0 else ""
        run_name = f"{trial.run_name}_{stage}{block_token}_side_{side}_{games_per_side}"
        state.begin_operation(
            "evaluation", trial.run_name, trial_index, trial.hyperparameters,
            games_per_side, stage=stage, side=side,
        )
        reporter.refresh()
        summary_path = evaluation_summary_path(repo_root, run_name)
        run_names.append(run_name)
        model_a = candidate_checkpoint if side == "a" else parent_checkpoint
        model_b = parent_checkpoint if side == "a" else candidate_checkpoint
        can_resume = (
            args.resume
            and trial.training_reused
            and summary_path.exists()
            and evaluation_artifacts_match(summary_path, model_a, model_b, games_per_side)
        )
        if not can_resume:
            run_command(build_eval_command(
                args, repo_root, run_name, candidate_checkpoint, parent_checkpoint,
                side, games_per_side, seed_base + side_index,
            ), repo_root, reporter, logs_dir / f"{run_name}.log" if args.dashboard_write_raw_logs else None)
        summary = load_json(summary_path)
        if summary.get("status") != "completed":
            raise RuntimeError(f"evaluation did not complete: {run_name}")
        group = ((summary.get("group_stats", {}) or {}).get(side, {}) or {})
        baseline_side = "b" if side == "a" else "a"
        baseline_group = ((summary.get("group_stats", {}) or {}).get(baseline_side, {}) or {})
        candidate_stats = aggregate_group_stats(candidate_stats, group)
        baseline_stats = aggregate_group_stats(baseline_stats, baseline_group)
        games = int(group.get("matches_played", 0) or 0)
        total_games += games
        total_wins += int(group.get("wins", 0) or 0)
        total_earned_wins += int(group.get("earned_wins", 0) or 0)
        state.finish_operation(games, record_duration=not can_resume)
        if stage == "screen":
            state.screen_games_completed += min(games, games_per_side)
        else:
            state.final_games_completed += min(games, games_per_side)
        reporter.refresh()
    low, high = wilson_interval(total_wins, total_games)
    return EvaluationResult(
        stage=stage,
        games=total_games,
        wins=total_wins,
        earned_wins=total_earned_wins,
        win_rate=(total_wins / total_games) if total_games else 0.0,
        earned_win_rate=(total_earned_wins / total_games) if total_games else 0.0,
        confidence_low=low,
        confidence_high=high,
        collapse_flags=aggregate_collapse_flags(candidate_stats, baseline_stats),
        run_names=run_names,
        candidate_stats=candidate_stats,
        baseline_stats=baseline_stats,
    )


def evaluation_rank_key(trial: TrialResult, final: bool = False) -> tuple[int, float, float, float]:
    result = trial.final_evaluation if final else trial.screen_evaluation
    if result is None:
        return (0, 0.0, 0.0, 0.0)
    return (1 if not result.collapse_flags else 0, result.confidence_low, result.win_rate, -trial.anchor_kl_mean)


def adaptive_screen_contenders(
    ranked_trials: list[TrialResult],
    finalist_count: int,
    requested_games_per_side: dict[str, int],
    max_games_per_side: int,
) -> list[TrialResult]:
    usable = [
        trial for trial in ranked_trials
        if trial.screen_evaluation is not None and not trial.screen_evaluation.collapse_flags
    ]
    if finalist_count <= 0 or len(usable) <= finalist_count:
        return []
    provisional = usable[:finalist_count]
    cutoff = provisional[-1].screen_evaluation
    assert cutoff is not None
    challengers = [
        trial for trial in usable[finalist_count:]
        if trial.screen_evaluation is not None
        and trial.screen_evaluation.confidence_high >= cutoff.confidence_low
    ]
    if not challengers:
        return []
    unresolved = provisional + challengers
    return [
        trial for trial in unresolved
        if requested_games_per_side.get(trial.run_name, 0) < max_games_per_side
    ]


def promotion_assessment(
    ranked_final: list[TrialResult],
    minimum_win_rate: float,
    confidence_threshold: float,
) -> tuple[TrialResult | None, dict[str, object]]:
    if not ranked_final:
        return None, {
            "status": "no_finalist",
            "candidate": None,
            "minimum_win_rate": minimum_win_rate,
            "confidence_threshold": confidence_threshold,
            "promotion_confident": False,
        }
    top = ranked_final[0]
    result = top.final_evaluation
    assert result is not None
    collapse_free = not result.collapse_flags
    clears_point_gate = result.win_rate > minimum_win_rate
    clears_confidence_gate = result.confidence_low > confidence_threshold
    winner = top if collapse_free and clears_point_gate else None
    if not collapse_free:
        status = "collapse_rejected"
    elif not clears_point_gate:
        status = "no_winner"
    elif clears_confidence_gate:
        status = "confident_winner"
    else:
        status = "tentative_winner"
    return winner, {
        "status": status,
        "candidate": top.run_name,
        "minimum_win_rate": minimum_win_rate,
        "confidence_threshold": confidence_threshold,
        "win_rate": result.win_rate,
        "confidence_low": result.confidence_low,
        "confidence_high": result.confidence_high,
        "collapse_free": collapse_free,
        "clears_point_gate": clears_point_gate,
        "promotion_confident": bool(winner is not None and clears_confidence_gate),
    }


def trial_payload(trial: TrialResult) -> dict[str, object]:
    return asdict(trial)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled anchored PPO hyperparameter search")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Search defaults file; CLI flags override it")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--episode-batch", required=True)
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--eval-model-b", default="")
    parser.add_argument("--policy-tag-expected", default="")
    parser.add_argument("--learning-rates", type=csv_floats, default=[5e-6, 1e-5, 2.5e-5, 5e-5])
    parser.add_argument("--entropy-coefs", type=csv_floats, default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--anchor-kl-coefs", type=csv_floats, default=[0.003, 0.01, 0.03])
    parser.add_argument("--ppo-clip-epsilons", type=csv_floats, default=[0.2])
    parser.add_argument("--shuffle-seeds", type=csv_ints, default=[101])
    parser.add_argument("--ppo-minibatch-episodes", type=positive_int, default=int_default("ppo_minibatch_episodes"))
    parser.add_argument("--max-trials", type=positive_int, default=18)
    parser.add_argument("--screen-candidates", type=positive_int, default=8)
    parser.add_argument("--finalists", type=positive_int, default=3)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--gamma", type=float, default=float_default("ppo_gamma"))
    parser.add_argument("--advantage-norm", type=parse_bool, default=bool_default("advantage_norm"))
    parser.add_argument("--gae-lambda", type=float, default=float_default("gae_lambda"))
    parser.add_argument("--ppo-value-clip-epsilon", type=float, default=float_default("ppo_value_clip_epsilon"))
    parser.add_argument("--target-kl", type=float, default=float_default("ppo_target_kl"))
    parser.add_argument("--target-kl-min-episodes", type=positive_int, default=int_default("ppo_target_kl_min_episodes"))
    parser.add_argument("--target-kl-min-labels", type=positive_int, default=int_default("ppo_target_kl_min_labels"))
    parser.add_argument("--target-kl-hard-multiplier", type=positive_float, default=float_default("ppo_target_kl_hard_multiplier"))
    parser.add_argument("--target-kl-hard-consecutive-updates", type=positive_int, default=int_default("ppo_target_kl_hard_consecutive_updates"))
    parser.add_argument("--adam-beta1", type=float, default=float_default("adam_beta1"))
    parser.add_argument("--adam-beta2", type=float, default=float_default("adam_beta2"))
    parser.add_argument("--adam-epsilon", type=float, default=float_default("adam_epsilon"))
    parser.add_argument("--reward-mode", choices=["terminal", "dense_additive"], default="terminal")
    parser.add_argument("--dense-additive-hp-swing-weight", type=float, default=0.1)
    parser.add_argument("--dense-additive-faint-swing-weight", type=float, default=0.25)
    parser.add_argument("--dense-additive-reward-clip", type=float, default=0.4)
    parser.add_argument("--max-approx-kl", type=float, default=float_default("ppo_search_prune_approx_kl"))
    parser.add_argument("--max-anchor-kl-mean", type=float, default=0.05)
    parser.add_argument("--max-anchor-kl-max", type=float, default=0.20)
    parser.add_argument("--max-clip-fraction", type=float, default=0.25)
    parser.add_argument("--screen-games-per-side", type=positive_int, default=250)
    parser.add_argument("--adaptive-screen", type=parse_bool, default=True)
    parser.add_argument("--screen-game-block-per-side", type=positive_int, default=100)
    parser.add_argument("--screen-max-games-per-side", type=positive_int, default=500)
    parser.add_argument("--final-games-per-side", type=positive_int, default=1000)
    parser.add_argument("--promotion-min-win-rate", type=float, default=0.5)
    parser.add_argument("--promotion-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--eval-concurrent-games", type=positive_int, default=70)
    parser.add_argument("--eval-worker-pairs", type=positive_int, default=125)
    parser.add_argument("--format", default="gen9randomdoublesbattle")
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.25)
    parser.add_argument("--resource-check-seconds", type=float, default=2.0)
    parser.add_argument("--min-available-memory-gb", type=float, default=2.0)
    parser.add_argument("--min-available-pagefile-gb", type=float, default=4.0)
    parser.add_argument("--startup-timeout-seconds", type=positive_int, default=120)
    parser.add_argument("--omp-threads", type=nonnegative_int, default=8)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--dashboard", type=parse_bool, default=True)
    parser.add_argument("--dashboard-refresh-per-second", type=positive_float, default=8.0)
    parser.add_argument("--dashboard-visible-trials", type=positive_int, default=10)
    parser.add_argument("--dashboard-leaderboard-size", type=positive_int, default=5)
    parser.add_argument("--dashboard-write-raw-logs", type=parse_bool, default=True)
    return parser


def parse_search_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    config_args = load_config_args(config_path_from_args(cli_args))
    return build_parser().parse_args(config_args + cli_args)


def main() -> None:
    args = parse_search_args()
    if args.screen_max_games_per_side < args.screen_games_per_side:
        raise SystemExit("screen-max-games-per-side must be >= screen-games-per-side")
    for label, value in (
        ("promotion-min-win-rate", args.promotion_min_win_rate),
        ("promotion-confidence-threshold", args.promotion_confidence_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{label} must be between 0 and 1")
    repo_root = resolve_repo_root()
    trainer_exe = resolve_path(repo_root, DEFAULT_TRAINER_EXE)
    init_checkpoint = resolve_path(repo_root, args.init_checkpoint)
    episode_batch = resolve_path(repo_root, args.episode_batch)
    anchor_checkpoint = resolve_path(repo_root, args.anchor_checkpoint) if args.anchor_checkpoint else init_checkpoint
    parent_checkpoint = resolve_path(repo_root, args.eval_model_b) if args.eval_model_b else init_checkpoint
    config_path = resolve_path(repo_root, args.config)
    for label, path in (
        ("trainer executable", trainer_exe),
        ("initial checkpoint", init_checkpoint),
        ("episode batch", episode_batch),
        ("anchor checkpoint", anchor_checkpoint),
        ("evaluation parent", parent_checkpoint),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    search_dir = (repo_root / DEFAULT_SEARCH_ROOT / args.run_prefix).resolve()
    search_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = search_dir / f"{args.run_prefix}_search_manifest.json"
    combinations = [
        Hyperparameters(lr, entropy, anchor, clip, shuffle_seed)
        for lr, entropy, anchor, clip, shuffle_seed in itertools.product(
            args.learning_rates, args.entropy_coefs, args.anchor_kl_coefs,
            args.ppo_clip_epsilons, args.shuffle_seeds,
        )
    ]
    random.Random(args.search_seed).shuffle(combinations)
    combinations = combinations[: args.max_trials]
    manifest: dict[str, object] = {
        "status": "running",
        "run_prefix": args.run_prefix,
        "started_at_unix": time.time(),
        "fixed_inputs": {
            "trainer_executable": str(trainer_exe),
            "init_checkpoint": str(init_checkpoint),
            "episode_batch": str(episode_batch),
            "anchor_checkpoint": str(anchor_checkpoint),
            "evaluation_parent": str(parent_checkpoint),
        },
        "config_path": str(config_path),
        "search_space": {
            "learning_rates": args.learning_rates,
            "entropy_coefs": args.entropy_coefs,
            "anchor_kl_coefs": args.anchor_kl_coefs,
            "ppo_clip_epsilons": args.ppo_clip_epsilons,
            "shuffle_seeds": args.shuffle_seeds,
            "max_trials": args.max_trials,
        },
        "staging": {
            "screen_candidates": args.screen_candidates,
            "finalists": args.finalists,
            "screen_games_per_side": args.screen_games_per_side,
            "adaptive_screen": args.adaptive_screen,
            "screen_game_block_per_side": args.screen_game_block_per_side,
            "screen_max_games_per_side": args.screen_max_games_per_side,
            "final_games_per_side": args.final_games_per_side,
            "promotion_min_win_rate": args.promotion_min_win_rate,
            "promotion_confidence_threshold": args.promotion_confidence_threshold,
        },
        "search_seed": args.search_seed,
        "trials": [],
    }
    state = SearchDisplayState(
        run_prefix=args.run_prefix,
        max_trials=len(combinations),
        configured_screen_candidates=min(args.screen_candidates, len(combinations)),
        configured_finalists=min(args.finalists, args.screen_candidates, len(combinations)),
        screen_games_per_side=args.screen_games_per_side,
        final_games_per_side=args.final_games_per_side,
    )
    progress_writer = ManifestProgressWriter(manifest_path, manifest, state)
    reporter = select_search_reporter(args, state, progress_writer)
    logs_dir = search_dir / "logs"
    trials = state.trials
    trainer_env = {"PORYGON_OMP_THREADS": str(args.omp_threads)} if args.omp_threads > 0 else None
    progress_writer.update(force=True)
    reporter.start()
    try:
        state.phase = "training"
        reporter.refresh()
        for index, params in enumerate(combinations, start=1):
            run_name = trial_run_name(args.run_prefix, index, params)
            run_dir = (repo_root / DEFAULT_RUNS_ROOT / run_name).resolve()
            checkpoint_path = run_dir / "candidate.chk"
            summary_path = run_dir / "training_summary.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            state.begin_operation("training", run_name, index, params, 0)
            reporter.refresh()
            can_resume = (
                args.resume
                and checkpoint_path.exists()
                and summary_path.exists()
                and training_artifacts_match(
                    summary_path, episode_batch, init_checkpoint, anchor_checkpoint,
                    checkpoint_path, params, args,
                )
            )
            if not can_resume:
                shutil.copy2(init_checkpoint, checkpoint_path)
                run_command(
                    build_train_command(
                        args, trainer_exe, episode_batch, init_checkpoint, anchor_checkpoint,
                        checkpoint_path, summary_path, params,
                    ),
                    repo_root,
                    reporter,
                    logs_dir / f"{run_name}_training.log" if args.dashboard_write_raw_logs else None,
                    trainer_env,
                )
            trial = collect_training_result(
                run_name, params, checkpoint_path, summary_path, args,
                training_reused=can_resume,
            )
            state.active_total = trial.available_episode_count or trial.episode_count
            state.finish_operation(trial.episode_count, record_duration=not can_resume)
            trials.append(trial)
            state.trained_count += 1
            if trial.safety_flags:
                state.rejected_count += 1
                state.trial_status[run_name] = "rejected"
            else:
                state.safe_count += 1
                state.trial_status[run_name] = "safe"
            manifest["trials"] = [trial_payload(item) for item in trials]
            reporter.notice(
                f"trained {run_name} safe={not trial.safety_flags} labels={trial.labels} "
                f"approx_kl={trial.approx_kl:.6f}"
            )
            progress_writer.update(force=True)

        safe_trials = [trial for trial in trials if not trial.safety_flags]
        safe_trials.sort(key=lambda trial: training_screen_key(trial, args.target_kl), reverse=True)
        screen_trials = safe_trials[: args.screen_candidates]
        state.phase = "screening"
        state.screen_planned_candidates = len(screen_trials)
        state.screen_plan_finalized = True
        state.screen_games_planned = 2 * len(screen_trials) * args.screen_games_per_side
        reporter.refresh()
        trial_indices = {trial.run_name: index for index, trial in enumerate(trials, start=1)}
        requested_screen_games = {trial.run_name: 0 for trial in screen_trials}
        for index, trial in enumerate(screen_trials, start=1):
            trial.screen_evaluation = run_balanced_evaluation(
                args, repo_root, trial, parent_checkpoint, "screen",
                args.screen_games_per_side, args.search_seed * 1000 + index * 10,
                trial_indices[trial.run_name], reporter, state, logs_dir, block_index=1,
            )
            requested_screen_games[trial.run_name] = args.screen_games_per_side
            state.screened_count += 1
            state.trial_status[trial.run_name] = "screened"
            manifest["trials"] = [trial_payload(item) for item in trials]
            reporter.notice(
                f"screened {trial.run_name} win_rate={trial.screen_evaluation.win_rate:.4f} "
                f"lower95={trial.screen_evaluation.confidence_low:.4f}"
            )
            progress_writer.update(force=True)

        ranked_screen = sorted(screen_trials, key=evaluation_rank_key, reverse=True)
        adaptive_block_index = 1
        while args.adaptive_screen:
            contenders = adaptive_screen_contenders(
                ranked_screen,
                min(args.finalists, len(ranked_screen)),
                requested_screen_games,
                args.screen_max_games_per_side,
            )
            if not contenders:
                break
            adaptive_block_index += 1
            reporter.notice(
                f"adaptive screening block {adaptive_block_index}: refining {len(contenders)} unresolved candidates"
            )
            for trial in contenders:
                remaining = args.screen_max_games_per_side - requested_screen_games[trial.run_name]
                block_games = min(args.screen_game_block_per_side, remaining)
                if block_games <= 0:
                    continue
                state.screen_games_planned += 2 * block_games
                state.trial_status[trial.run_name] = "refining"
                addition = run_balanced_evaluation(
                    args, repo_root, trial, parent_checkpoint, "screen",
                    block_games,
                    args.search_seed * 1000000 + adaptive_block_index * 10000 + trial_indices[trial.run_name] * 10,
                    trial_indices[trial.run_name], reporter, state, logs_dir,
                    block_index=adaptive_block_index,
                )
                trial.screen_evaluation = merge_evaluation_results(trial.screen_evaluation, addition)
                requested_screen_games[trial.run_name] += block_games
                state.trial_status[trial.run_name] = "screened"
                manifest["trials"] = [trial_payload(item) for item in trials]
                progress_writer.update(force=True)
            ranked_screen = sorted(screen_trials, key=evaluation_rank_key, reverse=True)

        finalists = [
            trial for trial in ranked_screen
            if trial.screen_evaluation and not trial.screen_evaluation.collapse_flags
        ][: args.finalists]
        state.phase = "final evaluation"
        state.final_planned_candidates = len(finalists)
        state.final_plan_finalized = True
        state.final_games_planned = 2 * len(finalists) * args.final_games_per_side
        reporter.refresh()
        for index, trial in enumerate(finalists, start=1):
            trial.final_evaluation = run_balanced_evaluation(
                args, repo_root, trial, parent_checkpoint, "final",
                args.final_games_per_side, args.search_seed * 100000 + index * 10,
                trial_indices[trial.run_name], reporter, state, logs_dir, block_index=1,
            )
            state.finalized_count += 1
            state.trial_status[trial.run_name] = "finalized"
            manifest["trials"] = [trial_payload(item) for item in trials]
            reporter.notice(
                f"finalized {trial.run_name} win_rate={trial.final_evaluation.win_rate:.4f} "
                f"lower95={trial.final_evaluation.confidence_low:.4f}"
            )
            progress_writer.update(force=True)

        ranked_final = sorted(finalists, key=lambda trial: evaluation_rank_key(trial, final=True), reverse=True)
        winner, assessment = promotion_assessment(
            ranked_final,
            args.promotion_min_win_rate,
            args.promotion_confidence_threshold,
        )
        state.phase = "completed"
        state.clear_operation()
        manifest["status"] = "completed"
        manifest["completed_at_unix"] = time.time()
        manifest["ranked_screen_results"] = [trial_payload(trial) for trial in ranked_screen]
        manifest["ranked_final_results"] = [trial_payload(trial) for trial in ranked_final]
        manifest["top_result"] = trial_payload(ranked_final[0]) if ranked_final else None
        manifest["best_result"] = trial_payload(winner) if winner else None
        manifest["promotion_assessment"] = assessment
        if winner:
            assert winner.final_evaluation is not None
            reporter.notice(
                f"winner run={winner.run_name} lr={winner.hyperparameters.learning_rate:g} "
                f"entropy={winner.hyperparameters.entropy_coef:g} anchor={winner.hyperparameters.anchor_kl_coef:g} "
                f"win_rate={winner.final_evaluation.win_rate:.4f} "
                f"lower95={winner.final_evaluation.confidence_low:.4f} "
                f"promotion_confident={assessment['promotion_confident']}"
            )
        elif ranked_final:
            top = ranked_final[0]
            assert top.final_evaluation is not None
            reporter.notice(
                f"no winner; top run={top.run_name} win_rate={top.final_evaluation.win_rate:.4f} "
                f"lower95={top.final_evaluation.confidence_low:.4f}"
            )
        else:
            reporter.notice("no safe finalist completed")
        progress_writer.update(force=True)
    except BaseException as exc:
        state.phase = "failed"
        manifest["status"] = "failed"
        manifest["failed_at_unix"] = time.time()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        progress_writer.update(force=True)
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
