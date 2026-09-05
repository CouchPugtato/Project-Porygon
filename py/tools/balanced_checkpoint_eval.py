from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_io import write_json_atomically
from league_rl_orchestrator import (
    DEFAULT_PROMOTION_CONFIDENCE_THRESHOLD,
    league_promotion_assessment,
    run_balanced_valid_evaluation,
)
from live_rl_orchestrator import (
    DashboardProgressWriter,
    WorkflowDashboardState,
    select_workflow_reporter,
)
from rl_defaults import float_default, load_cli_defaults


DEFAULT_MATCH_RUNS_ROOT = Path("matches") / "runs"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "balanced_checkpoint_eval.toml"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
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


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def write_json(path: Path, payload: dict[str, object]) -> None:
    write_json_atomically(path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing candidate checkpoint against a baseline on both player sides.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument(
        "--baseline",
        "--baseline-checkpoint",
        "--champion-checkpoint",
        dest="baseline_checkpoint",
        required=True,
        help="Checkpoint path or 'random'",
    )
    parser.add_argument("--games-per-side", type=positive_int, default=250)
    parser.add_argument("--block-games", type=positive_int, default=250)
    parser.add_argument("--concurrent-games", type=positive_int, default=70)
    parser.add_argument("--worker-pairs", type=positive_int, default=125)
    parser.add_argument("--max-replacement-attempts", type=positive_int, default=5)
    parser.add_argument("--pool-seed", type=int, default=1)
    parser.add_argument("--battle-seed-base", type=int, default=None)
    parser.add_argument("--format", default="gen9randomdoublesbattle")
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.25)
    parser.add_argument("--resource-check-seconds", type=float, default=2.0)
    parser.add_argument("--min-available-memory-gb", type=float, default=2.0)
    parser.add_argument("--min-available-pagefile-gb", type=float, default=4.0)
    parser.add_argument("--startup-timeout-seconds", type=positive_int, default=120)
    parser.add_argument("--promotion-min-win-rate", type=float, default=float_default("promotion_earned_win_rate"))
    parser.add_argument(
        "--promotion-confidence-threshold",
        type=float,
        default=DEFAULT_PROMOTION_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--min-promotion-tera-ratio",
        type=float,
        default=float_default("promotion_min_tera_baseline_ratio"),
    )
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--dashboard", type=parse_bool, default=True)
    parser.add_argument("--dashboard-refresh-per-second", type=positive_float, default=8.0)
    parser.add_argument("--dashboard-write-raw-logs", type=parse_bool, default=True)
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    configured, _ = config_parser.parse_known_args(argv)
    config_path = resolve_path(resolve_repo_root(), configured.config)
    parser = build_parser()
    try:
        defaults = load_cli_defaults(config_path)
    except (OSError, RuntimeError) as exc:
        parser.error(str(exc))
    args = parser.parse_args(defaults + argv)
    args.config = str(config_path)
    return args


def prepare_shared_args(args: argparse.Namespace) -> argparse.Namespace:
    """Expose the names consumed by the shared league evaluation implementation."""
    args.eval_run_name = args.run_name
    args.eval_games = args.games_per_side
    args.eval_block_games = args.block_games
    args.eval_concurrent_games = args.concurrent_games
    args.eval_worker_pairs = args.worker_pairs
    args.eval_max_replacement_attempts = args.max_replacement_attempts
    args.eval_battle_seed_base = args.battle_seed_base
    args.promote_threshold = args.promotion_min_win_rate
    return args


def validate_unit_interval(parser: argparse.ArgumentParser, label: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        parser.error(f"{label} must be between 0 and 1")


def main() -> None:
    parser = build_parser()
    args = prepare_shared_args(parse_args(sys.argv[1:]))
    validate_unit_interval(parser, "promotion-min-win-rate", args.promotion_min_win_rate)
    validate_unit_interval(parser, "promotion-confidence-threshold", args.promotion_confidence_threshold)
    validate_unit_interval(parser, "min-promotion-tera-ratio", args.min_promotion_tera_ratio)

    repo_root = resolve_repo_root()
    candidate_checkpoint = resolve_path(repo_root, args.candidate_checkpoint)
    baseline_checkpoint: str | Path
    if args.baseline_checkpoint.strip().lower() == "random":
        baseline_checkpoint = "random"
    else:
        baseline_checkpoint = resolve_path(repo_root, args.baseline_checkpoint)
    if not candidate_checkpoint.exists():
        raise SystemExit(f"candidate checkpoint not found: {candidate_checkpoint}")
    if isinstance(baseline_checkpoint, Path) and not baseline_checkpoint.exists():
        raise SystemExit(f"baseline checkpoint not found: {baseline_checkpoint}")

    run_dir = (repo_root / DEFAULT_MATCH_RUNS_ROOT / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / f"{args.run_name}_manifest.json"
    summary_path = run_dir / f"{args.run_name}_summary.json"
    logs_dir = run_dir / "logs"
    manifest: dict[str, object] = {
        "run_name": args.run_name,
        "config": args.config,
        "status": "running",
        "started_at_unix": time.time(),
        "evaluation_mode": "balanced_valid_games",
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "summary_path": str(summary_path),
        "valid_games_per_side": args.games_per_side,
        "block_games": args.block_games,
        "max_replacement_attempts": args.max_replacement_attempts,
        "pool_seed": args.pool_seed,
        "battle_seed_base": args.battle_seed_base,
        "format": args.format,
        "worker_config": {
            "concurrent_games": args.concurrent_games,
            "worker_pairs": args.worker_pairs,
            "launch_stagger_seconds": args.launch_stagger_seconds,
            "resource_check_seconds": args.resource_check_seconds,
            "min_available_memory_gb": args.min_available_memory_gb,
            "min_available_pagefile_gb": args.min_available_pagefile_gb,
            "startup_timeout_seconds": args.startup_timeout_seconds,
        },
        "promotion_config": {
            "minimum_win_rate": args.promotion_min_win_rate,
            "confidence_threshold": args.promotion_confidence_threshold,
            "minimum_tera_ratio": args.min_promotion_tera_ratio,
        },
        "dashboard_config": {
            "enabled": args.dashboard,
            "refresh_per_second": args.dashboard_refresh_per_second,
            "write_raw_logs": args.dashboard_write_raw_logs,
        },
    }
    write_json(manifest_path, manifest)

    state = WorkflowDashboardState(
        run_name=args.run_name,
        rounds_total=0,
        games_per_round=0,
        evaluation_games_per_side=args.games_per_side,
        title="Balanced Checkpoint Evaluation",
    )
    progress_writer = DashboardProgressWriter(manifest_path, manifest, state)
    reporter = select_workflow_reporter(
        args.dashboard, args.dashboard_refresh_per_second, state, progress_writer,
    )
    reporter.start()
    try:
        combined_path, summary = run_balanced_valid_evaluation(
            args,
            repo_root,
            candidate_checkpoint,
            baseline_checkpoint,
            reporter,
            logs_dir,
        )
        collapse_flags = [str(flag) for flag in (summary.get("candidate_collapse_flags", []) or [])]
        assessment = league_promotion_assessment(args, summary, collapse_flags)
        summary["promotion_assessment"] = assessment
        write_json(combined_path, summary)

        state.collapse_flags = collapse_flags
        state.set_promotion(assessment)
        reporter.notice(f"assessment: {assessment['status']}")
        manifest["status"] = "completed"
        manifest["completed_at_unix"] = time.time()
        manifest["summary_path"] = str(combined_path)
        manifest["run_names"] = list(summary.get("run_names", []) or [])
        manifest["promotion_assessment"] = assessment
        manifest["candidate_collapse_flags"] = collapse_flags
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_at_unix"] = time.time()
        manifest["failure_reason"] = str(exc)
        state.phase = "failed"
        reporter.notice(str(exc))
        write_json(manifest_path, manifest)
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
