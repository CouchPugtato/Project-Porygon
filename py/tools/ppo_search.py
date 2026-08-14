from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rl_defaults import bool_default, float_default


DEFAULT_RUNS_ROOT = Path("models") / "runs"
DEFAULT_MATCH_RUNS_ROOT = Path("matches") / "runs"


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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def log(message: str) -> None:
    print(f"[ppo-search] {message}", flush=True)


def run_command(command: list[str], cwd: Path) -> None:
    log("exec: " + " ".join(f'"{part}"' if " " in part else part for part in command))
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass
class TrialResult:
    run_name: str
    learning_rate: float
    entropy_coef: float
    approx_kl: float
    clip_fraction: float
    tera_rate_train: float
    labels: int
    earned_win_rate: float
    completed_games: int
    collapse_flags: list[str]


def build_train_command(args: argparse.Namespace, repo_root: Path, run_name: str, learning_rate: float, entropy_coef: float, pool_seed: int) -> list[str]:
    return [
        sys.executable,
        str((repo_root / "py" / "tools" / "live_rl_orchestrator.py").resolve()),
        "--run-name",
        run_name,
        "--training-mode",
        "ppo",
        "--init-checkpoint",
        str(resolve_path(repo_root, args.init_checkpoint)),
        "--rounds",
        "1",
        "--games",
        str(args.games),
        "--concurrent-games",
        str(args.concurrent_games),
        "--worker-pairs",
        str(args.worker_pairs),
        "--ensure-shard-count",
        "true" if args.ensure_shard_count else "false",
        "--model-b",
        args.model_b,
        "--pool-seed",
        str(pool_seed),
        "--learning-rate",
        str(learning_rate),
        "--gamma",
        str(args.gamma),
        "--entropy-coef",
        str(entropy_coef),
        "--advantage-norm",
        "true" if args.advantage_norm else "false",
        "--reward-mode",
        args.reward_mode,
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
        "--resource-check-seconds",
        str(args.resource_check_seconds),
        "--min-available-memory-gb",
        str(args.min_available_memory_gb),
        "--min-available-pagefile-gb",
        str(args.min_available_pagefile_gb),
        "--stop-on-collapse",
        "true",
        "--omp-threads",
        str(args.omp_threads),
        "--resume",
        "true",
    ]


def build_eval_command(args: argparse.Namespace, repo_root: Path, run_name: str) -> list[str]:
    return [
        sys.executable,
        str((repo_root / "py" / "tools" / "selfplay_server.py").resolve()),
        "--run-name",
        f"{run_name}_vs_g4_{args.eval_games}",
        "--games",
        str(args.eval_games),
        "--concurrent-games",
        str(args.eval_concurrent_games),
        "--worker-pairs",
        str(args.eval_worker_pairs),
        "--ensure-shard-count",
        "false",
        "--model-a-pool",
        "",
        "--model-a",
        str((repo_root / DEFAULT_RUNS_ROOT / run_name / "round01" / "live_rl_round01.chk").resolve()),
        "--model-b",
        str(resolve_path(repo_root, args.eval_model_b)),
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
        "--resource-check-seconds",
        str(args.resource_check_seconds),
        "--min-available-memory-gb",
        str(args.min_available_memory_gb),
        "--min-available-pagefile-gb",
        str(args.min_available_pagefile_gb),
        "--startup-timeout-seconds",
        str(args.startup_timeout_seconds),
    ]


def collect_trial_result(repo_root: Path, run_name: str, learning_rate: float, entropy_coef: float, eval_games: int) -> TrialResult:
    training_summary = load_json((repo_root / DEFAULT_RUNS_ROOT / run_name / "round01" / f"{run_name}_round01_training_summary.json").resolve())
    eval_summary = load_json((repo_root / DEFAULT_MATCH_RUNS_ROOT / f"{run_name}_vs_g4_{eval_games}" / f"{run_name}_vs_g4_{eval_games}_summary.json").resolve())
    group_stats = eval_summary.get("group_stats", {}) or {}
    side_a = group_stats.get("a", {}) or {}
    collapse = ((eval_summary.get("group_collapse_flags", {}) or {}).get("a", []) or [])
    return TrialResult(
        run_name=run_name,
        learning_rate=learning_rate,
        entropy_coef=entropy_coef,
        approx_kl=float(training_summary.get("approx_kl", 0.0) or 0.0),
        clip_fraction=float(training_summary.get("clip_fraction", 0.0) or 0.0),
        tera_rate_train=float(training_summary.get("tera_action_rate", training_summary.get("tera_rate", 0.0)) or 0.0),
        labels=int(training_summary.get("labels", 0) or 0),
        earned_win_rate=float(side_a.get("earned_win_rate", 0.0) or 0.0),
        completed_games=int(eval_summary.get("completed_games", 0) or 0),
        collapse_flags=[str(flag) for flag in collapse],
    )


def rank_key(result: TrialResult) -> tuple[float, float, float]:
    collapse_penalty = 0.0 if not result.collapse_flags else -1.0
    return (collapse_penalty, result.earned_win_rate, -result.approx_kl)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--eval-model-b", required=True)
    parser.add_argument("--games", type=positive_int, default=300)
    parser.add_argument("--concurrent-games", type=positive_int, default=60)
    parser.add_argument("--worker-pairs", type=positive_int, default=60)
    parser.add_argument("--ensure-shard-count", type=parse_bool, default=True)
    parser.add_argument("--model-b", default="random")
    parser.add_argument("--gamma", type=float, default=float_default("ppo_gamma"))
    parser.add_argument("--advantage-norm", type=parse_bool, default=bool_default("advantage_norm"))
    parser.add_argument("--reward-mode", choices=["terminal", "dense_additive"], default="terminal")
    parser.add_argument("--learning-rates", type=csv_floats, default=[5e-5, 1e-4, 2e-4])
    parser.add_argument("--entropy-coefs", type=csv_floats, default=[1e-4, 3e-4, 5e-4])
    parser.add_argument("--max-trials", type=positive_int, default=9)
    parser.add_argument("--eval-games", type=positive_int, default=500)
    parser.add_argument("--eval-concurrent-games", type=positive_int, default=60)
    parser.add_argument("--eval-worker-pairs", type=positive_int, default=120)
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.35)
    parser.add_argument("--resource-check-seconds", type=float, default=2.0)
    parser.add_argument("--min-available-memory-gb", type=float, default=3.0)
    parser.add_argument("--min-available-pagefile-gb", type=float, default=6.0)
    parser.add_argument("--startup-timeout-seconds", type=positive_int, default=120)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--prune-approx-kl", type=float, default=float_default("ppo_search_prune_approx_kl"))
    parser.add_argument(
        "--prune-train-tera-action-rate-below",
        "--prune-train-tera-rate-below",
        dest="prune_train_tera_action_rate_below",
        type=float,
        default=float_default("ppo_search_prune_tera_action_rate"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = resolve_repo_root()
    search_dir = (repo_root / "models" / "search" / args.run_prefix).resolve()
    search_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = search_dir / f"{args.run_prefix}_search_manifest.json"

    trials: list[TrialResult] = []
    grid = list(itertools.product(args.learning_rates, args.entropy_coefs))[: args.max_trials]
    manifest: dict[str, object] = {
        "run_prefix": args.run_prefix,
        "started_at_unix": time.time(),
        "grid": [{"learning_rate": lr, "entropy_coef": ent} for lr, ent in grid],
        "trials": [],
    }
    write_json(manifest_path, manifest)

    for index, (learning_rate, entropy_coef) in enumerate(grid, start=1):
        run_name = f"{args.run_prefix}_{index:02d}_lr{learning_rate:g}_ent{entropy_coef:g}".replace(".", "p")
        run_command(build_train_command(args, repo_root, run_name, learning_rate, entropy_coef, 100 + index), repo_root)
        training_summary = load_json((repo_root / DEFAULT_RUNS_ROOT / run_name / "round01" / f"{run_name}_round01_training_summary.json").resolve())
        approx_kl = float(training_summary.get("approx_kl", 0.0) or 0.0)
        tera_rate_train = float(training_summary.get("tera_action_rate", training_summary.get("tera_rate", 0.0)) or 0.0)
        pruned = approx_kl > args.prune_approx_kl or tera_rate_train < args.prune_train_tera_action_rate_below
        trial_payload: dict[str, object] = {
            "run_name": run_name,
            "learning_rate": learning_rate,
            "entropy_coef": entropy_coef,
            "approx_kl": approx_kl,
            "tera_rate_train": tera_rate_train,
            "tera_action_rate_train": tera_rate_train,
            "pruned": pruned,
        }
        if not pruned:
            run_command(build_eval_command(args, repo_root, run_name), repo_root)
            result = collect_trial_result(repo_root, run_name, learning_rate, entropy_coef, args.eval_games)
            trials.append(result)
            trial_payload.update({
                "earned_win_rate": result.earned_win_rate,
                "completed_games": result.completed_games,
                "clip_fraction": result.clip_fraction,
                "collapse_flags": result.collapse_flags,
            })
        manifest["trials"].append(trial_payload)
        write_json(manifest_path, manifest)

    ranked = sorted(trials, key=rank_key, reverse=True)
    manifest["completed_at_unix"] = time.time()
    manifest["ranked_results"] = [result.__dict__ for result in ranked]
    manifest["best_result"] = ranked[0].__dict__ if ranked else None
    write_json(manifest_path, manifest)

    if ranked:
        best = ranked[0]
        log(
            f"best run={best.run_name} lr={best.learning_rate:g} entropy={best.entropy_coef:g} "
            f"earned_win_rate={best.earned_win_rate:.4f} approx_kl={best.approx_kl:.4f}"
        )
    else:
        log("no unpruned trials completed")


if __name__ == "__main__":
    main()
