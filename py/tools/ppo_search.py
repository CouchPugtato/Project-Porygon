from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_defaults import bool_default, float_default, int_default


DEFAULT_RUNS_ROOT = Path("models") / "runs"
DEFAULT_MATCH_RUNS_ROOT = Path("matches") / "runs"
DEFAULT_SEARCH_ROOT = Path("models") / "search"
DEFAULT_TRAINER_EXE = Path("build-fresh") / "showdown_client.exe"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ppo_search.toml"


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


def run_command(command: list[str], cwd: Path, env_updates: dict[str, str] | None = None) -> None:
    log("exec: " + " ".join(f'"{part}"' if " " in part else part for part in command))
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    completed = subprocess.run(command, cwd=str(cwd), env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}")


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
    anchor_kl_mean: float
    anchor_kl_max: float
    clip_fraction: float
    labels: int
    episode_count: int
    available_episode_count: int
    training_reused: bool
    screen_evaluation: EvaluationResult | None = None
    final_evaluation: EvaluationResult | None = None


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
) -> EvaluationResult:
    total_games = 0
    total_wins = 0
    total_earned_wins = 0
    collapse_flags: list[str] = []
    run_names: list[str] = []
    candidate_checkpoint = Path(trial.checkpoint_path)
    for side_index, side in enumerate(("a", "b")):
        run_name = f"{trial.run_name}_{stage}_side_{side}_{games_per_side}"
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
            ), repo_root)
        summary = load_json(summary_path)
        if summary.get("status") != "completed":
            raise RuntimeError(f"evaluation did not complete: {run_name}")
        group = ((summary.get("group_stats", {}) or {}).get(side, {}) or {})
        games = int(group.get("matches_played", 0) or 0)
        total_games += games
        total_wins += int(group.get("wins", 0) or 0)
        total_earned_wins += int(group.get("earned_wins", 0) or 0)
        side_flags = ((summary.get("group_collapse_flags", {}) or {}).get(side, []) or [])
        collapse_flags.extend(str(flag) for flag in side_flags)
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
        collapse_flags=sorted(set(collapse_flags)),
        run_names=run_names,
    )


def evaluation_rank_key(trial: TrialResult, final: bool = False) -> tuple[int, float, float, float]:
    result = trial.final_evaluation if final else trial.screen_evaluation
    if result is None:
        return (0, 0.0, 0.0, 0.0)
    return (1 if not result.collapse_flags else 0, result.confidence_low, result.win_rate, -trial.anchor_kl_mean)


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
    parser.add_argument("--trainer-exe", default=str(DEFAULT_TRAINER_EXE))
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
    parser.add_argument("--final-games-per-side", type=positive_int, default=1000)
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
    return parser


def parse_search_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    config_args = load_config_args(config_path_from_args(cli_args))
    return build_parser().parse_args(config_args + cli_args)


def main() -> None:
    args = parse_search_args()
    repo_root = resolve_repo_root()
    trainer_exe = resolve_path(repo_root, args.trainer_exe)
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
            "final_games_per_side": args.final_games_per_side,
        },
        "search_seed": args.search_seed,
        "trials": [],
    }
    write_json(manifest_path, manifest)

    trials: list[TrialResult] = []
    trainer_env = {"PORYGON_OMP_THREADS": str(args.omp_threads)} if args.omp_threads > 0 else None
    for index, params in enumerate(combinations, start=1):
        run_name = trial_run_name(args.run_prefix, index, params)
        run_dir = (repo_root / DEFAULT_RUNS_ROOT / run_name).resolve()
        checkpoint_path = run_dir / "candidate.chk"
        summary_path = run_dir / "training_summary.json"
        run_dir.mkdir(parents=True, exist_ok=True)
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
            run_command(build_train_command(
                args, trainer_exe, episode_batch, init_checkpoint, anchor_checkpoint,
                checkpoint_path, summary_path, params,
            ), repo_root, trainer_env)
        trial = collect_training_result(
            run_name, params, checkpoint_path, summary_path, args,
            training_reused=can_resume,
        )
        trials.append(trial)
        manifest["trials"] = [trial_payload(item) for item in trials]
        write_json(manifest_path, manifest)
        log(f"trained {run_name} safe={not trial.safety_flags} labels={trial.labels} approx_kl={trial.approx_kl:.6f}")

    safe_trials = [trial for trial in trials if not trial.safety_flags]
    safe_trials.sort(key=lambda trial: training_screen_key(trial, args.target_kl), reverse=True)
    screen_trials = safe_trials[: args.screen_candidates]
    for index, trial in enumerate(screen_trials, start=1):
        trial.screen_evaluation = run_balanced_evaluation(
            args, repo_root, trial, parent_checkpoint, "screen",
            args.screen_games_per_side, args.search_seed * 1000 + index * 10,
        )
        manifest["trials"] = [trial_payload(item) for item in trials]
        write_json(manifest_path, manifest)
        log(
            f"screened {trial.run_name} win_rate={trial.screen_evaluation.win_rate:.4f} "
            f"lower95={trial.screen_evaluation.confidence_low:.4f}"
        )

    ranked_screen = sorted(screen_trials, key=evaluation_rank_key, reverse=True)
    finalists = [
        trial for trial in ranked_screen
        if trial.screen_evaluation and not trial.screen_evaluation.collapse_flags
    ][: args.finalists]
    for index, trial in enumerate(finalists, start=1):
        trial.final_evaluation = run_balanced_evaluation(
            args, repo_root, trial, parent_checkpoint, "final",
            args.final_games_per_side, args.search_seed * 100000 + index * 10,
        )
        manifest["trials"] = [trial_payload(item) for item in trials]
        write_json(manifest_path, manifest)
        log(
            f"finalized {trial.run_name} win_rate={trial.final_evaluation.win_rate:.4f} "
            f"lower95={trial.final_evaluation.confidence_low:.4f}"
        )

    ranked_final = sorted(finalists, key=lambda trial: evaluation_rank_key(trial, final=True), reverse=True)
    manifest["status"] = "completed"
    manifest["completed_at_unix"] = time.time()
    manifest["ranked_screen_results"] = [trial_payload(trial) for trial in ranked_screen]
    manifest["ranked_final_results"] = [trial_payload(trial) for trial in ranked_final]
    manifest["best_result"] = trial_payload(ranked_final[0]) if ranked_final else None
    write_json(manifest_path, manifest)
    if ranked_final:
        best = ranked_final[0]
        assert best.final_evaluation is not None
        log(
            f"best run={best.run_name} lr={best.hyperparameters.learning_rate:g} "
            f"entropy={best.hyperparameters.entropy_coef:g} anchor={best.hyperparameters.anchor_kl_coef:g} "
            f"win_rate={best.final_evaluation.win_rate:.4f} lower95={best.final_evaluation.confidence_low:.4f}"
        )
    else:
        log("no safe finalist completed")


if __name__ == "__main__":
    main()
