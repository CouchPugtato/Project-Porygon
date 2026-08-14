from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rl_defaults import bool_default, float_default, int_default, reward_float_default


DEFAULT_ARGS_PATH = Path("config/live_rl_orchestrator.toml")
DEFAULT_TRAINER_EXE = Path("build-fresh") / "showdown_client.exe"
DEFAULT_RUNS_ROOT = Path("matches") / "runs"
DEFAULT_MODELS_ROOT = Path("models") / "runs"


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


def run_command(command: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    log("exec: " + " ".join(f'"{part}"' if " " in part else part for part in command))
    env = None
    if extra_env:
        env = dict(**extra_env)
        merged = dict(os.environ)
        merged.update(env)
        env = merged
    completed = subprocess.run(command, cwd=str(cwd), env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


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


def build_selfplay_command(args: argparse.Namespace, repo_root: Path, run_name: str, checkpoint_path: Path) -> list[str]:
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
    if args.model_b_pool:
        command.extend(["--model-b", "", "--model-b-pool", args.model_b_pool])
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
    return output_checkpoint.exists() and training_summary.exists()


def trainer_env(args: argparse.Namespace) -> dict[str, str] | None:
    if args.omp_threads <= 0:
        return None
    return {"PORYGON_OMP_THREADS": str(args.omp_threads)}


def collapse_flags_from_training_summary(summary: dict[str, object], baseline_tera_rate: float, min_episodes_warn: int, anchor_kl_warn_threshold: float) -> list[str]:
    flags: list[str] = []
    episode_count = int(summary.get("episode_count", 0) or 0)
    tera_rate = float(summary.get("tera_action_rate", summary.get("tera_rate", 0.0)) or 0.0)
    move_slot_rates = summary.get("move_slot_rates", {}) or {}
    anchor_kl_mean = float(summary.get("anchor_kl_mean", 0.0) or 0.0)
    if episode_count < min_episodes_warn:
        flags.append(f"warn_low_episode_count:{episode_count}")
    if baseline_tera_rate > 0.0 and tera_rate < (float_default("warn_tera_baseline_ratio") * baseline_tera_rate):
        flags.append(f"warn_tera_rate_low:{tera_rate:.3f}:{baseline_tera_rate:.3f}")
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
            "opponent": {
                "model_b": args.model_b,
                "model_b_pool": args.model_b_pool,
            },
            "round_manifests": [],
        }
    write_json(workflow_manifest_path, workflow_manifest)

    completed_round_paths = [Path(path) for path in workflow_manifest.get("round_manifests", []) if round_manifest_completed(Path(path))]
    workflow_manifest["round_manifests"] = [str(path) for path in completed_round_paths]
    current_checkpoint = Path(str(workflow_manifest.get("latest_checkpoint", init_checkpoint))).resolve()
    if not completed_round_paths or not current_checkpoint.exists():
        current_checkpoint = init_checkpoint
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
                if str(round_manifest_path) not in workflow_manifest["round_manifests"]:
                    workflow_manifest["round_manifests"].append(str(round_manifest_path))
                workflow_manifest["latest_checkpoint"] = str(current_checkpoint)
                write_json(workflow_manifest_path, workflow_manifest)
                log(f"skipping completed {round_token}")
                continue

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
            }
            write_json(round_manifest_path, round_manifest)

            selfplay_command = build_selfplay_command(args, repo_root, collect_run_name, current_checkpoint)
            round_manifest["selfplay_command"] = selfplay_command
            write_json(round_manifest_path, round_manifest)
            run_command(selfplay_command, repo_root)

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
            run_command(train_command, repo_root, extra_env=trainer_env(args))

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
            if args.stop_on_collapse and any(flag.startswith("hard_move_slot_collapse") for flag in collapse_flags):
                workflow_manifest["status"] = "stopped_on_collapse"
                workflow_manifest["completed_at_unix"] = time.time()
                workflow_manifest["latest_checkpoint"] = str(current_checkpoint)
                workflow_manifest["stop_reason"] = collapse_flags
                write_json(workflow_manifest_path, workflow_manifest)
                return

        workflow_manifest["status"] = "completed"
        workflow_manifest["completed_at_unix"] = time.time()
        workflow_manifest["latest_checkpoint"] = str(current_checkpoint)
        write_json(workflow_manifest_path, workflow_manifest)
    except Exception as exc:
        workflow_manifest["status"] = "failed"
        workflow_manifest["failure_reason"] = str(exc)
        workflow_manifest["completed_at_unix"] = time.time()
        write_json(workflow_manifest_path, workflow_manifest)
        raise


if __name__ == "__main__":
    main()
