from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_RUNS_ROOT = Path("matches") / "runs"
DEFAULT_TRAINER_EXE = Path("build-fresh") / "showdown_client.exe"
DEFAULT_ARGS_PATH = Path("config/train_batch_selfplay.toml")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


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


def batch_stats_dir(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / f"{checkpoint_path.stem}_batch_training_stats"


def batch_stats_path(
    checkpoint_path: Path,
    epoch: int,
    shard_index: int,
    replay_path: Path,
) -> Path:
    safe_replay = replay_path.stem.replace(" ", "_")
    return batch_stats_dir(checkpoint_path) / (
        f"epoch{epoch:03d}_shard{shard_index:04d}_{safe_replay}_training_stats.json"
    )


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
        elif line.startswith("[train] split "):
            stats["split"] = parse_key_values(line)
        elif line.startswith("[train] epoch=") and "supervised_profile" in line:
            profiles = stats.setdefault("supervised_profile", [])
            if isinstance(profiles, list):
                profiles.append(parse_key_values(line))
        elif line.startswith("[train] epoch=") and " validation action_loss=" in line:
            stats["validation_summary"] = parse_key_values(line)
        elif line.startswith("[train] epoch=") and " validation top3_accuracy=" in line:
            stats["validation_breakdown"] = parse_key_values(line)
        elif line.startswith("[train] epoch=") and " validation elapsed=" in line:
            stats["validation_timing"] = parse_key_values(line)
        elif line.startswith("trained mode="):
            stats["final_train"] = parse_key_values(line)
        elif line.startswith("[train] saved checkpoint "):
            stats["saved_checkpoint"] = line.removeprefix("[train] saved checkpoint ").strip()
    return stats


def run_trainer_and_capture(command: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, list[str]]:
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
    for line in process.stdout:
        print(line, end="")
        captured_lines.append(line.rstrip("\r\n"))
    return_code = process.wait()
    return return_code, captured_lines


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
        "stats": parsed_stats,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run name under matches/runs/")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path/name to train into")
    parser.add_argument("--mode", choices=["supervised", "rl"], default="supervised")
    parser.add_argument("--epochs", type=positive_int, default=1, help="Passes over all worker files")
    parser.add_argument("--pattern", default="worker_*_raw.jsonl", help="Glob for training shards")
    parser.add_argument("--trainer-exe", default=str(DEFAULT_TRAINER_EXE))
    parser.add_argument("--shuffle", type=parse_bool01, default=True, help="Shuffle shard order each epoch")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N matched files")
    parser.add_argument("--limit-files", type=positive_int, default=0, help="Optional cap on matched files")
    parser.add_argument("--sample-files", type=positive_int, default=0, help="Train on a random subset of up to N shards per epoch")
    parser.add_argument("--epochs-per-file", type=positive_int, default=1, help="Epochs to pass to showdown_client for each shard")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
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
    if args.mode == "rl":
        command.extend(
            [
                "--gamma",
                str(args.gamma),
                "--entropy-coef",
                str(args.entropy_coef),
                "--advantage-norm",
                args.advantage_norm,
                "--reward-mode",
                args.reward_mode,
            ]
        )
    return command


def main() -> None:
    parser = build_parser()
    argv = load_default_args(DEFAULT_ARGS_PATH) + sys.argv[1:]
    configured_env = load_default_env(DEFAULT_ARGS_PATH)
    args = parser.parse_args(argv)
    repo_root = Path.cwd()
    run_dir = repo_root / DEFAULT_RUNS_ROOT / args.run
    trainer_exe = repo_root / Path(args.trainer_exe)
    args.resolved_checkpoint = resolve_batch_checkpoint(args.run, args.checkpoint)

    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")
    if not trainer_exe.exists():
        raise SystemExit(f"trainer executable not found: {trainer_exe}")
    args.resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    subprocess_env = os.environ.copy()
    subprocess_env.update(configured_env)

    all_files = sorted(run_dir.glob(args.pattern))
    if args.start_index > 0:
        all_files = all_files[args.start_index :]
    if args.limit_files > 0:
        all_files = all_files[: args.limit_files]
    if not all_files:
        raise SystemExit(f"no input files matched {args.pattern!r} in {run_dir}")

    started_at = time.monotonic()
    total_shards = len(all_files)
    print(
        f"[train_batch_selfplay] run={args.run} mode={args.mode} epochs={args.epochs} "
        f"epochs_per_file={args.epochs_per_file} shards={total_shards} "
        f"sample_files={args.sample_files if args.sample_files > 0 else 'all'} "
        f"checkpoint={args.resolved_checkpoint} "
        f"config_env={configured_env if configured_env else '{}'}",
        flush=True,
    )

    completed_files = 0
    shard_rate_ema = 0.0
    eta_alpha = 0.2
    eta_min_elapsed = 30.0
    eta_min_completed = 3
    planned_total_shards = (
        args.epochs * min(total_shards, args.sample_files)
        if args.sample_files > 0
        else args.epochs * total_shards
    )
    for epoch in range(1, args.epochs + 1):
        epoch_files = list(all_files)
        if args.shuffle:
            random.shuffle(epoch_files)
        if args.sample_files > 0 and len(epoch_files) > args.sample_files:
            epoch_files = epoch_files[: args.sample_files]
        epoch_started_at = time.monotonic()
        print(
            f"[train_batch_selfplay] epoch {epoch}/{args.epochs} start shards={len(epoch_files)} "
            f"shuffle={int(args.shuffle)} sample_files={args.sample_files if args.sample_files > 0 else 'all'}",
            flush=True,
        )

        for shard_index, replay_path in enumerate(epoch_files, start=1):
            shard_started_at = time.monotonic()
            command = trainer_command_for_file(args, replay_path)
            print(
                f"[train_batch_selfplay] epoch {epoch}/{args.epochs} shard {shard_index}/{len(epoch_files)} "
                f"path={replay_path.name}",
                flush=True,
            )
            return_code, captured_lines = run_trainer_and_capture(command, repo_root, subprocess_env)
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
            )
            print(f"[train_batch_selfplay] wrote stats {stats_path}", flush=True)
            if return_code != 0:
                raise SystemExit(
                    f"[train_batch_selfplay] shard failed epoch={epoch} shard={shard_index} "
                    f"path={replay_path} exit_code={return_code}"
                )
            completed_files += 1
            total_elapsed = time.monotonic() - started_at
            shard_rate_sample = (completed_files / total_elapsed) if total_elapsed > 0.0 else 0.0
            if shard_rate_sample > 0.0:
                shard_rate_ema = ema_update(shard_rate_ema, shard_rate_sample, eta_alpha)
            shards_per_minute = shard_rate_ema * 60.0 if shard_rate_ema > 0.0 else 0.0
            remaining_shards = planned_total_shards - completed_files
            eta_ready = completed_files >= eta_min_completed or total_elapsed >= eta_min_elapsed
            if eta_ready and shard_rate_ema > 0.0:
                eta_seconds = remaining_shards / shard_rate_ema
                print(
                    f"[train_batch_selfplay] shard complete elapsed={shard_elapsed:.1f}s "
                    f"overall={completed_files}/{planned_total_shards} "
                    f"shards_per_minute={shards_per_minute:.2f} eta_minutes={eta_seconds / 60.0:.1f}",
                    flush=True,
                )
            else:
                print(
                    f"[train_batch_selfplay] shard complete elapsed={shard_elapsed:.1f}s "
                    f"overall={completed_files}/{planned_total_shards} "
                    f"shards_per_minute={shards_per_minute:.2f} eta_minutes=estimating",
                    flush=True,
                )

        epoch_elapsed = time.monotonic() - epoch_started_at
        print(
            f"[train_batch_selfplay] epoch {epoch}/{args.epochs} complete elapsed={epoch_elapsed:.1f}s",
            flush=True,
        )

    total_elapsed = time.monotonic() - started_at
    print(
        f"[train_batch_selfplay] complete files={completed_files} total_elapsed={total_elapsed:.1f}s "
        f"checkpoint={args.resolved_checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
