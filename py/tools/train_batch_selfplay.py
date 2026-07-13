from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_RUNS_ROOT = Path("matches") / "runs"
DEFAULT_TRAINER_EXE = Path("build-fresh") / "showdown_client.exe"
DEFAULT_ARGS_PATH = Path("config/train_batch_selfplay.toml")


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


def has_path_separator(path: str) -> bool:
    return "/" in path or "\\" in path


def resolve_batch_checkpoint(run_name: str, checkpoint_arg: str) -> Path:
    checkpoint_path = Path(checkpoint_arg)
    if has_path_separator(checkpoint_arg):
        return checkpoint_path
    stem = checkpoint_path.stem or checkpoint_path.name
    return Path("models") / "runs" / run_name / stem / checkpoint_path.name


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
        f"checkpoint={args.resolved_checkpoint}",
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
            completed = subprocess.run(command, cwd=str(repo_root))
            if completed.returncode != 0:
                raise SystemExit(
                    f"[train_batch_selfplay] shard failed epoch={epoch} shard={shard_index} "
                    f"path={replay_path} exit_code={completed.returncode}"
                )
            completed_files += 1
            shard_elapsed = time.monotonic() - shard_started_at
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
