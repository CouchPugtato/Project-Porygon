from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run name under matches/runs/")
    parser.add_argument("--output", default="", help="Output JSONL path; defaults to <run-dir>/combined_raw.jsonl")
    parser.add_argument("--pattern", default="worker_*_raw.jsonl", help="Glob for input replay files")
    return parser


def load_battles(path: Path) -> list[list[str]]:
    battles: list[list[str]] = []
    current_battle: list[str] = []
    current_battle_id: str | None = None

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            record = json.loads(line)
            record_type = str(record.get("type", "")).strip()
            battle_id = str(record.get("battle_id", "")).strip()

            if record_type == "battle_start":
                if current_battle:
                    raise ValueError(f"{path}: encountered battle_start before previous battle ended")
                current_battle = [line]
                current_battle_id = battle_id
                continue

            if not current_battle:
                raise ValueError(f"{path}: record outside battle block: {record_type}")

            if battle_id != current_battle_id:
                raise ValueError(
                    f"{path}: battle_id changed inside block from {current_battle_id} to {battle_id}"
                )

            current_battle.append(line)
            if record_type == "battle_end":
                battles.append(current_battle)
                current_battle = []
                current_battle_id = None

    if current_battle:
        raise ValueError(f"{path}: file ended before battle_end for {current_battle_id}")

    return battles


def combine_battle_streams(per_file_battles: list[list[list[str]]], rng: random.Random) -> list[str]:
    queues = [list(battles) for battles in per_file_battles if battles]
    combined_lines: list[str] = []
    while queues:
        queue_index = rng.randrange(len(queues))
        combined_lines.extend(queues[queue_index].pop(0))
        if not queues[queue_index]:
            queues.pop(queue_index)
    return combined_lines


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = Path("matches") / "runs" / args.run
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    output_path = Path(args.output) if args.output else run_dir / "combined_raw.jsonl"
    input_paths = sorted(run_dir.glob(args.pattern))
    if not input_paths:
        raise SystemExit(f"no input files matched {args.pattern!r} in {run_dir}")

    rng = random.Random()
    per_file_battles = [load_battles(path) for path in input_paths]
    combined_lines = combine_battle_streams(per_file_battles, rng)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(combined_lines) + ("\n" if combined_lines else ""), encoding="utf-8")

    print(f"[combine_match_jsons] wrote {len(combined_lines)} records to {output_path}")
    print(
        f"[combine_match_jsons] inputs={len(input_paths)} total_battles={sum(len(battles) for battles in per_file_battles)}"
    )


if __name__ == "__main__":
    main()
