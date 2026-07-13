from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_RUNS_ROOT = Path("matches") / "runs"


@dataclass
class BattleBuffer:
    battle_id: str
    lines: list[str] = field(default_factory=list)
    source_files: set[str] = field(default_factory=set)
    terminal_result: str | None = None
    reward: float | None = None
    terminal_conflict: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Source run under matches/runs/")
    parser.add_argument("--output-run", required=True, help="Destination filtered run name under matches/runs/")
    parser.add_argument("--side", choices=["a", "b"], default="a")
    return parser


def load_battle_buffers(files: list[Path]) -> dict[str, BattleBuffer]:
    battles: dict[str, BattleBuffer] = {}
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                battle_id = payload.get("battle_id")
                if not battle_id:
                    continue
                battle_id = str(battle_id)
                battle = battles.get(battle_id)
                if battle is None:
                    battle = BattleBuffer(battle_id=battle_id)
                    battles[battle_id] = battle
                battle.lines.append(line if line.endswith("\n") else line + "\n")
                battle.source_files.add(path.name)
                record_type = str(payload.get("type") or payload.get("message_type") or "").strip().lower()
                if record_type == "terminal":
                    result = str(payload.get("result") or "").strip().lower()
                    reward = payload.get("reward")
                    if battle.terminal_result is None:
                        battle.terminal_result = result
                        battle.reward = float(reward) if reward is not None else None
                    elif battle.terminal_result != result:
                        battle.terminal_conflict = True
    return battles


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = repo_root / DEFAULT_RUNS_ROOT / args.run
    output_dir = repo_root / DEFAULT_RUNS_ROOT / args.output_run
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    all_files = sorted(run_dir.glob(f"worker_*_{args.side}_raw.jsonl"))
    if not all_files:
        raise SystemExit(f"no replay files found for side {args.side} in {run_dir}")

    battles = load_battle_buffers(all_files)
    selected_battles = {
        battle_id: battle
        for battle_id, battle in battles.items()
        if not battle.terminal_conflict and battle.terminal_result == "win" and (battle.reward or 0.0) > 0.0
    }
    if not selected_battles:
        raise SystemExit("no winning battles matched the filter")

    output_dir.mkdir(parents=True, exist_ok=True)
    lines_by_file: dict[str, list[str]] = defaultdict(list)
    selected_ids = set(selected_battles.keys())
    for path in all_files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                battle_id = payload.get("battle_id")
                if battle_id and str(battle_id) in selected_ids:
                    lines_by_file[path.name].append(line if line.endswith("\n") else line + "\n")

    written_files = 0
    written_lines = 0
    for filename, lines in sorted(lines_by_file.items()):
        target_path = output_dir / filename
        target_path.write_text("".join(lines), encoding="utf-8")
        written_files += 1
        written_lines += len(lines)

    summary = {
        "source_run": args.run,
        "output_run": args.output_run,
        "side": args.side,
        "selected_battles": len(selected_battles),
        "written_files": written_files,
        "written_lines": written_lines,
        "battle_ids": sorted(selected_ids),
    }
    (output_dir / f"{args.output_run}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[replay_extract_wins] source_run={args.run} output_run={args.output_run} "
        f"side={args.side} selected_battles={len(selected_battles)} written_files={written_files} written_lines={written_lines}",
        flush=True,
    )


if __name__ == "__main__":
    main()
