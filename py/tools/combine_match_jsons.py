from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import TextIO


STAT_INT_KEYS = {
    "matches_played",
    "wins",
    "earned_wins",
    "losses",
    "draws",
    "max_rating",
    "total_invalid_choices",
    "total_fallbacks",
    "total_accepted_proposals",
    "total_forced_switches",
    "total_voluntary_switches",
    "total_moves",
    "total_protects",
    "total_passes",
    "total_teras",
    "tera_battles",
}

PROGRESS_INTERVAL_BATTLES = 100
PROGRESS_INTERVAL_SECONDS = 10.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run name under matches/runs/")
    parser.add_argument("--output", default="", help="Output JSONL path; defaults to <run-dir>/combined_raw.jsonl")
    parser.add_argument("--pattern", default="worker_*_raw.jsonl", help="Glob for input replay files")
    return parser


class BattleStream:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: TextIO | None = None
        self._finished = False
        self._battle_count = 0

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def battle_count(self) -> int:
        return self._battle_count

    def _ensure_open(self) -> TextIO:
        if self._handle is None:
            self._handle = self.path.open(encoding="utf-8")
        return self._handle

    def next_battle(self) -> list[str] | None:
        if self._finished:
            return None
        handle = self._ensure_open()
        current_battle: list[str] = []
        current_battle_id: str | None = None

        while True:
            raw_line = handle.readline()
            if not raw_line:
                self._finished = True
                if current_battle:
                    raise ValueError(f"{self.path}: file ended before battle_end for {current_battle_id}")
                return None
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            record = json.loads(line)
            record_type = str(record.get("type", "")).strip()
            battle_id = str(record.get("battle_id", "")).strip()

            if not current_battle:
                if record_type != "battle_start":
                    raise ValueError(f"{self.path}: record outside battle block: {record_type}")
                current_battle = [line]
                current_battle_id = battle_id
                continue

            if battle_id != current_battle_id:
                raise ValueError(
                    f"{self.path}: battle_id changed inside block from {current_battle_id} to {battle_id}"
                )

            current_battle.append(line)
            if record_type == "battle_end":
                self._battle_count += 1
                return current_battle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def parse_stats_file(path: Path) -> dict[str, str]:
    stats: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            stats[key.strip()] = value.strip()
    return stats


def combine_stats(run_dir: Path) -> dict[str, str]:
    combined: dict[str, str] = {}
    stats_paths = sorted(run_dir.glob("worker_*_raw.stats.txt"))
    weighted_tera_turns = 0.0
    total_tera_battles = 0

    for path in stats_paths:
        stats = parse_stats_file(path)
        for key in STAT_INT_KEYS:
            if key not in stats:
                continue
            value = int(stats[key])
            if key == "max_rating":
                current = int(combined.get(key, "0"))
                combined[key] = str(max(current, value))
            else:
                combined[key] = str(int(combined.get(key, "0")) + value)
        tera_battles = int(stats.get("tera_battles", "0"))
        avg_turns = float(stats.get("avg_turns_until_tera", "0"))
        weighted_tera_turns += tera_battles * avg_turns
        total_tera_battles += tera_battles

    wins = int(combined.get("wins", "0"))
    losses = int(combined.get("losses", "0"))
    draws = int(combined.get("draws", "0"))
    combined["record"] = f"{wins}-{losses}-{draws}"
    combined["avg_turns_until_tera"] = f"{(weighted_tera_turns / total_tera_battles) if total_tera_battles else 0.0:.3f}"
    return combined


def write_stats_file(path: Path, stats: dict[str, str]) -> None:
    ordered_keys = [
        "matches_played",
        "record",
        "wins",
        "earned_wins",
        "losses",
        "draws",
        "max_rating",
        "total_invalid_choices",
        "total_fallbacks",
        "total_accepted_proposals",
        "total_forced_switches",
        "total_voluntary_switches",
        "total_moves",
        "total_protects",
        "total_passes",
        "total_teras",
        "tera_battles",
        "avg_turns_until_tera",
    ]
    lines = [f"{key}={stats[key]}" for key in ordered_keys if key in stats]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
    streams = [BattleStream(path) for path in input_paths]
    active_streams: list[tuple[BattleStream, list[str]]] = []
    total_records = 0
    total_battles = 0
    discovered_battles = 0
    started_at = time.monotonic()
    last_progress_at = started_at

    try:
        for stream in streams:
            battle = stream.next_battle()
            if battle is not None:
                active_streams.append((stream, battle))
                discovered_battles += 1

        print(
            f"[combine_match_jsons] starting run={args.run} inputs={len(input_paths)} "
            f"active_streams={len(active_streams)} output={output_path}",
            flush=True,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as out_handle:
            while active_streams:
                stream_index = rng.randrange(len(active_streams))
                stream, battle = active_streams[stream_index]
                for line in battle:
                    out_handle.write(line + "\n")
                total_records += len(battle)
                total_battles += 1

                next_battle = stream.next_battle()
                if next_battle is None:
                    active_streams.pop(stream_index)
                else:
                    active_streams[stream_index] = (stream, next_battle)
                    discovered_battles += 1

                now = time.monotonic()
                if (
                    total_battles % PROGRESS_INTERVAL_BATTLES == 0
                    or (now - last_progress_at) >= PROGRESS_INTERVAL_SECONDS
                ):
                    elapsed = max(now - started_at, 1e-9)
                    bpm = (total_battles * 60.0) / elapsed
                    eta_seconds: float | None = None
                    remaining_known_battles = discovered_battles - total_battles
                    if bpm > 0.0 and remaining_known_battles > 0:
                        eta_seconds = (remaining_known_battles * 60.0) / bpm
                    eta_text = "unknown"
                    if eta_seconds is not None:
                        eta_minutes = eta_seconds / 60.0
                        eta_text = f"{eta_minutes:.1f}m"
                    print(
                        f"[combine_match_jsons] progress battles={total_battles} "
                        f"records={total_records} active_streams={len(active_streams)} "
                        f"battles_per_minute={bpm:.1f} "
                        f"known_remaining_battles={remaining_known_battles} eta={eta_text}",
                        flush=True,
                    )
                    last_progress_at = now
    finally:
        for stream in streams:
            stream.close()

    combined_stats = combine_stats(run_dir)
    write_stats_file(output_path.with_suffix(".stats.txt"), combined_stats)

    print(f"[combine_match_jsons] wrote {total_records} records to {output_path}", flush=True)
    print(f"[combine_match_jsons] wrote combined stats to {output_path.with_suffix('.stats.txt')}", flush=True)
    print(f"[combine_match_jsons] inputs={len(input_paths)} total_battles={total_battles}", flush=True)


if __name__ == "__main__":
    main()
