from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from league_manage import (
    DEFAULT_REGISTRY_PATH,
    LeagueMember,
    LeagueRegistry,
    OpponentStats,
    find_member,
    league_member_from_json,
    load_registry,
    member_to_json_dict,
    save_registry,
)
from opponent_sampling import ADAPTIVE_STRATEGY, matchup_difficulty_weight as calculate_matchup_difficulty_weight
from rl_defaults import bool_default, float_default, int_default


DEFAULT_RUNS_ROOT = Path("models") / "runs"
DEFAULT_LEAGUE_ROOT = Path("models") / "league"
DEFAULT_MATCHUP_TARGET_WIN_RATE = float_default("league_matchup_target_win_rate")
DEFAULT_MATCHUP_MIN_WEIGHT = float_default("league_matchup_min_weight")
DEFAULT_MATCHUP_CONFIDENCE_GAMES = int_default("league_matchup_confidence_games")
DEFAULT_MIN_CATEGORY_STARTS = int_default("league_min_category_starts")


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


def positive_unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be greater than 0 and at most 1")
    return parsed


def target_win_rate_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be strictly between 0 and 1")
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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def log(message: str) -> None:
    print(f"[league-rl] {message}", flush=True)


def run_command(command: list[str], cwd: Path) -> None:
    log("exec: " + " ".join(f'"{part}"' if " " in part else part for part in command))
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def registry_has_member(registry: LeagueRegistry, member_id: str) -> bool:
    return any(member.id == member_id for member in registry.members)


def utc_now_unix() -> float:
    return time.time()


def matchup_reference(
    learner: LeagueMember,
    opponent: LeagueMember,
    champion_id: str,
    confidence_games: int,
) -> tuple[float | None, int]:
    stats = learner.opponent_stats.get(opponent.id)
    if stats is not None and stats.matches_played > 0:
        return min(1.0, max(0.0, stats.recent_win_rate)), stats.matches_played
    if opponent.id == champion_id and learner.eval.vs_champion_win_rate is not None:
        return min(1.0, max(0.0, learner.eval.vs_champion_win_rate)), confidence_games
    return None, 0


def matchup_difficulty_weight(
    win_rate: float | None,
    matches_played: int,
    *,
    target_win_rate: float = DEFAULT_MATCHUP_TARGET_WIN_RATE,
    min_weight: float = DEFAULT_MATCHUP_MIN_WEIGHT,
    confidence_games: int = DEFAULT_MATCHUP_CONFIDENCE_GAMES,
) -> float:
    return calculate_matchup_difficulty_weight(
        win_rate,
        matches_played,
        target_win_rate=target_win_rate,
        min_weight=min_weight,
        confidence_games=confidence_games,
    )


def sort_recent(members: list[LeagueMember]) -> list[LeagueMember]:
    return sorted(members, key=lambda item: (item.generation, item.promoted_at or item.created_at, item.id), reverse=True)


def choose_parent_member(registry: LeagueRegistry, parent_id: str, learner_role: str) -> LeagueMember:
    if parent_id:
        return find_member(registry, parent_id)
    if learner_role == "main":
        if registry.champion_id:
            return find_member(registry, registry.champion_id)
    candidates = [member for member in registry.members if member.role == learner_role and member.status in {"active", "candidate"}]
    if candidates:
        return sort_recent(candidates)[0]
    if registry.champion_id:
        return find_member(registry, registry.champion_id)
    raise SystemExit("could not infer parent member; provide --parent-id or initialize the registry with a champion")


def build_weighted_pool(
    registry: LeagueRegistry,
    learner_role: str,
    parent_id: str,
    include_random_weight: float,
    *,
    target_win_rate: float = DEFAULT_MATCHUP_TARGET_WIN_RATE,
    min_difficulty_weight: float = DEFAULT_MATCHUP_MIN_WEIGHT,
    confidence_games: int = DEFAULT_MATCHUP_CONFIDENCE_GAMES,
    min_category_starts: int = DEFAULT_MIN_CATEGORY_STARTS,
) -> dict[str, object]:
    members: list[dict[str, object]] = []
    parent_member = find_member(registry, parent_id)
    champion = find_member(registry, registry.champion_id) if registry.champion_id else None
    historical = [m for m in registry.members if m.role == "historical_snapshot" and m.status == "active"]
    recent_main = [
        m for m in registry.members
        if m.role == "main"
        and m.status in {"active", "candidate"}
        and m.id not in {parent_id, registry.champion_id}
    ]
    main_exploiters = [m for m in registry.members if m.role == "main_exploiter" and m.status in {"active", "candidate"}]
    league_exploiters = [m for m in registry.members if m.role == "league_exploiter" and m.status in {"active", "candidate"}]

    bucket_members: list[tuple[float, list[LeagueMember], str]] = []
    if learner_role == "main":
        bucket_members.extend([
            (0.40, [champion] if champion else [], "champion"),
            (0.25, sort_recent(recent_main)[:4], "recent_main"),
            (0.20, historical, "historical"),
            (0.10, sort_recent(main_exploiters)[:3], "main_exploiter"),
            (0.05, sort_recent(league_exploiters)[:3], "league_exploiter"),
        ])
    elif learner_role == "main_exploiter":
        targets = [parent_member]
        bucket_members.extend([
            (0.60, targets, "exploit_target"),
            (0.20, sort_recent(recent_main)[:4], "recent_main"),
            (0.20, [champion] if champion else [], "champion"),
        ])
    elif learner_role == "league_exploiter":
        bucket_members.extend([
            (0.50, [champion] if champion else [], "champion"),
            (0.30, historical, "historical"),
            (0.20, sort_recent(recent_main)[:4], "recent_main"),
        ])
    else:
        bucket_members.append((1.0, [champion] if champion else [parent_member], "champion"))

    combined: dict[str, dict[str, object]] = {}
    for bucket_weight, bucket, category in bucket_members:
        filtered = [member for member in bucket if member is not None]
        if not filtered or bucket_weight <= 0.0:
            continue
        raw_weights: list[float] = []
        matchup_metadata: list[tuple[float | None, int, float]] = []
        for member in filtered:
            reference_rate, matchup_games = matchup_reference(
                parent_member,
                member,
                registry.champion_id,
                confidence_games,
            )
            difficulty_weight = matchup_difficulty_weight(
                reference_rate,
                matchup_games,
                target_win_rate=target_win_rate,
                min_weight=min_difficulty_weight,
                confidence_games=confidence_games,
            )
            raw_weights.append(member.collection_weight * difficulty_weight)
            matchup_metadata.append((reference_rate, matchup_games, difficulty_weight))
        total = sum(raw_weights)
        if total <= 0.0:
            continue
        for member, raw_weight, metadata in zip(filtered, raw_weights, matchup_metadata):
            reference_rate, matchup_games, difficulty_weight = metadata
            weight = bucket_weight * (raw_weight / total)
            existing = combined.get(member.id)
            if existing is None:
                combined[member.id] = {
                    "name": member.id,
                    "kind": "checkpoint",
                    "path": member.path,
                    "weight": weight,
                    "category": category,
                    "bucket_weight": bucket_weight,
                    "learner_win_rate": reference_rate,
                    "matchup_games": matchup_games,
                    "difficulty_weight": difficulty_weight,
                    "base_weight": member.collection_weight,
                }
            else:
                existing["weight"] = float(existing["weight"]) + weight

    members = sorted(combined.values(), key=lambda item: (-float(item["weight"]), str(item["name"])))
    if include_random_weight > 0.0:
        members.append({
            "name": "random",
            "kind": "random",
            "weight": include_random_weight,
            "category": "random",
        })
    if not members:
        members.append({
            "name": champion.id if champion else parent_member.id,
            "kind": "checkpoint",
            "path": (champion or parent_member).path,
            "weight": 1.0,
            "category": "fallback",
            "bucket_weight": 1.0,
            "difficulty_weight": 1.0,
            "base_weight": 1.0,
            "learner_win_rate": None,
            "matchup_games": 0,
        })
    return {
        "coverage": {
            "enabled": min_category_starts > 0,
            "min_category_starts": min_category_starts,
            "prefer_under_sampled_members": True,
        },
        "sampling": {
            "strategy": ADAPTIVE_STRATEGY,
            "target_win_rate": target_win_rate,
            "min_difficulty_weight": min_difficulty_weight,
            "confidence_games": confidence_games,
        },
        "members": members,
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def opponent_stats_payload(stats_by_opponent: dict[str, OpponentStats]) -> dict[str, dict[str, object]]:
    return {
        opponent_id: {
            "matches_played": stats.matches_played,
            "wins": stats.wins,
            "losses": stats.losses,
            "draws": stats.draws,
            "recent_win_rate": stats.recent_win_rate,
            "source_run": stats.source_run,
        }
        for opponent_id, stats in sorted(stats_by_opponent.items())
    }


def collect_recent_opponent_stats(
    repo_root: Path,
    round_manifest_paths: list[Path],
    source_run: str,
) -> dict[str, OpponentStats]:
    aggregate: dict[str, OpponentStats] = {}
    for manifest_path in round_manifest_paths:
        resolved_manifest = resolve_path(repo_root, manifest_path)
        if not resolved_manifest.exists():
            continue
        round_manifest = load_json(resolved_manifest)
        collection_run = str(round_manifest.get("collection_run", "")).strip()
        if not collection_run:
            continue
        summary_path = (
            repo_root
            / "matches"
            / "runs"
            / collection_run
            / f"{collection_run}_summary.json"
        ).resolve()
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        group_member_stats = summary.get("group_member_stats", {}) or {}
        opponent_members = group_member_stats.get("b", {}) if isinstance(group_member_stats, dict) else {}
        if not isinstance(opponent_members, dict):
            continue
        for opponent_id, raw_stats in opponent_members.items():
            if not isinstance(raw_stats, dict) or not str(opponent_id).strip():
                continue
            opponent_wins = max(0, int(raw_stats.get("wins", 0)))
            opponent_losses = max(0, int(raw_stats.get("losses", 0)))
            draws = max(0, int(raw_stats.get("draws", 0)))
            matches_played = opponent_wins + opponent_losses + draws
            if matches_played <= 0:
                continue
            learner_wins = opponent_losses
            learner_losses = opponent_wins
            opponent_key = str(opponent_id)
            stats = aggregate.setdefault(opponent_key, OpponentStats(source_run=source_run))
            stats.matches_played += matches_played
            stats.wins += learner_wins
            stats.losses += learner_losses
            stats.draws += draws
            stats.recent_win_rate = learner_wins / matches_played
            stats.source_run = source_run
    return aggregate


def build_live_command(args: argparse.Namespace, repo_root: Path, pool_path: Path, parent_checkpoint: Path) -> list[str]:
    command = [
        sys.executable,
        str((repo_root / "py" / "tools" / "live_rl_orchestrator.py").resolve()),
        "--run-name",
        args.run_name,
        "--training-mode",
        args.training_mode,
        "--init-checkpoint",
        str(parent_checkpoint),
        "--rounds",
        str(args.rounds),
        "--games",
        str(args.games),
        "--concurrent-games",
        str(args.concurrent_games),
        "--worker-pairs",
        str(args.worker_pairs),
        "--ensure-shard-count",
        "true" if args.ensure_shard_count else "false",
        "--model-b-pool",
        str(pool_path),
        "--model-b",
        "",
        "--pool-seed",
        str(args.pool_seed),
        "--learning-rate",
        str(args.learning_rate),
        "--gamma",
        str(args.gamma),
        "--entropy-coef",
        str(args.entropy_coef),
        "--advantage-norm",
        "1" if args.advantage_norm else "0",
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
        "true" if args.stop_on_collapse else "false",
        "--omp-threads",
        str(args.omp_threads),
        "--resume",
        "true" if args.resume else "false",
    ]
    return command


def build_eval_command(args: argparse.Namespace, repo_root: Path, candidate_checkpoint: Path, champion_checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        str((repo_root / "py" / "tools" / "selfplay_server.py").resolve()),
        "--run-name",
        args.eval_run_name,
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
        str(candidate_checkpoint),
        "--model-b",
        str(champion_checkpoint),
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


def create_member_from_run(
    args: argparse.Namespace,
    registry: LeagueRegistry,
    candidate_checkpoint: Path,
    parent_member: LeagueMember,
    eval_summary_path: Path | None,
    eval_summary: dict[str, object] | None,
    collapse_flags: list[str],
    opponent_stats: dict[str, OpponentStats],
) -> LeagueMember:
    generation = registry.current_generation + 1
    vs_champion_win_rate = None
    if eval_summary is not None:
        group_stats = eval_summary.get("group_stats", {}) or {}
        side_a = group_stats.get("a", {}) or {}
        vs_champion_win_rate = float(side_a.get("earned_win_rate", 0.0) or 0.0)
    return LeagueMember(
        id=args.member_id or args.run_name,
        path=str(candidate_checkpoint),
        generation=generation,
        status="candidate",
        collection_weight=1.0,
        role=args.learner_role,
        parent_id=parent_member.id,
        exploit_target_id=args.exploit_target_id,
        regime="rl",
        source_run=args.run_name,
        experiment_id=args.experiment_id,
        training_config_id=args.training_config_id,
        snapshot_eligible=True,
        notes=args.notes,
        opponent_stats=opponent_stats,
        eval=league_member_from_json({
            "id": "tmp",
            "path": str(candidate_checkpoint),
            "generation": generation,
            "status": "candidate",
            "collection_weight": 1.0,
            "eval": {
                "vs_champion_win_rate": vs_champion_win_rate,
                "summary_path": str(eval_summary_path) if eval_summary_path else "",
                "collapse_flags": collapse_flags,
            },
        }).eval,
    )


def maybe_promote_candidate(args: argparse.Namespace, registry: LeagueRegistry, candidate: LeagueMember, eval_summary: dict[str, object], collapse_flags: list[str]) -> bool:
    group_stats = eval_summary.get("group_stats", {}) or {}
    side_a = group_stats.get("a", {}) or {}
    earned_win_rate = float(side_a.get("earned_win_rate", 0.0) or 0.0)
    tera_rate = float(side_a.get("tera_battle_rate", side_a.get("tera_rate", 0.0)) or 0.0)
    champion = find_member(registry, registry.champion_id) if registry.champion_id else None
    champion_tera_baseline = 1.0
    if champion and champion.eval.summary_path:
        try:
            champion_eval = load_json(Path(champion.eval.summary_path))
            champion_stats = (champion_eval.get("group_stats", {}) or {}).get("a", {}) or {}
            champion_tera_baseline = float(champion_stats.get("tera_battle_rate", champion_stats.get("tera_rate", 1.0)) or 1.0)
        except Exception:
            champion_tera_baseline = 1.0
    if earned_win_rate < args.promote_threshold:
        return False
    if collapse_flags:
        return False
    if tera_rate < (args.min_promotion_tera_ratio * champion_tera_baseline):
        return False
    candidate.status = "active"
    candidate.role = "champion" if args.learner_role == "main" else candidate.role
    candidate.promoted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    registry.champion_id = candidate.id
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--parent-id", default="")
    parser.add_argument("--learner-role", choices=["main", "main_exploiter", "league_exploiter"], default="main")
    parser.add_argument("--member-id", default="")
    parser.add_argument("--exploit-target-id", default="")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--training-config-id", default="live_ppo_default")
    parser.add_argument("--notes", default="")
    parser.add_argument("--rounds", type=positive_int, default=1)
    parser.add_argument("--games", type=nonnegative_int, required=True)
    parser.add_argument("--concurrent-games", type=positive_int, required=True)
    parser.add_argument("--worker-pairs", type=positive_int, default=60)
    parser.add_argument("--ensure-shard-count", type=parse_bool, default=True)
    parser.add_argument("--pool-seed", type=int, default=1)
    parser.add_argument("--training-mode", choices=["rl", "ppo"], default="ppo")
    parser.add_argument("--learning-rate", type=float, default=float_default("league_ppo_learning_rate"))
    parser.add_argument("--gamma", type=float, default=float_default("ppo_gamma"))
    parser.add_argument("--entropy-coef", type=float, default=float_default("league_ppo_entropy_coef"))
    parser.add_argument("--advantage-norm", type=parse_bool, default=bool_default("advantage_norm"))
    parser.add_argument("--reward-mode", choices=["terminal", "dense_additive"], default="terminal")
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.35)
    parser.add_argument("--resource-check-seconds", type=float, default=2.0)
    parser.add_argument("--min-available-memory-gb", type=float, default=3.0)
    parser.add_argument("--min-available-pagefile-gb", type=float, default=6.0)
    parser.add_argument("--stop-on-collapse", type=parse_bool, default=True)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--include-random-weight", type=float, default=0.0)
    parser.add_argument("--matchup-target-win-rate", type=target_win_rate_float, default=DEFAULT_MATCHUP_TARGET_WIN_RATE)
    parser.add_argument("--matchup-min-weight", type=positive_unit_float, default=DEFAULT_MATCHUP_MIN_WEIGHT)
    parser.add_argument("--matchup-confidence-games", type=positive_int, default=DEFAULT_MATCHUP_CONFIDENCE_GAMES)
    parser.add_argument("--min-category-starts", type=nonnegative_int, default=DEFAULT_MIN_CATEGORY_STARTS)
    parser.add_argument("--eval-games", type=positive_int, default=500)
    parser.add_argument("--eval-concurrent-games", type=positive_int, default=40)
    parser.add_argument("--eval-worker-pairs", type=positive_int, default=120)
    parser.add_argument("--startup-timeout-seconds", type=positive_int, default=120)
    parser.add_argument("--promote-threshold", type=float, default=float_default("promotion_earned_win_rate"))
    parser.add_argument("--min-promotion-tera-ratio", type=float, default=float_default("promotion_min_tera_baseline_ratio"))
    parser.add_argument("--snapshot-cadence", type=positive_int, default=5)
    parser.add_argument("--max-active-historical", type=positive_int, default=20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    repo_root = resolve_repo_root()
    registry_path = resolve_path(repo_root, args.registry)
    registry = load_registry(registry_path)
    parent_member = choose_parent_member(registry, args.parent_id, args.learner_role)

    workflow_dir = (repo_root / DEFAULT_LEAGUE_ROOT / args.run_name).resolve()
    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_manifest_path = workflow_dir / f"{args.run_name}_league_manifest.json"
    pool_path = workflow_dir / f"{args.run_name}_opponent_pool.json"
    workflow_run_name = args.run_name
    args.eval_run_name = f"{args.run_name}_vs_champion_{args.eval_games}"

    pool_payload = build_weighted_pool(
        registry,
        args.learner_role,
        parent_member.id,
        args.include_random_weight,
        target_win_rate=args.matchup_target_win_rate,
        min_difficulty_weight=args.matchup_min_weight,
        confidence_games=args.matchup_confidence_games,
        min_category_starts=args.min_category_starts,
    )
    write_json(pool_path, pool_payload)

    if args.resume and workflow_manifest_path.exists():
        workflow_manifest = load_json(workflow_manifest_path)
        workflow_manifest["status"] = "running"
        workflow_manifest["resumed_at_unix"] = utc_now_unix()
        workflow_manifest["parent_id"] = parent_member.id
        workflow_manifest["parent_checkpoint"] = parent_member.path
        workflow_manifest["pool_path"] = str(pool_path)
        workflow_manifest["pool_payload"] = pool_payload
        log(f"resuming league workflow from {workflow_manifest_path}")
    else:
        workflow_manifest = {
            "run_name": args.run_name,
            "status": "running",
            "started_at_unix": utc_now_unix(),
            "registry": str(registry_path),
            "learner_role": args.learner_role,
            "parent_id": parent_member.id,
            "parent_checkpoint": parent_member.path,
            "pool_path": str(pool_path),
            "pool_payload": pool_payload,
        }
    write_json(workflow_manifest_path, workflow_manifest)

    live_command = build_live_command(args, repo_root, pool_path, Path(parent_member.path))
    workflow_manifest["live_command"] = live_command
    write_json(workflow_manifest_path, workflow_manifest)
    run_command(live_command, repo_root)

    live_manifest_path = (repo_root / DEFAULT_RUNS_ROOT / workflow_run_name / f"{workflow_run_name}_live_rl_manifest.json").resolve()
    live_manifest = load_json(live_manifest_path)
    candidate_checkpoint = Path(str(live_manifest.get("latest_checkpoint", ""))).resolve()
    if not candidate_checkpoint.exists():
        raise SystemExit(f"candidate checkpoint not found after live run: {candidate_checkpoint}")

    round_manifests = [Path(path) for path in live_manifest.get("round_manifests", [])]
    final_round_manifest = load_json(round_manifests[-1]) if round_manifests else {}
    collapse_flags = [str(flag) for flag in final_round_manifest.get("training_round_collapse_flags", [])]
    recent_opponent_stats = collect_recent_opponent_stats(repo_root, round_manifests, args.run_name)
    if recent_opponent_stats:
        parent_member.opponent_stats = recent_opponent_stats

    eval_summary = None
    eval_summary_path = None
    if registry.champion_id:
        champion = find_member(registry, registry.champion_id)
        eval_command = build_eval_command(args, repo_root, candidate_checkpoint, Path(champion.path))
        workflow_manifest["eval_command"] = eval_command
        write_json(workflow_manifest_path, workflow_manifest)
        eval_summary_path = (repo_root / "matches" / "runs" / args.eval_run_name / f"{args.eval_run_name}_summary.json").resolve()
        if not (args.resume and eval_summary_path.exists()):
            run_command(eval_command, repo_root)
        eval_summary = load_json(eval_summary_path)
        collapse_flags.extend([str(flag) for flag in ((eval_summary.get("group_collapse_flags", {}) or {}).get("a", []) or [])])

    candidate_id = args.member_id or args.run_name
    if registry_has_member(registry, candidate_id):
        candidate = find_member(registry, candidate_id)
        if recent_opponent_stats:
            candidate.opponent_stats = recent_opponent_stats
    else:
        candidate = create_member_from_run(
            args,
            registry,
            candidate_checkpoint,
            parent_member,
            eval_summary_path,
            eval_summary,
            collapse_flags,
            recent_opponent_stats,
        )
        registry.members.append(candidate)
        registry.current_generation = max(registry.current_generation, candidate.generation)

    if args.learner_role == "main" and args.rounds >= args.snapshot_cadence:
        snapshot_member = LeagueMember(
            id=f"{candidate.id}_snapshot",
            path=str(candidate_checkpoint),
            generation=candidate.generation,
            status="active",
            collection_weight=1.0,
            role="historical_snapshot",
            parent_id=candidate.id,
            regime="rl",
            source_run=args.run_name,
            experiment_id=args.experiment_id,
            training_config_id=args.training_config_id,
            snapshot_eligible=False,
            opponent_stats=dict(recent_opponent_stats),
        )
        registry.members.append(snapshot_member)

    if eval_summary is not None:
        maybe_promote_candidate(args, registry, candidate, eval_summary, collapse_flags)

    historical_active = [member for member in registry.members if member.role == "historical_snapshot" and member.status == "active"]
    for member in sort_recent(historical_active)[args.max_active_historical:]:
        member.status = "inactive"

    save_registry(registry_path, registry)

    workflow_manifest["status"] = "completed"
    workflow_manifest["completed_at_unix"] = utc_now_unix()
    workflow_manifest["candidate_id"] = candidate.id
    workflow_manifest["candidate_checkpoint"] = str(candidate_checkpoint)
    workflow_manifest["eval_summary_path"] = str(eval_summary_path) if eval_summary_path else ""
    workflow_manifest["post_eval_collapse_flags"] = collapse_flags
    workflow_manifest["recent_opponent_stats"] = opponent_stats_payload(recent_opponent_stats)
    workflow_manifest["promoted"] = bool(registry.champion_id == candidate.id)
    write_json(workflow_manifest_path, workflow_manifest)


if __name__ == "__main__":
    main()
