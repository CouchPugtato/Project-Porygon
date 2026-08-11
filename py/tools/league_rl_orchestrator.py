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
    find_member,
    league_member_from_json,
    load_registry,
    member_to_json_dict,
    save_registry,
)


DEFAULT_RUNS_ROOT = Path("models") / "runs"
DEFAULT_LEAGUE_ROOT = Path("models") / "league"


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


def clamp_win_rate(value: float | None) -> float:
    if value is None:
        return 0.5
    return min(0.95, max(0.05, float(value)))


def member_reference_win_rate(member: LeagueMember) -> float:
    if member.eval.vs_champion_win_rate is not None:
        return clamp_win_rate(member.eval.vs_champion_win_rate)
    if member.eval.vs_random_win_rate is not None:
        return clamp_win_rate(member.eval.vs_random_win_rate)
    return 0.5


def pfs_weight(member: LeagueMember) -> float:
    win_rate = member_reference_win_rate(member)
    return (1.0 - win_rate) ** 2


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


def build_weighted_pool(registry: LeagueRegistry, learner_role: str, parent_id: str, include_random_weight: float) -> dict[str, object]:
    members: list[dict[str, object]] = []
    parent_member = find_member(registry, parent_id)
    champion = find_member(registry, registry.champion_id) if registry.champion_id else None
    historical = [m for m in registry.members if m.role == "historical_snapshot" and m.status == "active"]
    recent_main = [m for m in registry.members if m.role == "main" and m.status in {"active", "candidate"} and m.id != parent_id]
    main_exploiters = [m for m in registry.members if m.role == "main_exploiter" and m.status in {"active", "candidate"}]
    league_exploiters = [m for m in registry.members if m.role == "league_exploiter" and m.status in {"active", "candidate"}]

    bucket_members: list[tuple[float, list[LeagueMember], str]] = []
    if learner_role == "main":
        bucket_members.extend([
            (0.40, [champion] if champion and champion.id != parent_id else [], "uniform"),
            (0.25, sort_recent(recent_main)[:4], "uniform"),
            (0.20, historical, "pfsp"),
            (0.10, sort_recent(main_exploiters)[:3], "uniform"),
            (0.05, sort_recent(league_exploiters)[:3], "uniform"),
        ])
    elif learner_role == "main_exploiter":
        targets = [parent_member]
        bucket_members.extend([
            (0.60, targets, "uniform"),
            (0.20, sort_recent(recent_main)[:4], "uniform"),
            (0.20, [champion] if champion else [], "uniform"),
        ])
    elif learner_role == "league_exploiter":
        bucket_members.extend([
            (0.50, [champion] if champion else [], "uniform"),
            (0.30, historical, "pfsp"),
            (0.20, sort_recent(recent_main)[:4], "uniform"),
        ])
    else:
        bucket_members.append((1.0, [champion] if champion else [parent_member], "uniform"))

    combined: dict[str, dict[str, object]] = {}
    for bucket_weight, bucket, mode in bucket_members:
        filtered = [member for member in bucket if member is not None and member.id != parent_id]
        if not filtered or bucket_weight <= 0.0:
            continue
        raw_weights: list[float] = []
        for member in filtered:
            raw_weights.append(pfs_weight(member) if mode == "pfsp" else 1.0)
        total = sum(raw_weights)
        if total <= 0.0:
            continue
        for member, raw_weight in zip(filtered, raw_weights):
            weight = bucket_weight * (raw_weight / total)
            existing = combined.get(member.id)
            if existing is None:
                combined[member.id] = {
                    "name": member.id,
                    "kind": "checkpoint",
                    "path": member.path,
                    "weight": weight,
                }
            else:
                existing["weight"] = float(existing["weight"]) + weight

    members = sorted(combined.values(), key=lambda item: (-float(item["weight"]), str(item["name"])))
    if include_random_weight > 0.0:
        members.append({"name": "random", "kind": "random", "weight": include_random_weight})
    if not members:
        members.append({"name": champion.id if champion else parent_member.id, "kind": "checkpoint", "path": (champion or parent_member).path, "weight": 1.0})
    return {"members": members}


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def create_member_from_run(args: argparse.Namespace, registry: LeagueRegistry, candidate_checkpoint: Path, parent_member: LeagueMember, eval_summary_path: Path | None, eval_summary: dict[str, object] | None, collapse_flags: list[str]) -> LeagueMember:
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
    tera_rate = float(side_a.get("tera_rate", 0.0) or 0.0)
    champion = find_member(registry, registry.champion_id) if registry.champion_id else None
    champion_tera_baseline = 1.0
    if champion and champion.eval.summary_path:
        try:
            champion_eval = load_json(Path(champion.eval.summary_path))
            champion_stats = (champion_eval.get("group_stats", {}) or {}).get("a", {}) or {}
            champion_tera_baseline = float(champion_stats.get("tera_rate", 1.0) or 1.0)
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
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.0003)
    parser.add_argument("--advantage-norm", type=parse_bool, default=True)
    parser.add_argument("--reward-mode", choices=["terminal", "dense_additive"], default="terminal")
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.35)
    parser.add_argument("--resource-check-seconds", type=float, default=2.0)
    parser.add_argument("--min-available-memory-gb", type=float, default=3.0)
    parser.add_argument("--min-available-pagefile-gb", type=float, default=6.0)
    parser.add_argument("--stop-on-collapse", type=parse_bool, default=True)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--include-random-weight", type=float, default=0.0)
    parser.add_argument("--eval-games", type=positive_int, default=500)
    parser.add_argument("--eval-concurrent-games", type=positive_int, default=40)
    parser.add_argument("--eval-worker-pairs", type=positive_int, default=120)
    parser.add_argument("--startup-timeout-seconds", type=positive_int, default=120)
    parser.add_argument("--promote-threshold", type=float, default=0.52)
    parser.add_argument("--min-promotion-tera-ratio", type=float, default=0.60)
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

    pool_payload = build_weighted_pool(registry, args.learner_role, parent_member.id, args.include_random_weight)
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
    else:
        candidate = create_member_from_run(args, registry, candidate_checkpoint, parent_member, eval_summary_path, eval_summary, collapse_flags)
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
    workflow_manifest["promoted"] = bool(registry.champion_id == candidate.id)
    write_json(workflow_manifest_path, workflow_manifest)


if __name__ == "__main__":
    main()
