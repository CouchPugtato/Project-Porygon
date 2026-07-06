from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REGISTRY_PATH = Path("models") / "league" / "league_registry.json"
DEFAULT_MAX_ACTIVE_MEMBERS = 5


def parse_bool01(value: str) -> bool:
    if value not in {"0", "1"}:
        raise argparse.ArgumentTypeError("value must be 0 or 1")
    return value == "1"


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


def win_rate_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_registry_path() -> Path:
    return DEFAULT_REGISTRY_PATH


def resolve_checkpoint_path(path_str: str) -> str:
    resolved = Path(path_str).resolve()
    if not resolved.exists():
        raise SystemExit(f"checkpoint path not found: {resolved}")
    return str(resolved)


@dataclass
class LeagueEval:
    vs_random_win_rate: float | None = None
    vs_champion_win_rate: float | None = None


@dataclass
class LeagueMember:
    id: str
    path: str
    generation: int
    status: str
    collection_weight: float
    notes: str = ""
    parent_id: str = ""
    eval: LeagueEval = field(default_factory=LeagueEval)
    promoted_at: str = ""
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class LeagueRegistry:
    current_generation: int
    champion_id: str
    max_active_members: int
    members: list[LeagueMember]


def member_to_json_dict(member: LeagueMember) -> dict[str, object]:
    return {
        "id": member.id,
        "path": member.path,
        "generation": member.generation,
        "status": member.status,
        "collection_weight": member.collection_weight,
        "notes": member.notes,
        "parent_id": member.parent_id,
        "eval": {
            "vs_random_win_rate": member.eval.vs_random_win_rate,
            "vs_champion_win_rate": member.eval.vs_champion_win_rate,
        },
        "promoted_at": member.promoted_at,
        "created_at": member.created_at,
    }


def registry_to_json_dict(registry: LeagueRegistry) -> dict[str, object]:
    return {
        "current_generation": registry.current_generation,
        "champion_id": registry.champion_id,
        "max_active_members": registry.max_active_members,
        "members": [member_to_json_dict(member) for member in registry.members],
    }


def league_eval_from_json(raw: object) -> LeagueEval:
    payload = raw if isinstance(raw, dict) else {}
    return LeagueEval(
        vs_random_win_rate=payload.get("vs_random_win_rate"),
        vs_champion_win_rate=payload.get("vs_champion_win_rate"),
    )


def league_member_from_json(raw: dict[str, object]) -> LeagueMember:
    return LeagueMember(
        id=str(raw.get("id", "")).strip(),
        path=str(raw.get("path", "")).strip(),
        generation=int(raw.get("generation", 0)),
        status=str(raw.get("status", "")).strip(),
        collection_weight=float(raw.get("collection_weight", 0.0)),
        notes=str(raw.get("notes", "")),
        parent_id=str(raw.get("parent_id", "")).strip(),
        eval=league_eval_from_json(raw.get("eval")),
        promoted_at=str(raw.get("promoted_at", "")),
        created_at=str(raw.get("created_at", "")),
    )


def empty_registry(max_active_members: int) -> LeagueRegistry:
    return LeagueRegistry(
        current_generation=0,
        champion_id="",
        max_active_members=max_active_members,
        members=[],
    )


def load_registry(path: Path) -> LeagueRegistry:
    resolved = path.resolve()
    if not resolved.exists():
        raise SystemExit(f"registry not found: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to parse registry {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid registry {resolved}: expected top-level object")
    members_raw = payload.get("members")
    if not isinstance(members_raw, list):
        raise SystemExit(f"invalid registry {resolved}: 'members' must be a list")
    registry = LeagueRegistry(
        current_generation=int(payload.get("current_generation", 0)),
        champion_id=str(payload.get("champion_id", "")).strip(),
        max_active_members=int(payload.get("max_active_members", DEFAULT_MAX_ACTIVE_MEMBERS)),
        members=[league_member_from_json(member) for member in members_raw if isinstance(member, dict)],
    )
    validate_registry(registry)
    return registry


def save_registry(path: Path, registry: LeagueRegistry) -> None:
    validate_registry(registry)
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = registry_to_json_dict(registry)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(resolved.parent), suffix=".tmp") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        temp_path = Path(handle.name)
    temp_path.replace(resolved)


def find_member(registry: LeagueRegistry, member_id: str) -> LeagueMember:
    for member in registry.members:
        if member.id == member_id:
            return member
    raise SystemExit(f"member not found: {member_id}")


def validate_registry(registry: LeagueRegistry) -> None:
    if registry.current_generation < 0:
        raise SystemExit("invalid registry: current_generation must be >= 0")
    if registry.max_active_members < 1:
        raise SystemExit("invalid registry: max_active_members must be >= 1")
    ids: set[str] = set()
    active_ids: set[str] = set()
    for member in registry.members:
        if not member.id:
            raise SystemExit("invalid registry member: id must be non-empty")
        if member.id in ids:
            raise SystemExit(f"invalid registry: duplicate member id '{member.id}'")
        ids.add(member.id)
        if not member.path:
            raise SystemExit(f"invalid registry member '{member.id}': path must be non-empty")
        if not Path(member.path).exists():
            raise SystemExit(f"invalid registry member '{member.id}': checkpoint path not found: {member.path}")
        if member.generation < 0:
            raise SystemExit(f"invalid registry member '{member.id}': generation must be >= 0")
        if member.status not in {"active", "inactive"}:
            raise SystemExit(f"invalid registry member '{member.id}': bad status '{member.status}'")
        if member.collection_weight <= 0.0:
            raise SystemExit(f"invalid registry member '{member.id}': collection_weight must be > 0")
        if member.parent_id and member.parent_id == member.id:
            raise SystemExit(f"invalid registry member '{member.id}': parent_id cannot self-reference")
        if member.eval.vs_random_win_rate is not None and not (0.0 <= member.eval.vs_random_win_rate <= 1.0):
            raise SystemExit(f"invalid registry member '{member.id}': vs_random_win_rate must be between 0 and 1")
        if member.eval.vs_champion_win_rate is not None and not (0.0 <= member.eval.vs_champion_win_rate <= 1.0):
            raise SystemExit(f"invalid registry member '{member.id}': vs_champion_win_rate must be between 0 and 1")
        if member.status == "active":
            active_ids.add(member.id)
    for member in registry.members:
        if member.parent_id and member.parent_id not in ids:
            raise SystemExit(
                f"invalid registry member '{member.id}': parent_id references unknown member '{member.parent_id}'"
            )
    if registry.champion_id:
        if registry.champion_id not in ids:
            raise SystemExit(f"invalid registry: champion_id references unknown member '{registry.champion_id}'")
        if registry.champion_id not in active_ids:
            raise SystemExit(f"invalid registry: champion member '{registry.champion_id}' must be active")


def build_pool_payload(
    registry: LeagueRegistry,
    *,
    status_filter: str,
    include_random: bool,
    random_weight: float,
    selected_member_ids: list[str],
    normalize_member_weights: bool,
) -> dict[str, object]:
    selected: list[LeagueMember] = []
    allowlist = set(selected_member_ids)
    for member in registry.members:
        if status_filter != "all" and member.status != status_filter:
            continue
        if allowlist and member.id not in allowlist:
            continue
        selected.append(member)
    if allowlist:
        missing = sorted(allowlist - {member.id for member in selected})
        if missing:
            raise SystemExit(f"build-pool selected unknown/ineligible members: {', '.join(missing)}")
    if not selected and not include_random:
        raise SystemExit("build-pool selected zero checkpoint members and random is disabled")

    members: list[dict[str, object]] = []
    total_weight = sum(member.collection_weight for member in selected)
    for member in selected:
        weight = member.collection_weight
        if normalize_member_weights and total_weight > 0.0:
            weight = weight / total_weight
        members.append(
            {
                "name": member.id,
                "kind": "checkpoint",
                "path": member.path,
                "weight": weight,
            }
        )
    if include_random:
        members.append({"name": "random", "kind": "random", "weight": random_weight})
    if not members:
        raise SystemExit("build-pool produced an empty member list")
    return {"members": members}


def print_registry_summary(registry: LeagueRegistry) -> None:
    active_count = sum(1 for member in registry.members if member.status == "active")
    print(f"current_generation: {registry.current_generation}")
    print(f"champion_id: {registry.champion_id or '(none)'}")
    print(f"active_members: {active_count}/{registry.max_active_members}")
    print("members:")
    if not registry.members:
        print("  (none)")
        return
    for member in sorted(registry.members, key=lambda item: (item.generation, item.id)):
        champion_marker = " *champion" if registry.champion_id == member.id else ""
        print(
            f"  - id={member.id} generation={member.generation} status={member.status} "
            f"weight={member.collection_weight:g} path={member.path}{champion_marker}"
        )


def command_init(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry).resolve()
    if registry_path.exists() and not args.force:
        raise SystemExit(f"registry already exists: {registry_path} (use --force 1 to overwrite)")
    save_registry(registry_path, empty_registry(args.max_active_members))
    print(f"[league_manage] initialized registry: {registry_path}")


def command_show(args: argparse.Namespace) -> None:
    registry = load_registry(Path(args.registry))
    print_registry_summary(registry)


def command_add_checkpoint(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    member_id = args.id.strip()
    if not member_id:
        raise SystemExit("member id must be non-empty")
    if any(member.id == member_id for member in registry.members):
        raise SystemExit(f"duplicate member id: {member_id}")
    parent_id = args.parent_id.strip()
    if parent_id:
        find_member(registry, parent_id)
    generation = registry.current_generation if args.generation is None else args.generation
    member = LeagueMember(
        id=member_id,
        path=resolve_checkpoint_path(args.path),
        generation=generation,
        status=args.status,
        collection_weight=args.collection_weight,
        notes=args.notes,
        parent_id=parent_id,
        created_at=utc_now_iso(),
    )
    registry.members.append(member)
    if generation > registry.current_generation:
        registry.current_generation = generation
    save_registry(registry_path, registry)
    print(f"[league_manage] added member id={member.id} generation={member.generation}")


def command_update_member(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    member = find_member(registry, args.id)
    if args.collection_weight is not None:
        member.collection_weight = args.collection_weight
    if args.status is not None:
        member.status = args.status
    if args.notes is not None:
        member.notes = args.notes
    if args.parent_id is not None:
        parent_id = args.parent_id.strip()
        if parent_id:
            find_member(registry, parent_id)
        member.parent_id = parent_id
    if args.vs_random_win_rate is not None:
        member.eval.vs_random_win_rate = args.vs_random_win_rate
    if args.vs_champion_win_rate is not None:
        member.eval.vs_champion_win_rate = args.vs_champion_win_rate
    save_registry(registry_path, registry)
    print(f"[league_manage] updated member id={member.id}")


def command_promote(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    member = find_member(registry, args.id)
    if args.generation is not None:
        member.generation = args.generation
    member.status = "active"
    member.promoted_at = utc_now_iso()
    registry.champion_id = member.id
    registry.current_generation = max(registry.current_generation, member.generation)
    save_registry(registry_path, registry)
    active_count = sum(1 for item in registry.members if item.status == "active")
    warning = ""
    if active_count > registry.max_active_members:
        warning = f" warning: active members {active_count} exceed max_active_members {registry.max_active_members}"
    print(f"[league_manage] promoted member id={member.id} champion_id={registry.champion_id}{warning}")


def command_deactivate(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    member = find_member(registry, args.id)
    if registry.champion_id == member.id and not args.allow_champion:
        raise SystemExit("cannot deactivate current champion without --allow-champion 1")
    member.status = "inactive"
    if registry.champion_id == member.id and args.allow_champion:
        registry.champion_id = ""
    save_registry(registry_path, registry)
    print(f"[league_manage] deactivated member id={member.id}")


def command_build_pool(args: argparse.Namespace) -> None:
    registry = load_registry(Path(args.registry))
    payload = build_pool_payload(
        registry,
        status_filter=args.status,
        include_random=args.include_random,
        random_weight=args.random_weight,
        selected_member_ids=args.member,
        normalize_member_weights=args.normalize_member_weights,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[league_manage] wrote pool: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--registry", default=str(default_registry_path()))
    init_parser.add_argument("--max-active-members", type=positive_int, default=DEFAULT_MAX_ACTIVE_MEMBERS)
    init_parser.add_argument("--force", type=parse_bool01, default=False)
    init_parser.set_defaults(func=command_init)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--registry", default=str(default_registry_path()))
    show_parser.set_defaults(func=command_show)

    add_parser = subparsers.add_parser("add-checkpoint")
    add_parser.add_argument("--registry", default=str(default_registry_path()))
    add_parser.add_argument("--id", required=True)
    add_parser.add_argument("--path", required=True)
    add_parser.add_argument("--generation", type=nonnegative_int, default=None)
    add_parser.add_argument("--collection-weight", type=positive_float, default=1.0)
    add_parser.add_argument("--status", choices=["active", "inactive"], default="active")
    add_parser.add_argument("--parent-id", default="")
    add_parser.add_argument("--notes", default="")
    add_parser.set_defaults(func=command_add_checkpoint)

    update_parser = subparsers.add_parser("update-member")
    update_parser.add_argument("--registry", default=str(default_registry_path()))
    update_parser.add_argument("--id", required=True)
    update_parser.add_argument("--collection-weight", type=positive_float, default=None)
    update_parser.add_argument("--status", choices=["active", "inactive"], default=None)
    update_parser.add_argument("--notes", default=None)
    update_parser.add_argument("--parent-id", default=None)
    update_parser.add_argument("--vs-random-win-rate", type=win_rate_float, default=None)
    update_parser.add_argument("--vs-champion-win-rate", type=win_rate_float, default=None)
    update_parser.set_defaults(func=command_update_member)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--registry", default=str(default_registry_path()))
    promote_parser.add_argument("--id", required=True)
    promote_parser.add_argument("--generation", type=nonnegative_int, default=None)
    promote_parser.set_defaults(func=command_promote)

    deactivate_parser = subparsers.add_parser("deactivate")
    deactivate_parser.add_argument("--registry", default=str(default_registry_path()))
    deactivate_parser.add_argument("--id", required=True)
    deactivate_parser.add_argument("--allow-champion", type=parse_bool01, default=False)
    deactivate_parser.set_defaults(func=command_deactivate)

    pool_parser = subparsers.add_parser("build-pool")
    pool_parser.add_argument("--registry", default=str(default_registry_path()))
    pool_parser.add_argument("--output", required=True)
    pool_parser.add_argument("--status", choices=["active", "inactive", "all"], default="active")
    pool_parser.add_argument("--include-random", type=parse_bool01, default=True)
    pool_parser.add_argument("--random-weight", type=positive_float, default=1.0)
    pool_parser.add_argument("--member", action="append", default=[])
    pool_parser.add_argument("--normalize-member-weights", type=parse_bool01, default=False)
    pool_parser.set_defaults(func=command_build_pool)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
