from __future__ import annotations

import copy
import math


ADAPTIVE_STRATEGY = "category_matchup_target"


def matchup_difficulty_weight(
    win_rate: float | None,
    matches_played: int,
    *,
    target_win_rate: float,
    min_weight: float,
    confidence_games: int,
) -> float:
    if win_rate is None or matches_played <= 0:
        return 1.0
    confidence = min(1.0, matches_played / max(1, confidence_games))
    smoothed_rate = target_win_rate + ((min(1.0, max(0.0, win_rate)) - target_win_rate) * confidence)
    span = max(target_win_rate, 1.0 - target_win_rate, 1e-9)
    closeness = 1.0 - min(1.0, abs(smoothed_rate - target_win_rate) / span)
    return max(min_weight, closeness)


def is_adaptive_pool(payload: dict[str, object]) -> bool:
    sampling = payload.get("sampling", {})
    return isinstance(sampling, dict) and sampling.get("strategy") == ADAPTIVE_STRATEGY


def refresh_adaptive_pool(
    payload: dict[str, object],
    opponent_member_stats: dict[str, object],
    collection_run: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    refreshed = copy.deepcopy(payload)
    if not is_adaptive_pool(refreshed):
        return refreshed, {}

    sampling = refreshed.get("sampling", {})
    members = refreshed.get("members", [])
    if not isinstance(sampling, dict) or not isinstance(members, list):
        return refreshed, {}

    target_win_rate = float(sampling.get("target_win_rate", 0.5))
    min_weight = float(sampling.get("min_difficulty_weight", 0.1))
    confidence_games = max(1, int(sampling.get("confidence_games", 20)))
    if not math.isfinite(target_win_rate) or not 0.0 < target_win_rate < 1.0:
        raise ValueError("adaptive pool target_win_rate must be strictly between 0 and 1")
    if not math.isfinite(min_weight) or not 0.0 < min_weight <= 1.0:
        raise ValueError("adaptive pool min_difficulty_weight must be greater than 0 and at most 1")

    category_members: dict[str, list[dict[str, object]]] = {}
    refresh_results: dict[str, dict[str, object]] = {}
    for raw_member in members:
        if not isinstance(raw_member, dict):
            continue
        name = str(raw_member.get("name", "")).strip()
        category = str(raw_member.get("category", "uncategorized")).strip() or "uncategorized"
        kind = str(raw_member.get("kind", "")).strip()
        if kind == "random" or category == "random":
            continue

        old_difficulty = max(float(raw_member.get("difficulty_weight", 1.0)), 1e-9)
        base_weight = float(raw_member.get("base_weight", float(raw_member.get("weight", 1.0)) / old_difficulty))
        if not math.isfinite(base_weight) or base_weight <= 0.0:
            raise ValueError(f"adaptive pool member '{name}' has invalid base_weight")
        raw_member["base_weight"] = base_weight
        raw_member["last_round_games"] = 0
        raw_member["last_round_learner_win_rate"] = None

        result = opponent_member_stats.get(name)
        if isinstance(result, dict):
            opponent_wins = max(0, int(result.get("wins", 0)))
            opponent_losses = max(0, int(result.get("losses", 0)))
            draws = max(0, int(result.get("draws", 0)))
            matches_played = opponent_wins + opponent_losses + draws
            if matches_played > 0:
                learner_wins = opponent_losses
                learner_losses = opponent_wins
                learner_win_rate = learner_wins / matches_played
                raw_member["learner_win_rate"] = learner_win_rate
                raw_member["matchup_games"] = matches_played
                raw_member["last_round_games"] = matches_played
                raw_member["last_round_learner_win_rate"] = learner_win_rate
                refresh_results[name] = {
                    "matches_played": matches_played,
                    "wins": learner_wins,
                    "losses": learner_losses,
                    "draws": draws,
                    "win_rate": learner_win_rate,
                }

        raw_rate = raw_member.get("learner_win_rate")
        learner_win_rate = None if raw_rate is None else float(raw_rate)
        matchup_games = max(0, int(raw_member.get("matchup_games", 0)))
        difficulty_weight = matchup_difficulty_weight(
            learner_win_rate,
            matchup_games,
            target_win_rate=target_win_rate,
            min_weight=min_weight,
            confidence_games=confidence_games,
        )
        raw_member["difficulty_weight"] = difficulty_weight
        category_members.setdefault(category, []).append(raw_member)

    for category, grouped_members in category_members.items():
        explicit_budgets = [
            float(member["bucket_weight"])
            for member in grouped_members
            if member.get("bucket_weight") is not None
        ]
        category_budget = explicit_budgets[0] if explicit_budgets else sum(
            float(member.get("weight", 0.0)) for member in grouped_members
        )
        raw_weights = [
            float(member["base_weight"]) * float(member["difficulty_weight"])
            for member in grouped_members
        ]
        raw_total = sum(raw_weights)
        if category_budget <= 0.0 or raw_total <= 0.0:
            raise ValueError(f"adaptive pool category '{category}' has no positive sampling weight")
        for member, raw_weight in zip(grouped_members, raw_weights):
            member["weight"] = category_budget * (raw_weight / raw_total)

    refreshed["members"] = sorted(
        members,
        key=lambda member: (
            -float(member.get("weight", 0.0)) if isinstance(member, dict) else 0.0,
            str(member.get("name", "")) if isinstance(member, dict) else "",
        ),
    )
    sampling["refresh_count"] = int(sampling.get("refresh_count", 0)) + 1
    sampling["last_collection_run"] = collection_run
    sampling["last_refresh_results"] = refresh_results
    return refreshed, refresh_results
