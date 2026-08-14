from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from league_manage import LeagueEval, LeagueMember, LeagueRegistry, OpponentStats, build_pool_payload, league_member_from_json, member_to_json_dict
from league_rl_orchestrator import build_weighted_pool, collect_recent_opponent_stats, matchup_difficulty_weight, maybe_promote_candidate
from live_rl_orchestrator import collapse_flags_from_training_summary, round_manifest_completed
from selfplay_server import validate_pool_member


def member(
    member_id: str,
    *,
    role: str = "main",
    generation: int = 1,
    win_rate: float | None = None,
) -> LeagueMember:
    return LeagueMember(
        id=member_id,
        path=f"{member_id}.chk",
        generation=generation,
        status="active",
        collection_weight=1.0,
        role=role,
        eval=LeagueEval(vs_champion_win_rate=win_rate),
    )


class LeaguePoolTests(unittest.TestCase):
    def test_champion_plus_random_preset_is_explicit(self) -> None:
        champion = member("champion")
        registry = LeagueRegistry(1, champion.id, 5, [champion, member("other")])
        payload = build_pool_payload(
            registry,
            preset="champion-plus-random",
            status_filter="active",
            include_random=False,
            random_weight=0.25,
            selected_member_ids=[],
            normalize_member_weights=False,
            top_k=3,
        )
        self.assertEqual([item["name"] for item in payload["members"]], ["champion", "random"])

    def test_weighted_pool_keeps_role_mixture_and_champion_parent(self) -> None:
        champion = member("champion", role="champion", generation=10)
        parent = member("parent", generation=11)
        recent = member("recent", generation=9)
        history = member("history", role="historical_snapshot")
        main_exploiter = member("main_exploiter", role="main_exploiter")
        league_exploiter = member("league_exploiter", role="league_exploiter")
        registry = LeagueRegistry(
            11,
            champion.id,
            8,
            [champion, parent, recent, history, main_exploiter, league_exploiter],
        )
        payload = build_weighted_pool(registry, "main", parent.id, 0.0)
        members = {str(item["name"]): item for item in payload["members"]}
        self.assertEqual(members[champion.id]["category"], "champion")
        self.assertEqual(members[recent.id]["category"], "recent_main")
        self.assertEqual(members[history.id]["category"], "historical")
        self.assertEqual(members[main_exploiter.id]["category"], "main_exploiter")
        self.assertEqual(members[league_exploiter.id]["category"], "league_exploiter")
        self.assertAlmostEqual(float(members[champion.id]["weight"]), 0.40)
        self.assertAlmostEqual(float(members[recent.id]["weight"]), 0.25)
        self.assertAlmostEqual(float(members[history.id]["weight"]), 0.20)
        self.assertAlmostEqual(float(members[main_exploiter.id]["weight"]), 0.10)
        self.assertAlmostEqual(float(members[league_exploiter.id]["weight"]), 0.05)

        champion_parent_registry = LeagueRegistry(11, champion.id, 8, [champion, history])
        champion_parent_pool = build_weighted_pool(champion_parent_registry, "main", champion.id, 0.0)
        self.assertIn(champion.id, {str(item["name"]) for item in champion_parent_pool["members"]})

    def test_matchup_weight_favors_opponents_near_fifty_percent(self) -> None:
        champion = member("champion", role="champion", generation=10)
        parent = member("parent", generation=11)
        balanced = member("balanced", role="historical_snapshot")
        easy = member("easy", role="historical_snapshot")
        hard = member("hard", role="historical_snapshot")
        parent.opponent_stats = {
            balanced.id: OpponentStats(20, 10, 10, 0, 0.50, "latest"),
            easy.id: OpponentStats(20, 18, 2, 0, 0.90, "latest"),
            hard.id: OpponentStats(20, 2, 18, 0, 0.10, "latest"),
        }
        registry = LeagueRegistry(11, champion.id, 8, [champion, parent, balanced, easy, hard])
        payload = build_weighted_pool(registry, "main", parent.id, 0.0)
        weights = {str(item["name"]): float(item["weight"]) for item in payload["members"]}
        self.assertGreater(weights[balanced.id], weights[easy.id])
        self.assertAlmostEqual(weights[easy.id], weights[hard.id])
        self.assertEqual(payload["sampling"]["strategy"], "category_matchup_target")

    def test_matchup_weight_shrinks_small_samples_toward_target(self) -> None:
        low_confidence = matchup_difficulty_weight(1.0, 2, confidence_games=20)
        high_confidence = matchup_difficulty_weight(1.0, 20, confidence_games=20)
        self.assertGreater(low_confidence, high_confidence)
        self.assertEqual(matchup_difficulty_weight(0.5, 20), 1.0)

    def test_recent_opponent_results_are_aggregated_from_learner_perspective(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            manifest_paths: list[Path] = []
            for round_index, opponent_wins, opponent_losses in ((1, 3, 7), (2, 6, 4)):
                collection_run = f"collect_{round_index}"
                summary_dir = repo_root / "matches" / "runs" / collection_run
                summary_dir.mkdir(parents=True)
                (summary_dir / f"{collection_run}_summary.json").write_text(json.dumps({
                    "group_member_stats": {
                        "b": {
                            "opponent": {
                                "wins": opponent_wins,
                                "losses": opponent_losses,
                                "draws": 0,
                            }
                        }
                    }
                }), encoding="utf-8")
                manifest_path = repo_root / f"round_{round_index}.json"
                manifest_path.write_text(json.dumps({"collection_run": collection_run}), encoding="utf-8")
                manifest_paths.append(manifest_path)

            stats = collect_recent_opponent_stats(repo_root, manifest_paths, "league_run")["opponent"]
            self.assertEqual((stats.matches_played, stats.wins, stats.losses), (20, 11, 9))
            self.assertEqual(stats.recent_win_rate, 0.4)
            self.assertEqual(stats.source_run, "league_run")

    def test_opponent_stats_round_trip_through_registry_member_json(self) -> None:
        original = member("learner")
        original.opponent_stats["opponent"] = OpponentStats(10, 6, 3, 1, 0.6, "latest")
        restored = league_member_from_json(member_to_json_dict(original))
        self.assertEqual(restored.opponent_stats["opponent"], original.opponent_stats["opponent"])

    def test_selfplay_pool_preserves_adaptive_member_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "opponent.chk"
            checkpoint.write_bytes(b"checkpoint")
            parsed = validate_pool_member({
                "name": "opponent",
                "kind": "checkpoint",
                "path": str(checkpoint),
                "weight": 0.25,
                "category": "historical",
                "learner_win_rate": 0.52,
                "matchup_games": 40,
                "difficulty_weight": 0.96,
            }, Path(temp_dir) / "pool.json", set())
            self.assertEqual(parsed.category, "historical")
            self.assertEqual(parsed.learner_win_rate, 0.52)
            self.assertEqual(parsed.matchup_games, 40)
            self.assertEqual(parsed.difficulty_weight, 0.96)


class PromotionTests(unittest.TestCase):
    def test_promotion_requires_win_rate_and_no_collapse_flags(self) -> None:
        champion = member("champion")
        candidate = member("candidate", generation=2)
        candidate.status = "candidate"
        registry = LeagueRegistry(2, champion.id, 5, [champion, candidate])
        args = SimpleNamespace(promote_threshold=0.52, min_promotion_tera_ratio=0.60, learner_role="main")
        summary = {"group_stats": {"a": {"earned_win_rate": 0.53, "tera_battle_rate": 0.90}}}

        self.assertFalse(maybe_promote_candidate(args, registry, candidate, summary, ["warn_move_slot_concentration"]))
        self.assertEqual(registry.champion_id, champion.id)
        self.assertTrue(maybe_promote_candidate(args, registry, candidate, summary, []))
        self.assertEqual(registry.champion_id, candidate.id)


class ResumeAndCollapseTests(unittest.TestCase):
    def test_completed_round_requires_manifest_checkpoint_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "candidate.chk"
            summary = root / "summary.json"
            manifest = root / "round.json"
            checkpoint.write_bytes(b"checkpoint")
            summary.write_text("{}\n", encoding="utf-8")
            manifest.write_text(json.dumps({
                "status": "completed",
                "output_checkpoint": str(checkpoint),
                "training_round_stats_path": str(summary),
            }), encoding="utf-8")
            self.assertTrue(round_manifest_completed(manifest))
            summary.unlink()
            self.assertFalse(round_manifest_completed(manifest))

    def test_training_collapse_uses_action_level_tera_metric(self) -> None:
        flags = collapse_flags_from_training_summary(
            {
                "episode_count": 200,
                "tera_action_rate": 0.10,
                "tera_rate": 0.90,
                "move_slot_rates": {"slot_1": 0.71},
                "anchor_kl_mean": 0.11,
            },
            baseline_tera_rate=0.50,
            min_episodes_warn=100,
            anchor_kl_warn_threshold=0.10,
        )
        self.assertTrue(any(flag.startswith("warn_tera_rate_low") for flag in flags))
        self.assertTrue(any(flag.startswith("hard_move_slot_collapse") for flag in flags))
        self.assertTrue(any(flag.startswith("warn_anchor_kl_high") for flag in flags))


if __name__ == "__main__":
    unittest.main()
