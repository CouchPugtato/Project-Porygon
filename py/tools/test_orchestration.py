from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from league_manage import LeagueEval, LeagueMember, LeagueRegistry, OpponentStats, build_pool_payload, league_member_from_json, member_to_json_dict
from league_rl_orchestrator import (
    balanced_evaluation_artifacts_match,
    build_eval_command as build_league_eval_command,
    build_live_command as build_league_live_command,
    build_weighted_pool,
    collect_recent_opponent_stats,
    league_promotion_assessment,
    matchup_difficulty_weight,
    maybe_promote_candidate,
)
from live_rl_orchestrator import build_selfplay_command, build_train_command, collapse_flags_from_training_summary, round_manifest_completed
from opponent_sampling import refresh_adaptive_pool
from rl_defaults import float_default
from selfplay_server import load_model_pool, pool_coverage_summary, sample_pool_member, validate_pool_member


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

    def test_round_refresh_preserves_categories_and_targets_fifty_percent(self) -> None:
        champion = member("champion", role="champion", generation=10)
        parent = member("parent", generation=11)
        balanced = member("balanced", role="historical_snapshot")
        easy = member("easy", role="historical_snapshot")
        registry = LeagueRegistry(11, champion.id, 8, [champion, parent, balanced, easy])
        payload = build_weighted_pool(registry, "main", parent.id, 0.0)
        refreshed, results = refresh_adaptive_pool(payload, {
            balanced.id: {"wins": 10, "losses": 10, "draws": 0},
            easy.id: {"wins": 2, "losses": 18, "draws": 0},
        }, "round01_collect")
        members = {str(item["name"]): item for item in refreshed["members"]}
        historical_total = sum(
            float(item["weight"])
            for item in refreshed["members"]
            if item.get("category") == "historical"
        )
        self.assertAlmostEqual(historical_total, 0.20)
        self.assertAlmostEqual(float(members[champion.id]["weight"]), 0.40)
        self.assertGreater(float(members[balanced.id]["weight"]), float(members[easy.id]["weight"]))
        self.assertEqual(results[easy.id]["wins"], 18)
        self.assertEqual(results[easy.id]["losses"], 2)
        self.assertEqual(refreshed["sampling"]["refresh_count"], 1)
        self.assertEqual(refreshed["sampling"]["last_collection_run"], "round01_collect")

        second_refresh, _ = refresh_adaptive_pool(refreshed, {
            balanced.id: {"wins": 10, "losses": 10, "draws": 0},
            easy.id: {"wins": 10, "losses": 10, "draws": 0},
        }, "round02_collect")
        second_members = {str(item["name"]): item for item in second_refresh["members"]}
        self.assertAlmostEqual(
            float(second_members[balanced.id]["weight"]),
            float(second_members[easy.id]["weight"]),
        )
        self.assertEqual(second_refresh["sampling"]["refresh_count"], 2)

    def test_static_pool_is_not_adapted(self) -> None:
        payload = {"members": [{"name": "random", "kind": "random", "weight": 1.0}]}
        refreshed, results = refresh_adaptive_pool(payload, {}, "round01_collect")
        self.assertEqual(refreshed, payload)
        self.assertEqual(results, {})

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

    def test_category_quota_assigns_every_category_before_weighted_balance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pool_path = Path(temp_dir) / "pool.json"
            pool_path.write_text(json.dumps({
                "coverage": {
                    "enabled": True,
                    "min_category_starts": 1,
                    "prefer_under_sampled_members": True,
                },
                "members": [
                    {"name": "champion", "kind": "random", "weight": 0.6, "category": "champion"},
                    {"name": "recent", "kind": "random", "weight": 0.3, "category": "recent"},
                    {"name": "history", "kind": "random", "weight": 0.1, "category": "historical"},
                ],
            }), encoding="utf-8")
            pool = load_model_pool(pool_path)
            rng = random.Random(7)
            starts: dict[str, int] = {}
            samples = []
            for _ in range(3):
                sample = sample_pool_member(pool, rng, starts)
                samples.append(sample)
                starts[sample.member_name] = starts.get(sample.member_name, 0) + 1

            self.assertEqual({sample.category for sample in samples}, {"champion", "recent", "historical"})
            self.assertTrue(all(sample.selection_reason == "category_quota" for sample in samples))
            self.assertEqual(sample_pool_member(pool, rng, starts).selection_reason, "weighted_balance")

    def test_category_assignment_prefers_member_behind_its_weighted_share(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pool_path = Path(temp_dir) / "pool.json"
            pool_path.write_text(json.dumps({
                "coverage": {"enabled": True, "min_category_starts": 1},
                "members": [
                    {"name": "common", "kind": "random", "weight": 0.75, "category": "historical"},
                    {"name": "rare", "kind": "random", "weight": 0.25, "category": "historical"},
                ],
            }), encoding="utf-8")
            pool = load_model_pool(pool_path)
            sample = sample_pool_member(pool, random.Random(11), {"common": 3, "rare": 0})
            self.assertEqual(sample.member_name, "rare")
            self.assertEqual(sample.selection_reason, "weighted_balance")

    def test_pool_coverage_reports_assignment_and_completed_game_shortfalls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pool_path = Path(temp_dir) / "pool.json"
            pool_path.write_text(json.dumps({
                "coverage": {"enabled": True, "min_category_starts": 1},
                "members": [
                    {"name": "champion", "kind": "random", "weight": 0.7, "category": "champion"},
                    {"name": "history", "kind": "random", "weight": 0.3, "category": "historical"},
                ],
            }), encoding="utf-8")
            pool = load_model_pool(pool_path)
            summary = pool_coverage_summary(pool, {
                "champion": {"worker_starts": 1, "completed_games": 1},
            })
            self.assertFalse(summary["assignment_coverage_met"])
            self.assertEqual(summary["categories"]["historical"]["assignment_shortfall"], 1)

            summary = pool_coverage_summary(pool, {
                "champion": {"worker_starts": 1, "completed_games": 1},
                "history": {"worker_starts": 1, "completed_games": 0},
            })
            self.assertTrue(summary["assignment_coverage_met"])
            self.assertFalse(summary["completed_game_coverage_met"])
            self.assertEqual(summary["categories"]["historical"]["completed_game_shortfall"], 1)

    def test_static_pool_keeps_weighted_random_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pool_path = Path(temp_dir) / "pool.json"
            pool_path.write_text(json.dumps({
                "members": [{"name": "random", "kind": "random", "weight": 1.0}],
            }), encoding="utf-8")
            pool = load_model_pool(pool_path)
            sample = sample_pool_member(pool, random.Random(3), {})
            self.assertFalse(pool.coverage_enabled)
            self.assertEqual(sample.selection_reason, "weighted_random")


class PromotionTests(unittest.TestCase):
    def test_promotion_requires_confidence_win_rate_and_no_collapse_flags(self) -> None:
        champion = member("champion")
        candidate = member("candidate", generation=2)
        candidate.status = "candidate"
        registry = LeagueRegistry(2, champion.id, 5, [champion, candidate])
        args = SimpleNamespace(
            promote_threshold=0.52,
            promotion_confidence_threshold=0.50,
            min_promotion_tera_ratio=0.60,
            learner_role="main",
        )
        summary = {
            "valid_win_rate": 0.54,
            "valid_games": 1000,
            "invalid_games": 20,
            "confidence_low": 0.509,
            "confidence_high": 0.571,
            "group_stats": {
                "candidate": {"tera_battle_rate": 0.90},
                "champion": {"tera_battle_rate": 0.92},
            },
        }

        self.assertFalse(maybe_promote_candidate(args, registry, candidate, summary, ["warn_move_slot_concentration"]))
        self.assertEqual(registry.champion_id, champion.id)
        summary["confidence_low"] = 0.499
        assessment = league_promotion_assessment(args, summary, [])
        self.assertEqual(assessment["status"], "tentative_winner")
        self.assertFalse(maybe_promote_candidate(args, registry, candidate, summary, []))
        summary["confidence_low"] = 0.509
        self.assertTrue(maybe_promote_candidate(args, registry, candidate, summary, []))
        self.assertEqual(registry.champion_id, candidate.id)

    def test_balanced_eval_command_swaps_candidate_side_and_requires_shards(self) -> None:
        args = SimpleNamespace(
            eval_concurrent_games=40,
            eval_worker_pairs=120,
            format="gen9randomdoublesbattle",
            launch_stagger_seconds=0.25,
            resource_check_seconds=2.0,
            min_available_memory_gb=2.0,
            min_available_pagefile_gb=4.0,
            startup_timeout_seconds=120,
        )
        command = build_league_eval_command(
            args, Path("repo"), "eval-side-b", Path("candidate.chk"),
            Path("champion.chk"), "b", 250, 77,
        )
        self.assertEqual(command[command.index("--model-a") + 1], "champion.chk")
        self.assertEqual(command[command.index("--model-b") + 1], "candidate.chk")
        self.assertEqual(command[command.index("--games") + 1], "250")
        self.assertEqual(command[command.index("--ensure-shard-count") + 1], "true")
        self.assertEqual(command[command.index("--model-a-pool") + 1], "")
        self.assertEqual(command[command.index("--model-b-pool") + 1], "")

    def test_balanced_eval_resume_checks_models_and_valid_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            candidate = root / "candidate.chk"
            champion = root / "champion.chk"
            summary = {
                "status": "completed",
                "evaluation_mode": "balanced_valid_games",
                "target_valid_games_per_side": 500,
                "model_specs": {
                    "candidate": {"path": str(candidate)},
                    "champion": {"path": str(champion)},
                },
            }
            self.assertTrue(balanced_evaluation_artifacts_match(summary, candidate, champion, 500))
            self.assertFalse(balanced_evaluation_artifacts_match(summary, champion, candidate, 500))
            self.assertFalse(balanced_evaluation_artifacts_match(summary, candidate, champion, 250))

    def test_league_live_command_forwards_anchored_ppo_controls(self) -> None:
        args = SimpleNamespace(
            run_name="league-round",
            training_mode="ppo",
            rounds=3,
            games=2000,
            concurrent_games=70,
            worker_pairs=125,
            ensure_shard_count=True,
            pool_seed=11,
            learning_rate=1e-5,
            gamma=0.99,
            entropy_coef=1e-4,
            advantage_norm=True,
            gae_lambda=0.95,
            ppo_clip_epsilon=0.2,
            ppo_value_clip_epsilon=0.2,
            target_kl=0.02,
            target_kl_min_episodes=20,
            target_kl_min_labels=500,
            target_kl_hard_multiplier=4.0,
            target_kl_hard_consecutive_updates=2,
            shuffle_seed=1337,
            ppo_minibatch_episodes=8,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            anchor_checkpoint="anchor.chk",
            anchor_kl_coef=0.01,
            reward_mode="terminal",
            launch_stagger_seconds=0.25,
            resource_check_seconds=2.0,
            min_available_memory_gb=2.0,
            min_available_pagefile_gb=4.0,
            stop_on_collapse=True,
            omp_threads=8,
            resume=True,
        )
        command = build_league_live_command(
            args, Path("repo"), Path("pool.json"), Path("parent.chk"),
        )
        self.assertEqual(command[command.index("--anchor-checkpoint") + 1], "anchor.chk")
        self.assertEqual(command[command.index("--anchor-kl-coef") + 1], "0.01")
        self.assertEqual(command[command.index("--target-kl") + 1], "0.02")
        self.assertEqual(command[command.index("--ppo-minibatch-episodes") + 1], "8")
        self.assertEqual(command[command.index("--learning-rate") + 1], "1e-05")


class ResumeAndCollapseTests(unittest.TestCase):
    def test_selfplay_command_uses_round_pool_snapshot(self) -> None:
        args = SimpleNamespace(
            games=100,
            concurrent_games=4,
            worker_pairs=8,
            worker_games=0,
            ensure_shard_count=True,
            pool_seed=7,
            format="gen9randomdoublesbattle",
            worker_think_mode="live",
            serve_client=False,
            worker_log_stdout=False,
            launch_stagger_seconds=0.0,
            resource_check_seconds=1.0,
            min_available_memory_gb=1.0,
            min_available_pagefile_gb=1.0,
            model_b_pool="initial_pool.json",
            model_b="",
        )
        snapshot = Path("round_pool.json")
        command = build_selfplay_command(args, Path.cwd(), "collect", Path("actor.chk"), snapshot)
        pool_index = command.index("--model-b-pool")
        self.assertEqual(command[pool_index + 1], str(snapshot))

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

    def test_completed_round_requires_recorded_pool_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "candidate.chk"
            summary = root / "summary.json"
            used_pool = root / "used_pool.json"
            next_pool = root / "next_pool.json"
            manifest = root / "round.json"
            for path in (checkpoint, summary, used_pool, next_pool):
                path.write_bytes(b"artifact")
            manifest.write_text(json.dumps({
                "status": "completed",
                "output_checkpoint": str(checkpoint),
                "training_round_stats_path": str(summary),
                "opponent_pool_path": str(used_pool),
                "next_opponent_pool_path": str(next_pool),
            }), encoding="utf-8")
            self.assertTrue(round_manifest_completed(manifest))
            next_pool.unlink()
            self.assertFalse(round_manifest_completed(manifest))

    def test_ppo_train_command_preserves_anchor_and_target_kl(self) -> None:
        args = SimpleNamespace(
            training_mode="ppo",
            current_policy_tag="parent.chk",
            epochs=1,
            learning_rate=0.001,
            gamma=0.99,
            entropy_coef=0.001,
            advantage_norm=True,
            gae_lambda=0.95,
            ppo_clip_epsilon=0.2,
            ppo_value_clip_epsilon=0.2,
            target_kl=0.02,
            target_kl_min_episodes=20,
            target_kl_min_labels=500,
            target_kl_hard_multiplier=4.0,
            target_kl_hard_consecutive_updates=2,
            shuffle_seed=1337,
            ppo_minibatch_episodes=8,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            reward_mode="terminal",
            training_summary_path=Path("summary.json"),
            anchor_checkpoint="anchor.chk",
            anchor_kl_coef=0.01,
            dense_additive_hp_swing_weight=0.1,
            dense_additive_faint_swing_weight=0.25,
            dense_additive_reward_clip=0.4,
        )
        command = build_train_command(args, Path("trainer.exe"), Path("episodes.jsonl"), Path("candidate.chk"))
        self.assertEqual(command[command.index("--anchor-checkpoint") + 1], "anchor.chk")
        self.assertEqual(command[command.index("--anchor-kl-coef") + 1], "0.01")
        self.assertEqual(command[command.index("--target-kl") + 1], "0.02")
        self.assertEqual(command[command.index("--target-kl-min-episodes") + 1], "20")
        self.assertEqual(command[command.index("--target-kl-min-labels") + 1], "500")
        self.assertEqual(command[command.index("--target-kl-hard-consecutive-updates") + 1], "2")
        self.assertEqual(command[command.index("--shuffle-seed") + 1], "1337")
        self.assertEqual(command[command.index("--ppo-minibatch-episodes") + 1], "8")
        self.assertEqual(command[command.index("--parent-checkpoint") + 1], "parent.chk")

    def test_training_collapse_uses_action_level_tera_metric(self) -> None:
        flags = collapse_flags_from_training_summary(
            {
                "episode_count": 200,
                "tera_action_rate": 0.10,
                "tera_rate": 0.90,
                "move_slot_rates": {"slot_1": 0.71},
                "anchor_kl_mean": 0.11,
            },
            baseline_tera_action_rate=0.50,
            min_episodes_warn=100,
            anchor_kl_warn_threshold=0.10,
        )
        self.assertTrue(any(flag.startswith("warn_tera_action_rate_low") for flag in flags))
        self.assertTrue(any(flag.startswith("hard_move_slot_collapse") for flag in flags))
        self.assertTrue(any(flag.startswith("warn_anchor_kl_high") for flag in flags))

    def test_default_tera_guardrail_uses_action_level_units(self) -> None:
        baseline = float_default("baseline_tera_action_rate")
        self.assertGreater(baseline, 0.0)
        self.assertLess(baseline, 0.10)
        flags = collapse_flags_from_training_summary(
            {
                "episode_count": 200,
                "tera_action_rate": baseline,
                "move_slot_rates": {},
            },
            baseline_tera_action_rate=baseline,
            min_episodes_warn=100,
            anchor_kl_warn_threshold=0.10,
        )
        self.assertFalse(any(flag.startswith("warn_tera_action_rate_low") for flag in flags))


if __name__ == "__main__":
    unittest.main()
