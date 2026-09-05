from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from balanced_checkpoint_eval import (
    build_parser as build_balanced_eval_parser,
    parse_args as parse_balanced_eval_args,
    prepare_shared_args,
)
from league_manage import (
    LeagueEval,
    LeagueMember,
    LeagueRegistry,
    OpponentStats,
    build_pool_payload,
    is_untrusted_legacy_checkpoint,
    league_member_from_json,
    member_to_json_dict,
)
from league_rl_orchestrator import (
    balanced_evaluation_artifacts_match,
    build_eval_command as build_league_eval_command,
    build_live_command as build_league_live_command,
    build_parser as build_league_parser,
    build_weighted_pool,
    candidate_runtime_appears_broken,
    choose_parent_member,
    collect_recent_opponent_stats,
    evaluation_block_limit,
    league_promotion_assessment,
    load_fixed_opponent_pool,
    matchup_difficulty_weight,
    maybe_promote_candidate,
    parse_args as parse_league_args,
    round_candidate_rank,
    round_evaluation_baseline,
    round_screen_should_expand,
    run_balanced_valid_evaluation,
    select_evaluation_block_run,
    RoundMappedWorkflowReporter,
)
from live_rl_orchestrator import (
    BaseWorkflowReporter,
    DashboardProgressWriter,
    WorkflowDashboardState,
    build_selfplay_command,
    build_train_command,
    collapse_flags_from_training_summary,
    extract_episode_batch,
    round_manifest_completed,
    run_reported_command,
    terminate_recorded_child_processes,
)
from model_spec import model_spec_payload
from opponent_sampling import refresh_adaptive_pool
from rl_defaults import float_default
from selfplay_server import (
    WorkerLaunchIdentity,
    WorkerProcess,
    WorkerSpec,
    load_model_pool,
    pool_coverage_summary,
    sample_pool_member,
    validate_pool_member,
)


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
    def test_league_config_defaults_allow_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "league.toml"
            config_path.write_text(
                "games = 100\n"
                "concurrent_games = 4\n"
                "worker_pairs = 6\n"
                "registry_update = false\n",
                encoding="utf-8",
            )

            args = parse_league_args([
                "--config", str(config_path), "--run-name", "configured",
                "--games", "200",
            ])

            self.assertEqual(args.games, 200)
            self.assertEqual(args.concurrent_games, 4)
            self.assertEqual(args.worker_pairs, 6)
            self.assertFalse(args.registry_update)
            self.assertEqual(args.config, str(config_path.resolve()))

    def test_fixed_pool_preserves_configured_members_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pool_path = root / "pool.json"
            configured = {
                "members": [
                    {"name": "random", "kind": "random", "weight": 0.4},
                    {"name": "parent", "kind": "random", "weight": 0.35},
                    {"name": "alternate", "kind": "random", "weight": 0.25},
                ]
            }
            pool_path.write_text(json.dumps(configured), encoding="utf-8")

            resolved, loaded = load_fixed_opponent_pool(root, "pool.json")

            self.assertEqual(resolved, pool_path.resolve())
            self.assertEqual(loaded, configured)

    def test_legacy_epoch_artifacts_are_not_automatic_pool_candidates(self) -> None:
        legacy = member("legacy")
        legacy.path = "models/runs/run/current_arch_full_ep10.chk"
        current = member("current")
        registry = LeagueRegistry(1, current.id, 1, [legacy, current])

        payload = build_pool_payload(
            registry,
            preset="active-mixed",
            status_filter="active",
            include_random=False,
            random_weight=0.0,
            selected_member_ids=[],
            normalize_member_weights=False,
            top_k=0,
        )

        self.assertTrue(is_untrusted_legacy_checkpoint(legacy.path))
        self.assertFalse(is_untrusted_legacy_checkpoint(current.path))
        checkpoint_names = [
            item["name"] for item in payload["members"] if item["kind"] == "checkpoint"
        ]
        self.assertEqual(checkpoint_names, ["current"])

    def test_legacy_champion_is_not_inferred_as_a_parent(self) -> None:
        legacy = member("legacy", role="champion", generation=10)
        legacy.path = "current_arch_full_ep10.chk"
        current = member("current", role="main", generation=9)
        registry = LeagueRegistry(10, legacy.id, 2, [legacy, current])

        chosen = choose_parent_member(registry, "", "main")

        self.assertEqual(chosen.id, "current")

    def test_loading_legacy_epoch_artifact_disables_snapshot_eligibility(self) -> None:
        restored = league_member_from_json(
            {
                "id": "legacy",
                "path": "models/runs/run/current_arch_full_ep20.chk",
                "generation": 1,
                "status": "active",
                "collection_weight": 1.0,
                "role": "historical_snapshot",
                "snapshot_eligible": True,
            }
        )

        self.assertFalse(restored.snapshot_eligible)

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

    def test_round_gate_expands_promising_safe_candidate_and_ranks_evidence(self) -> None:
        promising = {
            "valid_win_rate": 0.53,
            "confidence_low": 0.47,
            "valid_games": 200,
        }
        weaker = {
            "valid_win_rate": 0.51,
            "confidence_low": 0.45,
            "valid_games": 1000,
        }
        self.assertTrue(round_screen_should_expand(promising, [], 0.50))
        self.assertFalse(round_screen_should_expand(promising, ["collapse"], 0.50))
        self.assertGreater(round_candidate_rank(promising, []), round_candidate_rank(weaker, []))

    def test_balanced_eval_command_bounds_workers_for_small_replacement_batch(self) -> None:
        args = SimpleNamespace(
            eval_concurrent_games=70,
            eval_worker_pairs=125,
            format="gen9randomdoublesbattle",
            launch_stagger_seconds=0.25,
            resource_check_seconds=2.0,
            min_available_memory_gb=2.0,
            min_available_pagefile_gb=4.0,
            startup_timeout_seconds=120,
        )
        command = build_league_eval_command(
            args, Path("repo"), "replacement", Path("candidate.chk"),
            Path("champion.chk"), "a", 16, 78, 9001,
        )
        self.assertEqual(command[command.index("--games") + 1], "16")
        self.assertEqual(command[command.index("--worker-pairs") + 1], "16")
        self.assertEqual(command[command.index("--concurrent-games") + 1], "16")
        self.assertEqual(command[command.index("--battle-seed-base") + 1], "9001")

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
            format="gen9randomdoublesbattle",
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
            dashboard_write_raw_logs=True,
        )
        command = build_league_live_command(
            args, Path("repo"), Path("pool.json"), Path("parent.chk"),
        )
        self.assertEqual(command[command.index("--anchor-checkpoint") + 1], "anchor.chk")
        self.assertEqual(command[command.index("--anchor-kl-coef") + 1], "0.01")
        self.assertEqual(command[command.index("--target-kl") + 1], "0.02")
        self.assertEqual(command[command.index("--ppo-minibatch-episodes") + 1], "8")
        self.assertEqual(command[command.index("--learning-rate") + 1], "1e-05")
        self.assertEqual(command[command.index("--format") + 1], "gen9randomdoublesbattle")
        self.assertEqual(command[command.index("--dashboard") + 1], "false")
        self.assertEqual(command[command.index("--dashboard-write-raw-logs") + 1], "true")

    def test_league_parser_defines_format_used_by_collection_and_evaluation(self) -> None:
        args = build_league_parser().parse_args([
            "--run-name", "league-run", "--games", "10", "--concurrent-games", "2",
        ])
        self.assertEqual(args.format, "gen9randomdoublesbattle")
        self.assertEqual(args.episode_limit, 256)
        self.assertTrue(args.round_gating)
        self.assertEqual(args.round_screen_games, 100)
        self.assertEqual(args.round_eval_baseline, "parent")
        self.assertTrue(args.registry_update)
        live_command = build_league_live_command(
            args, Path("repo"), Path("pool.json"), Path("parent.chk"),
        )
        eval_command = build_league_eval_command(
            args, Path("repo"), "eval-run", Path("candidate.chk"),
            Path("champion.chk"), "a", 10, 12,
        )
        self.assertEqual(live_command[live_command.index("--format") + 1], args.format)
        self.assertEqual(live_command[live_command.index("--episode-limit") + 1], "256")
        self.assertEqual(eval_command[eval_command.index("--format") + 1], args.format)

    def test_controlled_round_can_gate_against_random_without_registry_updates(self) -> None:
        args = build_league_parser().parse_args([
            "--run-name", "controlled", "--games", "10", "--concurrent-games", "2",
            "--round-eval-baseline", "random", "--registry-update", "false",
            "--opponent-pool", "config/pool.json", "--eval-battle-seed-base", "1180000",
        ])
        parent = Path("parent.chk")

        self.assertEqual(round_evaluation_baseline(args.round_eval_baseline, parent), "random")
        self.assertFalse(args.registry_update)
        self.assertEqual(args.opponent_pool, "config/pool.json")
        self.assertEqual(args.eval_battle_seed_base, 1180000)


class WorkflowDashboardTests(unittest.TestCase):
    def test_gated_child_round_maps_to_outer_dashboard_round(self) -> None:
        state = WorkflowDashboardState("league-run", 4, 500, 100, "League PPO")
        base = BaseWorkflowReporter(state)
        mapped = RoundMappedWorkflowReporter(base, 3, 4)
        mapped.child_line("[live-rl] dashboard phase=collection round=1/1 total=500\n")
        self.assertEqual(state.round_index, 3)
        self.assertEqual(state.rounds_total, 4)
        mapped.child_line("[live-rl] dashboard round_completed=1/1\n")
        self.assertEqual(state.rounds_completed, 3)

    def test_reporter_tracks_nested_collection_training_and_round_progress(self) -> None:
        state = WorkflowDashboardState("league-run", 4, 500, 500, "League PPO")
        reporter = BaseWorkflowReporter(state)

        reporter.child_line("[live-rl] dashboard phase=collection round=2/4 total=500\n")
        reporter.child_line("[selfplay] completed_games=125/500\n")
        self.assertEqual(state.phase, "collection")
        self.assertEqual((state.round_index, state.collection_current, state.collection_total), (2, 125, 500))

        reporter.child_line("[live-rl] dashboard phase=training round=2/4 total=480\n")
        reporter.child_line(
            "[train-ppo] epoch=1 episodes=16/480 policy_loss=0.0123 value_loss=0.1010 "
            "entropy=2.2000 approx_kl=0.0040 anchor_kl_mean=0.0010 "
            "clip_fraction=0.020 hard_kl_breaches=0/2 labels=320\n"
        )
        reporter.child_line("[train] epoch=1 elapsed=2.0s episodes_per_sec=8.00 eta=58.0s\n")
        self.assertEqual((state.training_current, state.training_total), (16, 480))
        self.assertEqual(state.metrics["hard_kl_breaches"], "0/2")
        self.assertAlmostEqual(float(state.metrics["approx_kl"]), 0.004)
        self.assertEqual(state.active_eta_seconds, 58.0)

        reporter.child_line('[live-rl] dashboard collapse_flags=["warn_anchor_kl_high:0.120"]\n')
        reporter.child_line("[live-rl] dashboard round_completed=2/4\n")
        self.assertEqual(state.rounds_completed, 2)
        self.assertEqual(state.collapse_flags, ["warn_anchor_kl_high:0.120"])

    def test_reporter_separates_valid_evaluation_progress_from_raw_attempts(self) -> None:
        state = WorkflowDashboardState("league-run", 2, 500, 500, "League PPO")
        reporter = BaseWorkflowReporter(state)
        state.begin_evaluation("a", valid=490, invalid=3, attempt_total=10)
        reporter.child_line("[selfplay] completed_games=7/10\n")
        self.assertEqual(state.evaluation_valid["a"], 490)
        self.assertEqual(state.evaluation_attempt_current, 7)
        state.update_evaluation("a", valid=500, invalid=4)
        state.finish_evaluation({"valid_win_rate": 0.53, "confidence_low": 0.501, "confidence_high": 0.559})
        state.set_promotion({"status": "confident_winner"})
        payload = state.progress_payload()
        self.assertEqual(payload["evaluation"]["valid"]["a"], 500)
        self.assertEqual(payload["evaluation"]["invalid"]["a"], 4)
        self.assertEqual(payload["promotion_status"], "confident_winner")

    def test_reporter_caps_evaluation_progress_when_draining_overshoots(self) -> None:
        state = WorkflowDashboardState("eval-run", 0, 0, 250, "Balanced evaluation")
        reporter = BaseWorkflowReporter(state)
        state.begin_evaluation("b", valid=234, invalid=12, attempt_total=16)
        reporter.child_line("[selfplay] completed_games=37/16\n")
        self.assertEqual(state.evaluation_attempt_current, 16)
        self.assertEqual(state.evaluation_attempt_total, 16)

    def test_reporter_tracks_evaluation_worker_startup(self) -> None:
        state = WorkflowDashboardState("eval-run", 0, 0, 500, "Balanced evaluation")
        reporter = BaseWorkflowReporter(state)
        state.begin_evaluation("a", valid=0, invalid=0, attempt_total=500)
        reporter.child_line(
            "[selfplay] launching worker pool: initial_pairs=70 initial_workers=140 total_pairs=125\n"
        )
        reporter.child_line("[selfplay] started worker_091_a user=PoryPoolA045 mode=live\n")
        self.assertEqual(state.evaluation_workers_started, 92)
        self.assertEqual(state.evaluation_workers_total, 140)
        payload = state.progress_payload()
        self.assertEqual(payload["evaluation"]["workers_started"], 92)
        self.assertEqual(payload["evaluation"]["workers_total"], 140)

    def test_reporter_tracks_resumable_evaluation_blocks(self) -> None:
        state = WorkflowDashboardState("eval-run", 0, 0, 1000, "Balanced evaluation")
        state.begin_evaluation("a", valid=250, invalid=10, attempt_total=250, block_current=2, blocks_total=9)
        state.complete_evaluation_block()

        evaluation = state.progress_payload()["evaluation"]
        self.assertEqual(evaluation["block_current"], 2)
        self.assertEqual(evaluation["blocks_completed"], 2)
        self.assertEqual(evaluation["blocks_total"], 9)

    def test_reported_command_captures_logs_and_persists_manifest_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.json"
            raw_log_path = root / "child.log"
            manifest: dict[str, object] = {"status": "running"}
            state = WorkflowDashboardState("live-run", 1, 10)
            writer = DashboardProgressWriter(manifest_path, manifest, state)
            reporter = BaseWorkflowReporter(state, writer)
            command = [
                sys.executable,
                "-c",
                "print('[live-rl] dashboard phase=collection round=1/1 total=10'); "
                "print('[selfplay] completed_games=10/10')",
            ]
            run_reported_command(command, root, reporter, raw_log_path)
            reporter.close()
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["progress"]["collection"], {"current": 10, "total": 10})
            self.assertIn("completed_games=10/10", raw_log_path.read_text(encoding="utf-8"))

    def test_failed_child_cleanup_uses_only_matching_process_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "processes.json"
            manifest_path.write_text(json.dumps({
                "owner_pid": 1234,
                "status": "running",
                "children": [{"role": "worker_001_a", "pid": 5678}],
            }), encoding="utf-8")
            completed = SimpleNamespace(returncode=0)
            with patch("live_rl_orchestrator.subprocess.run", return_value=completed) as run:
                stopped = terminate_recorded_child_processes(manifest_path, 1234)
                ignored = terminate_recorded_child_processes(manifest_path, 9999)

            self.assertEqual(stopped, [5678])
            self.assertEqual(ignored, [])
            run.assert_called_once()

    def test_episode_batch_extraction_copies_episode_records_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            episode = '{"type":"episode_complete","battle_id":"battle-1","values":[1,2,3]}'
            (run_dir / "worker_001_a_raw.jsonl").write_text(
                '{"type":"battle_start","battle_id":"battle-1"}\n'
                + episode + "\n"
                + '{"type":"battle_end","battle_id":"battle-1"}\n',
                encoding="utf-8",
            )
            (run_dir / "worker_002_b_raw.jsonl").write_text(
                '{"type":"episode_complete","battle_id":"wrong-side"}\n',
                encoding="utf-8",
            )
            output = run_dir / "episode_batch_a.jsonl"

            stats = extract_episode_batch(run_dir, "a", output)

            self.assertEqual(stats, {
                "source_files": 1,
                "scanned_lines": 3,
                "written_episodes": 1,
            })
            self.assertEqual(output.read_text(encoding="utf-8"), episode + "\n")


class BalancedCheckpointEvalTests(unittest.TestCase):
    def test_balanced_eval_splits_targets_into_base_and_replacement_blocks(self) -> None:
        self.assertEqual(evaluation_block_limit(1000, 250, 5), 9)
        self.assertEqual(evaluation_block_limit(501, 250, 2), 5)

    def test_balanced_eval_config_defaults_allow_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "balanced.toml"
            config_path.write_text(
                "games_per_side = 100\n"
                "concurrent_games = 4\n"
                "worker_pairs = 6\n"
                "dashboard = false\n",
                encoding="utf-8",
            )

            args = parse_balanced_eval_args([
                "--config", str(config_path),
                "--run-name", "configured-eval",
                "--candidate-checkpoint", "candidate.chk",
                "--baseline", "random",
                "--games-per-side", "200",
            ])

            self.assertEqual(args.games_per_side, 200)
            self.assertEqual(args.concurrent_games, 4)
            self.assertEqual(args.worker_pairs, 6)
            self.assertEqual(args.block_games, 250)
            self.assertFalse(args.dashboard)
            self.assertEqual(args.config, str(config_path.resolve()))

    def test_balanced_eval_detects_a_nonacting_candidate_runtime(self) -> None:
        self.assertTrue(
            candidate_runtime_appears_broken(
                {"matches_played": 250, "total_moves": 0}, 0, 0,
            )
        )
        self.assertFalse(
            candidate_runtime_appears_broken(
                {"matches_played": 250, "total_moves": 1}, 0, 0,
            )
        )
        self.assertFalse(
            candidate_runtime_appears_broken(
                {"matches_played": 250, "total_moves": 0}, 0, 1,
            )
        )

    def test_cli_maps_to_shared_balanced_evaluation_arguments(self) -> None:
        args = prepare_shared_args(build_balanced_eval_parser().parse_args([
            "--run-name", "round03-eval",
            "--candidate-checkpoint", "round03.chk",
            "--baseline-checkpoint", "champion.chk",
            "--games-per-side", "300",
            "--concurrent-games", "12",
            "--worker-pairs", "20",
            "--max-replacement-attempts", "7",
        ]))
        self.assertEqual(args.eval_run_name, "round03-eval")
        self.assertEqual(args.eval_games, 300)
        self.assertEqual(args.eval_block_games, 250)
        self.assertEqual(args.eval_concurrent_games, 12)
        self.assertEqual(args.eval_worker_pairs, 20)
        self.assertEqual(args.eval_max_replacement_attempts, 7)
        self.assertEqual(args.promote_threshold, args.promotion_min_win_rate)
        command = build_league_eval_command(
            args, Path("repo"), args.run_name, Path("round03.chk"),
            Path("champion.chk"), "b", args.eval_games, args.pool_seed,
        )
        self.assertEqual(command[command.index("--games") + 1], "300")
        self.assertEqual(command[command.index("--format") + 1], "gen9randomdoublesbattle")

    def test_failed_evaluation_block_gets_a_clean_retry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / "candidate.chk"
            candidate.write_bytes(b"candidate")
            args = SimpleNamespace(eval_run_name="resume-eval", resume=True)

            run_name, summary_path, reusable = select_evaluation_block_run(
                args, root, candidate, "random", "a", 1, 250, 7000,
            )
            self.assertEqual(run_name, "resume-eval_side_a_block_01")
            self.assertFalse(reusable)

            summary_path.parent.mkdir(parents=True, exist_ok=True)
            retry_name, retry_path, reusable = select_evaluation_block_run(
                args, root, candidate, "random", "a", 1, 250, 7000,
            )
            self.assertEqual(retry_name, "resume-eval_side_a_block_01_retry_02")
            self.assertFalse(reusable)

            retry_path.parent.mkdir(parents=True, exist_ok=True)
            retry_path.write_text(json.dumps({
                "status": "completed",
                "target_games": 250,
                "battle_seed_base": 7000,
                "model_specs": {
                    "a": model_spec_payload(candidate),
                    "b": model_spec_payload("random"),
                },
            }), encoding="utf-8")
            selected_name, _, reusable = select_evaluation_block_run(
                args, root, candidate, "random", "a", 1, 250, 7000,
            )
            self.assertEqual(selected_name, retry_name)
            self.assertTrue(reusable)

    def test_balanced_eval_resumes_completed_blocks_after_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / "candidate.chk"
            candidate.write_bytes(b"candidate")
            args = SimpleNamespace(
                eval_run_name="crash-resume",
                eval_games=500,
                eval_block_games=250,
                eval_max_replacement_attempts=2,
                eval_battle_seed_base=9000,
                eval_worker_pairs=4,
                eval_concurrent_games=4,
                pool_seed=12,
                format="gen9randomdoublesbattle",
                launch_stagger_seconds=0.0,
                resource_check_seconds=1.0,
                min_available_memory_gb=1.0,
                min_available_pagefile_gb=1.0,
                startup_timeout_seconds=30,
                dashboard_write_raw_logs=False,
                resume=True,
            )
            reporter = BaseWorkflowReporter(
                WorkflowDashboardState("crash-resume", 0, 0, 500, "Balanced evaluation")
            )
            calls: list[str] = []

            def finish_block(command: list[str], *_args: object, **_kwargs: object) -> None:
                run_name = command[command.index("--run-name") + 1]
                calls.append(run_name)
                run_dir = root / "matches" / "runs" / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                if len(calls) == 2:
                    raise RuntimeError("simulated native crash")
                games = int(command[command.index("--games") + 1])
                model_a = command[command.index("--model-a") + 1]
                model_b = command[command.index("--model-b") + 1]
                seed = int(command[command.index("--battle-seed-base") + 1])
                half = games // 2
                group_a = {"matches_played": games, "earned_wins": half, "total_moves": games}
                group_b = {"matches_played": games, "earned_wins": games - half, "total_moves": games}
                (run_dir / f"{run_name}_summary.json").write_text(json.dumps({
                    "status": "completed",
                    "target_games": games,
                    "battle_seed_base": seed,
                    "model_specs": {
                        "a": model_spec_payload(model_a),
                        "b": model_spec_payload(model_b),
                    },
                    "group_stats": {"a": group_a, "b": group_b},
                }), encoding="utf-8")

            with patch("league_rl_orchestrator.run_reported_command", side_effect=finish_block):
                with self.assertRaisesRegex(RuntimeError, "simulated native crash"):
                    run_balanced_valid_evaluation(
                        args, root, candidate, "random", reporter, root / "logs",
                    )
                _, summary = run_balanced_valid_evaluation(
                    args, root, candidate, "random", reporter, root / "logs",
                )

            self.assertEqual(calls.count("crash-resume_side_a_block_01"), 1)
            self.assertIn("crash-resume_side_a_block_02_retry_02", calls)
            self.assertEqual(summary["valid_games"], 1000)
            self.assertEqual(len(summary["run_names"]), 4)


class ResumeAndCollapseTests(unittest.TestCase):
    def test_selfplay_worker_forwards_dense_rewards_to_battle_agent(self) -> None:
        spec = WorkerSpec(
            worker_id=1,
            pair_index=0,
            model_group="a",
            checkpoint_path="parent.chk",
            mode="live",
            username="PoryA",
            replay_save_token="dense-worker",
            replay_path=Path("dense-worker.jsonl"),
            stdout_log_path=Path("dense-worker.log"),
            shutdown_path=Path("dense-worker.stop"),
        )
        worker = WorkerProcess(
            spec,
            WorkerLaunchIdentity("live", "parent.chk", "parent.chk"),
            Path.cwd(),
            sys.executable,
            "ws://127.0.0.1:8000/showdown/websocket",
            "gen9randomdoublesbattle",
            5.0,
            "dense_additive",
            0.1,
            0.25,
            0.4,
            lambda *_args: None,
            False,
        )

        command = worker.command()

        self.assertEqual(command[command.index("--reward-mode") + 1], "dense_additive")
        self.assertEqual(command[command.index("--dense-additive-faint-swing-weight") + 1], "0.25")

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
            reward_mode="dense_additive",
            dense_additive_hp_swing_weight=0.1,
            dense_additive_faint_swing_weight=0.25,
            dense_additive_reward_clip=0.4,
        )
        snapshot = Path("round_pool.json")
        command = build_selfplay_command(args, Path.cwd(), "collect", Path("actor.chk"), snapshot)
        pool_index = command.index("--model-b-pool")
        self.assertEqual(command[pool_index + 1], str(snapshot))
        self.assertEqual(command[command.index("--reward-mode") + 1], "dense_additive")
        self.assertEqual(command[command.index("--dense-additive-hp-swing-weight") + 1], "0.1")

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
            summary.write_text('{"checkpoint_published": false}\n', encoding="utf-8")
            self.assertFalse(round_manifest_completed(manifest))
            summary.write_text("{}\n", encoding="utf-8")
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
            for path in (checkpoint, used_pool, next_pool):
                path.write_bytes(b"artifact")
            summary.write_text("{}\n", encoding="utf-8")
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
