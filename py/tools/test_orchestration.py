from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from league_manage import LeagueEval, LeagueMember, LeagueRegistry, build_pool_payload
from league_rl_orchestrator import build_weighted_pool, maybe_promote_candidate
from live_rl_orchestrator import collapse_flags_from_training_summary, round_manifest_completed


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

    def test_weighted_pool_excludes_parent_and_pfavors_harder_history(self) -> None:
        champion = member("champion", generation=10)
        parent = member("parent", generation=11)
        hard = member("hard_history", role="historical_snapshot", win_rate=0.20)
        easy = member("easy_history", role="historical_snapshot", win_rate=0.80)
        registry = LeagueRegistry(11, champion.id, 8, [champion, parent, hard, easy])
        payload = build_weighted_pool(registry, "main", parent.id, 0.0)
        weights = {str(item["name"]): float(item["weight"]) for item in payload["members"]}
        self.assertNotIn(parent.id, weights)
        self.assertGreater(weights[hard.id], weights[easy.id])


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
