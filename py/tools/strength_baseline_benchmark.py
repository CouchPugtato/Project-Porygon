from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_io import write_json_atomically
from league_rl_orchestrator import balanced_evaluation_artifacts_match
from strength_baseline_audit import (
    MODEL_FIELDS,
    BenchmarkModel,
    benchmark_contract,
    benchmark_models,
    build_learning_audit,
    evaluation_record,
    head_to_head_record,
    rank_evaluations,
    resolve_path,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "strength_baseline_benchmark.toml"
DEFAULT_BENCHMARK_ROOT = Path("models") / "benchmarks"
DEFAULT_MATCH_ROOT = Path("matches") / "runs"
DEFAULT_CLIENT_EXE = Path("build-fresh") / "showdown_client.exe"


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true"}:
        return True
    if lowered in {"0", "false"}:
        return False
    raise argparse.ArgumentTypeError("value must be true or false")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def load_config_args(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"strength benchmark config not found: {path}")
    arguments: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise SystemExit(f"invalid config {path}:{line_number}: expected key = value")
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip()
        value = raw_value.strip()
        if not key or not value:
            raise SystemExit(f"invalid config {path}:{line_number}: empty key or value")
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = bytes(value[1:-1], "utf-8").decode("unicode_escape")
        arguments.extend(["--" + key.replace("_", "-"), value])
    return arguments


def config_path_from_args(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    known, _ = parser.parse_known_args(argv)
    path = Path(known.config)
    return path if path.is_absolute() else (resolve_repo_root() / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the model-learning recovery strength benchmark.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-name", required=True)
    for _, _, checkpoint_field, provenance_field, _ in MODEL_FIELDS:
        parser.add_argument("--" + checkpoint_field.replace("_", "-"), required=True)
        parser.add_argument("--" + provenance_field.replace("_", "-"), required=True)
    parser.add_argument("--supervised-overfit-report", required=True)
    parser.add_argument("--reconstruction-tests-exe", required=True)
    parser.add_argument("--screen-games-per-side", type=positive_int, default=250)
    parser.add_argument("--final-games-per-side", type=positive_int, default=1000)
    parser.add_argument("--head-to-head-games-per-side", type=positive_int, default=1000)
    parser.add_argument("--finalist-count", type=positive_int, default=2)
    parser.add_argument("--screen-pool-seed", type=int, default=12001)
    parser.add_argument("--final-pool-seed", type=int, default=12002)
    parser.add_argument("--head-to-head-pool-seed", type=int, default=12003)
    parser.add_argument("--concurrent-games", type=positive_int, default=50)
    parser.add_argument("--worker-pairs", type=positive_int, default=50)
    parser.add_argument("--max-replacement-attempts", type=positive_int, default=5)
    parser.add_argument("--format", default="gen9randomdoublesbattle")
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.25)
    parser.add_argument("--resource-check-seconds", type=float, default=2.0)
    parser.add_argument("--min-available-memory-gb", type=float, default=2.0)
    parser.add_argument("--min-available-pagefile-gb", type=float, default=4.0)
    parser.add_argument("--startup-timeout-seconds", type=positive_int, default=120)
    parser.add_argument("--dashboard", type=parse_bool, default=True)
    parser.add_argument("--dashboard-refresh-per-second", type=positive_float, default=8.0)
    parser.add_argument("--dashboard-write-raw-logs", type=parse_bool, default=True)
    parser.add_argument("--resume", type=parse_bool, default=True)
    return parser


def load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return payload


def run_trainer_preflight(
    repo_root: Path,
    overfit_report_path: Path,
    reconstruction_tests_exe: Path,
) -> dict[str, object]:
    if not overfit_report_path.exists():
        raise SystemExit(f"supervised overfit report not found: {overfit_report_path}")
    overfit = load_json(overfit_report_path)
    overfit_checks = overfit.get("checks", {})
    required_checks = (
        "action_loss_reduced_50_percent",
        "full_turn_accuracy_at_least_80_percent",
        "selected_action_probability_increased",
        "selected_target_probability_increased",
        "value_error_decreased",
        "outputs_finite",
        "predictions_legal",
    )
    supervised_passed = (
        overfit.get("diagnostic") == "supervised_overfit"
        and int(overfit.get("metrics_version", 0) or 0) == 2
        and bool(overfit.get("passed"))
        and isinstance(overfit_checks, dict)
        and all(bool(overfit_checks.get(check)) for check in required_checks)
    )

    if not reconstruction_tests_exe.exists():
        raise SystemExit(f"reconstruction test executable not found: {reconstruction_tests_exe}")
    completed = subprocess.run(
        [str(reconstruction_tests_exe)],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    ppo_passed = completed.returncode == 0
    results: dict[str, object] = {
        "supervised_overfit": {
            "passed": supervised_passed,
            "report_path": str(overfit_report_path),
            "seed": overfit.get("seed"),
            "epochs": overfit.get("epochs"),
            "optimizer": overfit.get("optimizer"),
            "checks": overfit_checks,
            "failure_reasons": overfit.get("failure_reasons", []),
        },
        "ppo_direction": {
            "passed": ppo_passed,
            "test_case": "test_ppo_update_moves_policy_and_value_in_expected_directions",
            "executable": str(reconstruction_tests_exe),
            "exit_code": completed.returncode,
            "output": completed.stdout.strip(),
        },
    }
    results["passed"] = supervised_passed and ppo_passed
    return results


def runtime_reported_ready(stdout: str) -> bool:
    for line in stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and message.get("type") == "ready":
            return True
    return False


def run_checkpoint_preflight(
    repo_root: Path,
    models: list[BenchmarkModel],
    timeout_seconds: int,
) -> dict[str, object]:
    client_exe = (repo_root / DEFAULT_CLIENT_EXE).resolve()
    if not client_exe.exists():
        raise SystemExit(f"runtime client not found: {client_exe}")

    model_results: dict[str, object] = {}
    all_passed = True
    for model in models:
        try:
            completed = subprocess.run(
                [str(client_exe), "--battle-agent", model.checkpoint],
                cwd=repo_root,
                input="",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
            ready = runtime_reported_ready(completed.stdout)
            passed = completed.returncode == 0 and ready
            details: dict[str, object] = {
                "passed": passed,
                "checkpoint": model.checkpoint,
                "exit_code": completed.returncode,
                "ready": ready,
                "stderr_tail": completed.stderr.strip()[-2000:],
            }
        except subprocess.TimeoutExpired as exc:
            passed = False
            details = {
                "passed": False,
                "checkpoint": model.checkpoint,
                "exit_code": None,
                "ready": False,
                "failure": f"runtime load probe exceeded {timeout_seconds}s",
                "stderr_tail": str(exc.stderr or "")[-2000:],
            }
        except OSError as exc:
            passed = False
            details = {
                "passed": False,
                "checkpoint": model.checkpoint,
                "exit_code": None,
                "ready": False,
                "failure": str(exc),
                "stderr_tail": "",
            }
        model_results[model.id] = details
        all_passed = all_passed and passed

    return {
        "passed": all_passed,
        "client_exe": str(client_exe),
        "models": model_results,
    }


def safe_run_token(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_").lower()


def update_manifest_progress(manifest: dict[str, object]) -> None:
    evaluations = manifest.get("evaluations", {})
    records = list(evaluations.values()) if isinstance(evaluations, dict) else []
    manifest["completed_evaluations"] = len(records)
    manifest["completed_valid_games"] = sum(
        int(record.get("valid_games", 0)) for record in records if isinstance(record, dict)
    )


def evaluation_summary_path(repo_root: Path, run_name: str) -> Path:
    return repo_root / DEFAULT_MATCH_ROOT / run_name / f"{run_name}_summary.json"


def balanced_eval_command(
    args: argparse.Namespace,
    repo_root: Path,
    run_name: str,
    candidate_checkpoint: str,
    baseline: str,
    games_per_side: int,
    pool_seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "py" / "tools" / "balanced_checkpoint_eval.py"),
        "--run-name",
        run_name,
        "--candidate-checkpoint",
        candidate_checkpoint,
        "--baseline",
        baseline,
        "--games-per-side",
        str(games_per_side),
        "--concurrent-games",
        str(args.concurrent_games),
        "--worker-pairs",
        str(args.worker_pairs),
        "--max-replacement-attempts",
        str(args.max_replacement_attempts),
        "--pool-seed",
        str(pool_seed),
        "--battle-seed-base",
        str(pool_seed),
        "--format",
        args.format,
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
        "--dashboard",
        "true" if args.dashboard else "false",
        "--dashboard-refresh-per-second",
        str(args.dashboard_refresh_per_second),
        "--dashboard-write-raw-logs",
        "true" if args.dashboard_write_raw_logs else "false",
        "--resume",
        "true" if args.resume else "false",
    ]


def run_or_reuse_evaluation(
    args: argparse.Namespace,
    repo_root: Path,
    run_name: str,
    candidate_checkpoint: str,
    baseline: str,
    games_per_side: int,
    pool_seed: int,
) -> tuple[Path, dict[str, object], bool]:
    summary_path = evaluation_summary_path(repo_root, run_name)
    if args.resume and summary_path.exists():
        summary = load_json(summary_path)
        if balanced_evaluation_artifacts_match(
            summary,
            Path(candidate_checkpoint),
            baseline if baseline == "random" else Path(baseline),
            games_per_side,
            pool_seed,
        ):
            print(f"[strength-benchmark] reusing {run_name}", flush=True)
            return summary_path, summary, True

    command = balanced_eval_command(
        args,
        repo_root,
        run_name,
        candidate_checkpoint,
        baseline,
        games_per_side,
        pool_seed,
    )
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"balanced evaluation failed with exit code {completed.returncode}: {run_name}")
    summary = load_json(summary_path)
    if not balanced_evaluation_artifacts_match(
        summary,
        Path(candidate_checkpoint),
        baseline if baseline == "random" else Path(baseline),
        games_per_side,
        pool_seed,
    ):
        raise SystemExit(f"balanced evaluation produced mismatched artifacts: {run_name}")
    return summary_path, summary, False


def main() -> None:
    cli_args = sys.argv[1:]
    config_path = config_path_from_args(cli_args)
    args = build_parser().parse_args(load_config_args(config_path) + cli_args)
    if args.finalist_count != 2:
        raise SystemExit("the canonical benchmark requires --finalist-count 2")

    repo_root = resolve_repo_root()
    models = benchmark_models(args, repo_root)
    overfit_report = resolve_path(repo_root, args.supervised_overfit_report)
    reconstruction_tests = resolve_path(repo_root, args.reconstruction_tests_exe)
    trainer_tests = run_trainer_preflight(repo_root, overfit_report, reconstruction_tests)
    checkpoint_loading = run_checkpoint_preflight(
        repo_root, models, args.startup_timeout_seconds,
    )
    trainer_tests["checkpoint_loading"] = checkpoint_loading
    trainer_tests["passed"] = bool(trainer_tests["passed"]) and bool(checkpoint_loading["passed"])
    if not trainer_tests["passed"]:
        failed_models = [
            model_id
            for model_id, result in checkpoint_loading["models"].items()
            if not bool(result.get("passed"))
        ]
        suffix = f"; checkpoint load failed for: {', '.join(failed_models)}" if failed_models else ""
        raise SystemExit(f"learnability preflight failed; benchmark was not started{suffix}")

    benchmark_dir = (repo_root / DEFAULT_BENCHMARK_ROOT / args.run_name).resolve()
    manifest_path = benchmark_dir / f"{args.run_name}_manifest.json"
    audit_path = benchmark_dir / f"{args.run_name}_learning_audit.json"
    contract = benchmark_contract(args, models)
    existing_manifest: dict[str, object] | None = None
    if args.resume and manifest_path.exists():
        existing_manifest = load_json(manifest_path)
        if existing_manifest.get("contract") != contract:
            raise SystemExit("cannot resume: benchmark config or checkpoint set changed")
        if existing_manifest.get("status") == "completed" and audit_path.exists():
            print(f"[strength-benchmark] benchmark already completed: {audit_path}", flush=True)
            return

    if existing_manifest is None:
        manifest: dict[str, object] = {
            "run_name": args.run_name,
            "started_at_unix": time.time(),
            "evaluations": {},
        }
    else:
        manifest = existing_manifest
        manifest["resumed_at_unix"] = time.time()
    manifest.update(
        {
            "status": "running",
            "config_path": str(config_path),
            "contract": contract,
            "trainer_test_results": trainer_tests,
            "audit_path": str(audit_path),
            "planned_evaluations": len(models) + args.finalist_count + 1,
            "planned_valid_games": (
                len(models) * 2 * args.screen_games_per_side
                + args.finalist_count * 2 * args.final_games_per_side
                + 2 * args.head_to_head_games_per_side
            ),
        }
    )
    if not isinstance(manifest.get("evaluations"), dict):
        raise SystemExit("cannot resume: benchmark manifest evaluations are invalid")
    manifest.pop("failure_reason", None)
    update_manifest_progress(manifest)
    write_json_atomically(manifest_path, manifest)

    try:
        screening: list[dict[str, object]] = []
        for index, model in enumerate(models, start=1):
            print(f"[strength-benchmark] screening {index}/{len(models)}: {model.label}", flush=True)
            run_name = f"{args.run_name}_screen_{safe_run_token(model.id)}"
            summary_path, summary, _ = run_or_reuse_evaluation(
                args,
                repo_root,
                run_name,
                model.checkpoint,
                "random",
                args.screen_games_per_side,
                args.screen_pool_seed,
            )
            record = evaluation_record(model, "screen", summary_path, summary)
            screening.append(record)
            manifest["evaluations"][f"screen:{model.id}"] = record
            update_manifest_progress(manifest)
            write_json_atomically(manifest_path, manifest)

        screening_ranking = rank_evaluations(screening)
        model_by_id = {model.id: model for model in models}
        finalists = [
            model_by_id[str(record["model"]["id"])]
            for record in screening_ranking[: args.finalist_count]
        ]
        manifest["screening_ranking"] = screening_ranking
        manifest["finalists"] = [asdict(model) for model in finalists]
        write_json_atomically(manifest_path, manifest)

        final_random: list[dict[str, object]] = []
        for index, model in enumerate(finalists, start=1):
            print(f"[strength-benchmark] final vs random {index}/2: {model.label}", flush=True)
            run_name = f"{args.run_name}_final_random_{safe_run_token(model.id)}"
            summary_path, summary, _ = run_or_reuse_evaluation(
                args,
                repo_root,
                run_name,
                model.checkpoint,
                "random",
                args.final_games_per_side,
                args.final_pool_seed,
            )
            record = evaluation_record(model, "final_random", summary_path, summary)
            final_random.append(record)
            manifest["evaluations"][f"final_random:{model.id}"] = record
            update_manifest_progress(manifest)
            write_json_atomically(manifest_path, manifest)

        final_ranking = rank_evaluations(final_random)
        leading = model_by_id[str(final_ranking[0]["model"]["id"])]
        runner_up = model_by_id[str(final_ranking[1]["model"]["id"])]
        print(
            f"[strength-benchmark] head-to-head: {leading.label} vs {runner_up.label}",
            flush=True,
        )
        head_run_name = (
            f"{args.run_name}_head_{safe_run_token(leading.id)}_vs_{safe_run_token(runner_up.id)}"
        )
        summary_path, summary, _ = run_or_reuse_evaluation(
            args,
            repo_root,
            head_run_name,
            leading.checkpoint,
            runner_up.checkpoint,
            args.head_to_head_games_per_side,
            args.head_to_head_pool_seed,
        )
        head_record = head_to_head_record(leading, runner_up, summary_path, summary)
        manifest["evaluations"]["head_to_head"] = head_record
        update_manifest_progress(manifest)
        write_json_atomically(manifest_path, manifest)

        audit = build_learning_audit(
            args.run_name,
            trainer_tests,
            screening_ranking,
            final_ranking,
            head_record,
        )
        write_json_atomically(audit_path, audit)
        manifest["status"] = "completed"
        manifest["completed_at_unix"] = time.time()
        manifest["final_random_ranking"] = final_ranking
        manifest["head_to_head"] = head_record
        manifest["learning_audit"] = audit
        write_json_atomically(manifest_path, manifest)
        print(f"[strength-benchmark] learning audit: {audit_path}", flush=True)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["stopped_at_unix"] = time.time()
        manifest["failure_reason"] = str(exc)
        write_json_atomically(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
