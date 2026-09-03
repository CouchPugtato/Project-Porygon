from __future__ import annotations

from pathlib import Path


def is_random_model_spec(value: str | Path) -> bool:
    return str(value).strip().lower() == "random"


def model_spec_payload(value: str | Path) -> dict[str, str]:
    if is_random_model_spec(value):
        return {"kind": "random"}
    return {"kind": "checkpoint", "path": str(Path(value))}


def recorded_model_spec_matches(
    recorded: object,
    expected: str | Path,
    repo_root: Path,
) -> bool:
    if not isinstance(recorded, dict):
        return False
    if is_random_model_spec(expected):
        return str(recorded.get("kind", "")).strip().lower() == "random"

    recorded_path = str(recorded.get("path", "")).strip()
    if not recorded_path:
        return False
    path = Path(recorded_path)
    resolved_recorded = path if path.is_absolute() else (repo_root / path).resolve()
    return resolved_recorded == Path(expected).resolve()
