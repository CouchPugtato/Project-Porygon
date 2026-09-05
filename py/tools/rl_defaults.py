from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPO_ROOT / "config" / "rl_defaults.toml"
REWARD_DEFAULTS_PATH = REPO_ROOT / "config" / "reward_weights.toml"


def load_rl_defaults(path: Path = DEFAULTS_PATH) -> dict[str, object]:
    payload: dict[str, object] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise RuntimeError(f"invalid defaults file {path}:{line_number}: expected key = value")
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip()
        value_text = raw_value.strip()
        if not key or not value_text:
            raise RuntimeError(f"invalid defaults file {path}:{line_number}: empty key or value")
        lowered = value_text.lower()
        if lowered in {"true", "false"}:
            value: object = lowered == "true"
        elif len(value_text) >= 2 and value_text[0] == '"' and value_text[-1] == '"':
            value = value_text[1:-1]
        else:
            try:
                value = float(value_text) if any(char in value_text.lower() for char in (".", "e")) else int(value_text)
            except ValueError as exc:
                raise RuntimeError(f"invalid defaults file {path}:{line_number}: bad value for {key}") from exc
        payload[key] = value
    return payload


def load_cli_defaults(path: Path) -> list[str]:
    """Translate a flat config file into arguments that explicit CLI flags can override."""
    args: list[str] = []
    for key, value in load_rl_defaults(path).items():
        if isinstance(value, bool):
            value_text = "true" if value else "false"
        else:
            value_text = str(value)
        args.extend(["--" + key.replace("_", "-"), value_text])
    return args


RL_DEFAULTS = load_rl_defaults()
REWARD_DEFAULTS = load_rl_defaults(REWARD_DEFAULTS_PATH)


def float_default(name: str) -> float:
    value = RL_DEFAULTS.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"RL default '{name}' must be numeric")
    return float(value)


def int_default(name: str) -> int:
    value = RL_DEFAULTS.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"RL default '{name}' must be an integer")
    return value


def bool_default(name: str) -> bool:
    value = RL_DEFAULTS.get(name)
    if not isinstance(value, bool):
        raise RuntimeError(f"RL default '{name}' must be a boolean")
    return value


def reward_float_default(name: str) -> float:
    value = REWARD_DEFAULTS.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"reward default '{name}' must be numeric")
    return float(value)
