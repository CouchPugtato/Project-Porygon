from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data"

FILE_MAP = {
    "species_ids.txt": "pokedex.ts",
    "move_ids.txt": "moves.ts",
    "item_ids.txt": "items.ts",
    "ability_ids.txt": "abilities.ts",
    "conditions_ids.txt": "conditions.ts",
}


def normalize_token(token: str) -> str:
    return "".join(ch.lower() for ch in token if ch.isalnum())


def load_text(source_dir: Path | None, base_url: str | None, filename: str) -> str:
    if source_dir is not None:
        return (source_dir / filename).read_text(encoding="utf-8")
    if not base_url:
        raise ValueError(f"no source for {filename}")
    url = f"{base_url.rstrip('/')}/{filename}"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def extract_top_level_keys(ts_text: str) -> list[str]:
    keys: list[str] = []
    root_open = ts_text.find("{")
    if root_open < 0:
        return keys

    depth = 0
    in_string = False
    string_quote = ""
    escape = False
    pending_key = []
    reading_key = False
    candidate = ""
    expecting_value = False

    for ch in ts_text[root_open:]:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == string_quote:
                in_string = False
                if reading_key:
                    candidate = "".join(pending_key)
                    pending_key = []
                    reading_key = False
                    expecting_value = True
            elif reading_key:
                pending_key.append(ch)
            continue

        if ch in ("'", '"'):
            if depth == 1:
                in_string = True
                string_quote = ch
                pending_key = []
                reading_key = True
            continue

        if ch == "{":
            if depth == 1 and expecting_value and candidate:
                keys.append(candidate)
                candidate = ""
                expecting_value = False
            depth += 1
            continue

        if ch == "}":
            depth -= 1
            if depth <= 0:
                break
            continue

        if depth != 1:
            continue

        if ch == ":" and candidate:
            expecting_value = True
            continue

        if ch == ",":
            candidate = ""
            expecting_value = False
            continue

        if ch.isspace():
            continue

        if not candidate and re.match(r"[A-Za-z0-9_]", ch):
            candidate = ch
            continue

        if candidate and re.match(r"[A-Za-z0-9_]", ch):
            candidate += ch
            continue

        if candidate and ch == ":":
            expecting_value = True
            continue

        if candidate and ch not in "{":
            candidate = ""

    normalized: list[str] = []
    seen: set[str] = set()
    for key in keys:
        token = normalize_token(key)
        if token and token not in seen:
            normalized.append(token)
            seen.add(token)
    return normalized


def write_vocab(out_path: Path, source_filename: str, tokens: list[str]) -> None:
    out_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Pokemon Showdown *.ts data keys into the current txt vocab format."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing local Showdown data files like pokedex.ts, moves.ts, items.ts, abilities.ts.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Raw base URL for Showdown data files when --source-dir is not used.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write species_ids.txt, move_ids.txt, item_ids.txt, ability_ids.txt, and conditions_ids.txt.",
    )
    args = parser.parse_args()

    if args.source_dir is None and not args.base_url:
        parser.error("either --source-dir or --base-url is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for out_name, source_name in FILE_MAP.items():
        ts_text = load_text(args.source_dir, args.base_url, source_name)
        tokens = extract_top_level_keys(ts_text)
        if not tokens:
            raise RuntimeError(f"failed to extract tokens from {source_name}")
        write_vocab(args.out_dir / out_name, source_name, tokens)
        print(f"wrote {out_name}: {len(tokens)} entries")

    return 0


if __name__ == "__main__":
    sys.exit(main())
