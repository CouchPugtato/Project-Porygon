from __future__ import annotations

from pathlib import Path


PATCH_VERSION = 1
LADDERS_MARKER = "/* project-porygon deterministic battle seeds v1 */"
ROOM_BATTLE_MARKER = "/* project-porygon deterministic player seeds v1 */"


def _replace_once(source: str, needle: str, replacement: str, path: Path) -> str:
    count = source.count(needle)
    if count != 1:
        raise RuntimeError(
            f"cannot apply deterministic Showdown patch to {path}: "
            f"expected one patch anchor, found {count}"
        )
    return source.replace(needle, replacement, 1)


def apply_deterministic_battle_seed_patch(showdown_dir: Path) -> bool:
    """Install the opt-in local Showdown hook used for matched evaluations.

    The hook is dormant unless PORYGON_BATTLE_SEED_BASE is present in the
    Showdown server environment. It seeds the battle and both random teams.
    """
    server_dir = showdown_dir / "server"
    ladders_path = server_dir / "ladders.ts"
    room_battle_path = server_dir / "room-battle.ts"
    if not ladders_path.exists() or not room_battle_path.exists():
        raise FileNotFoundError(
            f"Pokemon Showdown server sources not found under {showdown_dir}"
        )

    changed = False
    ladders = ladders_path.read_text(encoding="utf-8")
    if LADDERS_MARKER not in ladders:
        constants_anchor = "const PERIODIC_MATCH_INTERVAL = 60 * SECONDS;"
        seed_support = f"""const PERIODIC_MATCH_INTERVAL = 60 * SECONDS;

{LADDERS_MARKER}
const porygonBattleSeedBase = Number.parseInt(process.env.PORYGON_BATTLE_SEED_BASE || '', 10);
let porygonBattleSeedIndex = 0;

function porygonSeed(sequenceIndex: number): PRNGSeed {{
\tlet value = (porygonBattleSeedBase + Math.imul(sequenceIndex + 1, 0x9E3779B9)) >>> 0;
\tconst words: number[] = [];
\tfor (let index = 0; index < 4; index++) {{
\t\tvalue = (Math.imul(value, 1664525) + 1013904223) >>> 0;
\t\twords.push(value);
\t}}
\treturn `gen5,${{words.join(',')}}` as PRNGSeed;
}}

function nextPorygonBattleSeeds() {{
\tif (!Number.isSafeInteger(porygonBattleSeedBase) || porygonBattleSeedBase < 0) return null;
\tconst offset = porygonBattleSeedIndex++ * 3;
\treturn {{battle: porygonSeed(offset), players: [porygonSeed(offset + 1), porygonSeed(offset + 2)]}};
}}"""
        ladders = _replace_once(ladders, constants_anchor, seed_support, ladders_path)
        battle_anchor = """\t\tconst delayedStart = format.playerCount > players.length ? 'multi' : false;
\t\treturn Rooms.createBattle({
\t\t\tformat: formatid,
\t\t\tplayers,
\t\t\trated: minRating,
\t\t\tchallengeType: readies[0].challengeType,
\t\t\tdelayedStart,
\t\t});"""
        battle_replacement = """\t\tconst delayedStart = format.playerCount > players.length ? 'multi' : false;
\t\tconst porygonSeeds = nextPorygonBattleSeeds();
\t\treturn Rooms.createBattle({
\t\t\tformat: formatid,
\t\t\tplayers,
\t\t\trated: minRating,
\t\t\tchallengeType: readies[0].challengeType,
\t\t\tdelayedStart,
\t\t\tseed: porygonSeeds?.battle,
\t\t\tplayerSeeds: porygonSeeds?.players,
\t\t});"""
        ladders = _replace_once(ladders, battle_anchor, battle_replacement, ladders_path)
        ladders_path.write_text(ladders, encoding="utf-8")
        changed = True

    room_battle = room_battle_path.read_text(encoding="utf-8")
    if ROOM_BATTLE_MARKER not in room_battle:
        interface_anchor = "\tseed?: PRNGSeed;\n\troomid?: RoomID;"
        interface_replacement = (
            f"\tseed?: PRNGSeed;\n\t{ROOM_BATTLE_MARKER}\n"
            "\tplayerSeeds?: PRNGSeed[];\n\troomid?: RoomID;"
        )
        room_battle = _replace_once(
            room_battle, interface_anchor, interface_replacement, room_battle_path,
        )
        player_anchor = """\t\t\tconst options = {
\t\t\t\tname: player.name,
\t\t\t\tavatar: user.avatar,
\t\t\t\tteam: playerOpts?.team,
\t\t\t};"""
        player_replacement = """\t\t\tconst options = {
\t\t\t\tname: player.name,
\t\t\t\tavatar: user.avatar,
\t\t\t\tteam: playerOpts?.team,
\t\t\t\tseed: this.options.playerSeeds?.[player.num - 1],
\t\t\t};"""
        room_battle = _replace_once(
            room_battle, player_anchor, player_replacement, room_battle_path,
        )
        room_battle_path.write_text(room_battle, encoding="utf-8")
        changed = True

    return changed
