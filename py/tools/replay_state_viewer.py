from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


DEFAULT_BINARY = Path("build-fresh/showdown_client.exe")
LEFT_PANEL_WIDTH = 280


def is_earned_victory_record(payload: dict) -> bool:
    record_type = str(payload.get("type") or payload.get("message_type") or "").strip().lower()
    if record_type != "terminal":
        return False
    result = str(payload.get("result") or "").strip().lower()
    if result != "win":
        return False
    try:
        reward = float(payload.get("reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        reward = 0.0
    return reward > 0.0


def is_clean_terminal_breaker(payload: dict) -> bool:
    record_type = str(payload.get("type") or payload.get("message_type") or "").strip().lower()
    if record_type != "event":
        return False
    line = str(payload.get("line") or ((payload.get("message") or {}).get("line")) or "").strip().lower()
    if not line:
        return False
    return "lost due to inactivity." in line or "forfeited." in line


def scan_battle_entries(replay_path: Path) -> list[dict[str, object]]:
    battle_entries: list[dict[str, object]] = []
    by_battle_id: dict[str, dict[str, object]] = {}
    with replay_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            battle_id = payload.get("battle_id")
            if not battle_id:
                continue
            battle_id = str(battle_id)
            entry = by_battle_id.get(battle_id)
            if entry is None:
                entry = {
                    "battle_id": battle_id,
                    "earned_win": False,
                    "has_terminal": False,
                    "clean_terminal_breaker": False,
                }
                by_battle_id[battle_id] = entry
                battle_entries.append(entry)
            if is_earned_victory_record(payload):
                entry["earned_win"] = True
            record_type = str(payload.get("type") or payload.get("message_type") or "").strip().lower()
            if record_type == "terminal":
                entry["has_terminal"] = True
            if is_clean_terminal_breaker(payload):
                entry["clean_terminal_breaker"] = True
    for entry in battle_entries:
        entry["clean_match"] = bool(entry.get("has_terminal")) and not bool(entry.get("clean_terminal_breaker"))
    return battle_entries


def snapshot_turn_numbers(snapshots: list[dict]) -> list[int]:
    turns: list[int] = []
    current_turn = 0
    for snapshot in snapshots:
        line = str(((snapshot.get("message") or {}).get("line")) or "")
        parts = line.split("|")
        if len(parts) >= 3 and parts[1] == "turn":
            try:
                current_turn = int(parts[2])
            except ValueError:
                pass
        turns.append(current_turn)
    return turns


def turn_group_key(snapshot: dict, turn: int) -> str:
    if str(snapshot.get("message_type") or "").strip().lower() == "terminal":
        return f"final:{turn}"
    return str(turn)


def parse_turn_group_number(turn_key: str) -> int:
    if turn_key.startswith("final:"):
        try:
            return int(turn_key.split(":", 1)[1])
        except ValueError:
            return 0
    try:
        return int(turn_key)
    except ValueError:
        return 0


def display_turn_label(turn_key: str) -> str:
    if turn_key.startswith("final:"):
        return "Final"
    return f"Turn {parse_turn_group_number(turn_key)}"


def build_turn_index(snapshots: list[dict], turn_numbers: list[int]) -> list[tuple[str, int]]:
    grouped: dict[str, list[tuple[int, dict]]] = {}
    ordered_turns: list[str] = []
    for idx, snapshot in enumerate(snapshots):
        turn = turn_numbers[idx] if idx < len(turn_numbers) else 0
        key = turn_group_key(snapshot, turn)
        if key not in grouped:
            grouped[key] = []
            ordered_turns.append(key)
        grouped[key].append((idx, snapshot))

    turns: list[tuple[str, int]] = []
    for turn_key in ordered_turns:
        candidates = grouped[turn_key]
        best_idx, _ = max(candidates, key=canonical_turn_snapshot_score)
        turns.append((turn_key, best_idx))
    return turns


def build_turn_groups(snapshots: list[dict], turn_numbers: list[int]) -> list[tuple[str, list[int], int]]:
    grouped: dict[str, list[tuple[int, dict]]] = {}
    ordered_turns: list[str] = []
    for idx, snapshot in enumerate(snapshots):
        turn = turn_numbers[idx] if idx < len(turn_numbers) else 0
        key = turn_group_key(snapshot, turn)
        if key not in grouped:
            grouped[key] = []
            ordered_turns.append(key)
        grouped[key].append((idx, snapshot))

    turn_groups: list[tuple[str, list[int], int]] = []
    for turn_key in ordered_turns:
        candidates = grouped[turn_key]
        best_idx, _ = max(candidates, key=canonical_turn_snapshot_score)
        turn_groups.append((turn_key, [idx for idx, _ in candidates], best_idx))
    return turn_groups


def canonical_turn_snapshot_score(candidate: tuple[int, dict]) -> tuple[int, int, int, int]:
    idx, snapshot = candidate
    session = snapshot.get("session") or {}
    raw_state = session.get("canonical_state") or session.get("raw_state") or {}
    request = session.get("request") or {}
    is_doubles = int(raw_state.get("is_doubles", 0) or 0)
    expected_active = 2 if is_doubles else 1
    self_active = int(raw_state.get("self_active_count", 0) or 0)
    opp_active = int(raw_state.get("opp_active_count", 0) or 0)
    active_completeness = min(self_active, expected_active) + min(opp_active, expected_active)
    has_full_field = int(self_active >= expected_active and opp_active >= expected_active)
    is_request = int(snapshot.get("message_type") == "request")
    request_needs_choice = int(bool(request) and not int(request.get("wait", 0) or 0))
    ready_for_decision = int(session.get("ready_for_decision", 0) or 0)
    known_count = 0
    for team_name in ("self_team", "opp_team"):
        for mon in raw_state.get(team_name) or []:
            if mon.get("known") or mon.get("revealed") or mon.get("active"):
                known_count += 1
    return (
        has_full_field,
        active_completeness,
        is_request + request_needs_choice + ready_for_decision,
        known_count,
        idx,
    )


def tracked_name(tracked: dict | None, fallback_prefix: str = "id") -> str:
    tracked = tracked or {}
    name = (tracked.get("name") or "").strip()
    if name:
        return name
    value = tracked.get("value", 0)
    return f"{fallback_prefix}={value}" if value else "unknown"


def tracked_knowledge(tracked: dict | None) -> str:
    tracked = tracked or {}
    knowledge = str(tracked.get("knowledge") or "unknown").strip()
    return knowledge if knowledge else "unknown"


def session_state(session: dict) -> dict:
    if not session:
        return {}
    state = session.get("canonical_state")
    if isinstance(state, dict) and state:
        return state
    state = session.get("raw_state")
    if isinstance(state, dict) and state:
        return state
    return {}


def merge_team_debug(team: list[dict], debug_team: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for idx, mon in enumerate(team):
        if not isinstance(mon, dict):
            merged.append({})
            continue
        combined = dict(mon)
        if idx < len(debug_team) and isinstance(debug_team[idx], dict):
            for key, value in debug_team[idx].items():
                if key not in combined:
                    combined[key] = value
        merged.append(combined)
    return merged


def pretty_knowledge(tracked: dict | None) -> str:
    knowledge = tracked_knowledge(tracked)
    if knowledge == "confirmed":
        return "confirmed"
    if knowledge == "inferred":
        return "inferred"
    return "unknown"


def short_result_label(snapshot: dict) -> str:
    msg = snapshot.get("message") or {}
    result = str(msg.get("result") or "").strip().lower()
    if result == "win":
        return "W"
    if result == "loss":
        return "L"
    if result == "draw":
        return "D"
    return ""


def side_badges(side: dict) -> list[str]:
    badges: list[str] = []
    if side.get("stealth_rock"):
        badges.append("Rocks")
    spikes = int(side.get("spikes", 0) or 0)
    if spikes:
        badges.append(f"Spikes {spikes}")
    toxic_spikes = int(side.get("toxic_spikes", 0) or 0)
    if toxic_spikes:
        badges.append(f"TSpikes {toxic_spikes}")
    if side.get("sticky_web"):
        badges.append("Web")
    if side.get("reflect"):
        badges.append(f"Reflect {side.get('reflect_turns', 0)}")
    if side.get("light_screen"):
        badges.append(f"Light Screen {side.get('light_screen_turns', 0)}")
    if side.get("aurora_veil"):
        badges.append(f"Aurora Veil {side.get('aurora_veil_turns', 0)}")
    if side.get("tailwind"):
        badges.append(f"Tailwind {side.get('tailwind_turns', 0)}")
    if side.get("safeguard"):
        badges.append(f"Safeguard {side.get('safeguard_turns', 0)}")
    if side.get("mist"):
        badges.append(f"Mist {side.get('mist_turns', 0)}")
    if side.get("lucky_chant"):
        badges.append(f"Lucky Chant {side.get('lucky_chant_turns', 0)}")
    return badges


def field_badges(raw_state: dict) -> list[str]:
    badges: list[str] = []
    weather_id = int(raw_state.get("weather_id", 0) or 0)
    if weather_id:
        badges.append(f"Weather {weather_id}")
    terrain_id = int(raw_state.get("terrain_id", 0) or 0)
    if terrain_id:
        badges.append(f"Terrain {terrain_id}")
    if raw_state.get("trick_room"):
        badges.append(f"Trick Room {raw_state.get('trick_room_turns_remaining', 0)}")
    if raw_state.get("gravity"):
        badges.append(f"Gravity {raw_state.get('gravity_turns_remaining', 0)}")
    if raw_state.get("magic_room"):
        badges.append(f"Magic Room {raw_state.get('magic_room_turns_remaining', 0)}")
    if raw_state.get("wonder_room"):
        badges.append(f"Wonder Room {raw_state.get('wonder_room_turns_remaining', 0)}")
    if raw_state.get("mud_sport"):
        badges.append("Mud Sport")
    if raw_state.get("water_sport"):
        badges.append("Water Sport")
    if raw_state.get("ion_deluge"):
        badges.append("Ion Deluge")
    return badges


def active_mons(team: list[dict]) -> list[dict]:
    mons = [mon for mon in team if mon.get("active") and not mon.get("fainted")]
    mons.sort(key=lambda mon: int(mon.get("active_slot", 0) or 0))
    return mons


def reserve_mons(team: list[dict]) -> list[dict]:
    return [mon for mon in team if not mon.get("active") or mon.get("fainted")]


def move_names(mon: dict) -> list[str]:
    names: list[str] = []
    for move in mon.get("moves") or []:
        move_data = move.get("effective_move") or move.get("move") or move.get("base_move") or {}
        known = move.get("effective_known")
        if known is None:
            known = move.get("known")
        if known or move_data.get("value", 0):
            names.append(tracked_name(move_data, "move"))
    return names


def mon_flags(mon: dict) -> list[str]:
    flags: list[str] = []
    if mon.get("fainted"):
        flags.append("fainted")
    if mon.get("tera_used"):
        flags.append("tera used")
    if mon.get("can_tera"):
        flags.append("can tera")
    if mon.get("substitute_active"):
        flags.append("substitute")
    if mon.get("protect_active"):
        flags.append("protect")
    if mon.get("flinch_active"):
        flags.append("flinched")
    if mon.get("confusion_active"):
        flags.append("confused")
    if mon.get("seed_active"):
        flags.append("seeded")
    if mon.get("encore_active"):
        flags.append("encore")
    if mon.get("disable_active"):
        flags.append("disable")
    if mon.get("taunt_active"):
        flags.append("taunt")
    if mon.get("trapped"):
        flags.append("trapped")
    elif mon.get("maybe_trapped"):
        flags.append("maybe trapped")
    if mon.get("commanding_active"):
        flags.append("commanding")
    if mon.get("reviving"):
        flags.append("reviving")
    return flags


def base_stats_text(mon: dict) -> str | None:
    stats = mon.get("base_stats") or {}
    values = [
        ("HP", int(stats.get("hp", 0) or 0)),
        ("Atk", int(stats.get("atk", 0) or 0)),
        ("Def", int(stats.get("def", 0) or 0)),
        ("SpA", int(stats.get("spa", 0) or 0)),
        ("SpD", int(stats.get("spd", 0) or 0)),
        ("Spe", int(stats.get("spe", 0) or 0)),
    ]
    if not any(value for _, value in values):
        return None
    return " / ".join(f"{label} {value}" for label, value in values)


def slot_flag_text(mon: dict) -> str | None:
    details: list[str] = []
    encore_slot = int(mon.get("encore_move_slot", -1) or -1)
    disable_slot = int(mon.get("disable_move_slot", -1) or -1)
    if encore_slot >= 0:
        details.append(f"encore slot {encore_slot + 1}")
    if disable_slot >= 0:
        details.append(f"disable slot {disable_slot + 1}")
    return ", ".join(details) if details else None


def hp_text(mon: dict) -> str:
    if mon.get("fainted"):
        return f"0/{mon.get('max_hp', 0)}"
    return f"{mon.get('current_hp', 0)}/{mon.get('max_hp', 0)}"


def boost_text(mon: dict) -> str:
    boosts = mon.get("boosts") or []
    if mon.get("fainted"):
        boosts = [0] * len(boosts) if boosts else [0, 0, 0, 0, 0, 0, 0]
    return f"Boosts {boosts}"


def compact_ident(text: str) -> str:
    if not text:
        return "unknown"
    if ": " in text:
        return text.split(": ", 1)[1]
    return text


def summarize_action_line(line: str) -> str | None:
    if not line:
        return None
    parts = line.split("|")
    if len(parts) < 2:
        return None
    tag = parts[1]
    if tag in {"", "request", "turn", "upkeep", "done"}:
        return None
    if tag == "move" and len(parts) >= 4:
        actor = compact_ident(parts[2])
        move = parts[3]
        target = compact_ident(parts[4]) if len(parts) >= 5 and parts[4] else ""
        return f"{actor}: {move}{f' -> {target}' if target else ''}"
    if tag == "switch" and len(parts) >= 4:
        actor = compact_ident(parts[2])
        species = parts[3].split(",")[0]
        return f"{actor}: switch to {species}"
    if tag == "faint" and len(parts) >= 3:
        return f"{compact_ident(parts[2])} fainted"
    if tag == "cant" and len(parts) >= 4:
        return f"{compact_ident(parts[2])}: can't act ({parts[3]})"
    if tag == "-status" and len(parts) >= 4:
        return f"{compact_ident(parts[2])}: {parts[3]}"
    if tag == "-boost" and len(parts) >= 5:
        return f"{compact_ident(parts[2])}: +{parts[4]} {parts[3]}"
    if tag == "-unboost" and len(parts) >= 5:
        return f"{compact_ident(parts[2])}: -{parts[4]} {parts[3]}"
    if tag == "win" and len(parts) >= 3:
        return f"Winner: {parts[2]}"
    if tag == "tie":
        return "Tie"
    return None


def summarize_turn_actions(snapshot_indices: list[int], snapshots: list[dict]) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    for idx in snapshot_indices:
        snapshot = snapshots[idx]
        line = str(((snapshot.get("message") or {}).get("line")) or "")
        summary = summarize_action_line(line)
        if not summary or summary in seen:
            continue
        seen.add(summary)
        actions.append(summary)
    return actions


class ReplayStateViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Porygon Replay State Viewer")
        self.geometry("1680x980")
        self.minsize(1280, 800)

        self.replay_var = tk.StringVar()
        self.loading_var = tk.StringVar(value="")
        self.current_battle_var = tk.StringVar(value="No battle loaded")
        self.current_turn_var = tk.StringVar(value="Turn 0")
        self.current_meta_var = tk.StringVar(value="")
        self.earned_victories_only_var = tk.BooleanVar(value=False)
        self.clean_matches_only_var = tk.BooleanVar(value=False)

        self.snapshot_path: Path | None = None
        self.snapshots: list[dict] = []
        self.snapshot_turns: list[int] = []
        self.turns: list[tuple[str, int]] = []
        self.turn_groups: list[tuple[str, list[int], int]] = []
        self.battle_entries: list[dict[str, object]] = []
        self.visible_battle_ids: list[str] = []
        self.current_turn_idx = 0
        self.loading_active = False
        self._programmatic_battle_select = False
        self._programmatic_turn_select = False
        self.turn_chip_buttons: list[tk.Button] = []

        self._build_style()
        self._build_ui()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(bg="#f4f6fb")
        style.configure("App.TFrame", background="#f4f6fb")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Battlefield.TFrame", background="#f8fafc")
        style.configure("Card.TLabelframe", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background="#ffffff", foreground="#1f2937", font=("Segoe UI", 10, "bold"))
        style.configure("Section.TLabel", background="#f4f6fb", foreground="#0f172a", font=("Segoe UI", 11, "bold"))
        style.configure("Meta.TLabel", background="#f4f6fb", foreground="#475569", font=("Segoe UI", 9))
        style.configure("Header.TLabel", background="#f4f6fb", foreground="#0f172a", font=("Segoe UI", 14, "bold"))
        style.configure("Subtle.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 9))
        style.configure("Value.TLabel", background="#ffffff", foreground="#111827", font=("Segoe UI", 9))
        style.configure("CenterCard.TFrame", background="#eef2f7")
        style.configure("CompactMeta.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 8))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="App.TFrame", padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root, style="App.TFrame")
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="Replay", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.replay_var, width=100).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.browse_button = ttk.Button(top, text="Browse", command=self.browse_replay)
        self.browse_button.pack(side=tk.LEFT)

        self.content = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        self.content.pack(fill=tk.BOTH, expand=True)

        self.left_panel = ttk.Frame(self.content, style="Panel.TFrame", padding=8, width=LEFT_PANEL_WIDTH)
        self.center_panel = ttk.Frame(self.content, style="App.TFrame", padding=(12, 0, 12, 0))
        self.content.add(self.left_panel, weight=1)
        self.content.add(self.center_panel, weight=6)

        self._build_left_panel()
        self._build_center_panel()

    def _build_left_panel(self) -> None:
        ttk.Label(self.left_panel, text="Battles", style="Section.TLabel").pack(anchor="w")
        ttk.Label(self.left_panel, text="Select a replay to scan and load battles.", style="Meta.TLabel", wraplength=240).pack(anchor="w", pady=(2, 8))
        ttk.Checkbutton(
            self.left_panel,
            text="Earned victories only",
            variable=self.earned_victories_only_var,
            command=self.on_filter_change,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Checkbutton(
            self.left_panel,
            text="Clean matches only",
            variable=self.clean_matches_only_var,
            command=self.on_filter_change,
        ).pack(anchor="w", pady=(0, 8))
        self.battle_list = tk.Listbox(
            self.left_panel,
            exportselection=False,
            activestyle="none",
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#111827",
            highlightthickness=0,
            selectbackground="#dbeafe",
            selectforeground="#111827",
        )
        self.battle_list.pack(fill=tk.BOTH, expand=True)
        self.battle_list.bind("<<ListboxSelect>>", self.on_battle_select)

    def _build_center_panel(self) -> None:
        header = ttk.Frame(self.center_panel, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 8))

        title_col = ttk.Frame(header, style="App.TFrame")
        title_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title_col, text="Current Battle", style="Header.TLabel").pack(anchor="w")
        ttk.Label(title_col, textvariable=self.current_battle_var, style="Meta.TLabel", wraplength=760).pack(anchor="w", pady=(2, 0))
        ttk.Label(title_col, textvariable=self.current_meta_var, style="Meta.TLabel").pack(anchor="w", pady=(2, 0))

        right_controls = ttk.Frame(header, style="App.TFrame")
        right_controls.pack(side=tk.RIGHT)
        self.prev_button = ttk.Button(right_controls, text="Prev Turn", command=self.prev_turn)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(right_controls, text="Next Turn", command=self.next_turn)
        self.next_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(right_controls, textvariable=self.current_turn_var, style="Header.TLabel").pack(side=tk.LEFT, padx=(12, 0))

        self.battle_surface = ttk.Frame(self.center_panel, style="Battlefield.TFrame", padding=8)
        self.battle_surface.pack(fill=tk.BOTH, expand=True)

        self.field_frame = ttk.LabelFrame(self.battle_surface, text="Field", style="Card.TLabelframe", padding=8)
        self.field_frame.pack(fill=tk.X, pady=(0, 6))
        self.opp_side_frame = ttk.LabelFrame(self.battle_surface, text="Opponent", style="Card.TLabelframe", padding=8)
        self.opp_side_frame.pack(fill=tk.X, pady=(0, 6))
        self.self_side_frame = ttk.LabelFrame(self.battle_surface, text="Player", style="Card.TLabelframe", padding=8)
        self.self_side_frame.pack(fill=tk.X)

        turns_frame = ttk.Frame(self.center_panel, style="Panel.TFrame", padding=8)
        turns_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(turns_frame, text="Turns", style="Section.TLabel").pack(anchor="w")
        turn_nav = ttk.Frame(turns_frame, style="Panel.TFrame")
        turn_nav.pack(fill=tk.X, pady=(6, 0))
        self.turn_left_button = ttk.Button(turn_nav, text="<", width=3, command=self.prev_turn)
        self.turn_left_button.pack(side=tk.LEFT, padx=(0, 6))
        self.turn_canvas = tk.Canvas(turn_nav, height=78, bg="#ffffff", highlightthickness=0)
        self.turn_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.turn_right_button = ttk.Button(turn_nav, text=">", width=3, command=self.next_turn)
        self.turn_right_button.pack(side=tk.LEFT, padx=(6, 0))
        turn_scroll = ttk.Scrollbar(turns_frame, orient=tk.HORIZONTAL, command=self.turn_canvas.xview)
        turn_scroll.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        self.turn_canvas.configure(xscrollcommand=turn_scroll.set)
        self.turn_chip_frame = ttk.Frame(self.turn_canvas, style="Panel.TFrame")
        self.turn_canvas_window = self.turn_canvas.create_window((0, 0), window=self.turn_chip_frame, anchor="nw")
        self.turn_chip_frame.bind("<Configure>", self.on_turn_chip_frame_configure)
        self.turn_canvas.bind("<Configure>", self.on_turn_canvas_configure)

        self.loading_overlay = ttk.Frame(self.battle_surface, style="Panel.TFrame")
        self.loading_label = ttk.Label(self.loading_overlay, textvariable=self.loading_var, style="Header.TLabel")
        self.loading_label.place(relx=0.5, rely=0.45, anchor="center")
        self.loading_bar = ttk.Progressbar(self.loading_overlay, mode="indeterminate", length=260)
        self.loading_bar.place(relx=0.5, rely=0.56, anchor="center")

    def browse_replay(self) -> None:
        path = filedialog.askopenfilename(title="Select replay jsonl", filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")])
        if path:
            self.replay_var.set(path)
            self.scan_replay()

    def set_loading(self, loading: bool, message: str = "Loading...") -> None:
        self.loading_active = loading
        state = tk.DISABLED if loading else tk.NORMAL
        if loading:
            self.loading_var.set(message)
            self.loading_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.loading_bar.start(12)
        else:
            self.loading_bar.stop()
            self.loading_overlay.place_forget()
            self.loading_var.set("")
        self.browse_button.configure(state=state)
        self.prev_button.configure(state=state)
        self.next_button.configure(state=state)
        self.turn_left_button.configure(state=state)
        self.turn_right_button.configure(state=state)
        self.update_idletasks()

    def scan_replay(self) -> None:
        replay_path = Path(self.replay_var.get())
        if not replay_path.exists():
            messagebox.showerror("Replay missing", f"Replay file not found:\n{replay_path}")
            return
        self.set_loading(True, "Scanning battles...")
        try:
            self.clear_battle_state()
            self.battle_entries = scan_battle_entries(replay_path)
            self.refresh_battle_list()
        finally:
            self.set_loading(False)

    def filtered_battle_ids(self) -> list[str]:
        earned_only = bool(self.earned_victories_only_var.get())
        clean_only = bool(self.clean_matches_only_var.get())
        battle_ids: list[str] = []
        for entry in self.battle_entries:
            if earned_only and not bool(entry.get("earned_win")):
                continue
            if clean_only and not bool(entry.get("clean_match")):
                continue
            battle_ids.append(str(entry.get("battle_id") or ""))
        return [battle_id for battle_id in battle_ids if battle_id]

    def refresh_battle_list(self, preferred_battle_id: str | None = None) -> None:
        if preferred_battle_id is None and self.battle_list.curselection():
            preferred_battle_id = self.battle_list.get(self.battle_list.curselection()[0])
        self.visible_battle_ids = self.filtered_battle_ids()
        self.battle_list.delete(0, tk.END)
        for battle_id in self.visible_battle_ids:
            self.battle_list.insert(tk.END, battle_id)
        if not self.visible_battle_ids:
            self.clear_battle_state()
            if self.battle_entries:
                self.current_battle_var.set("No battles match current filter")
            return

        selected_idx = 0
        if preferred_battle_id and preferred_battle_id in self.visible_battle_ids:
            selected_idx = self.visible_battle_ids.index(preferred_battle_id)

        self._programmatic_battle_select = True
        self.battle_list.selection_clear(0, tk.END)
        self.battle_list.selection_set(selected_idx)
        self.battle_list.see(selected_idx)
        self._programmatic_battle_select = False
        self.export_and_load()

    def on_filter_change(self) -> None:
        if self.loading_active:
            return
        current_battle_id = self.current_battle_var.get()
        preferred_battle_id = current_battle_id if current_battle_id not in {"No battle loaded", "No battles match current filter"} else None
        self.refresh_battle_list(preferred_battle_id)

    def export_and_load(self) -> None:
        replay_path = Path(self.replay_var.get())
        selection = self.battle_list.curselection()
        if not replay_path.exists():
            messagebox.showerror("Replay missing", f"Replay file not found:\n{replay_path}")
            return
        if not DEFAULT_BINARY.exists():
            messagebox.showerror("Binary missing", f"Binary not found:\n{DEFAULT_BINARY}")
            return
        if not selection:
            return
        battle_id = self.battle_list.get(selection[0])
        snapshot_path = Path(tempfile.gettempdir()) / f"{battle_id}_snapshots.json"
        self.set_loading(True, "Exporting battle...")
        try:
            try:
                result = subprocess.run(
                    [str(DEFAULT_BINARY), "--export-battle", str(replay_path), battle_id, str(snapshot_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as exc:
                messagebox.showerror("Export failed", str(exc))
                return
            if result.returncode != 0:
                messagebox.showerror("Export failed", (result.stderr or result.stdout or "Unknown error").strip())
                return
            self.set_loading(True, "Loading turn states...")
            self.snapshot_path = snapshot_path
            with snapshot_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.snapshots = list(payload.get("snapshots") or [])
            self.snapshot_turns = snapshot_turn_numbers(self.snapshots)
            self.turn_groups = build_turn_groups(self.snapshots, self.snapshot_turns)
            self.turns = build_turn_index(self.snapshots, self.snapshot_turns)
            self.current_battle_var.set(battle_id)
            self.populate_turn_list()
            if self.turns:
                self.select_turn(0)
        finally:
            self.set_loading(False)

    def populate_turn_list(self) -> None:
        for child in self.turn_chip_frame.winfo_children():
            child.destroy()
        self.turn_chip_buttons = []
        for idx, (turn_key, snapshot_idx) in enumerate(self.turns):
            snapshot = self.snapshots[snapshot_idx]
            msg_type = snapshot.get("message_type", "")
            badge = short_result_label(snapshot)
            if turn_key.startswith("final:"):
                label = f"Final\n{msg_type}{f' [{badge}]' if badge else ''}"
            else:
                turn = parse_turn_group_number(turn_key)
                label = f"T{turn:02d}\n{msg_type}{f' [{badge}]' if badge else ''}"
            button = tk.Button(
                self.turn_chip_frame,
                text=label,
                command=lambda turn_idx=idx: self.select_turn(turn_idx),
                font=("Segoe UI", 9, "bold"),
                bg="#ffffff",
                fg="#0f172a",
                activebackground="#dbeafe",
                activeforeground="#0f172a",
                relief=tk.RIDGE,
                bd=1,
                padx=10,
                pady=8,
                width=11,
                justify=tk.CENTER,
            )
            button.pack(side=tk.LEFT, padx=(0, 8))
            self.turn_chip_buttons.append(button)
        self.update_idletasks()
        self.turn_canvas.configure(scrollregion=self.turn_canvas.bbox("all"))

    def clear_battle_state(self) -> None:
        self.snapshots = []
        self.snapshot_turns = []
        self.turns = []
        self.turn_groups = []
        self.current_turn_idx = 0
        self.populate_turn_list()
        self.current_battle_var.set("No battle loaded")
        self.current_turn_var.set("Turn 0")
        self.current_meta_var.set("")
        self.clear_frame(self.field_frame)
        self.clear_frame(self.opp_side_frame)
        self.clear_frame(self.self_side_frame)

    def on_battle_select(self, _event: object | None = None) -> None:
        if self.loading_active:
            return
        if self._programmatic_battle_select:
            return
        if not self.battle_list.curselection():
            return
        self.export_and_load()

    def select_turn(self, turn_idx: int) -> None:
        if not self.turns:
            return
        turn_idx = max(0, min(turn_idx, len(self.turns) - 1))
        self.current_turn_idx = turn_idx
        self.refresh_turn_chip_selection()
        self.render_current_turn()

    def prev_turn(self) -> None:
        self.select_turn(self.current_turn_idx - 1)

    def next_turn(self) -> None:
        self.select_turn(self.current_turn_idx + 1)

    def on_turn_select(self, _event: object | None = None) -> None:
        return

    def on_turn_chip_frame_configure(self, _event: object | None = None) -> None:
        self.turn_canvas.configure(scrollregion=self.turn_canvas.bbox("all"))

    def on_turn_canvas_configure(self, event: object | None = None) -> None:
        if event is not None:
            self.turn_canvas.itemconfigure(self.turn_canvas_window, height=max(1, getattr(event, "height", 78) - 2))

    def refresh_turn_chip_selection(self) -> None:
        for idx, button in enumerate(self.turn_chip_buttons):
            selected = idx == self.current_turn_idx
            button.configure(
                bg="#dbeafe" if selected else "#ffffff",
                fg="#0f172a",
                relief=tk.SOLID if selected else tk.RIDGE,
                bd=2 if selected else 1,
            )
        self.scroll_selected_turn_into_view()

    def scroll_selected_turn_into_view(self) -> None:
        if not (0 <= self.current_turn_idx < len(self.turn_chip_buttons)):
            return
        self.update_idletasks()
        button = self.turn_chip_buttons[self.current_turn_idx]
        total_width = max(1, self.turn_chip_frame.winfo_reqwidth())
        viewport_width = max(1, self.turn_canvas.winfo_width())
        left_visible = self.turn_canvas.canvasx(0)
        right_visible = left_visible + viewport_width
        button_left = button.winfo_x()
        button_right = button_left + button.winfo_width()

        if button_left < left_visible:
            new_left = button_left
        elif button_right > right_visible:
            new_left = button_right - viewport_width
        else:
            return

        max_left = max(0, total_width - viewport_width)
        new_left = max(0, min(new_left, max_left))
        self.turn_canvas.xview_moveto(new_left / max(1, total_width))

    def render_current_turn(self) -> None:
        if not self.turns:
            return
        turn_key, snapshot_idx = self.turns[self.current_turn_idx]
        snapshot = self.snapshots[snapshot_idx]
        session = snapshot.get("session") or {}
        raw_state = session_state(session)
        debug_only = session.get("debug_only") or {}
        self_team = merge_team_debug(raw_state.get("self_team") or [], debug_only.get("self_team") or [])
        opp_team = merge_team_debug(raw_state.get("opp_team") or [], debug_only.get("opp_team") or [])
        self.current_turn_var.set(display_turn_label(turn_key))
        result = str(((snapshot.get("message") or {}).get("result")) or "").strip()
        self.current_meta_var.set(
            f"message={snapshot.get('message_type', '')}  "
            f"ready={session.get('ready_for_decision', 0)}  "
            f"terminal={session.get('terminal', 0)}"
            f"{f'  result={result}' if result else ''}"
        )
        self.render_field_state(snapshot)
        self.render_side_section(
            self.opp_side_frame,
            active_mons(opp_team),
            reserve_mons(opp_team),
            align="right",
        )
        self.render_side_section(
            self.self_side_frame,
            active_mons(self_team),
            reserve_mons(self_team),
            align="left",
        )

    def clear_frame(self, frame: ttk.Widget) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def render_field_state(self, snapshot: dict) -> None:
        self.clear_frame(self.field_frame)
        session = snapshot.get("session") or {}
        raw_state = session_state(session)
        turn_key, snapshot_indices, _ = self.turn_groups[self.current_turn_idx]

        row = ttk.Frame(self.field_frame, style="CenterCard.TFrame")
        row.pack(fill=tk.X)

        actions_panel = ttk.Frame(row, style="Panel.TFrame")
        actions_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        field_panel = ttk.Frame(row, style="CenterCard.TFrame")
        field_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(actions_panel, text="Actions This Turn", style="Section.TLabel").pack(anchor="w")
        actions = summarize_turn_actions(snapshot_indices, self.snapshots)
        if actions:
            for action in actions[:8]:
                ttk.Label(actions_panel, text=f"- {action}", style="Value.TLabel", wraplength=420).pack(anchor="w", pady=(2, 0))
        else:
            ttk.Label(actions_panel, text="No action messages recorded yet.", style="Meta.TLabel").pack(anchor="w", pady=(2, 0))

        top = ttk.Frame(field_panel, style="CenterCard.TFrame")
        top.pack(fill=tk.X)
        ttk.Label(top, text=display_turn_label(turn_key), style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(top, text=f"Player active: {raw_state.get('self_active_count', 0)}", style="Meta.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(top, text=f"Opp active: {raw_state.get('opp_active_count', 0)}", style="Meta.TLabel").pack(side=tk.LEFT)

        summary = ttk.Frame(field_panel, style="CenterCard.TFrame")
        summary.pack(fill=tk.X, pady=(6, 0))
        self.render_badge_group(summary, "Field", field_badges(raw_state))
        self.render_badge_group(summary, "Opponent side", side_badges(raw_state.get("opp_side") or {}))
        self.render_badge_group(summary, "Player side", side_badges(raw_state.get("self_side") or {}))

    def render_badge_group(self, parent: ttk.Frame, label: str, badges: list[str]) -> None:
        row = ttk.Frame(parent, style="CenterCard.TFrame")
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        if not badges:
            ttk.Label(row, text="none", style="Meta.TLabel").pack(side=tk.LEFT)
            return
        for badge in badges:
            self.make_badge(row, badge).pack(side=tk.LEFT, padx=(0, 6))

    def render_side_section(self, frame: ttk.LabelFrame, active_mons_list: list[dict], reserve_mons_list: list[dict], align: str) -> None:
        self.clear_frame(frame)
        outer = ttk.Frame(frame, style="Panel.TFrame")
        outer.pack(fill=tk.X)

        active_section = ttk.Frame(outer, style="Panel.TFrame")
        divider = ttk.Separator(outer, orient=tk.VERTICAL)
        reserve_section = ttk.Frame(outer, style="Panel.TFrame")

        if align == "right":
            reserve_section.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            divider.pack(side=tk.RIGHT, fill=tk.Y, padx=8)
            active_section.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        else:
            active_section.pack(side=tk.LEFT, fill=tk.X, expand=True)
            divider.pack(side=tk.LEFT, fill=tk.Y, padx=8)
            reserve_section.pack(side=tk.LEFT, fill=tk.X, expand=True)

        active_header = ttk.Frame(active_section, style="Panel.TFrame")
        active_header.pack(fill=tk.X, pady=(0, 4))
        reserve_header = ttk.Frame(reserve_section, style="Panel.TFrame")
        reserve_header.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(active_header, text="Active", style="Section.TLabel").pack(side=tk.RIGHT)
        ttk.Label(reserve_header, text="Reserve", style="Section.TLabel").pack(side=tk.LEFT)

        active_row = ttk.Frame(active_section, style="Panel.TFrame")
        active_row.pack(fill=tk.X)
        reserve_row = ttk.Frame(reserve_section, style="Panel.TFrame")
        reserve_row.pack(fill=tk.X)

        display_active = active_mons_list[:2]
        while len(display_active) < 2:
            display_active.append(None)
        display_reserve = reserve_mons_list[:4]
        while len(display_reserve) < 4:
            display_reserve.append(None)

        self.render_card_row(active_row, display_active, align=align, active=True)
        self.render_card_row(reserve_row, display_reserve, align="left", active=False)

    def render_card_row(self, parent: ttk.Frame, mons: list[dict | None], align: str, active: bool) -> None:
        cards = [self.render_pokemon_card(parent, mon, active=active) for mon in mons]
        if align == "right":
            for card in reversed(cards):
                card.pack(side=tk.RIGHT, padx=(6, 0), fill=tk.X if active else tk.Y, expand=active)
        else:
            for card in cards:
                card.pack(side=tk.LEFT, padx=(0, 6), fill=tk.X if active else tk.Y, expand=active)

    def render_pokemon_card(self, parent: ttk.Frame, mon: dict | None, active: bool) -> ttk.Frame:
        card = ttk.Frame(parent, style="Panel.TFrame", padding=6)
        if mon is None:
            ttk.Label(card, text="Unknown", style="Section.TLabel").pack(anchor="w")
            ttk.Label(card, text="No known data", style="CompactMeta.TLabel", wraplength=160 if active else 100).pack(anchor="w", pady=(2, 0))
            return card

        ident = mon.get("ident") or mon.get("canonical_ident") or tracked_name(mon.get("species"), "species")
        ttk.Label(card, text=ident, style="Section.TLabel").pack(anchor="w")

        species_row = ttk.Frame(card, style="Panel.TFrame")
        species_row.pack(anchor="w", fill=tk.X, pady=(2, 0))
        ttk.Label(species_row, text=tracked_name(mon.get("species"), "species"), style="Value.TLabel").pack(side=tk.LEFT)
        self.make_badge(species_row, pretty_knowledge(mon.get("species"))).pack(side=tk.LEFT, padx=(6, 0))

        hp_status = ttk.Frame(card, style="Panel.TFrame")
        hp_status.pack(anchor="w", fill=tk.X, pady=(2, 0))
        ttk.Label(hp_status, text=f"HP {hp_text(mon)}", style="Value.TLabel").pack(side=tk.LEFT)
        ttk.Label(hp_status, text=f"Status {tracked_name(mon.get('status'), 'status')}", style="Value.TLabel").pack(side=tk.LEFT, padx=(10, 0))

        self.render_known_line(card, "Ability", mon.get("ability"), "ability")
        self.render_known_line(card, "Item", mon.get("item"), "item")
        self.render_known_line(card, "Tera", mon.get("tera_type"), "type")
        stats_text = base_stats_text(mon)
        if stats_text:
            ttk.Label(card, text=f"Stats {stats_text}", style="CompactMeta.TLabel", wraplength=240 if active else 140).pack(anchor="w", pady=(2, 0))

        ttk.Label(card, text=boost_text(mon), style="CompactMeta.TLabel", wraplength=240 if active else 140).pack(anchor="w", pady=(2, 0))

        moves = move_names(mon)
        ttk.Label(card, text=f"Moves: {', '.join(moves) if moves else '-'}", style="CompactMeta.TLabel", wraplength=240 if active else 140).pack(anchor="w", pady=(2, 0))

        slot_text = slot_flag_text(mon)
        if slot_text:
            ttk.Label(card, text=f"Slots: {slot_text}", style="CompactMeta.TLabel", wraplength=240 if active else 140).pack(anchor="w", pady=(2, 0))

        flags = mon_flags(mon)
        ttk.Label(card, text=f"Flags: {', '.join(flags) if flags else '-'}", style="CompactMeta.TLabel", wraplength=240 if active else 140).pack(anchor="w", pady=(2, 0))

        if not active:
            extra = []
            if mon.get("revealed"):
                extra.append("revealed")
            if mon.get("known"):
                extra.append("known")
            if mon.get("fainted"):
                extra.append("fainted")
            ttk.Label(card, text=f"State: {', '.join(extra) if extra else '-'}", style="CompactMeta.TLabel").pack(anchor="w", pady=(2, 0))

        return card

    def render_known_line(self, parent: ttk.Frame, label: str, tracked: dict | None, fallback_prefix: str) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(anchor="w", fill=tk.X, pady=(2, 0))
        ttk.Label(row, text=f"{label} {tracked_name(tracked, fallback_prefix)}", style="Value.TLabel").pack(side=tk.LEFT)
        self.make_badge(row, pretty_knowledge(tracked)).pack(side=tk.LEFT, padx=(6, 0))

    def make_badge(self, parent: ttk.Frame, text: str) -> ttk.Label:
        return ttk.Label(parent, text=text, style="CompactMeta.TLabel")


def main() -> None:
    app = ReplayStateViewer()
    app.mainloop()


if __name__ == "__main__":
    main()
