"""Build an offline snapshot so the pipeline can be run without the live API.

Useful in two situations: you are on a network that cannot reach
fantasy.premierleague.com, or you want a frozen dataset to develop against so
results do not move under you between runs.

It assembles a bootstrap-static-shaped payload from the public season mirror on
GitHub, synthesises the gameweek calendar from the real fixture list, and writes
everything into data/snapshots/<today>/ where `fplbot build --offline` looks.

    python tests/seed_offline_fixture.py --season 2026-27 --current-gw 4

Then:

    python -m fplbot build --offline
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MIRROR = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

INT_FIELDS = {
    "id", "element_type", "team", "team_code", "code", "now_cost", "minutes",
    "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_saved", "penalties_missed", "yellow_cards",
    "red_cards", "saves", "bonus", "bps", "total_points", "event_points",
    "transfers_in", "transfers_out", "transfers_in_event", "transfers_out_event",
    "cost_change_event", "cost_change_start", "dreamteam_count",
    "defensive_contribution", "tackles", "recoveries",
    "clearances_blocks_interceptions",
}
FLOAT_FIELDS = {
    "form", "points_per_game", "ep_next", "ep_this", "selected_by_percent",
    "value_form", "value_season", "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "expected_goals_per_90", "expected_assists_per_90",
    "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
    "defensive_contribution_per_90", "saves_per_90", "starts_per_90",
    "clean_sheets_per_90", "goals_conceded_per_90",
}
OPTIONAL_INT = {"chance_of_playing_next_round", "chance_of_playing_this_round",
                "penalties_order", "corners_and_indirect_freekicks_order",
                "direct_freekicks_order", "squad_number"}


def _rows(url: str) -> list[dict]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


EMPTY = ("", None, "None", "nan", "NaN")


def _coerce(row: dict) -> dict:
    out: dict = {}
    for key, value in row.items():
        if key in INT_FIELDS:
            out[key] = int(float(value)) if value not in EMPTY else 0
        elif key in FLOAT_FIELDS:
            out[key] = float(value) if value not in EMPTY else 0.0
        elif key in OPTIONAL_INT:
            out[key] = int(float(value)) if value not in EMPTY else None
        elif value in ("True", "False"):
            out[key] = value == "True"
        else:
            out[key] = value
    return out


def build_events(fixtures: list[dict], current_gw: int) -> list[dict]:
    """Derive the gameweek calendar from real kickoff times."""
    by_gw: dict[int, list[str]] = {}
    for fx in fixtures:
        if not fx.get("event"):
            continue
        by_gw.setdefault(int(fx["event"]), []).append(fx["kickoff_time"])
    events = []
    for gw in sorted(by_gw):
        first = min(k for k in by_gw[gw] if k)
        kickoff = datetime.fromisoformat(first.replace("Z", "+00:00"))
        deadline = (kickoff - timedelta(minutes=90)).replace(second=0, microsecond=0)
        events.append({
            "id": gw, "name": f"Gameweek {gw}",
            "deadline_time": deadline.astimezone(timezone.utc)
                                     .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": gw <= current_gw,
            "is_current": gw == current_gw,
            "is_next": gw == current_gw + 1,
            "is_previous": gw == current_gw - 1,
            "can_enter": gw == current_gw + 1,
            "chip_plays": [], "most_selected": None, "top_element": None,
        })
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--current-gw", type=int, default=4)
    ap.add_argument("--team-id", type=int, default=None)
    args = ap.parse_args()

    from fplbot.config import load_config
    cfg = load_config()
    team_id = args.team_id or cfg.team_id

    print(f"downloading the {args.season} mirror…")
    players = [_coerce(r) for r in _rows(f"{MIRROR}/{args.season}/players_raw.csv")]
    teams = [_coerce(r) for r in _rows(f"{MIRROR}/{args.season}/teams.csv")]
    fixtures = [_coerce(r) for r in _rows(f"{MIRROR}/{args.season}/fixtures.csv")]
    for fx in fixtures:
        fx["event"] = int(fx["event"]) if fx.get("event") not in ("", None) else None
        for key in ("team_h", "team_a", "team_h_difficulty", "team_a_difficulty", "id"):
            fx[key] = int(float(fx[key]))

    events = build_events(fixtures, args.current_gw)
    bootstrap = {
        "events": events, "teams": teams, "elements": players,
        "element_types": [
            {"id": 1, "singular_name_short": "GKP"},
            {"id": 2, "singular_name_short": "DEF"},
            {"id": 3, "singular_name_short": "MID"},
            {"id": 4, "singular_name_short": "FWD"},
        ],
        "total_players": 11_000_000,
    }

    # A plausible squad: the highest-scoring legal fifteen inside the budget,
    # picked greedily. Replace this by your real picks once the API is reachable.
    squad = _greedy_squad(players)
    picks = {"picks": [{"element": pid, "position": i + 1, "multiplier": 1}
                       for i, pid in enumerate(squad)],
             "entry_history": {"event": args.current_gw, "bank": 3, "value": 1003}}
    entry = {
        "id": team_id, "player_first_name": "G", "player_last_name": "Num",
        "name": "Gnum United", "summary_overall_points": 252,
        "summary_overall_rank": 1_756_736, "summary_event_points": 40,
        "current_event": args.current_gw, "last_deadline_bank": 3,
        "last_deadline_value": 1003, "last_deadline_total_transfers": 2,
    }
    history = {"current": [
        {"event": gw, "points": 60, "event_transfers": 0 if gw < 3 else 1,
         "event_transfers_cost": 0, "rank": 2_000_000}
        for gw in range(1, args.current_gw + 1)
    ], "chips": []}

    out = cfg.data_dir / "snapshots" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in [("bootstrap", bootstrap), ("fixtures", fixtures),
                          ("entry", entry), ("entry_history", history),
                          (f"picks_gw{args.current_gw}", picks)]:
        (out / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    print(f"wrote {len(players)} players, {len(fixtures)} fixtures, "
          f"{len(events)} gameweeks to {out}")
    print("now run:  python -m fplbot build --offline")
    return 0


def _greedy_squad(players: list[dict]) -> list[int]:
    need = {1: 2, 2: 5, 3: 5, 4: 3}
    budget, spent, per_club = 1000, 0, {}
    ranked = sorted(players, key=lambda p: -p.get("total_points", 0))
    squad: list[int] = []
    for p in ranked:
        pos = p["element_type"]
        if need.get(pos, 0) <= 0:
            continue
        club = p["team"]
        if per_club.get(club, 0) >= 3:
            continue
        remaining = sum(need.values()) - 1
        if spent + p["now_cost"] + remaining * 40 > budget:
            continue
        squad.append(p["id"])
        spent += p["now_cost"]
        need[pos] -= 1
        per_club[club] = per_club.get(club, 0) + 1
        if sum(need.values()) == 0:
            break
    return squad


if __name__ == "__main__":
    sys.exit(main())
