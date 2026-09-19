"""Render the dashboard: a single static HTML file plus PWA assets.

The output is deliberately self-contained and offline-capable. Drop the folder
on GitHub Pages and "Add to Home Screen" turns it into an app icon; open the
file straight from disk and it still works.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import POSITIONS, Config
from .optimize import outlook_by_player

MODEL_VERSION = "0.3.0"
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(WEB_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


THAI_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
THAI_MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
               "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]


def _thai_datetime(dt: datetime) -> str:
    """Thai weekday and month, with the Buddhist-era year people actually use."""
    return (f"{THAI_DAYS[dt.weekday()]} {dt.day} {THAI_MONTHS[dt.month - 1]} "
            f"{dt.year + 543}, {dt:%H:%M} น.")


def _rank(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _opponent_label(rows: pd.DataFrame, teams: pd.DataFrame) -> str:
    if rows.empty:
        return "ไม่มีเกม"
    parts = []
    for _, r in rows.iterrows():
        short = teams.short_name.get(int(r.opponent), "?")
        parts.append(f"{short} ({'H' if r.is_home else 'A'})")
    return " + ".join(parts)


def squad_alerts(players: pd.DataFrame, squad: list[int], xi: list[int],
                 captain: int | None) -> tuple[list[dict], list[str]]:
    """Split squad problems into page context and things worth interrupting for.

    `flags` is everything worth knowing. `alerts` is the subset that costs
    points if you do nothing before the deadline — an unavailable player in the
    XI, or a captain who may not play. Only alerts go into a push notification,
    because a reminder that lists six things is a reminder you scroll past.
    """
    flags: list[dict] = []
    alerts: list[str] = []
    xi_set = set(xi)

    for pid in squad:
        if pid not in players.index:
            continue
        p = players.loc[pid]
        in_xi = pid in xi_set
        news = (p.get("news") or "").strip()

        if p.p_available < 0.5:
            flags.append({"tag": "ไม่พร้อมลง", "severity": "high", "name": p["name"],
                          "in_xi": in_xi,
                          "detail": news or "FPL ติดธงว่าไม่พร้อมลงเล่น"})
            if in_xi:
                alerts.append(f"{p['name']} ไม่พร้อมลง แต่ยังอยู่ในตัวจริง"
                              + (f" — {news}" if news else ""))
        elif p.p_available < 1.0:
            flags.append({"tag": "ต้องลุ้น", "severity": "medium", "name": p["name"],
                          "in_xi": in_xi,
                          "detail": news or "FPL ระบุว่ามีโอกาสไม่ได้ลง"})
            if in_xi:
                alerts.append(f"{p['name']} ยังไม่ชัวร์ว่าได้ลง"
                              + (f" — {news}" if news else ""))
        elif p.p_start < 0.55 and p.minutes > 0:
            flags.append({"tag": "หมุนเวียน", "severity": "medium", "name": p["name"],
                          "in_xi": in_xi,
                          "detail": f"ลงตัวจริงแค่ {p.p_start:.0%} ของนัดที่ทีมเตะ"})

        if int(p.get("yellow_cards", 0)) >= 4:
            flags.append({"tag": "เสี่ยงโดนแบน", "severity": "medium", "name": p["name"],
                          "in_xi": in_xi,
                          "detail": f"ใบเหลือง {int(p.yellow_cards)} ใบ — อีกใบเดียวโดนแบน"})

    # A captain who does not play is the most expensive single thing that can go
    # wrong in a gameweek, so it leads regardless of what else is flagged.
    if captain is not None and captain in players.index:
        cap = players.loc[captain]
        if cap.p_available < 1.0:
            alerts.insert(0, f"กัปตัน {cap['name']} ไม่ชัวร์ว่าได้ลง — เปลี่ยนกัปตันด่วน")

    flags.sort(key=lambda f: (f["severity"] != "high", not f["in_xi"]))
    return flags, alerts


def chip_signals(cfg: Config, ep_rows: pd.DataFrame, players: pd.DataFrame,
                 squad: list[int], horizon_gws: list[int], plan=None) -> list[str]:
    """Flag structural chip opportunities and show their bounded EP upside."""
    signals: list[str] = []
    schedule = ep_rows[["team", "gw", "opponent", "is_home"]].drop_duplicates()
    owned = [p for p in squad if p in players.index]
    tc_available = not all(cfg.get("chips", name, default=False)
                           for name in ("triple_captain_1", "triple_captain_2"))
    bb_available = not all(cfg.get("chips", name, default=False)
                           for name in ("bench_boost_1", "bench_boost_2"))
    fh_available = not all(cfg.get("chips", name, default=False)
                           for name in ("free_hit_1", "free_hit_2"))
    plans = {week.gw: week for week in plan.gameweeks} if plan is not None else {}
    ep_by_gw = (ep_rows.groupby(["gw", "player_id"]).ep.sum().to_dict()
                if "player_id" in ep_rows else {})
    for gw in horizon_gws:
        games = schedule[schedule.gw == gw].groupby("team").size().to_dict()
        double_teams = sorted({int(players.team[p]) for p in owned
                               if games.get(int(players.team[p]), 0) >= 2})
        if double_teams and (tc_available or bb_available):
            chips = "/".join(name for name, available in (
                ("Triple Captain", tc_available), ("Bench Boost", bb_available)) if available)
            suffix = ""
            week = plans.get(gw)
            if week is not None:
                tc_gain = float(ep_by_gw.get((gw, week.captain), 0.0))
                bench_gain = sum(float(ep_by_gw.get((gw, pid), 0.0)) for pid in week.bench)
                estimates = []
                if tc_available:
                    estimates.append(f"TC เพิ่มได้ราว {tc_gain:.1f} EP")
                if bb_available:
                    estimates.append(f"BB ปลดล็อกม้านั่งราว {bench_gain:.1f} EP")
                if estimates:
                    suffix = " (" + " · ".join(estimates) + ")"
            signals.append(
                f"GW{gw} มี Double Gameweek ในทีม — พิจารณา {chips}{suffix} หลังเช็กข่าวตัวจริง")
        playable = sum(games.get(int(players.team[p]), 0) > 0
                       and float(players.p_available[p]) >= 0.5 for p in owned)
        if owned and playable < 11 and fh_available:
            signals.append(f"GW{gw} ทีมปัจจุบันมีผู้เล่นพร้อมโปรแกรมเพียง {playable} คน — พิจารณา Free Hit")
    return signals[:4]


def player_comparisons(cfg: Config, ep_rows: pd.DataFrame, players: pd.DataFrame,
                       teams: pd.DataFrame, squad: list[int], current_squad: list[int],
                       bank: float, selling_price: dict[int, float],
                       transfer_advice) -> dict[str, dict]:
    """Build an affordable same-position ranking for every player in the plan."""
    matches = int(cfg.get("planning", "outlook_matches", default=5))
    limit = int(cfg.get("planning", "comparison_count", default=8))
    min_start = float(cfg.get("planning", "min_candidate_start", default=0.5))
    totals, fixtures = outlook_by_player(ep_rows, matches)
    squad_set = set(squad)
    club_counts = players.loc[[p for p in squad if p in players.index]].team.value_counts()

    remaining_bank = float(bank)
    if transfer_advice.recommend and transfer_advice.out_id is not None:
        incoming = transfer_advice.candidates[0].in_id
        remaining_bank += float(selling_price.get(
            transfer_advice.out_id, players.price[transfer_advice.out_id]))
        remaining_bank -= float(players.price[incoming])

    def row(pid: int, selected: bool) -> dict:
        p = players.loc[pid]
        fx = fixtures.get(pid, [])
        ownership = pd.to_numeric(p.selected_by_percent, errors="coerce")
        return {
            "id": pid, "name": p["name"], "team": p.team_name,
            "position": POSITIONS[int(p.element_type)], "price": float(p.price),
            "ownership": 0.0 if pd.isna(ownership) else float(ownership),
            "form": float(pd.to_numeric(p.get("form", 0.0), errors="coerce") or 0.0),
            "availability": float(p.p_available), "starts": float(p.p_start),
            "ep": float(totals.get(pid, 0.0)), "selected": selected,
            "fixtures": [
                f"GW{item['gw']} {teams.short_name.get(item['opponent'], '?')} "
                f"({'H' if item['is_home'] else 'A'}) [{item['fdr']}]"
                for item in fx
            ],
        }

    result: dict[str, dict] = {}
    for pid in squad:
        if pid not in players.index:
            continue
        player = players.loc[pid]
        pos = int(player.element_type)
        club = int(player.team)
        sell_value = (float(selling_price.get(pid, player.price))
                      if pid in current_squad else float(player.price))
        budget = sell_value + remaining_bank
        band = players[(players.element_type == pos)
                       & (~players.index.isin(squad_set - {pid}))
                       & (players.p_available > 0.25)
                       & ((players.p_start >= min_start) | (players.index == pid))
                       & (players.price <= budget + 1e-9)]
        legal_ids = []
        for other_id, other in band.iterrows():
            other_id = int(other_id)
            other_club = int(other.team)
            after = int(club_counts.get(other_club, 0)) + 1 - int(other_club == club)
            if after <= 3:
                legal_ids.append(other_id)
        ranked = sorted(legal_ids, key=lambda p: totals.get(p, 0.0), reverse=True)
        selected_rank = ranked.index(pid) + 1 if pid in ranked else len(ranked) + 1
        # Send the complete legal set. Sorting in the browser must not be limited
        # to the players that happened to rank highest by EP first.
        rows = [row(other_id, other_id == pid) for other_id in ranked]
        rows.sort(key=lambda item: item["ep"], reverse=True)
        for item in rows:
            item["gain"] = item["ep"] - float(totals.get(pid, 0.0))
            item["rank"] = ranked.index(item["id"]) + 1 if item["id"] in ranked else selected_rank
        result[str(pid)] = {
            "name": player["name"], "position": POSITIONS[pos],
            "budget": round(budget, 1), "selected_rank": selected_rank,
            "total_ranked": len(ranked), "rows": rows,
        }
    return result


def build_context(*, cfg: Config, bootstrap: dict, teams: pd.DataFrame,
                  players: pd.DataFrame, ep_rows: pd.DataFrame, ep_grid: pd.DataFrame,
                  plan, entry: dict, current_squad: list[int], free_transfers: int,
                  bank: float, squad_value: float, event: dict,
                  transfer_advice, selling_price: dict[int, float],
                  scenarios=None, provenance: dict | None = None,
                  live: dict | None = None) -> dict:
    gw = int(event["id"])
    horizon_gws = list(ep_grid.columns)
    tz = ZoneInfo(cfg.get("notify", "timezone", default="Asia/Bangkok"))
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    hours_left = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds() / 3600)

    this_gw = ep_rows[ep_rows.gw == gw]
    ep_this = this_gw.groupby("player_id").ep.sum()
    ep_total = ep_grid.sum(axis=1)

    first = plan.gameweeks[0]
    id_to_name = players.name.to_dict()

    # ---- pitch -----------------------------------------------------------
    def card(pid: int, captain: int | None = None, vice: int | None = None) -> dict:
        rows = this_gw[this_gw.player_id == pid]
        captain = first.captain if captain is None else captain
        vice = first.vice if vice is None else vice
        return {
            "id": pid,
            "name": id_to_name.get(pid, str(pid)),
            "ep": float(ep_this.get(pid, 0.0)),
            "opponent": _opponent_label(rows, teams),
            "is_captain": pid == captain,
            "is_vice": pid == vice,
            "position": POSITIONS[int(players.element_type.get(pid, 3))],
        }

    xi_cards = [card(p) for p in first.xi]
    pitch_rows = [[c for c in xi_cards if c["position"] == pos] for pos in ("GKP", "DEF", "MID", "FWD")]
    pitch_rows = [row for row in pitch_rows if row]
    bench_gk = [card(p) for p in (first.bench_gk or [])]
    bench_outfield = [card(p) for p in (first.bench_outfield or [])]
    formation_counts = {
        POSITIONS[pos]: sum(int(players.element_type[p]) == pos for p in first.xi)
        for pos in (2, 3, 4)
    }
    formation = (f"{formation_counts['DEF']}-{formation_counts['MID']}-"
                 f"{formation_counts['FWD']}")

    scenario_cards = []
    plan_request_base = cfg.get("notify", "plan_request_url", default="") or ""
    for index, scenario in enumerate(scenarios or []):
        week = scenario.plan.gameweeks[0]
        cards = [card(p, week.captain, week.vice) for p in week.xi]
        counts = {pos: sum(int(players.element_type[p]) == pos for p in week.xi)
                  for pos in (2, 3, 4)}
        scenario_cards.append({
            "index": index, "transfers": scenario.transfers,
            "variant": scenario.variant,
            "recommended": scenario.recommended,
            "objective_gain": scenario.objective_gain,
            "next_gw_gain": scenario.next_gw_gain,
            "formation": f"{counts[2]}-{counts[3]}-{counts[4]}",
            "pitch_rows": [[c for c in cards if c["position"] == pos]
                           for pos in ("GKP", "DEF", "MID", "FWD")],
            "bench_gk": [card(p, week.captain, week.vice) for p in week.bench_gk],
            "bench_outfield": [card(p, week.captain, week.vice) for p in week.bench_outfield],
            "captain": id_to_name.get(week.captain, "—"),
            "moves": [{"out": id_to_name.get(o, str(o)),
                       "in": id_to_name.get(i, str(i))} for o, i in week.moves],
            "move_ids": [{"out": int(o), "in": int(i)} for o, i in week.moves],
            "select_url": (f"{plan_request_base}&title={quote(f'[select-plan] GW{gw} scenario {index}')}"
                           if plan_request_base else ""),
        })

    # ---- captain shortlist ----------------------------------------------
    cap_pool = (this_gw[this_gw.player_id.isin(first.xi)]
                .groupby("player_id")
                .agg(ep=("ep", "sum"), exp_goals=("exp_goals", "sum"),
                     exp_assists=("exp_assists", "sum"), ep_bonus=("ep_bonus", "sum"))
                .sort_values("ep", ascending=False).head(6))
    cap_max = float(cap_pool.ep.max() or 1.0)
    captains = []
    for pid, row in cap_pool.iterrows():
        captains.append({
            "name": id_to_name.get(pid, str(pid)),
            "team": teams.short_name.get(int(players.team.get(pid, 0)), "?"),
            "position": POSITIONS[int(players.element_type.get(pid, 3))],
            "opponent": _opponent_label(this_gw[this_gw.player_id == pid], teams),
            "ep": float(row.ep), "exp_goals": float(row.exp_goals),
            "exp_assists": float(row.exp_assists), "ep_bonus": float(row.ep_bonus),
            "bar": round(100 * float(row.ep) / cap_max),
        })

    # ---- one weak link, with ranked replacements -------------------------
    outlook_matches = int(cfg.get("planning", "outlook_matches", default=5))
    out_id = transfer_advice.out_id
    transfer_out = None
    if out_id is not None and out_id in players.index:
        outgoing = players.loc[out_id]
        transfer_out = {
            "name": outgoing["name"], "team": outgoing.team_name,
            "position": POSITIONS[int(outgoing.element_type)],
            "price": float(selling_price.get(out_id, outgoing.price)),
            "ep": float(transfer_advice.candidates[0].out_ep)
                   if transfer_advice.candidates else 0.0,
            "price_signal": str(outgoing.get("price_change_signal", "stable")),
        }

    transfer_candidates = []
    chosen_incoming = {incoming for _, incoming in first.moves}
    for rank, candidate in enumerate(transfer_advice.candidates, start=1):
        incoming = players.loc[candidate.in_id]
        fixture_labels = [
            f"GW{fx['gw']} {teams.short_name.get(fx['opponent'], '?')} "
            f"({'H' if fx['is_home'] else 'A'}) [{fx['fdr']}]"
            for fx in candidate.fixtures
        ]
        ownership = pd.to_numeric(incoming.selected_by_percent, errors="coerce")
        candidate_rows = ep_rows[ep_rows.player_id == candidate.in_id]
        component_cols = ["ep_minutes", "ep_goals", "ep_assists", "ep_clean_sheet",
                          "ep_defcon", "ep_bonus", "ep_saves", "ep_cards", "ep_conceded"]
        components = {name.removeprefix("ep_"): float(candidate_rows[name].head(outlook_matches).sum())
                      for name in component_cols if name in candidate_rows}
        transfer_candidates.append({
            "rank": rank, "name": incoming["name"], "team": incoming.team_name,
            "price": float(incoming.price),
            "ownership": 0.0 if pd.isna(ownership) else float(ownership),
            "availability": float(incoming.p_available),
            "starts": float(incoming.p_start),
            "ep_next": float(ep_this.get(candidate.in_id, 0.0)),
            "ep": candidate.in_ep, "gain": candidate.gain,
            "net_gain": candidate.net_gain, "fixtures": fixture_labels,
            "recommended": candidate.in_id in chosen_incoming,
            "price_signal": str(incoming.get("price_change_signal", "stable")),
            "price_risk": float(incoming.get("price_risk_score", 0.0)),
            "components": components,
        })

    # ---- this week's action ----------------------------------------------
    moves = [{"out": id_to_name.get(o, str(o)), "in_": id_to_name.get(i, str(i)),
              "in": id_to_name.get(i, str(i))} for o, i in first.moves]
    captain_name = id_to_name.get(first.captain, "—")
    if moves:
        if len(moves) == 1:
            headline = f"พิจารณา {moves[0]['out']} → {moves[0]['in_']} · กัปตัน {captain_name}"
        else:
            headline = f"พิจารณาย้าย {len(moves)} ตัว · กัปตัน {captain_name}"
        selected_scenario = next((s for s in scenario_cards if s["recommended"]), None)
        gain_text = (f"ดีกว่าแผนไม่ย้าย {selected_scenario['objective_gain']:.1f} คะแนนแผนสุทธิ"
                     if selected_scenario else transfer_advice.reason)
        detail = f"{gain_text}. คาดการณ์ {first.expected_points:.0f} แต้มใน GW{gw}"
        recommended = next((p for p in transfer_candidates if p["recommended"]), None)
        urgency = []
        if recommended and recommended["price_signal"] == "rising" and recommended["price_risk"] >= 0.5:
            urgency.append(f"{recommended['name']} มีแรงซื้อสูงและเสี่ยงขึ้นราคา")
        if transfer_out and transfer_out["price_signal"] == "falling":
            urgency.append(f"{transfer_out['name']} มีแรงขายสูงและเสี่ยงลงราคา")
        if urgency:
            detail += ". ราคา: " + " · ".join(urgency)
    else:
        headline = f"แนะนำให้เก็บ transfer · กัปตัน {captain_name}"
        detail = f"{transfer_advice.reason}. คาดการณ์ {first.expected_points:.0f} แต้มใน GW{gw}"
    action = {"headline": headline, "detail": detail, "moves": moves}

    clashes = []
    proposed = set(first.squad)
    for attacker in proposed:
        if attacker not in players.index or int(players.element_type[attacker]) not in (3, 4):
            continue
        opponents = set(this_gw[this_gw.player_id == attacker].opponent.astype(int).tolist())
        for defender in proposed:
            if defender not in players.index or int(players.element_type[defender]) not in (1, 2):
                continue
            if int(players.team[defender]) in opponents:
                clashes.append(f"{id_to_name[attacker]} เกมรุก พบแนวรับของ {id_to_name[defender]}")

    # ---- the horizon plan -------------------------------------------------
    max_ep = max((w.expected_points for w in plan.gameweeks), default=1.0) or 1.0
    plan_rows = []
    for w in plan.gameweeks:
        pairs = w.moves or list(zip(w.sells, w.buys))
        if pairs:
            summary = ", ".join(
                f"{id_to_name.get(o, o)} → {id_to_name.get(i, i)}" for o, i in pairs)
        else:
            summary = "เก็บ transfer ไว้"
        plan_rows.append({
            "gw": w.gw, "summary": summary, "hits": w.hits,
            "captain": id_to_name.get(w.captain, "—"),
            "ft": w.free_transfers_before, "ep": w.expected_points,
            "bar": round(100 * max(w.expected_points, 0) / max_ep),
        })

    # ---- fixture run for the clubs you own -------------------------------
    squad_clubs = sorted({int(players.team.get(p)) for p in first.squad if p in players.index})
    sched = ep_rows[["team", "gw", "opponent", "is_home", "fdr"]].drop_duplicates()
    fixture_grid = []
    for club in squad_clubs:
        cells, diffs = [], []
        for g in horizon_gws:
            games = sched[(sched.team == club) & (sched.gw == g)]
            cell = []
            for _, r in games.iterrows():
                cell.append({"label": f"{teams.short_name.get(int(r.opponent), '?')} "
                                      f"{'H' if r.is_home else 'A'}", "fdr": int(r.fdr)})
                diffs.append(int(r.fdr))
            cells.append({"games": cell})
        fixture_grid.append({
            "team": teams.name.get(club, "?"),
            "players": [{
                "id": p, "name": id_to_name.get(p, str(p)),
                "position": POSITIONS[int(players.element_type[p])],
            } for p in first.squad if p in players.index and int(players.team[p]) == club],
            "cells": cells,
            "avg": sum(diffs) / len(diffs) if diffs else 0.0,
        })
    fixture_grid.sort(key=lambda r: r["avg"])

    # ---- watch list -------------------------------------------------------
    flags, alerts = squad_alerts(players, first.squad, first.xi, first.captain)
    chips = chip_signals(cfg, ep_rows, players, first.squad, horizon_gws, plan)
    comparisons = player_comparisons(
        cfg, ep_rows, players, teams, first.squad, current_squad, bank,
        selling_price, transfer_advice)

    history = entry.get("__history__", {})
    last_gw_points = 0
    if history.get("current"):
        last_gw_points = history["current"][-1].get("points", 0)

    fetched_values = [v.get("fetched_at") for v in (provenance or {}).values()
                      if v.get("fetched_at")]
    built_iso = datetime.now(timezone.utc).isoformat()
    fetched_iso = max(fetched_values) if fetched_values else built_iso
    try:
        fetched_human = _thai_datetime(
            datetime.fromisoformat(fetched_iso).astimezone(tz))
    except ValueError:
        fetched_human = fetched_iso
    previous = None
    history_path = cfg.site_dir / "plan-history.json"
    if history_path.exists():
        try:
            rows = json.loads(history_path.read_text(encoding="utf-8"))
            previous = rows[-1] if rows else None
        except (OSError, ValueError, TypeError):
            previous = None
    signature = [{"out": m["out"], "in": m["in"]} for m in moves]
    if previous and previous.get("moves") == signature:
        delta = first.expected_points - float(previous.get("expected_points", first.expected_points))
        plan_change = f"แผนหลักยังเหมือนรายงานครั้งก่อน · EP GW นี้เปลี่ยน {delta:+.1f}"
    elif previous:
        old = previous.get("headline", "แผนเดิม")
        plan_change = f"แผนหลักเปลี่ยนจาก “{old}” หลังดึงข้อมูลและคำนวณทุก scenario ใหม่"
    else:
        plan_change = "นี่คือแผนฐานฉบับแรกที่บันทึกไว้"

    sources = sorted({v.get("source", "unknown") for v in (provenance or {}).values()})
    source_label = ({"network"} if set(sources) == {"network"} else set(sources))
    if source_label == {"network"}:
        data_status = "ข้อมูลสดจาก FPL"
    elif "offline" in sources:
        data_status = "ข้อมูลออฟไลน์ที่บันทึกไว้"
    elif "cache" in sources:
        data_status = "ข้อมูลแคช"
    else:
        data_status = "ไม่ทราบที่มาข้อมูล"

    return {
        "title": cfg.get("output", "title", default="FPL Assistant"),
        "gw": gw, "horizon": len(horizon_gws), "horizon_gws": horizon_gws,
        "captain_name": captain_name,
        "built_at": _thai_datetime(datetime.now(tz)), "built_at_iso": built_iso,
        "fetched_at_iso": fetched_iso, "fetched_at_human": fetched_human,
        "data_sources": sources, "data_status": data_status,
        "deadline_iso": deadline.isoformat(),
        "deadline_human": f"{int(hours_left // 24)}d {int(hours_left % 24)}h" if hours_left >= 24
                          else f"{int(hours_left)}h",
        "deadline_local": _thai_datetime(deadline.astimezone(tz)),
        "hours_left": hours_left,
        "entry": {
            "total_points": entry.get("summary_overall_points", 0),
            "rank_human": _rank(entry.get("summary_overall_rank")),
            "last_gw_points": last_gw_points,
        },
        "free_transfers": free_transfers, "bank": bank, "squad_value": squad_value,
        "this_gw_ep": first.expected_points,
        "action": action, "pitch_rows": pitch_rows,
        "bench_gk": bench_gk, "bench_outfield": bench_outfield,
        "formation": formation,
        "captains": captains, "transfer_out": transfer_out,
        "transfer_candidates": transfer_candidates,
        "outlook_matches": outlook_matches,
        "plan": plan_rows, "plan_notes": " ".join(plan.notes),
        "fixture_grid": fixture_grid, "flags": flags[:10], "alerts": alerts,
        "chip_signals": chips,
        "player_comparisons": comparisons,
        "scenario_cards": scenario_cards, "plan_change": plan_change,
        "clashes": sorted(set(clashes)), "live": live,
        "refresh_request_url": cfg.get("notify", "refresh_request_url", default="") or "",
        "site_url": (cfg.get("notify", "site_url", default="") or "").rstrip("/"),
        "remind_hours": [int(h) for h in cfg.get(
            "notify", "remind_hours_before", default=[24])],
        "model_version": MODEL_VERSION, "solver_status": plan.status,
    }


def render(context: dict, cfg: Config) -> Path:
    out_dir = cfg.site_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep render() useful for lightweight callers and older saved contexts.
    context.setdefault("built_at_iso", datetime.now(timezone.utc).isoformat())
    context.setdefault("fetched_at_iso", context["built_at_iso"])
    context.setdefault("fetched_at_human", context.get("built_at", context["built_at_iso"]))
    context.setdefault("data_sources", ["unknown"])
    context.setdefault("data_status", "ไม่ทราบที่มาข้อมูล")
    context.setdefault("scenario_cards", [])
    context.setdefault("plan_change", "")
    context.setdefault("clashes", [])
    context.setdefault("live", None)
    context.setdefault("refresh_request_url", "")
    html = _env().get_template("template.html").render(**context)
    index = out_dir / "index.html"
    index_tmp = out_dir / "index.html.tmp"
    index_tmp.write_text(html, encoding="utf-8")
    index_tmp.replace(index)

    for asset in ("manifest.webmanifest", "sw.js", "icon.svg",
                  "icon-180.png", "icon-512.png", "sports-ui.css"):
        src = WEB_DIR / asset
        if src.exists():
            shutil.copy2(src, out_dir / asset)

    # A machine-readable copy, for the notifier and for future backtesting.
    # The captain comes from the plan, not from the captain shortlist: the
    # shortlist is sorted by expected points and can disagree with the armband
    # the solver actually chose, and the alert must match the dashboard.
    summary = {
        "gw": context["gw"], "deadline": context["deadline_iso"],
        "deadline_local": context["deadline_local"],
        "headline": context["action"]["headline"], "detail": context["action"]["detail"],
        "captain": context["captain_name"],
        "moves": [{"out": m["out"], "in": m["in_"]} for m in context["action"]["moves"]],
        "alerts": context["alerts"],
        "expected_points": context["this_gw_ep"],
        "site_url": context["site_url"],
        "candidates": [{
            "rank": p["rank"], "name": p["name"], "price": p["price"],
            "ep": round(p["ep"], 2), "gain": round(p["gain"], 2),
            "fixtures": p["fixtures"],
        } for p in context["transfer_candidates"]],
        "built_at": context["built_at"], "built_at_iso": context["built_at_iso"],
        "fetched_at": context["fetched_at_iso"],
        "data_sources": context["data_sources"], "model_version": MODEL_VERSION,
        "plan_change": context["plan_change"], "scenarios": [{
            "transfers": s["transfers"], "variant": s["variant"],
            "recommended": s["recommended"],
            "objective_gain": round(s["objective_gain"], 3),
            "next_gw_gain": round(s["next_gw_gain"], 3), "moves": s["moves"],
            "move_ids": s["move_ids"],
        } for s in context["scenario_cards"]],
    }
    summary_path = out_dir / "summary.json"
    summary_tmp = out_dir / "summary.json.tmp"
    summary_tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_tmp.replace(summary_path)

    history_path = out_dir / "plan-history.json"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    except (OSError, ValueError, TypeError):
        history = []
    history.append({
        "built_at": context["built_at_iso"], "fetched_at": context["fetched_at_iso"],
        "gw": context["gw"], "model_version": MODEL_VERSION,
        "moves": [{"out": m["out"], "in": m["in_"]} for m in context["action"]["moves"]],
        "expected_points": round(float(context["this_gw_ep"]), 3),
        "headline": context["action"]["headline"],
    })
    history_tmp = out_dir / "plan-history.json.tmp"
    history_tmp.write_text(json.dumps(history[-100:], indent=2, ensure_ascii=False), encoding="utf-8")
    history_tmp.replace(history_path)
    return index
