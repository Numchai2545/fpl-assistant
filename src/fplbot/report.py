"""Render the dashboard: a single static HTML file plus PWA assets.

The output is deliberately self-contained and offline-capable. Drop the folder
on GitHub Pages and "Add to Home Screen" turns it into an app icon; open the
file straight from disk and it still works.
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import POSITIONS, Config

MODEL_VERSION = "0.1.0"
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(WEB_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _rank(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _opponent_label(rows: pd.DataFrame, teams: pd.DataFrame) -> str:
    if rows.empty:
        return "blank"
    parts = []
    for _, r in rows.iterrows():
        short = teams.short_name.get(int(r.opponent), "?")
        parts.append(f"{short} ({'H' if r.is_home else 'A'})")
    return " + ".join(parts)


def build_context(*, cfg: Config, bootstrap: dict, teams: pd.DataFrame,
                  players: pd.DataFrame, ep_rows: pd.DataFrame, ep_grid: pd.DataFrame,
                  plan, entry: dict, current_squad: list[int], free_transfers: int,
                  bank: float, squad_value: float, event: dict) -> dict:
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
    def card(pid: int) -> dict:
        rows = this_gw[this_gw.player_id == pid]
        return {
            "name": id_to_name.get(pid, str(pid)),
            "ep": float(ep_this.get(pid, 0.0)),
            "opponent": _opponent_label(rows, teams),
            "is_captain": pid == first.captain,
            "is_vice": pid == first.vice,
            "position": POSITIONS[int(players.element_type.get(pid, 3))],
        }

    xi_cards = [card(p) for p in first.xi]
    pitch_rows = [[c for c in xi_cards if c["position"] == pos] for pos in ("GKP", "DEF", "MID", "FWD")]
    pitch_rows = [row for row in pitch_rows if row]
    bench = [card(p) for p in first.bench]

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

    # ---- transfer targets -------------------------------------------------
    board = players.assign(ep_total=ep_total, ep_next=ep_this).dropna(subset=["ep_total"])
    board = board[(board.p_available > 0.25) & (~board.index.isin(current_squad))]
    board["ep_per_m"] = board.ep_total / board.price.clip(lower=3.8)
    targets: dict[str, list[dict]] = {}
    for code, label in POSITIONS.items():
        top = board[board.element_type == code].nlargest(10, "ep_total")
        targets[label] = [{
            "name": r.name, "team": r.team_name, "price": float(r.price),
            "ownership": float(pd.to_numeric(r.selected_by_percent, errors="coerce") or 0.0),
            "price_trend": r.price_trend, "ep_next": float(r.ep_next or 0.0),
            "ep_total": float(r.ep_total), "ep_per_m": float(r.ep_per_m),
        } for r in top.itertuples()]

    # ---- sell candidates --------------------------------------------------
    owned = players.loc[[p for p in current_squad if p in players.index]].assign(
        ep_total=ep_total, ep_next=ep_this)
    owned["ep_per_m"] = owned.ep_total / owned.price.clip(lower=3.8)
    sells = []
    for r in owned.nsmallest(6, "ep_total").itertuples():
        reasons = []
        if r.p_available < 0.75:
            reasons.append("availability doubt")
        if r.p_start < 0.6:
            reasons.append(f"starts {r.p_start:.0%} of the time")
        if r.price_trend == "falling":
            reasons.append("price falling")
        if not reasons:
            reasons.append("fixtures and form both below the alternatives")
        sells.append({
            "name": r.name, "team": r.team_name,
            "position": POSITIONS[int(r.element_type)], "price": float(r.price),
            "ep_total": float(r.ep_total or 0.0), "ep_per_m": float(r.ep_per_m or 0.0),
            "reason": ", ".join(reasons),
        })

    # ---- this week's action ----------------------------------------------
    moves = [{"out": id_to_name.get(o, str(o)), "in_": id_to_name.get(i, str(i))}
             for o, i in zip(first.sells, first.buys)]
    captain_name = id_to_name.get(first.captain, "—")
    if moves:
        n = len(moves)
        headline = f"Make {n} transfer{'s' if n > 1 else ''}, captain {captain_name}"
        free_used = n - first.hits
        if first.hits:
            detail = (f"{free_used} free, {first.hits} paid — costs "
                      f"{first.hits * 4} points. ")
        else:
            detail = (f"Uses {free_used} of your {free_transfers} free "
                      f"transfer{'s' if free_transfers > 1 else ''}, no hit. ")
        detail += f"Projected {first.expected_points:.0f} points for GW{gw}."
    else:
        headline = f"No transfer this week — captain {captain_name}"
        nxt = next((w for w in plan.gameweeks[1:] if w.buys), None)
        detail = (f"Bank the free transfer; the plan spends it in GW{nxt.gw}. "
                  if nxt else "Bank the free transfer. ")
        detail += f"Projected {first.expected_points:.0f} points for GW{gw}."
    action = {"headline": headline, "detail": detail, "moves": moves}

    # ---- the horizon plan -------------------------------------------------
    max_ep = max((w.expected_points for w in plan.gameweeks), default=1.0) or 1.0
    plan_rows = []
    for w in plan.gameweeks:
        if w.buys:
            summary = ", ".join(
                f"{id_to_name.get(o, o)} → {id_to_name.get(i, i)}"
                for o, i in zip(w.sells, w.buys))
        else:
            summary = "Hold — bank the transfer"
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
            "team": teams.name.get(club, "?"), "cells": cells,
            "avg": sum(diffs) / len(diffs) if diffs else 0.0,
        })
    fixture_grid.sort(key=lambda r: r["avg"])

    # ---- watch list -------------------------------------------------------
    flags = []
    for pid in first.squad:
        if pid not in players.index:
            continue
        p = players.loc[pid]
        if p.p_available < 0.5:
            flags.append({"tag": "availability", "severity": "high", "name": p["name"],
                          "detail": (p.get("news") or "flagged by FPL").strip()})
        elif p.p_available < 1.0:
            flags.append({"tag": "doubt", "severity": "medium", "name": p["name"],
                          "detail": (p.get("news") or "listed as doubtful").strip()})
        elif p.p_start < 0.55 and p.minutes > 0:
            flags.append({"tag": "rotation", "severity": "medium", "name": p["name"],
                          "detail": f"started only {p.p_start:.0%} of his club's matches"})
        if int(p.get("yellow_cards", 0)) >= 4:
            flags.append({"tag": "suspension", "severity": "medium", "name": p["name"],
                          "detail": f"{int(p.yellow_cards)} yellow cards — one away from a ban"})

    history = entry.get("__history__", {})
    last_gw_points = 0
    if history.get("current"):
        last_gw_points = history["current"][-1].get("points", 0)

    return {
        "title": cfg.get("output", "title", default="FPL Assistant"),
        "gw": gw, "horizon": len(horizon_gws), "horizon_gws": horizon_gws,
        "built_at": datetime.now(tz).strftime("%d %b %Y, %H:%M"),
        "deadline_iso": deadline.isoformat(),
        "deadline_human": f"{int(hours_left // 24)}d {int(hours_left % 24)}h" if hours_left >= 24
                          else f"{int(hours_left)}h",
        "deadline_local": deadline.astimezone(tz).strftime("%a %d %b, %H:%M %Z"),
        "hours_left": hours_left,
        "entry": {
            "total_points": entry.get("summary_overall_points", 0),
            "rank_human": _rank(entry.get("summary_overall_rank")),
            "last_gw_points": last_gw_points,
        },
        "free_transfers": free_transfers, "bank": bank, "squad_value": squad_value,
        "this_gw_ep": first.expected_points,
        "action": action, "pitch_rows": pitch_rows, "bench": bench,
        "captains": captains, "targets": targets, "sells": sells,
        "plan": plan_rows, "plan_notes": " ".join(plan.notes),
        "fixture_grid": fixture_grid, "flags": flags[:10],
        "model_version": MODEL_VERSION, "solver_status": plan.status,
    }


def render(context: dict, cfg: Config) -> Path:
    out_dir = cfg.site_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    html = _env().get_template("template.html").render(**context)
    index = out_dir / "index.html"
    index.write_text(html, encoding="utf-8")

    for asset in ("manifest.webmanifest", "sw.js", "icon.svg",
                  "icon-180.png", "icon-512.png"):
        src = WEB_DIR / asset
        if src.exists():
            shutil.copy2(src, out_dir / asset)

    # A machine-readable copy, for the notifier and for future backtesting.
    (out_dir / "summary.json").write_text(json.dumps({
        "gw": context["gw"], "deadline": context["deadline_iso"],
        "headline": context["action"]["headline"], "detail": context["action"]["detail"],
        "captain": context["captains"][0]["name"] if context["captains"] else None,
        "expected_points": context["this_gw_ep"],
        "plan": context["plan"], "built_at": context["built_at"],
    }, indent=2), encoding="utf-8")
    return index
