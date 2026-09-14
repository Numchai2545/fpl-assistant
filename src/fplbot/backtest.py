"""Freeze honest pre-deadline projections and score them after matches finish.

The projection timestamp is part of the data contract: a record created after
the deadline is rejected, preventing future information from leaking into a
backtest. Outcomes are fetched only for the small evaluated set, never for the
entire league.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import Config, POSITIONS


def write_projection(cfg: Config, event: dict, players: pd.DataFrame,
                     ep_rows: pd.DataFrame, plan, advice, *,
                     model_version: str) -> Path | None:
    now = datetime.now(timezone.utc)
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    if now >= deadline or not plan.gameweeks:
        return None
    gw = int(event["id"])
    first = plan.gameweeks[0]
    priority = list(first.squad)
    priority += [c.in_id for c in advice.candidates]
    priority += [first.captain, first.vice]
    limit = int(cfg.get("backtest", "max_players_per_gameweek", default=30))
    ids = list(dict.fromkeys(int(pid) for pid in priority if pid in players.index))[:limit]
    this_gw = ep_rows[ep_rows.gw == gw].groupby("player_id").ep.sum()
    rows = []
    for pid in ids:
        p = players.loc[pid]
        rows.append({
            "player_id": pid, "name": str(p["name"]),
            "position": POSITIONS[int(p.element_type)],
            "model_ep": round(float(this_gw.get(pid, 0.0)), 4),
            "fpl_ep_next": round(float(pd.to_numeric(p.get("ep_next", 0.0), errors="coerce") or 0.0), 4),
        })
    payload = {
        "schema_version": 1, "model_version": model_version, "gw": gw,
        "built_at": now.isoformat(), "deadline": deadline.isoformat(), "players": rows,
    }
    out_dir = cfg.data_dir / "projections"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"gw{gw}-{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _eligible_projection_files(cfg: Config, now: datetime) -> list[Path]:
    chosen: dict[int, tuple[datetime, Path]] = {}
    for path in sorted((cfg.data_dir / "projections").glob("gw*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            built = datetime.fromisoformat(payload["built_at"])
            deadline = datetime.fromisoformat(payload["deadline"])
            gw = int(payload["gw"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if built >= deadline or deadline >= now:
            continue
        if gw not in chosen or built > chosen[gw][0]:
            chosen[gw] = built, path
    return [chosen[gw][1] for gw in sorted(chosen)]


def evaluate_projections(cfg: Config, history_loader: Callable[[int], dict],
                         *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    records = []
    history_cache: dict[int, dict] = {}
    files = _eligible_projection_files(cfg, now)
    for path in files:
        projection = json.loads(path.read_text(encoding="utf-8"))
        gw = int(projection["gw"])
        for row in projection.get("players", []):
            pid = int(row["player_id"])
            if pid not in history_cache:
                try:
                    history_cache[pid] = history_loader(pid)
                except Exception:
                    history_cache[pid] = {}
            matches = [h for h in history_cache[pid].get("history", [])
                       if int(h.get("round") or 0) == gw]
            if not matches:
                continue
            actual = sum(float(h.get("total_points") or 0.0) for h in matches)
            records.append({**row, "gw": gw, "actual_points": actual,
                            "model_version": projection.get("model_version", "unknown")})

    if not records:
        return {"schema_version": 1, "generated_at": now.isoformat(),
                "gameweeks": 0, "players": 0, "metrics": {}, "rows": []}
    frame = pd.DataFrame(records)
    metrics = {}
    for label, column in (("model", "model_ep"), ("fpl", "fpl_ep_next")):
        error = pd.to_numeric(frame[column]) - pd.to_numeric(frame.actual_points)
        metrics[label] = {
            "mae": round(float(error.abs().mean()), 4),
            "bias": round(float(error.mean()), 4),
            "mean_prediction": round(float(pd.to_numeric(frame[column]).mean()), 4),
        }
    return {
        "schema_version": 1, "generated_at": now.isoformat(),
        "gameweeks": int(frame.gw.nunique()), "players": len(frame),
        "metrics": metrics, "rows": records,
    }


def write_report(cfg: Config, result: dict) -> Path:
    out_dir = cfg.data_dir / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "latest.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def format_summary(result: dict) -> str:
    model = result["metrics"]["model"]
    fpl = result["metrics"]["fpl"]
    return (f"Backtest {result['gameweeks']} GW / {result['players']} player-GW\n"
            f"  model MAE {model['mae']:.2f}, bias {model['bias']:+.2f}\n"
            f"  FPL    MAE {fpl['mae']:.2f}, bias {fpl['bias']:+.2f}")
