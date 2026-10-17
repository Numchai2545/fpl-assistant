"""Freeze honest pre-deadline projections and score them after matches finish.

The projection timestamp is part of the data contract: a record created after
the deadline is rejected, preventing future information from leaking into a
backtest. Outcomes are fetched only for the small evaluated set, never for the
entire league.
"""
from __future__ import annotations

import json
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import Config, POSITIONS


def write_projection(cfg: Config, event: dict, players: pd.DataFrame,
                     ep_rows: pd.DataFrame, plan, advice, *,
                     model_version: str, provenance: dict | None = None) -> Path | None:
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
        fpl_ep = float(pd.to_numeric(p.get("ep_next", 0.0), errors="coerce"))
        rows.append({
            "player_id": pid, "name": str(p["name"]),
            "position": POSITIONS[int(p.element_type)],
            "model_ep": round(float(this_gw.get(pid, 0.0)), 4),
            "fpl_ep_next": round(fpl_ep if math.isfinite(fpl_ep) else 0.0, 4),
            "availability_band": "flagged" if float(p.get("p_available", 1.0)) < 1 else "fit",
        })
    config_hash = hashlib.sha256(json.dumps(
        {key: cfg.get(key, default={}) for key in ("model", "planning", "strategy")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = {
        "schema_version": 2, "model_version": model_version, "gw": gw,
        "config_hash": config_hash,
        "season": str(deadline.year if deadline.month >= 7 else deadline.year - 1),
        "provenance": provenance or {},
        "built_at": now.isoformat(), "deadline": deadline.isoformat(), "players": rows,
    }
    out_dir = cfg.data_dir / "projections"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    path = out_dir / f"gw{gw}-{stamp}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    return path


def _eligible_projection_files(cfg: Config, now: datetime) -> list[Path]:
    chosen: dict[tuple, tuple[datetime, Path]] = {}
    for path in sorted((cfg.data_dir / "projections").glob("gw*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            built = datetime.fromisoformat(payload["built_at"])
            deadline = datetime.fromisoformat(payload["deadline"])
            gw = int(payload["gw"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if built.tzinfo is None or deadline.tzinfo is None or built >= deadline or deadline >= now:
            continue
        key = (str(payload.get("season", "legacy")), gw,
               str(payload.get("model_version", "unknown")),
               str(payload.get("config_hash", "legacy")))
        if key not in chosen or built > chosen[key][0]:
            chosen[key] = built, path
    return [chosen[key][1] for key in sorted(chosen)]


def evaluate_projections(cfg: Config, history_loader: Callable[[int], dict],
                         *, now: datetime | None = None,
                         fixtures: list[dict] | None = None,
                         season: str | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    records = []
    history_cache: dict[int, dict] = {}
    files = _eligible_projection_files(cfg, now)
    for path in files:
        projection = json.loads(path.read_text(encoding="utf-8"))
        projected_deadline = datetime.fromisoformat(projection["deadline"])
        projected_season = str(projection.get("season") or (
            projected_deadline.year if projected_deadline.month >= 7 else projected_deadline.year - 1))
        if season is not None and projected_season != season:
            continue
        gw = int(projection["gw"])
        if fixtures is not None:
            scheduled = [f for f in fixtures if int(f.get("event") or 0) == gw]
            # A deadline passing is not an outcome. Wait until every scheduled
            # fixture is final so partial live points never enter calibration.
            if not scheduled or any(not f.get("finished") for f in scheduled):
                continue
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
            records[-1]["config_hash"] = projection.get("config_hash", "legacy")
            records[-1]["season"] = projected_season

    if not records:
        return {"schema_version": 1, "generated_at": now.isoformat(),
                "gameweeks": 0, "players": 0, "metrics": {}, "rows": []}
    frame = pd.DataFrame(records)
    groups = []
    if "availability_band" not in frame:
        frame["availability_band"] = "legacy"
    for keys, band in frame.groupby(["season", "model_version", "config_hash", "position", "availability_band"], dropna=False):
        error = pd.to_numeric(band.model_ep) - pd.to_numeric(band.actual_points)
        fpl_error = pd.to_numeric(band.fpl_ep_next) - pd.to_numeric(band.actual_points)
        groups.append(dict(zip(("season", "model_version", "config_hash", "position", "availability_band"), keys)) | {
            "samples": len(band), "gameweeks": int(band.gw.nunique()),
            "model_mae": round(float(error.abs().mean()), 4),
            "model_bias": round(float(error.mean()), 4),
            "fpl_mae": round(float(fpl_error.abs().mean()), 4)})
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
        "metrics": metrics, "groups": groups, "rows": records,
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
