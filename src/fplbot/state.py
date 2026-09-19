"""Persist an explicitly selected scenario without touching the FPL account."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def select_scenario(summary_path: Path, selected_path: Path, index: int) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scenarios = summary.get("scenarios") or []
    if index < 0 or index >= len(scenarios):
        raise ValueError(f"scenario {index} is not available")
    scenario = scenarios[index]
    payload = {
        "schema_version": 1,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "source_report_built_at": summary.get("built_at_iso"),
        "gw": int(summary["gw"]),
        "scenario_index": index,
        "transfers": int(scenario["transfers"]),
        "moves": scenario.get("moves", []),
        "move_ids": scenario.get("move_ids", []),
    }
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected_path.with_suffix(selected_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(selected_path)
    return payload


def load_selected(path: Path, gw: int) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if int(payload.get("gw", -1)) == int(gw) else None
    except (OSError, ValueError, TypeError):
        return None
